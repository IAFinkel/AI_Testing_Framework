import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

ENV_PATH = find_dotenv()
if not ENV_PATH:
    raise RuntimeError(
        "Could not locate a .env file by walking up from this script's location. "
        "Make sure a .env file exists at the project root."
    )
load_dotenv(ENV_PATH)
PROJECT_ROOT = Path(ENV_PATH).resolve().parent

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams

# Reuses the same OpenRouter setup as the other eval scripts, so the judge
# model ToolCorrectnessMetric constructs internally doesn't require a
# separate OpenAI key — it authenticates with OPENROUTER_API_KEY from the
# shared .env.
JUDGE_MODEL = OpenRouterModel(
    model=os.environ.get("OPENROUTER_MODEL", "alibaba/tongyi-deepresearch-30b-a3b:free"),
)


# ---------- FUNCTION: Load JSON ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- FUNCTION: Extract Tool Calls from JSON ----------
def extract_tool_calls(data, source="expected"):
    """
    Extracts tool invocations from the router section (which tool was
    chosen and with what input) and matches each one to its corresponding
    entry in the tools section (the output that call actually produced).

    FIX: this previously paired invocations to executions purely by
    POSITION (zip_longest), which breaks the moment the two lists have
    different lengths for a legitimate reason — e.g. a routing-only action
    that was given a synthetic tool name (no output of its own) sitting
    alongside real tool calls that do have outputs. From that point on,
    every later invocation would get paired with the wrong execution's
    output, corrupting scoring silently.

    This mirrors the fix already used in task_completion's
    extract_tool_calls_with_positions: match each execution to the
    EARLIEST still-unfilled invocation with the same tool name, in order,
    rather than assuming position alone tells you which output belongs to
    which call. This also correctly handles the same tool being invoked
    more than once in a trace, and leaves an invocation with no matching
    execution (like a routing-only step) with an empty output instead of
    stealing a different tool's result.
    """
    invocations = [r for r in data.get("router", []) if r.get("action") == "Invoke Tool"]
    executions = data.get("tools", [])

    pending = []
    for invocation in invocations:
        pending.append({
            "name": invocation.get("tool", ""),
            "input_parameters": invocation.get("tool_input", {}) or {},
            "output": None,  # None = not yet filled; distinct from a genuine empty-string output
        })

    unmatched_executions = []
    for execution in executions:
        exe_name = execution.get("tool", "")
        exe_output = execution.get("output", "")

        match = next((e for e in pending if e["name"] == exe_name and e["output"] is None), None)
        if match is not None:
            match["output"] = str(exe_output)
        else:
            # An execution with no available matching invocation — surface
            # it as its own entry rather than silently dropping it.
            unmatched_executions.append({
                "name": exe_name,
                "input_parameters": {},
                "output": str(exe_output),
            })

    tool_calls = []
    for entry in pending + unmatched_executions:
        tool_calls.append(
            ToolCall(
                name=entry["name"],
                description=f"{source.capitalize()} tool invocation: {entry['name']}",
                input_parameters=entry["input_parameters"],
                output=entry["output"] if entry["output"] is not None else "",
            )
        )

    return tool_calls


# ---------- FUNCTION: Evaluate Tool Correctness ----------
def evaluate_tool_correctness(predicted_path, ground_truth_path, output_path):
    predicted = load_json(predicted_path)
    ground_truth = load_json(ground_truth_path)

    actual_tools = extract_tool_calls(predicted, "actual")
    expected_tools = extract_tool_calls(ground_truth, "expected")

    if not actual_tools:
        print("⚠️  No tool invocations extracted from the predicted trace — check the 'router'/'tools' sections.")
    if not expected_tools:
        print("⚠️  No tool invocations extracted from the ground-truth trace — check the 'router'/'tools' sections.")

    test_case = LLMTestCase(
        input="Agent system executed several tools as part of a plan.",
        actual_output="Actual tool invocations and their outputs from the agent system.",
        expected_output="Expected tool invocations and outputs based on the planned workflow.",
        tools_called=actual_tools,      # what actually happened
        expected_tools=expected_tools,  # what should have happened
    )

    # FIX: without evaluation_params, ToolCorrectnessMetric only compares
    # tool NAMES — the input_parameters and output extracted above would be
    # ignored entirely. Including both here means the score actually
    # reflects whether the right tool was called with the right arguments
    # and produced the right result, not just whether the name matched.
    metric = ToolCorrectnessMetric(
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS, ToolCallParams.OUTPUT],
        include_reason=True,
        model=JUDGE_MODEL,
        # should_consider_ordering=True,   # uncomment to also require call order to match
        # should_exact_match=True,          # uncomment to require an identical set of tool calls
    )
    metric.measure(test_case)

    try:
        success = metric.is_successful()
    except Exception:
        success = None

    print("\n✅ Agent Tool Evaluation Complete")
    print("----------------------------------------------------")
    print(f"Score: {metric.score}")
    print(f"Reason: {metric.reason}")
    print(f"Success (score >= threshold): {success}")
    print(f"Actual tools called:   {[tc.name for tc in actual_tools]}")
    print(f"Expected tools called: {[tc.name for tc in expected_tools]}")

    summary = {
        "score": metric.score,
        "reason": metric.reason,
        "threshold": metric.threshold,
        "success": success,
        "actual_tools_called": [tc.name for tc in actual_tools],
        "expected_tools": [tc.name for tc in expected_tools],
    }

    # Same pattern as planner/precision/recall: caller-provided output path,
    # parent directory created if missing, timestamp appended so each run
    # keeps its own report instead of overwriting the previous one.
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    output_path = str(output_path)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Report saved to {output_path}")

    return summary


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepEval ToolCorrectnessMetric evaluation of an agent's tool calls against a ground-truth reference."
    )
    parser.add_argument(
        "--predicted",
        required=True,
        help="Path to the extracted/normalized agent trace JSON (output of data_extract.py)",
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        help="Path to the ground-truth reference JSON",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "report" / "tool_correctness_eval_report.json"),
        help=f"Path to write the evaluation report JSON (default: {PROJECT_ROOT / 'report' / 'tool_correctness_eval_report.json'})",
    )
    return parser.parse_args()


# ---------- MAIN ----------
if __name__ == "__main__":
    args = parse_args()
    evaluate_tool_correctness(
        predicted_path=args.predicted,
        ground_truth_path=args.ground_truth,
        output_path=args.output,
    )