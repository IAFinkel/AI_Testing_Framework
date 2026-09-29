import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv

ENV_PATH = find_dotenv()
if not ENV_PATH:
    raise RuntimeError(
        "Could not locate a .env file by walking up from this script's location. "
        "Make sure a .env file exists at the project root."
    )
load_dotenv(ENV_PATH)
PROJECT_ROOT = Path(ENV_PATH).resolve().parent

# DeepEval's own troubleshooting guidance for runs that appear to hang/time
# out against a slower provider is to increase the per-task time budget and
# REDUCE the number of internal retries — stacking more retries on top (as
# our outer loop below does) multiplies total wait time instead of fixing
# it. Must be set before deepeval.metrics is imported, since it reads these
# at import time. Override by setting the same names in the shared .env.
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_RETRY_MAX_ATTEMPTS", "2")

from deepeval.metrics import TaskCompletionMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase, ToolCall

# Reuses the same OpenRouter setup as the other eval scripts, so this
# doesn't require a separate OpenAI key.
JUDGE_MODEL = OpenRouterModel(
    model=os.environ.get("OPENROUTER_MODEL", ""),
)

DEFAULT_THRESHOLD = 0.5  # matches DeepEval's own TaskCompletionMetric default


# ---------- HELPERS: load / extract ----------
def load_agent_history(path: Path) -> List[Dict[str, Any]]:
    """
    Loads a raw agent trace file and returns its flat event list.

    FIX: previously assumed the JSON was always already a bare top-level
    list (true for TC3-style traces), which broke on a dict-wrapped trace
    like TC1's {"id": ..., "type": "chat_history", "steps": [...]}.
    Unwraps the same way data_extract.py's extract_agent_data does, rather
    than requiring every raw file to already be a plain list.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        unwrapped = data.get("steps") or data.get("history")
        if unwrapped is None:
            if {"planner", "router", "tools"} <= data.keys():
                # This is the pre-structured shape used by
                # planner/precision/recall/tool-correctness (e.g. TC2-style
                # traces authored directly in normalized form). It has no
                # recorded human turn and no distinct final answer to the
                # user for either of those to be judged against — task
                # completion's whole premise (did the response accomplish
                # what the user asked) has nothing to work with here. This
                # isn't a parsing failure to fix; it's a real limitation of
                # this input shape for this specific metric.
                raise RuntimeError(
                    "This file is a pre-structured planner/router/tools trace "
                    "with no recorded human turn and no distinct final answer "
                    "to the user. Task completion can't be scored without "
                    "both — it's a valid input for the planner/precision/"
                    "recall/tool-correctness scripts, but not for this one."
                )
            raise RuntimeError(
                "Agent history JSON is a dict but has neither a 'steps' nor "
                "'history' key containing the event list."
            )
        return unwrapped

    return data


def extract_tool_calls_with_positions(history: List[Dict[str, Any]]):
    """
    Returns list of dicts {toolcall: ToolCall, invoke_index: int, output_index: Optional[int]}
    where invoke_index is index of agent_action step, output_index is index of tool_output step (if present).
    If tool_output appears before agent_action, it's treated as output-only with invoke_index=None.

    Matches each tool_output to the earliest still-unfilled pending invocation
    of the same tool name, so a tool invoked more than once in a trace gets
    each output correctly attached to its own call rather than all outputs
    collapsing onto the first invocation found by name.
    """
    tool_entries = []
    pending = []

    for idx, step in enumerate(history):
        ttype = step.get("type")
        if ttype == "agent_action" and step.get("tool"):
            tc = ToolCall(
                name=step.get("tool"),
                description=step.get("log") or f"Invoked by {step.get('agent')}",
                input_parameters=step.get("tool_input", {}) or {},
                output=""  # fill once we see tool_output
            )
            pending.append({"toolcall": tc, "invoke_index": idx, "output_index": None})

        elif ttype == "tool_output" and step.get("tool"):
            tool_name = step.get("tool")
            output_val = step.get("output", "")
            # match earliest pending invocation with same tool name lacking output
            match_idx = None
            for i, entry in enumerate(pending):
                if entry["toolcall"].name == tool_name and (entry["toolcall"].output is None or entry["toolcall"].output == ""):
                    match_idx = i
                    break
            if match_idx is not None:
                entry = pending.pop(match_idx)
                if isinstance(output_val, (dict, list)):
                    entry["toolcall"].output = json.dumps(output_val, ensure_ascii=False)
                else:
                    entry["toolcall"].output = str(output_val)
                entry["output_index"] = idx
                tool_entries.append(entry)
            else:
                # output-only (no preceding invocation)
                tc = ToolCall(
                    name=tool_name,
                    description=f"Output-only for {tool_name}",
                    input_parameters={},
                    output=json.dumps(output_val, ensure_ascii=False) if isinstance(output_val, (dict, list)) else str(output_val),
                )
                tool_entries.append({"toolcall": tc, "invoke_index": None, "output_index": idx})

        # FIX: TC1-style tool_call events carry BOTH input and output on a
        # single event, unlike TC3-style agent_action + tool_output pairs.
        # Previously unhandled here entirely — this event type matched no
        # branch, so a TC1-style trace's tool calls were silently invisible
        # to task completion scoring (num_tools_used always came out 0),
        # even though the same event type was already fixed in
        # data_extract.py. Both input and output are already present, so no
        # pending/matching step is needed — build the complete ToolCall in
        # one pass.
        elif ttype == "tool_call":
            tool_name = step.get("tool") or step.get("id")
            output_val = step.get("output") or step.get("content")
            tc = ToolCall(
                name=tool_name,
                description=step.get("description") or f"Invoked by {step.get('agent')}",
                input_parameters=step.get("input", {}) or {},
                output=json.dumps(output_val, ensure_ascii=False) if isinstance(output_val, (dict, list)) else str(output_val),
            )
            tool_entries.append({"toolcall": tc, "invoke_index": idx, "output_index": idx})

    # any remaining pending invocations without outputs: include them with empty output
    for entry in pending:
        tool_entries.append(entry)

    def sort_key(e):
        return (e["invoke_index"] if e["invoke_index"] is not None else (e["output_index"] if e["output_index"] is not None else float("inf")))
    tool_entries.sort(key=sort_key)
    return tool_entries


def find_human_indices(history: List[Dict[str, Any]]) -> List[int]:
    return [i for i, s in enumerate(history) if s.get("type") == "human"]


def find_agent_answer_in_window(history: List[Dict[str, Any]], start_idx: int, end_idx: int) -> Optional[str]:
    """
    Finds the agent's final answer (agent_finish/agent_final/ai) in
    history[start_idx:end_idx]. Returns the content string or None.

    FIX: this previously returned the FIRST matching step in the window,
    which is wrong whenever a turn contains more than one such step — e.g.
    TC1-style traces use "ai" for both intermediate narration ("Let me use
    my tool...") and the actual final answer. Confirmed in practice: on a
    real TC1 run, this picked up the narration step instead of the real
    itinerary, and the judge (correctly, given that wrong input) scored it
    0.1 for not containing an itinerary. Scanning in reverse and returning
    the LAST match instead assumes the final answer is the last
    answer-typed step in a turn — true for every trace format seen in this
    project so far (TC2/TC3 only ever have one such step per turn anyway,
    so this is a no-op change for those), but worth re-checking if a future
    trace format's final answer isn't reliably the last one.
    """
    agent_answer_types = ("agent_finish", "agent_final", "ai")
    for j in range(end_idx - 1, start_idx - 1, -1):
        s = history[j]
        if s.get("type") in agent_answer_types and s.get("content"):
            return s["content"]
    return None


def tool_entries_for_window(tool_entries, window_start, window_end):
    """
    Returns a list of ToolCall objects whose invoke_index or output_index lies inside [start, end).
    """
    selected = []
    for e in tool_entries:
        inv = e.get("invoke_index")
        out = e.get("output_index")
        if (inv is not None and window_start <= inv < window_end) or (out is not None and window_start <= out < window_end):
            selected.append(e["toolcall"])
    return selected


def build_conversation_context(history: List[Dict[str, Any]], up_to_index: int) -> str:
    """
    Builds a transcript of prior USER turns up to and including up_to_index,
    so the judge sees what the user has said before (resolving references
    like a product named two turns back) without also seeing the agent's
    own past replies.

    FIX: an earlier version included the assistant's past replies too. That
    caused a real regression — a later turn's judge call would see, e.g.,
    "Assistant: order placed successfully" from a prior, already-scored
    turn, and mistakenly attribute that completed action to the CURRENT
    turn's outcome (a simple address change got scored as if it also
    placed an unrequested order). Dropping assistant replies from the
    context removes that specific confusion, while keeping the part that
    actually helped: the judge still knows what the user asked for earlier.
    """
    lines = []
    for step in history[: up_to_index + 1]:
        if step.get("type") == "human" and step.get("content"):
            lines.append(f"User: {step['content']}")
    return "\n".join(lines)


# ---------- MAIN: Build and Run Task Completion Eval ----------
def run_task_completion_eval(history_path, output_path, threshold=DEFAULT_THRESHOLD):
    history = load_agent_history(history_path)
    if not isinstance(history, list):
        raise RuntimeError("Agent history JSON must be a top-level list of steps.")

    tool_entries = extract_tool_calls_with_positions(history)
    human_indices = find_human_indices(history)
    total_humans = len(human_indices)

    results = []

    for idx_pos, h_idx in enumerate(human_indices):
        # window: from human step +1 up to next human step (or end)
        start = h_idx + 1
        end = human_indices[idx_pos + 1] if idx_pos + 1 < total_humans else len(history)

        human_text = history[h_idx].get("content", "")
        agent_answer = find_agent_answer_in_window(history, start, end)
        tools_in_window = tool_entries_for_window(tool_entries, start, end)
        conversation_context = build_conversation_context(history, h_idx)

        # TaskCompletionMetric is referenceless (no expected_output is used
        # in its scoring), so unlike the planner/precision/recall scripts,
        # there's no ground-truth field to build here.
        #
        # `input` carries the conversation so far (context that can resolve
        # ambiguity — e.g. a product named two turns back), while `task`
        # stays the clean, isolated request for this turn. Keeping them
        # separate avoids muddying what the judge is meant to grade (the
        # current request) with what's just background for interpreting it.
        test_case = LLMTestCase(
            input=conversation_context,
            actual_output=agent_answer or "",
            tools_called=tools_in_window,  # may be empty []
        )

        # `task` is passed explicitly since we already know the user's exact
        # request — no need to let DeepEval infer it from scratch.
        metric = TaskCompletionMetric(
            threshold=threshold,
            model=JUDGE_MODEL,
            include_reason=True,
            task=human_text,
        )

        score = None
        reason = None
        for attempt in range(2):
            print(f"    [DEBUG] attempt={attempt} calling measure() at {datetime.now().strftime('%H:%M:%S')} "
                  f"(up to {os.environ.get('DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE', '?')}s budget)...")
            try:
                metric.measure(test_case)
                score = metric.score
                reason = metric.reason
                break
            except Exception as ex:
                print(f"    [DEBUG] attempt={attempt} measure() raised an exception: {ex}")
                reason = f"Evaluation failed: {ex}"
                time.sleep(3)

        results.append({
            "human_index": idx_pos,
            "human_step_index": h_idx,
            "human_text": human_text,
            "agent_answer": agent_answer,
            "num_tools_used": len(tools_in_window),
            "tool_names": [t.name for t in tools_in_window],
            "context_turns_included": idx_pos,
            "score": score,
            "reason": reason,
        })

        print(f"[{idx_pos + 1}/{total_humans}] Human idx={h_idx} tools={len(tools_in_window)} score={score}")

    total_scored = sum(1 for r in results if r["score"] is not None)
    avg_score = (
        sum(r["score"] for r in results if r["score"] is not None) / total_scored
        if total_scored else None
    )

    report = {
        "history_file": str(history_path),
        "total_cases": len(results),
        "scored_cases": total_scored,
        "average_score": avg_score,
        "cases": results,
    }

    # Same pattern as the other eval scripts: project-root-anchored default,
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
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved report to: {output_path}")
    if avg_score is not None:
        print(f"📊 Average Task Completion Score: {avg_score:.2f} across {total_scored}/{len(results)} scored cases")

    return report


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepEval TaskCompletionMetric evaluation of an agent's full conversation history, one test case per human turn."
    )
    parser.add_argument(
        "--history",
        required=True,
        help="Path to the agent conversation history JSON (a top-level list of steps: human/agent_action/tool_output/agent_finish etc.)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "report" / "task_completion_eval_report.json"),
        help="Path to write the evaluation report JSON (default: <project root>/report/task_completion_eval_report.json)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum score to count as a passing case (default: {DEFAULT_THRESHOLD})",
    )
    return parser.parse_args()


# ---------- MAIN ----------
if __name__ == "__main__":
    args = parse_args()
    run_task_completion_eval(
        history_path=args.history,
        output_path=args.output,
        threshold=args.threshold,
    )