import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
ENV_PATH = find_dotenv()
load_dotenv(ENV_PATH)


PROJECT_ROOT = Path(ENV_PATH).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "report" / "plan_eval_report.json"

from itertools import zip_longest
from openai import OpenAI

# ---------- CONFIG ----------
# FIX: API key now comes from an environment variable instead of being
# hardcoded (the original file didn't even pass an api_key at all —
# it was a non-functional stub with a leftover comment).
API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Model is overridable via env var so you can swap models without editing code
# (useful if a free-tier model becomes unavailable or rate-limited).
MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "Generative AI Plan Evaluator",
}


def get_client():
    """
    Builds the OpenRouter client lazily (only when a judge call is actually
    made), so importing this module for testing doesn't require an API key
    to already be set.
    """
    if not API_KEY:
        raise RuntimeError(
            "Missing OPENROUTER_API_KEY environment variable.\n"
            "Get a free key at https://openrouter.ai/keys, then set it, e.g.:\n"
            "  export OPENROUTER_API_KEY='sk-or-...'\n"
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)


# ---------- FUNCTION: Load JSON ----------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- FUNCTION: Safe JSON Parse ----------
def safe_json_parse(text):
    """
    LLMs sometimes wrap JSON in prose or code fences even when told not to.
    This defensively extracts the substring between the first '{' and the
    last '}' before attempting to parse it.
    """
    try:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        return json.loads(text)
    except Exception:
        return None


# ---------- FUNCTION: LLM Judge ----------
def evaluate_plan_with_llm(actual_goal, expected_goal, retries=4):
    """
    Uses an LLM (via OpenRouter) to judge whether actual_goal is a
    semantically acceptable match for expected_goal.
    Returns a dict: {"status": "PASS"/"FAIL", "justification": str}
    """
    client = get_client()

    prompt = f"""
You are an expert LLM-as-a-Judge for Agent Plan Evaluation.

Compare two plans:
1. **Expected Plan** — the correct sequence of steps.
2. **Actual Plan** — the agent-generated sequence.

Rules:
- Minor wording or ordering variations are acceptable if meaning is preserved.
- Mark FAIL only if key steps are missing, incorrect, or clearly out of order.

Return STRICTLY in JSON:
{{
  "status": "PASS" or "FAIL",
  "justification": "Brief reason if FAIL, else 'All good'."
}}

Expected Plan:
{expected_goal}

Actual Plan:
{actual_goal}
"""

    content = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            extra_headers=EXTRA_HEADERS,
            extra_body={
                "models": [
                    "deepseek/deepseek-chat-v3-0324:free",
                    "qwen/qwen3-235b-a22b:free",
                ]
            }
        )
        except Exception as e:
            print(f"    [DEBUG] attempt={attempt} API call raised an exception: {e}")
            time.sleep(1)
            continue
        
        if not response.choices:
            err = getattr(response, "error", None)
            print(f"    [DEBUG] attempt={attempt} no choices returned (likely rate-limited or provider error). error={err}")
            time.sleep(2)
            continue

        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        reasoning_tokens = None
        if response.usage and response.usage.completion_tokens_details:
            reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
        print(f"    [DEBUG] attempt={attempt} model={response.model} finish_reason={choice.finish_reason} "
              f"reasoning_tokens={reasoning_tokens} content={content!r}")
        parsed = safe_json_parse(content)

        if parsed and "status" in parsed:
            parsed["status"] = str(parsed["status"]).upper().strip()
            if parsed["status"] not in ["PASS", "FAIL"]:
                parsed["status"] = "FAIL"
            return parsed

        time.sleep(1)  # brief pause before retrying a malformed response

    # All retries exhausted — degrade gracefully instead of crashing the run
    return {"status": "FAIL", "justification": f"Invalid or empty JSON: {content[:80]}..."}


# ---------- FUNCTION: Evaluate Planner ----------
def evaluate_planner(predicted_plan_path, ground_truth_plan_path, output_path):
    predicted = load_json(predicted_plan_path)
    ground_truth = load_json(ground_truth_plan_path)

    actual_plans = predicted.get("planner", [])
    expected_plans = ground_truth.get("planner", [])

    # FIX: the original used zip(), which silently truncates to the length
    # of the SHORTER list. If the agent produced fewer or more steps than
    # expected, those extra/missing steps were never evaluated or reported.
    # zip_longest ensures every step on either side is accounted for.
    results = []
    for i, (act, exp) in enumerate(zip_longest(actual_plans, expected_plans), start=1):
        if exp is None:
            results.append({
                "step": i,
                "expected_goal": None,
                "actual_goal": act.get("goal", ""),
                "status": "FAIL",
                "justification": "Actual plan contains an unexpected extra step not present in the ground truth.",
            })
            print(f"Step {i}: FAIL | unexpected extra step in actual plan")
            continue

        if act is None:
            results.append({
                "step": i,
                "expected_goal": exp.get("goal", ""),
                "actual_goal": None,
                "status": "FAIL",
                "justification": "Expected step was not present in the actual plan (agent produced fewer steps than expected).",
            })
            print(f"Step {i}: FAIL | expected step missing from actual plan")
            continue

        act_goal = act.get("goal", "")
        exp_goal = exp.get("goal", "")
        evaluation = evaluate_plan_with_llm(act_goal, exp_goal)
        time.sleep(1)
        results.append({
            "step": i,
            "expected_goal": exp_goal,
            "actual_goal": act_goal,
            "status": evaluation["status"],
            "justification": evaluation["justification"],
        })
        print(f"Step {i}: {evaluation['status']} | {evaluation['justification']}")

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    length_mismatch = len(actual_plans) != len(expected_plans)

    summary = {
        "total_cases": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": f"{(passed / total * 100):.1f}%" if total else "0.0%",
        # FIX: length mismatch is now an explicit, visible field in the report
        # instead of being silently absorbed by zip()'s truncation behavior.
        "length_mismatch": length_mismatch,
        "actual_step_count": len(actual_plans),
        "expected_step_count": len(expected_plans),
        "details": results,
    }

    # FIX: output_path is now caller-provided (via CLI) instead of hardcoded,
    # and the parent directory is created if it doesn't exist yet.
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    output_path = str(output_path)
    
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ LLM evaluation complete. Report saved to {output_path}")
    print(f"📊 Pass Rate: {summary['pass_rate']} ({passed}/{total} passed)")
    if length_mismatch:
        print(
            f"⚠️  Length mismatch: expected {summary['expected_step_count']} steps, "
            f"got {summary['actual_step_count']}"
        )

    return summary


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation of an agent's planner steps against a ground-truth reference."
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
        default=str(DEFAULT_OUTPUT),
        help="Path to write the evaluation report JSON (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


# ---------- MAIN ----------
if __name__ == "__main__":
    args = parse_args()
    evaluate_planner(
        predicted_plan_path=args.predicted,
        ground_truth_plan_path=args.ground_truth,
        output_path=args.output,
    )