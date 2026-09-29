import argparse
import json
import os
import time
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from pathlib import Path

ENV_PATH = find_dotenv()
if not ENV_PATH:
    raise RuntimeError(
        "Could not locate a .env file by walking up from this script's location. "
        "Make sure a .env file exists at the project root."
    )
load_dotenv(ENV_PATH)
PROJECT_ROOT = Path(ENV_PATH).resolve().parent

from itertools import zip_longest

from openai import OpenAI

# ---------- CONFIG ----------
API_KEY = os.environ.get("OPENROUTER_API_KEY")

MODEL_NAME = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

EXTRA_HEADERS = {
    "HTTP-Referer": "https://generativeaitechshila.com",
    "X-Title": "Agent Plan Precision Evaluator",
}

# How far a judge's reported precision may drift from tp/(tp+fp) before we
# treat it as a genuine arithmetic inconsistency rather than rounding noise.
PRECISION_TOLERANCE = 0.05


def _coerce_nonneg_int(value):
    """
    Returns value as a non-negative int if it's a valid count, else None.
    None signals "missing or invalid" — nothing reliable to cross-check
    precision against — as opposed to a present-but-wrong value.
    """
    try:
        i = int(value)
    except (TypeError, ValueError):
        return None
    return i if i >= 0 else None


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


# ---------- FUNCTION: Precision Metric via LLM ----------
def calculate_precision_with_llm(actual_plan, expected_plan, retries=4):
    """
    Uses an LLM (via OpenRouter) to estimate Precision between an expected
    and actual plan step.
    Returns: {"precision": float, "true_positives": int, "false_positives": int, "justification": str}
    """
    client = get_client()

    precision_prompt = f"""
You are an expert AI Plan Evaluator specializing in Precision Scoring.

Compare two agent plans step-by-step:

1. **Expected Plan** — the ground truth sequence of actions.
2. **Actual Plan** — the agent-generated sequence.

Your task:
- Identify which steps in the Actual Plan correctly match or partially match the Expected Plan.
- Ignore minor wording variations (e.g., synonyms or short paraphrases).
- Consider a step correct if its core action and purpose match.
- Penalize extra or irrelevant steps.

Compute:
- True Positives (TP): Steps correctly matching expected ones.
- False Positives (FP): Incorrect or extra steps.
- Precision = TP / (TP + FP)

Return JSON ONLY:
{{
  "precision": float (0.0 - 1.0),
  "true_positives": int,
  "false_positives": int,
  "justification": "brief explanation"
}}

Expected Plan:
{expected_plan}

Actual Plan:
{actual_plan}
"""

    content = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Always return valid JSON with numeric precision value."},
                    {"role": "user", "content": precision_prompt},
                ],
                temperature=0,
                extra_headers=EXTRA_HEADERS,
                extra_body={
                    "models": [
                        "deepseek/deepseek-chat-v3-0324:free",
                        "qwen/qwen3-235b-a22b:free",
                    ]
                },
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

        if parsed and "precision" in parsed:
            try:
                parsed["precision"] = float(parsed["precision"])
            except (TypeError, ValueError):
                parsed["precision"] = 0.0

            tp = _coerce_nonneg_int(parsed.get("true_positives"))
            fp = _coerce_nonneg_int(parsed.get("false_positives"))

            if tp is None or fp is None:
                # true_positives/false_positives missing or invalid — there's
                # nothing reliable to cross-check precision against, so this
                # counts as a malformed response worth retrying, same as
                # missing the "precision" key entirely.
                print(f"    [DEBUG] attempt={attempt} missing/invalid true_positives or false_positives "
                      f"(tp={parsed.get('true_positives')!r}, fp={parsed.get('false_positives')!r}) — retrying")
                time.sleep(1)
                continue

            parsed["true_positives"] = tp
            parsed["false_positives"] = fp
            parsed.setdefault("justification", "")

            if tp + fp > 0:
                expected_precision = tp / (tp + fp)
                if abs(parsed["precision"] - expected_precision) > PRECISION_TOLERANCE:
                    print(f"    [DEBUG] attempt={attempt} precision mismatch: judge reported "
                          f"{parsed['precision']:.3f}, but tp={tp}/fp={fp} implies {expected_precision:.3f} "
                          f"— correcting to the computed value")
                    parsed["precision"] = expected_precision
                    parsed["precision_corrected"] = True
                else:
                    parsed["precision_corrected"] = False
            else:
                # tp + fp == 0: nothing was actually scored, so there's
                # nothing to recompute from — trust the judge's own value.
                parsed["precision_corrected"] = False

            return parsed

        time.sleep(1)  # brief pause before retrying a malformed response

    # All retries exhausted — degrade gracefully instead of crashing the run
    return {
        "precision": 0.0,
        "true_positives": 0,
        "false_positives": 0,
        "justification": f"Invalid or empty JSON: {content[:80]}...",
        "precision_corrected": False,
    }


# ---------- FUNCTION: Evaluate Plans ----------
def evaluate_precision_for_plans(predicted_plan_path, ground_truth_plan_path, output_path):
    predicted = load_json(predicted_plan_path)
    ground_truth = load_json(ground_truth_plan_path)

    actual_plans = predicted.get("planner", [])
    expected_plans = ground_truth.get("planner", [])

    results = []
    print("🚀 Starting Precision Evaluation for Agent Plans...\n")

    # FIX: zip() silently truncates to the shorter list, so a step-count
    # mismatch would leave some steps never evaluated or reported.
    # zip_longest ensures every step on either side is accounted for.
    for i, (act, exp) in enumerate(zip_longest(actual_plans, expected_plans), start=1):
        if exp is None:
            # Actual plan has an extra step with no corresponding ground-truth
            # step — that's a false positive by definition, no LLM call needed.
            results.append({
                "step": i,
                "expected_plan": None,
                "actual_plan": act.get("goal", ""),
                "precision": 0.0,
                "true_positives": 0,
                "false_positives": 1,
                "justification": "Actual plan contains an unexpected extra step not present in the ground truth.",
                "precision_corrected": False,
                "counted_in_average": True,
            })
            print(f"Step {i}: Precision=0.00 | FP=1 | unexpected extra step in actual plan")
            continue

        if act is None:
            # Expected step never produced by the agent — this affects
            # recall, not precision (precision only scores steps the agent
            # actually produced), so it's reported but excluded from the average.
            results.append({
                "step": i,
                "expected_plan": exp.get("goal", ""),
                "actual_plan": None,
                "precision": None,
                "true_positives": 0,
                "false_positives": 0,
                "justification": "Expected step was not present in the actual plan (agent produced fewer steps than expected).",
                "precision_corrected": None,
                "counted_in_average": False,
            })
            print(f"Step {i}: N/A | expected step missing from actual plan")
            continue

        act_goal = act.get("goal", "")
        exp_goal = exp.get("goal", "")
        precision_eval = calculate_precision_with_llm(act_goal, exp_goal)
        results.append({
            "step": i,
            "expected_plan": exp_goal,
            "actual_plan": act_goal,
            "precision": precision_eval["precision"],
            "true_positives": precision_eval["true_positives"],
            "false_positives": precision_eval["false_positives"],
            "justification": precision_eval["justification"],
            "precision_corrected": precision_eval.get("precision_corrected", False),
            "counted_in_average": True,
        })
        print(f"Step {i}: Precision={precision_eval['precision']:.2f} | "
              f"TP={precision_eval['true_positives']} | FP={precision_eval['false_positives']}")

    # ---------- SUMMARY ----------
    scored = [r for r in results if r["counted_in_average"]]
    total = len(results)
    total_scored = len(scored)
    avg_precision = sum(r["precision"] for r in scored) / total_scored if total_scored else 0.0
    total_tp = sum(r["true_positives"] for r in results)
    total_fp = sum(r["false_positives"] for r in results)
    # Micro-average: pool TP/FP across all steps first, then divide once.
    # Differs from the macro-average above whenever steps carry unequal
    # tp+fp weight — this is expected/structural, not a bug (see the two
    # numbers diverge even when every step's own math is internally
    # consistent). corrected_count tracks a separate, real failure mode:
    # steps where the judge's own reported precision contradicted its own
    # tp/fp counts and had to be recomputed.
    micro_avg_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    corrected_count = sum(1 for r in results if r.get("precision_corrected"))
    length_mismatch = len(actual_plans) != len(expected_plans)

    summary = {
        "total_cases": total,
        "average_precision": f"{avg_precision:.2f}",
        "micro_average_precision": f"{micro_avg_precision:.2f}",
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "steps_with_corrected_precision": corrected_count,
        "length_mismatch": length_mismatch,
        "actual_step_count": len(actual_plans),
        "expected_step_count": len(expected_plans),
        "details": results,
    }

    # FIX: output_path is caller-provided (via CLI), the parent directory is
    # created if missing, and a timestamp is inserted so each run produces
    # its own report file instead of overwriting the previous one.
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    output_path = str(output_path)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Precision evaluation complete. Report saved to {output_path}")
    print(f"📊 Macro-Average Precision: {avg_precision:.2f} | Micro-Average Precision: {micro_avg_precision:.2f} "
          f"| Total TP={total_tp}, FP={total_fp}")
    if corrected_count:
        print(f"⚠️  Corrected {corrected_count} step(s) where the judge's reported precision "
              f"didn't match its own TP/FP counts")
    if length_mismatch:
        print(
            f"⚠️  Length mismatch: expected {summary['expected_step_count']} steps, "
            f"got {summary['actual_step_count']}"
        )

    return summary


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge precision evaluation of an agent's planner steps against a ground-truth reference."
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
        default=str(PROJECT_ROOT / "report" / "precision_eval_report.json"),
        help=f"Path to write the evaluation report JSON (default: {PROJECT_ROOT / 'report' / 'precision_eval_report.json'})",
    )
    return parser.parse_args()


# ---------- MAIN ----------
if __name__ == "__main__":
    args = parse_args()
    evaluate_precision_for_plans(
        predicted_plan_path=args.predicted,
        ground_truth_plan_path=args.ground_truth,
        output_path=args.output,
    )