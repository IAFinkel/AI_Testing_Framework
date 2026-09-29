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
    "X-Title": "Agent Plan Recall Evaluator",
}

# How far a judge's reported recall may drift from tp/(tp+fn) before we
# treat it as a genuine arithmetic inconsistency rather than rounding noise.
RECALL_TOLERANCE = 0.05


def _coerce_nonneg_int(value):
    """
    Returns value as a non-negative int if it's a valid count, else None.
    None signals "missing or invalid" — nothing reliable to cross-check
    recall against — as opposed to a present-but-wrong value.
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


# ---------- FUNCTION: Recall Metric via LLM ----------
def calculate_recall_with_llm(actual_plan, expected_plan, retries=4):
    """
    Uses an LLM (via OpenRouter) to judge whether a required expected step
    was actually accomplished by the agent.
    Returns: {"recall": float, "true_positives": int, "false_negatives": int, "justification": str}
    """
    client = get_client()

    recall_prompt = f"""
You are an expert AI Plan Evaluator specializing in Recall Scoring.

Compare two agent plan steps:

1. **Expected Step** — a required action from the ground truth plan.
2. **Actual Step** — the agent-generated step aligned to it.

Your task:
- Determine whether the Expected Step's core action and purpose was actually
  accomplished by the Actual Step.
- Ignore minor wording variations (e.g., synonyms or short paraphrases).
- Mark it as missed (a false negative) only if the required action or its
  purpose is absent, incorrect, or clearly not fulfilled.

Compute:
- True Positives (TP): The expected step was accomplished (1) or not (0).
- False Negatives (FN): The expected step was missed (1) or accomplished (0).
- Recall = TP / (TP + FN)  (will be 1.0 or 0.0 for a single step comparison,
  unless the step bundles multiple sub-requirements — in that case, score
  the fraction of sub-requirements actually accomplished.)

Return JSON ONLY:
{{
  "recall": float (0.0 - 1.0),
  "true_positives": int,
  "false_negatives": int,
  "justification": "brief explanation"
}}

Expected Step:
{expected_plan}

Actual Step:
{actual_plan}
"""

    content = ""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Always return valid JSON with numeric recall value."},
                    {"role": "user", "content": recall_prompt},
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

        if parsed and "recall" in parsed:
            try:
                parsed["recall"] = float(parsed["recall"])
            except (TypeError, ValueError):
                parsed["recall"] = 0.0

            tp = _coerce_nonneg_int(parsed.get("true_positives"))
            fn = _coerce_nonneg_int(parsed.get("false_negatives"))

            if tp is None or fn is None:
                # true_positives/false_negatives missing or invalid — there's
                # nothing reliable to cross-check recall against, so this
                # counts as a malformed response worth retrying, same as
                # missing the "recall" key entirely.
                print(f"    [DEBUG] attempt={attempt} missing/invalid true_positives or false_negatives "
                      f"(tp={parsed.get('true_positives')!r}, fn={parsed.get('false_negatives')!r}) — retrying")
                time.sleep(1)
                continue

            parsed["true_positives"] = tp
            parsed["false_negatives"] = fn
            parsed.setdefault("justification", "")

            if tp + fn > 0:
                expected_recall = tp / (tp + fn)
                if abs(parsed["recall"] - expected_recall) > RECALL_TOLERANCE:
                    print(f"    [DEBUG] attempt={attempt} recall mismatch: judge reported "
                          f"{parsed['recall']:.3f}, but tp={tp}/fn={fn} implies {expected_recall:.3f} "
                          f"— correcting to the computed value")
                    parsed["recall"] = expected_recall
                    parsed["recall_corrected"] = True
                else:
                    parsed["recall_corrected"] = False
            else:
                # tp + fn == 0: nothing was actually scored, so there's
                # nothing to recompute from — trust the judge's own value.
                parsed["recall_corrected"] = False

            return parsed

        time.sleep(1)  # brief pause before retrying a malformed response

    # All retries exhausted — degrade gracefully instead of crashing the run
    return {
        "recall": 0.0,
        "true_positives": 0,
        "false_negatives": 0,
        "justification": f"Invalid or empty JSON: {content[:80]}...",
        "recall_corrected": False,
    }


# ---------- FUNCTION: Evaluate Plans ----------
def evaluate_recall_for_plans(predicted_plan_path, ground_truth_plan_path, output_path):
    predicted = load_json(predicted_plan_path)
    ground_truth = load_json(ground_truth_plan_path)

    actual_plans = predicted.get("planner", [])
    expected_plans = ground_truth.get("planner", [])

    results = []
    print("🚀 Starting Recall Evaluation for Agent Plans...\n")

    # Same zip_longest safeguard used in the planner/precision scripts, so a
    # step-count mismatch never silently drops a step from evaluation.
    for i, (act, exp) in enumerate(zip_longest(actual_plans, expected_plans), start=1):
        if act is None:
            # An expected step with no corresponding actual step at all —
            # the agent never produced it. That's a false negative by
            # definition, no LLM call needed.
            results.append({
                "step": i,
                "expected_plan": exp.get("goal", ""),
                "actual_plan": None,
                "recall": 0.0,
                "true_positives": 0,
                "false_negatives": 1,
                "justification": "Expected step was not present in the actual plan (agent produced fewer steps than expected).",
                "recall_corrected": False,
                "counted_in_average": True,
            })
            print(f"Step {i}: Recall=0.00 | FN=1 | expected step missing from actual plan")
            continue

        if exp is None:
            # Actual plan has an extra step with no corresponding ground-truth
            # requirement — this affects precision, not recall (recall only
            # scores steps that were actually required), so it's reported
            # but excluded from the average.
            results.append({
                "step": i,
                "expected_plan": None,
                "actual_plan": act.get("goal", ""),
                "recall": None,
                "true_positives": 0,
                "false_negatives": 0,
                "justification": "Actual plan contains an unexpected extra step not present in the ground truth.",
                "recall_corrected": None,
                "counted_in_average": False,
            })
            print(f"Step {i}: N/A | unexpected extra step in actual plan")
            continue

        act_goal = act.get("goal", "")
        exp_goal = exp.get("goal", "")
        recall_eval = calculate_recall_with_llm(act_goal, exp_goal)
        results.append({
            "step": i,
            "expected_plan": exp_goal,
            "actual_plan": act_goal,
            "recall": recall_eval["recall"],
            "true_positives": recall_eval["true_positives"],
            "false_negatives": recall_eval["false_negatives"],
            "justification": recall_eval["justification"],
            "recall_corrected": recall_eval.get("recall_corrected", False),
            "counted_in_average": True,
        })
        print(f"Step {i}: Recall={recall_eval['recall']:.2f} | "
              f"TP={recall_eval['true_positives']} | FN={recall_eval['false_negatives']}")

    # ---------- SUMMARY ----------
    scored = [r for r in results if r["counted_in_average"]]
    total = len(results)
    total_scored = len(scored)
    avg_recall = sum(r["recall"] for r in scored) / total_scored if total_scored else 0.0
    total_tp = sum(r["true_positives"] for r in results)
    total_fn = sum(r["false_negatives"] for r in results)
    # Micro-average: pool TP/FN across all steps first, then divide once.
    # Differs from the macro-average above whenever steps carry unequal
    # tp+fn weight — structural, not a bug. corrected_count tracks a
    # separate failure mode: steps where the judge's own reported recall
    # contradicted its own tp/fn counts and had to be recomputed.
    micro_avg_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    corrected_count = sum(1 for r in results if r.get("recall_corrected"))
    length_mismatch = len(actual_plans) != len(expected_plans)

    summary = {
        "total_cases": total,
        "average_recall": f"{avg_recall:.2f}",
        "micro_average_recall": f"{micro_avg_recall:.2f}",
        "total_true_positives": total_tp,
        "total_false_negatives": total_fn,
        "steps_with_corrected_recall": corrected_count,
        "length_mismatch": length_mismatch,
        "actual_step_count": len(actual_plans),
        "expected_step_count": len(expected_plans),
        "details": results,
    }

    # Same pattern as planner/precision: caller-provided output path, parent
    # directory created if missing, timestamp appended so each run keeps its
    # own report instead of overwriting the previous one.
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    output_path = str(output_path)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Recall evaluation complete. Report saved to {output_path}")
    print(f"📊 Macro-Average Recall: {avg_recall:.2f} | Micro-Average Recall: {micro_avg_recall:.2f} "
          f"| Total TP={total_tp}, FN={total_fn}")
    if corrected_count:
        print(f"⚠️  Corrected {corrected_count} step(s) where the judge's reported recall "
              f"didn't match its own TP/FN counts")
    if length_mismatch:
        print(
            f"⚠️  Length mismatch: expected {summary['expected_step_count']} steps, "
            f"got {summary['actual_step_count']}"
        )

    return summary


# ---------- CLI ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM-as-judge recall evaluation of an agent's planner steps against a ground-truth reference."
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
        default=str(PROJECT_ROOT / "report" / "recall_eval_report.json"),
        help=f"Path to write the evaluation report JSON (default: {PROJECT_ROOT / 'report' / 'recall_eval_report.json'})",
    )
    return parser.parse_args()


# ---------- MAIN ----------
if __name__ == "__main__":
    args = parse_args()
    evaluate_recall_for_plans(
        predicted_plan_path=args.predicted,
        ground_truth_plan_path=args.ground_truth,
        output_path=args.output,
    )