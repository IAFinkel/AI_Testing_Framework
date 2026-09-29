import argparse
import json
from pathlib import Path


def _find_human_turn_slices(event_list):
    """
    Splits a flat event list into one slice per human/user turn: from each
    human step up to (but not including) the next one, or the end of the
    list. If no human/user step exists at all, the whole list is treated
    as a single turn — matches the original script's flat, un-segmented
    behavior for schemas without an explicit turn marker.
    """
    human_indices = [
        i for i, e in enumerate(event_list)
        if isinstance(e, dict) and e.get("type") in ("human", "user")
    ]
    if not human_indices:
        return [event_list]

    slices = []
    for pos, idx in enumerate(human_indices):
        end = human_indices[pos + 1] if pos + 1 < len(human_indices) else len(event_list)
        slices.append(event_list[idx:end])
    return slices


def _extract_events(event_list, step_start=1):
    """
    Core categorization logic for a flat list of events — either one
    turn's worth of events, or an entire trace, depending on the caller.
    step_start controls where step numbering begins, so a caller can
    restart numbering at 1 per turn, or run it once across the whole
    trace for a single continuous sequence.

    Returns (user_request, planner, router, tools).
    """
    planner, router, tools = [], [], []
    user_request = None
    step_count = step_start - 1

    for event in event_list:
        if not isinstance(event, dict):
            continue

        event_type = event.get("type")
        agent = event.get("agent") or event.get("type")  # Fallback for your specific schema

        # 0️⃣ Human / user request extraction. Only the FIRST human step in
        # this slice is kept — previously this was a plain overwrite, so
        # a multi-turn trace passed as one flat list silently lost every
        # user_request except the last one.
        if event_type in ["human", "user"]:
            if user_request is None:
                user_request = event.get("content")

        # 1️⃣ Planner / AI Reasoning extraction
        # NOTE (pre-existing, not addressed here): "ai" is used by TC1-style
        # traces both for intermediate narration ("Let me use my tool...")
        # AND for the actual final answer. Both land in "planner" as a
        # "goal" the same way — there's no signal in this schema to tell
        # them apart, so a TC1-style final answer currently gets treated as
        # a planning step rather than recognized as the outcome.
        elif event_type in ["agent_plan", "ai"]:
            step_count += 1
            planner.append({
                "step": step_count,
                "agent": agent,
                "goal": event.get("content")
            })

        # 2️⃣ Router / Action extraction
        # Any agent_action step is eligible, regardless of which agent
        # performed it (not just "RouterAgent") — a sub-agent invoking a
        # tool directly is just as real a tool call. route_to and tool are
        # independent, non-exclusive fields: a step can both route to a
        # sub-agent AND invoke a tool as part of that decision, and
        # "action" stays the exact string "Invoke Tool" whenever a tool is
        # present, since the precision/recall/tool-correctness scripts
        # rely on an exact match against this value.
        elif event_type == "agent_action":
            step_count += 1
            route_entry = {
                "step": step_count,
                "router": agent,
                "log": event.get("log"),
            }

            if "route_to" in event:
                route_entry["route_to"] = event["route_to"]

            if "tool" in event:
                route_entry["action"] = "Invoke Tool"
                route_entry["tool"] = event["tool"]
                route_entry["tool_input"] = event.get("tool_input")
            elif "route_to" in event:
                route_entry["action"] = "Route to Sub-Agent"

            router.append(route_entry)

        # 2.5️⃣ TC1-style tool_call: input AND output on a single event.
        # FIX: previously this only ever landed in "tools" (output-only,
        # via the branch below), which silently dropped the "input" field
        # and never produced a "router" entry at all — since the router
        # branch only triggers on "agent_action", a tool_call-shaped trace
        # could never populate router, meaning tool correctness had
        # nothing to compare tool arguments against. Record it in both
        # sections, matching how an agent_action + tool_output pair is
        # represented in the TC2/TC3 schema.
        elif event_type == "tool_call":
            step_count += 1
            tool_name = event.get("tool") or event.get("id")
            output_val = event.get("output") or event.get("content")

            router.append({
                "step": step_count,
                "router": agent,
                "log": event.get("description"),
                "action": "Invoke Tool",
                "tool": tool_name,
                "tool_input": event.get("input", {}),
            })
            tools.append({"tool": tool_name, "output": output_val})

        # 3️⃣ Tool output extraction
        elif event_type == "tool_output":
            tools.append({
                "tool": event.get("tool") or event.get("id"),
                "output": event.get("output") or event.get("content"),
            })

    return user_request, planner, router, tools


def extract_agent_data(agent_history_input):
    """
    Normalizes a raw agent execution trace into a consistent structure:
    {
        "user_request": str | None,   # the FIRST human turn's content
        "planner": [...],             # flattened across the whole trace
        "router": [...],              # flattened across the whole trace
        "tools": [...],                # flattened across the whole trace
        "turns": [                    # one block per human/user turn
            {"user_request": ..., "planner": [...], "router": [...], "tools": [...]},
            ...
        ]
    }

    The top-level planner/router/tools fields are kept for backward
    compatibility with single-turn traces and callers that only ever read
    the flat shape. "turns" is always present (even for a single-turn or
    already-structured trace, as a one-element list), and is the correct
    thing to read for a multi-turn conversation — each element only
    contains that turn's own steps, rather than the whole trace merged
    together.

    Handles three known input shapes:
      1. Flat event list (TC1-style): {"steps": [...]} or {"history": [...]}
         where each event has a "type" (human / ai / tool_call).
      2. Multi-agent flat trace (TC3-style): a flat event list using
         agent_plan / agent_action / tool_output / agent_finish, which may
         span several human turns.
      3. Pre-structured multi-agent trace (TC2-style): a dict that already
         has "planner", "router", and "tools" as separate top-level lists.
         This shape has no concept of turns, so it's wrapped as a single
         one-element "turns" list.
    """

    # Detect the already-structured shape first. If the input already
    # separates planner/router/tools, it's effectively pre-normalized —
    # pass it through, and wrap it as a single turn for a consistent
    # "turns" field regardless of input shape.
    if isinstance(agent_history_input, dict) and {"planner", "router", "tools"} <= agent_history_input.keys():
        single_turn = {
            "user_request": agent_history_input.get("user_request"),  # usually absent in this trace shape
            "planner": agent_history_input.get("planner", []),
            "router": agent_history_input.get("router", []),
            "tools": agent_history_input.get("tools", []),
        }
        return {**single_turn, "turns": [single_turn]}

    # --- Ensure we are iterating over the LIST of events ---
    # If the JSON is {"steps": [...]}, we want the list inside "steps"
    if isinstance(agent_history_input, dict):
        agent_history = agent_history_input.get("steps") or agent_history_input.get("history")
        if agent_history is None:
            raise ValueError("❌ JSON dictionary must contain a 'steps' or 'history' list.")
    else:
        agent_history = agent_history_input

    # Flattened, backward-compatible view: run extraction once across the
    # WHOLE trace with one continuous step counter, exactly like the
    # original single-turn-only script did.
    user_request, planner, router, tools = _extract_events(agent_history, step_start=1)

    # Turn-segmented view: split on human/user boundaries first, then run
    # the same extraction independently per slice, with step numbering
    # restarting at 1 for each turn so a turn read in isolation makes sense
    # on its own (rather than showing large step numbers inherited from
    # earlier turns).
    turns = []
    for slice_events in _find_human_turn_slices(agent_history):
        turn_request, turn_planner, turn_router, turn_tools = _extract_events(slice_events, step_start=1)
        turns.append({
            "user_request": turn_request,
            "planner": turn_planner,
            "router": turn_router,
            "tools": turn_tools,
        })

    return {
        "user_request": user_request,
        "planner": planner,
        "router": router,
        "tools": tools,
        "turns": turns,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract structured planner/router/tool data from a raw agent history JSON file."
    )
    parser.add_argument(
        "input_file",
        help="Path to the raw agent history JSON file (e.g. ../input_data/agent_history_tc_1.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default="extracted_data.json",
        help="Path to write the extracted JSON output (default: extracted_data.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"❌ Error: File not found at {input_path}")

    if input_path.stat().st_size == 0:
        raise ValueError(f"❌ Error: JSON file '{input_path}' is empty.")

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    structured_data = extract_agent_data(raw_data)

    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Extraction complete! Found {len(structured_data['planner'])} plan steps across {len(structured_data['turns'])} turn(s).")
    print(f"👤 First user request captured: {structured_data['user_request']!r}")
    print(f"📂 Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()