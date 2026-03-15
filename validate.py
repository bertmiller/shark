"""Validate a scheduler action tape JSON file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from schedulers.action_tape import ActionTape, load_action_tape_data

_VALID_TOP_LEVEL_KEYS = {"actions"}

_VALID_ENTRY_KEYS = {
    "time_ms",
    "on_arrival",
    "eligible",
    "prioritize",
    "deprioritize",
    "reorder_eligible",
    "preempt",
    "prefill_chunk_tokens",
    "evict",
    "decode_priority",
}

_LIST_FIELDS = [
    "eligible",
    "prioritize",
    "deprioritize",
    "reorder_eligible",
    "preempt",
    "evict",
    "decode_priority",
]


def strict_validate(payload: dict) -> list[str]:
    """Return a list of error strings for unknown keys, duplicates, and negative values."""
    errors: list[str] = []

    unknown_top = set(payload.keys()) - _VALID_TOP_LEVEL_KEYS
    if unknown_top:
        errors.append(f"Unknown top-level keys: {sorted(unknown_top)}")

    items = payload.get("actions", [])
    if not isinstance(items, list):
        return errors

    seen_arrival_keys: dict[str, int] = {}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        unknown = set(item.keys()) - _VALID_ENTRY_KEYS
        if unknown:
            errors.append(f"Entry {i}: unknown keys {sorted(unknown)}")

        for field_name in _LIST_FIELDS:
            values = item.get(field_name)
            if not isinstance(values, list):
                continue
            seen: set[str] = set()
            for v in values:
                if isinstance(v, str) and v in seen:
                    errors.append(
                        f"Entry {i}: duplicate ID {v!r} in {field_name}"
                    )
                elif isinstance(v, str):
                    seen.add(v)

        if "on_arrival" in item and isinstance(item["on_arrival"], str):
            key = item["on_arrival"]
            if key in seen_arrival_keys:
                errors.append(
                    f"Entry {i}: duplicate on_arrival {key!r} "
                    f"(first at entry {seen_arrival_keys[key]})"
                )
            else:
                seen_arrival_keys[key] = i

        pct = item.get("prefill_chunk_tokens")
        if isinstance(pct, int) and pct < 0:
            errors.append(
                f"Entry {i}: prefill_chunk_tokens must be non-negative, got {pct}"
            )

        time_ms = item.get("time_ms")
        if isinstance(time_ms, (int, float)) and time_ms < 0:
            errors.append(
                f"Entry {i}: time_ms must be non-negative, got {time_ms}"
            )

    return errors


def validate_tape_file(tape_path: Path) -> ActionTape:
    """Validate a tape file and return the parsed ActionTape.

    Raises ``ValueError`` on any validation failure.
    Prints a warning to stderr if the tape has no eligible actions.
    """
    if not tape_path.exists():
        raise ValueError(f"tape file not found: {tape_path}")

    try:
        payload = json.loads(tape_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {tape_path}: line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    errors = strict_validate(payload)
    if errors:
        raise ValueError(
            f"invalid tape in {tape_path}:\n" + "\n".join(f"  {e}" for e in errors)
        )

    tape = load_action_tape_data(payload)

    has_eligible = (
        any(a.eligible for _, a in tape.timed_actions)
        or any(a.eligible for a in tape.arrival_actions.values())
    )
    if not has_eligible:
        print(
            f"WARNING: tape {tape_path} has no eligible actions. "
            "No requests will be admitted — the scheduler will spin "
            "with all requests stuck in the waiting queue.",
            file=sys.stderr,
        )

    return tape


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate a tape-driven scheduler action tape JSON file"
    )
    parser.add_argument(
        "tape",
        nargs="?",
        default="tape.json",
        help="Path to the tape JSON file (default: tape.json)",
    )
    args = parser.parse_args()

    try:
        tape = validate_tape_file(Path(args.tape))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_timed = len(tape.timed_actions)
    n_arrival = len(tape.arrival_actions)
    total = n_timed + n_arrival
    if total:
        parts = []
        if n_timed:
            parts.append(f"{n_timed} time_ms")
        if n_arrival:
            parts.append(f"{n_arrival} on_arrival")
        print(f"OK: {args.tape} is valid ({', '.join(parts)}; {total} total entries)")
    else:
        print(f"OK: {args.tape} is valid (0 entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
