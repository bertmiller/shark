"""Utilities for loading and logging scheduler action tapes."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# After this many consecutive idle ticks (waiting > 0, nothing scheduled),
# stop writing full log entries and emit periodic summaries instead.
IDLE_TICK_LOG_THRESHOLD = 50
IDLE_TICK_LOG_INTERVAL = 500

ACTION_TAPE_ENV_VAR = "SHARK_ACTION_TAPE_PATH"
ACTION_TAPE_LOG_ENV_VAR = "SHARK_ACTION_TAPE_LOG_PATH"


@dataclass(slots=True)
class TickAction:
    tick: int
    eligible: list[str] = field(default_factory=list)
    prioritize: list[str] = field(default_factory=list)
    deprioritize: list[str] = field(default_factory=list)
    reorder_eligible: list[str] = field(default_factory=list)
    preempt: list[str] = field(default_factory=list)
    prefill_chunk_tokens: int | None = None
    evict: list[str] = field(default_factory=list)
    decode_priority: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, tick: int) -> "TickAction":
        return cls(tick=tick)

    def to_json(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "eligible": self.eligible,
            "prioritize": self.prioritize,
            "deprioritize": self.deprioritize,
            "reorder_eligible": self.reorder_eligible,
            "preempt": self.preempt,
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
            "evict": self.evict,
            "decode_priority": self.decode_priority,
        }


@dataclass(slots=True)
class ActionTape:
    timed_actions: list[tuple[int, TickAction]]  # sorted by time_ms
    arrival_actions: dict[str, TickAction]  # keyed by request alias


def _coerce_request_id_list(payload: dict[str, Any], field_name: str) -> list[str]:
    value = payload.get(field_name, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list[str], got {value!r}")
    return value


def _parse_action_fields(payload: dict[str, Any]) -> TickAction:
    """Parse action fields from a tape entry without requiring a tick number."""
    prefill_chunk_tokens = payload.get("prefill_chunk_tokens")
    if prefill_chunk_tokens is not None and not isinstance(
        prefill_chunk_tokens, int
    ):
        raise ValueError(
            "prefill_chunk_tokens must be an int or null, "
            f"got {prefill_chunk_tokens!r}"
        )

    return TickAction(
        tick=0,
        eligible=_coerce_request_id_list(payload, "eligible"),
        prioritize=_coerce_request_id_list(payload, "prioritize"),
        deprioritize=_coerce_request_id_list(payload, "deprioritize"),
        reorder_eligible=_coerce_request_id_list(payload, "reorder_eligible"),
        preempt=_coerce_request_id_list(payload, "preempt"),
        prefill_chunk_tokens=prefill_chunk_tokens,
        evict=_coerce_request_id_list(payload, "evict"),
        decode_priority=_coerce_request_id_list(payload, "decode_priority"),
    )


def _merge_actions(tick: int, actions: list[TickAction]) -> TickAction:
    """Merge multiple actions into one, concatenating list fields."""
    if not actions:
        return TickAction.empty(tick)
    if len(actions) == 1:
        a = actions[0]
        return TickAction(
            tick=tick,
            eligible=a.eligible,
            prioritize=a.prioritize,
            deprioritize=a.deprioritize,
            reorder_eligible=a.reorder_eligible,
            preempt=a.preempt,
            prefill_chunk_tokens=a.prefill_chunk_tokens,
            evict=a.evict,
            decode_priority=a.decode_priority,
        )
    eligible: list[str] = []
    prioritize: list[str] = []
    deprioritize: list[str] = []
    reorder_eligible: list[str] = []
    preempt: list[str] = []
    evict: list[str] = []
    decode_priority: list[str] = []
    prefill_values: list[int] = []
    for a in actions:
        eligible.extend(a.eligible)
        prioritize.extend(a.prioritize)
        deprioritize.extend(a.deprioritize)
        reorder_eligible.extend(a.reorder_eligible)
        preempt.extend(a.preempt)
        evict.extend(a.evict)
        decode_priority.extend(a.decode_priority)
        if a.prefill_chunk_tokens is not None:
            prefill_values.append(a.prefill_chunk_tokens)
    return TickAction(
        tick=tick,
        eligible=eligible,
        prioritize=prioritize,
        deprioritize=deprioritize,
        reorder_eligible=reorder_eligible,
        preempt=preempt,
        evict=evict,
        decode_priority=decode_priority,
        prefill_chunk_tokens=min(prefill_values) if prefill_values else None,
    )


class ActionTapeRuntime:
    """Loads an action tape and emits per-tick replay logs."""

    def __init__(
        self,
        tape: ActionTape,
        log_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._tape = tape
        self._timed_cursor: int = 0
        self._fired_arrivals: set[str] = set()
        self._seen_waiting_aliases: set[str] = set()
        self._run_start_monotonic: float = time.monotonic()
        self._eligible_queue: list[str] = []
        self._eligible_set: set[str] = set()
        self._log_base = Path(log_path) if log_path else None
        self._reset_count = 0
        self._log_path: Path | None = None
        self._consecutive_idle_ticks = 0
        self._idle_suppressed_count = 0
        if self._log_base is not None:
            self._log_base.parent.mkdir(parents=True, exist_ok=True)
            self._log_path = self._numbered_log_path()
            self._log_path.write_text("", encoding="utf-8")

    @classmethod
    def from_env(cls) -> "ActionTapeRuntime":
        tape_path = os.environ.get(ACTION_TAPE_ENV_VAR)
        log_path = os.environ.get(ACTION_TAPE_LOG_ENV_VAR)
        if not tape_path:
            raise ValueError(
                f"{ACTION_TAPE_ENV_VAR} must be set for tape-driven scheduling"
            )
        return cls(
            tape=load_action_tape(tape_path),
            log_path=log_path,
        )

    def _enqueue_eligible(self, aliases: list[str]) -> None:
        """Append new aliases to the eligible queue, preserving order."""
        for alias in aliases:
            if alias not in self._eligible_set:
                self._eligible_set.add(alias)
                self._eligible_queue.append(alias)

    def _apply_reorder_eligible(self, order: list[str]) -> None:
        """Reorder: listed items first (in order), then unlisted in current order."""
        ordered_set = set(order)
        front = [x for x in order if x in self._eligible_set]
        rest = [x for x in self._eligible_queue if x not in ordered_set]
        self._eligible_queue = front + rest

    def _apply_prioritize(self, aliases: list[str]) -> None:
        """Move listed items to front of eligible queue."""
        alias_set = set(aliases)
        front = [x for x in aliases if x in self._eligible_set]
        rest = [x for x in self._eligible_queue if x not in alias_set]
        self._eligible_queue = front + rest

    def _apply_deprioritize(self, aliases: list[str]) -> None:
        """Move listed items to back of eligible queue."""
        alias_set = set(aliases)
        front = [x for x in self._eligible_queue if x not in alias_set]
        back = [x for x in aliases if x in self._eligible_set]
        self._eligible_queue = front + back

    def actions_for_now(
        self,
        tick: int,
        now_monotonic: float,
        waiting_aliases: set[str],
    ) -> TickAction:
        """Return the merged action for this scheduler tick."""
        elapsed_ms = (now_monotonic - self._run_start_monotonic) * 1000
        collected: list[TickAction] = []

        # Advance timed cursor past all entries whose time has elapsed
        while self._timed_cursor < len(self._tape.timed_actions):
            time_ms, action = self._tape.timed_actions[self._timed_cursor]
            if time_ms <= elapsed_ms:
                collected.append(action)
                self._timed_cursor += 1
            else:
                break

        # Check for newly-arrived requests that have on_arrival triggers
        new_aliases = waiting_aliases - self._seen_waiting_aliases
        self._seen_waiting_aliases.update(waiting_aliases)
        for alias in new_aliases:
            if alias in self._tape.arrival_actions and alias not in self._fired_arrivals:
                collected.append(self._tape.arrival_actions[alias])
                self._fired_arrivals.add(alias)

        # Merge point-in-time fields from newly-fired actions
        merged = _merge_actions(tick, collected)

        # Accumulate eligible requests into the persistent queue
        self._enqueue_eligible(merged.eligible)

        # Apply queue manipulations (order: reorder first, then prioritize, then deprioritize)
        if merged.reorder_eligible:
            self._apply_reorder_eligible(merged.reorder_eligible)
        if merged.prioritize:
            self._apply_prioritize(merged.prioritize)
        if merged.deprioritize:
            self._apply_deprioritize(merged.deprioritize)

        # Return with cumulative eligible queue, point-in-time everything else
        return TickAction(
            tick=tick,
            eligible=list(self._eligible_queue),
            preempt=merged.preempt,
            evict=merged.evict,
            decode_priority=merged.decode_priority,
            prefill_chunk_tokens=merged.prefill_chunk_tokens,
        )

    def _reset_runtime_state(self) -> None:
        """Reset all runtime state for a new run."""
        self._timed_cursor = 0
        self._fired_arrivals = set()
        self._seen_waiting_aliases = set()
        self._run_start_monotonic = time.monotonic()
        self._eligible_queue = []
        self._eligible_set = set()
        self._consecutive_idle_ticks = 0
        self._idle_suppressed_count = 0

    def reset_from_data(self, tape: ActionTape) -> None:
        """Replace the current tape with new data and start a new log file."""
        self._tape = tape
        self._reset_runtime_state()
        if self._log_base is not None:
            self._reset_count += 1
            self._log_path = self._numbered_log_path()
            self._log_path.write_text("", encoding="utf-8")

    def current_log_name(self) -> str | None:
        if self._log_path is None:
            return None
        return self._log_path.name

    def _numbered_log_path(self) -> Path:
        base = self._log_base
        return base.with_stem(f"{base.stem}_{self._reset_count}")

    def begin_tick(self, scheduler: Any, tick: int, action: TickAction) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "tick": tick,
            "elapsed_ms": round((now - self._run_start_monotonic) * 1000, 1),
            "action": action.to_json(),
            "eligible_queue": list(self._eligible_queue),
            "tick_start_monotonic": now,
            "ignored_actions": [],
            "pre_state": self._snapshot(scheduler),
        }

    def record_ignored_action(
        self,
        record: dict[str, Any],
        *,
        kind: str,
        request_id: str,
        reason: str,
    ) -> None:
        record["ignored_actions"].append(
            {
                "kind": kind,
                "request_id": request_id,
                "reason": reason,
            }
        )

    def finish_tick(
        self,
        record: dict[str, Any],
        scheduler: Any,
        scheduler_output: Any,
    ) -> None:
        record["tick_end_monotonic"] = time.monotonic()
        record["post_state"] = self._snapshot(scheduler)
        scheduled_req_ids = list(scheduler_output.num_scheduled_tokens.keys())
        preempted_req_ids = sorted(scheduler_output.preempted_req_ids)
        finished_req_ids = sorted(scheduler_output.finished_req_ids)
        record["scheduled_req_ids"] = scheduled_req_ids
        record["preempted_req_ids"] = preempted_req_ids
        record["finished_req_ids"] = finished_req_ids
        record["total_num_scheduled_tokens"] = (
            scheduler_output.total_num_scheduled_tokens
        )

        queue_depth = record["post_state"]["queue_depth"]
        is_idle = (
            not scheduled_req_ids
            and not preempted_req_ids
            and not finished_req_ids
            and queue_depth > 0
        )

        if is_idle:
            self._consecutive_idle_ticks += 1
            if self._consecutive_idle_ticks == IDLE_TICK_LOG_THRESHOLD:
                logger.warning(
                    "Scheduler has been idle for %d consecutive ticks with "
                    "%d request(s) waiting. Does the tape have eligible actions?",
                    self._consecutive_idle_ticks,
                    queue_depth,
                )
            if self._consecutive_idle_ticks >= IDLE_TICK_LOG_THRESHOLD:
                self._idle_suppressed_count += 1
                if self._idle_suppressed_count % IDLE_TICK_LOG_INTERVAL == 0:
                    record["idle_suppressed_count"] = self._idle_suppressed_count
                    self._append_record(record)
                return
        else:
            if self._idle_suppressed_count > 0:
                self._append_record({
                    "tick": record["tick"],
                    "idle_summary": True,
                    "idle_ticks_suppressed": self._idle_suppressed_count,
                })
            self._consecutive_idle_ticks = 0
            self._idle_suppressed_count = 0

        self._append_record(record)

    def _append_record(self, record: dict[str, Any]) -> None:
        if self._log_path is None:
            return
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

    def _snapshot(self, scheduler: Any) -> dict[str, Any]:
        waiting = list(scheduler.waiting)
        running = list(scheduler.running)
        return {
            "queue_depth": len(waiting),
            "num_running": len(running),
            "waiting": [self._request_meta(r) for r in waiting],
            "running": [self._request_meta(r) for r in running],
            "kv_utilization": scheduler.kv_cache_manager.usage,
            "pause_state": scheduler._pause_state.name,
        }

    def _request_meta(self, request: Any) -> dict[str, Any]:
        info: dict[str, Any] = {
            "request_id": request.request_id,
            "input_length": request.num_prompt_tokens,
            "num_computed_tokens": request.num_computed_tokens,
            "num_output_tokens": request.num_output_tokens,
        }
        th = request.trace_headers
        if th:
            conversation_id = th.get("x-shark-conversation-id")
            turn_index = th.get("x-shark-turn-index")
            if conversation_id is not None:
                info["conversation_id"] = conversation_id
                info["trace_row"] = conversation_id
            if turn_index is not None:
                info["turn_index"] = turn_index
            if conversation_id is not None and turn_index is not None:
                info["action_tape_id"] = (
                    f"conv:{conversation_id}:turn:{turn_index}"
                )
            for key, target in (
                ("x-shark-trace-timestamp-ms", "trace_timestamp_ms"),
                ("x-shark-output-length", "output_length"),
            ):
                if key in th:
                    info[target] = th[key]
        return info


def load_action_tape_data(payload: Any) -> ActionTape:
    """Parse an already-loaded JSON value into an ActionTape."""
    if not isinstance(payload, dict) or "actions" not in payload:
        raise ValueError(
            "Action tape JSON must be an object with an 'actions' list"
        )
    items = payload["actions"]
    if not isinstance(items, list):
        raise ValueError("'actions' must be a list")

    timed_actions: list[tuple[int, TickAction]] = []
    arrival_actions: dict[str, TickAction] = {}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"Each tape entry must be an object, got {item!r}")
        has_time = "time_ms" in item
        has_arrival = "on_arrival" in item
        if has_time == has_arrival:
            raise ValueError(
                f"Entry {i}: must have exactly one of 'time_ms' or 'on_arrival', "
                f"got {'both' if has_time else 'neither'}"
            )
        action = _parse_action_fields(item)
        if has_time:
            time_ms = item["time_ms"]
            if not isinstance(time_ms, int) or time_ms < 0:
                raise ValueError(
                    f"Entry {i}: time_ms must be a non-negative int, got {time_ms!r}"
                )
            timed_actions.append((time_ms, action))
        else:
            key = item["on_arrival"]
            if not isinstance(key, str):
                raise ValueError(
                    f"Entry {i}: on_arrival must be a string, got {key!r}"
                )
            if key in arrival_actions:
                raise ValueError(f"Entry {i}: duplicate on_arrival key {key!r}")
            arrival_actions[key] = action

    timed_actions.sort(key=lambda x: x[0])
    return ActionTape(timed_actions=timed_actions, arrival_actions=arrival_actions)


def load_action_tape(path: str | os.PathLike[str]) -> ActionTape:
    tape_path = Path(path)
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    return load_action_tape_data(payload)
