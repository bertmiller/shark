"""Utilities for loading and logging scheduler action tapes."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ACTION_TAPE_ENV_VAR = "SHARK_ACTION_TAPE_PATH"
ACTION_TAPE_LOG_ENV_VAR = "SHARK_ACTION_TAPE_LOG_PATH"


@dataclass(slots=True)
class TickAction:
    tick: int
    admit: list[str] = field(default_factory=list)
    preempt: list[str] = field(default_factory=list)
    prefill_chunk_tokens: int | None = None
    evict: list[str] = field(default_factory=list)
    decode_priority: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls, tick: int) -> "TickAction":
        return cls(tick=tick)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "TickAction":
        tick = payload.get("tick")
        if not isinstance(tick, int):
            raise ValueError(f"tick must be an int, got {tick!r}")

        prefill_chunk_tokens = payload.get("prefill_chunk_tokens")
        if prefill_chunk_tokens is not None and not isinstance(
            prefill_chunk_tokens, int
        ):
            raise ValueError(
                "prefill_chunk_tokens must be an int or null, "
                f"got {prefill_chunk_tokens!r}"
            )

        return cls(
            tick=tick,
            admit=_coerce_request_id_list(payload, "admit"),
            preempt=_coerce_request_id_list(payload, "preempt"),
            prefill_chunk_tokens=prefill_chunk_tokens,
            evict=_coerce_request_id_list(payload, "evict"),
            decode_priority=_coerce_request_id_list(payload, "decode_priority"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "admit": self.admit,
            "preempt": self.preempt,
            "prefill_chunk_tokens": self.prefill_chunk_tokens,
            "evict": self.evict,
            "decode_priority": self.decode_priority,
        }


def _coerce_request_id_list(payload: dict[str, Any], field_name: str) -> list[str]:
    value = payload.get(field_name, [])
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list[str], got {value!r}")
    return value


class ActionTapeRuntime:
    """Loads an action tape and emits per-tick replay logs."""

    def __init__(
        self,
        actions_by_tick: dict[int, TickAction],
        log_path: str | os.PathLike[str] | None = None,
    ) -> None:
        self._actions_by_tick = actions_by_tick
        self._log_path = Path(log_path) if log_path else None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_path.write_text("", encoding="utf-8")

    @classmethod
    def from_env(cls) -> "ActionTapeRuntime":
        tape_path = os.environ.get(ACTION_TAPE_ENV_VAR)
        log_path = os.environ.get(ACTION_TAPE_LOG_ENV_VAR)
        if not tape_path:
            raise ValueError(
                f"{ACTION_TAPE_ENV_VAR} must be set for tape-driven scheduling"
            )
        return cls(actions_by_tick=load_action_tape(tape_path), log_path=log_path)

    def action_for_tick(self, tick: int) -> TickAction:
        return self._actions_by_tick.get(tick, TickAction.empty(tick))

    def begin_tick(self, scheduler: Any, tick: int, action: TickAction) -> dict[str, Any]:
        return {
            "tick": tick,
            "action": action.to_json(),
            "tick_start_monotonic": time.monotonic(),
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
        record["scheduled_req_ids"] = list(scheduler_output.num_scheduled_tokens.keys())
        record["preempted_req_ids"] = sorted(scheduler_output.preempted_req_ids)
        record["finished_req_ids"] = sorted(scheduler_output.finished_req_ids)
        record["total_num_scheduled_tokens"] = (
            scheduler_output.total_num_scheduled_tokens
        )
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
            "waiting_request_ids": [request.request_id for request in waiting],
            "running_request_ids": [request.request_id for request in running],
            "kv_utilization": scheduler.kv_cache_manager.usage,
            "pause_state": scheduler._pause_state.name,
        }


def load_action_tape(path: str | os.PathLike[str]) -> dict[int, TickAction]:
    tape_path = Path(path)
    payload = json.loads(tape_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("ticks")
        if not isinstance(items, list):
            raise ValueError(
                "Action tape JSON must be a list or an object with a 'ticks' list"
            )
    elif isinstance(payload, list):
        items = payload
    else:
        raise ValueError("Action tape JSON root must be a list or object")

    actions: dict[int, TickAction] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each tape entry must be an object, got {item!r}")
        action = TickAction.from_json(item)
        if action.tick in actions:
            raise ValueError(f"Duplicate action for tick {action.tick}")
        actions[action.tick] = action
    return actions
