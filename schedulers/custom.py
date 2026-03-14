"""Custom vLLM scheduler for experimentation."""

import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable

from schedulers.action_tape import ActionTapeRuntime, TickAction, load_action_tape_data

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

logger = logging.getLogger(__name__)


class _ControlHandler(BaseHTTPRequestHandler):
    """HTTP handler for the scheduler control server."""

    scheduler: "CustomScheduler"

    def do_POST(self):
        if self.path != "/reset":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)
            tape = load_action_tape_data(payload)
        except Exception as exc:
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Bad tape JSON: {exc}\n".encode())
            return

        sched = self.scheduler
        if sched._can_apply_reset_immediately():
            sched._apply_reset(tape)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
            return

        sched._pending_tape = tape
        sched._reset_done.clear()
        sched._reset_requested.set()

        if sched._reset_done.wait(timeout=30):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK\n")
        else:
            self.send_response(504)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Timeout waiting for schedule() to apply reset\n")

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK\n")

    def log_message(self, format, *args):
        logger.debug("control: %s", format % args)


class CustomScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._action_tape = ActionTapeRuntime.from_env()
        self._scheduler_tick = 0

        self._reset_requested = threading.Event()
        self._reset_done = threading.Event()
        self._pending_tape: dict[int, TickAction] | None = None

        control_port = int(os.environ.get("SHARK_CONTROL_PORT", "8001"))
        handler = type(
            "_BoundHandler",
            (_ControlHandler,),
            {"scheduler": self},
        )
        self._control_server = HTTPServer(("0.0.0.0", control_port), handler)
        thread = threading.Thread(
            target=self._control_server.serve_forever,
            daemon=True,
        )
        thread.start()
        logger.info("Control server listening on :%d", control_port)

    def _can_apply_reset_immediately(self) -> bool:
        return not self.running and not self.waiting

    def _apply_reset(self, tape: dict[int, TickAction]) -> None:
        self._action_tape.reset_from_data(tape)
        self._pending_tape = None
        self._scheduler_tick = 0

    def _record_ignored_action(
        self,
        record: dict[str, object],
        *,
        kind: str,
        request_id: str,
        reason: str,
    ) -> None:
        self._action_tape.record_ignored_action(
            record,
            kind=kind,
            request_id=request_id,
            reason=reason,
        )

    def _force_preempt_requests(
        self,
        request_ids: list[str],
        timestamp: float,
        record: dict[str, object],
        *,
        kind: str,
    ) -> list[Request]:
        running_by_id = self._request_alias_map(self.running)
        preempted: list[Request] = []
        seen_request_ids: set[str] = set()
        for request_id in request_ids:
            request = running_by_id.get(request_id)
            if request is None:
                self._record_ignored_action(
                    record,
                    kind=kind,
                    request_id=request_id,
                    reason="request is not running on this tick",
                )
                continue
            if request.request_id in seen_request_ids:
                continue
            self.running.remove(request)
            self._preempt_request(request, timestamp)
            preempted.append(request)
            seen_request_ids.add(request.request_id)
        return preempted

    def _force_evict_requests(
        self,
        request_ids: list[str],
        timestamp: float,
        record: dict[str, object],
    ) -> list[Request]:
        active_by_id = self._request_alias_map(self.requests.values())
        evicted_running: list[Request] = []
        seen_request_ids: set[str] = set()
        for request_id in request_ids:
            request = active_by_id.get(request_id)
            if request is None or request.is_finished():
                self._record_ignored_action(
                    record,
                    kind="evict",
                    request_id=request_id,
                    reason="request is not active",
                )
                continue
            if request.request_id in seen_request_ids:
                continue
            seen_request_ids.add(request.request_id)

            if request.status == RequestStatus.RUNNING:
                if request in self.running:
                    self.running.remove(request)
                self._preempt_request(request, timestamp)
                evicted_running.append(request)
                continue

            if request.status not in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                self._record_ignored_action(
                    record,
                    kind="evict",
                    request_id=request_id,
                    reason=f"request is in unsupported state {request.status.name}",
                )
                continue

            self.kv_cache_manager.free(request)
            self.encoder_cache_manager.free(request)
            request.num_computed_tokens = 0
            request.num_cached_tokens = -1
            request.num_external_computed_tokens = 0
            if request.spec_token_ids:
                request.spec_token_ids = []
        return evicted_running

    def _request_tape_alias(self, request: Request) -> str | None:
        trace_headers = request.trace_headers or {}
        conversation_id = trace_headers.get("x-shark-conversation-id")
        turn_index = trace_headers.get("x-shark-turn-index")
        if conversation_id is None or turn_index is None:
            return None
        return f"conv:{conversation_id}:turn:{turn_index}"

    def _request_aliases(self, request: Request) -> tuple[str, ...]:
        aliases = [request.request_id]
        tape_alias = self._request_tape_alias(request)
        if tape_alias is not None:
            aliases.append(tape_alias)
        return tuple(aliases)

    def _request_alias_map(self, requests: Iterable[Request]) -> dict[str, Request]:
        alias_map: dict[str, Request] = {}
        for request in requests:
            for alias in self._request_aliases(request):
                alias_map.setdefault(alias, request)
        return alias_map

    def _resolve_request_order(
        self,
        requests: list[Request],
        targets: list[str],
        record: dict[str, object],
        *,
        kind: str,
        missing_reason: str,
    ) -> dict[str, int]:
        alias_map = self._request_alias_map(requests)
        requested_order: dict[str, int] = {}
        for index, target in enumerate(targets):
            request = alias_map.get(target)
            if request is None:
                self._record_ignored_action(
                    record,
                    kind=kind,
                    request_id=target,
                    reason=missing_reason,
                )
                continue
            requested_order.setdefault(request.request_id, index)
        return requested_order

    def _order_running_requests(
        self,
        action: TickAction,
        record: dict[str, object],
    ) -> None:
        current_order = {
            request.request_id: index for index, request in enumerate(self.running)
        }
        requested_order = self._resolve_request_order(
            list(self.running),
            action.decode_priority,
            record,
            kind="decode_priority",
            missing_reason="request is not running on this tick",
        )
        self.running.sort(key=lambda request: (
            requested_order.get(request.request_id, len(action.decode_priority)),
            current_order[request.request_id],
        ))

    def _rebuild_waiting_queue(self, ordered_requests: list[Request]) -> None:
        rebuilt_waiting = create_request_queue(self.policy)
        for request in ordered_requests:
            rebuilt_waiting.add_request(request)
        self.waiting = rebuilt_waiting

    def _order_waiting_requests(
        self,
        action: TickAction,
        record: dict[str, object],
    ) -> set[str]:
        waiting_requests = list(self.waiting)
        if not waiting_requests:
            return set()

        current_order = {
            request.request_id: index for index, request in enumerate(waiting_requests)
        }
        requested_order = self._resolve_request_order(
            waiting_requests,
            action.admit,
            record,
            kind="admit",
            missing_reason="request is not waiting on this tick",
        )
        waiting_requests.sort(key=lambda request: (
            requested_order.get(request.request_id, len(action.admit)),
            current_order[request.request_id],
        ))
        self._rebuild_waiting_queue(waiting_requests)
        return set(requested_order)

    def _apply_prefill_chunk_action(
        self,
        request: Request,
        num_new_tokens: int,
        action: TickAction,
    ) -> int:
        if action.prefill_chunk_tokens is None or request.num_computed_tokens >= request.num_tokens:
            return num_new_tokens

        if action.prefill_chunk_tokens <= 0:
            return 0
        return min(num_new_tokens, action.prefill_chunk_tokens)

    def schedule(self) -> SchedulerOutput:
        if self._reset_requested.is_set():
            self._apply_reset(self._pending_tape)
            self._reset_requested.clear()
            self._reset_done.set()
        elif self._action_tape.maybe_reload():
            self._scheduler_tick = 0
        tick = self._scheduler_tick
        self._scheduler_tick += 1
        action = self._action_tape.action_for_tick(tick)
        record = self._action_tape.begin_tick(self, tick, action)

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        preempted_reqs.extend(
            self._force_preempt_requests(
                action.preempt,
                scheduled_timestamp,
                record,
                kind="preempt",
            )
        )
        preempted_reqs.extend(
            self._force_evict_requests(
                action.evict,
                scheduled_timestamp,
                record,
            )
        )

        self._order_running_requests(action, record)

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            num_new_tokens = self._apply_prefill_chunk_action(
                request,
                num_new_tokens,
                action,
            )
            num_new_tokens = min(num_new_tokens, token_budget)

            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                req_index += 1
                continue

            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        break

                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        break

            if new_blocks is None:
                break

            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                request.spec_token_ids = []

            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        if self._pause_state == PauseState.UNPAUSED and action.admit:
            allowed_request_ids = self._order_waiting_requests(action, record)

            skipped_waiting_requests = create_request_queue(self.policy)

            while self.waiting and token_budget > 0 and action.prefill_chunk_tokens != 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()
                request_id = request.request_id

                if request_id not in allowed_request_ids:
                    request = self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        if request.num_preemptions:
                            request.status = RequestStatus.PREEMPTED
                        else:
                            request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    assert not request.streaming_queue
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                if request.num_computed_tokens == 0:
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                else:
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    num_new_tokens = self._apply_prefill_chunk_action(
                        request,
                        num_new_tokens,
                        action,
                    )

                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    if num_new_tokens == 0:
                        break

                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    delay_cache_blocks=load_kv_async,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = self.waiting.pop_request()
                if load_kv_async:
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            if skipped_waiting_requests:
                self.waiting.prepend_requests(skipped_waiting_requests)

        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        self._action_tape.finish_tick(record, self, scheduler_output)
        return scheduler_output
