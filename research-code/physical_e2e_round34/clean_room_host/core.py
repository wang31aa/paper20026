"""Fail-closed host acquisition state machine; never computes control commands."""

from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

from .hashlog import AppendOnlyHashLog
from .messages import Arm, Frame, ProtocolViolation, validate_acl, validate_identity
from .transport import HardwareTransport


class HostState(str, Enum):
    BOOT_SAFE = "BOOT_SAFE"
    SELF_TEST = "SELF_TEST"
    CONFIG_VERIFIED = "CONFIG_VERIFIED"
    ARMED_ZERO = "ARMED_ZERO"
    RUNNING = "RUNNING"
    SAFE_ZERO_LATCHED = "SAFE_ZERO_LATCHED"
    POWER_ISOLATED = "POWER_ISOLATED"


class SafetyLatch(RuntimeError):
    pass


class HostOrchestrator:
    """Orchestrates identity and safety only; contains no observer/controller API."""

    def __init__(self, transport: HardwareTransport, log_path: Path, *,
                 protocol_version: int, experiment_prefix: str) -> None:
        self.transport = transport
        self.log = AppendOnlyHashLog(log_path)
        self.protocol_version = protocol_version
        self.experiment_prefix = experiment_prefix
        self.state = HostState.BOOT_SAFE
        self.run_id: Optional[int] = None
        self.arm: Optional[Arm] = None
        self.current_tick = -1
        self._last_sequences: Dict[Tuple[int, str], int] = {}
        self.transport.force_safe_zero()
        self._event("BOOT_SAFE", reason="constructor_default_zero")

    def _event(self, kind: str, **fields: object) -> None:
        self.log.append({
            "event_type": kind,
            "host_state": self.state.value,
            "evidence_class": self.transport.evidence_class.value,
            **fields,
        })

    def self_test_passed(self) -> None:
        self._transition(HostState.BOOT_SAFE, HostState.SELF_TEST)

    def verify_configuration(self, *, run_id: int, arm: Arm,
                             full_hashes_match: bool, clocks_valid: bool) -> None:
        if self.state != HostState.SELF_TEST:
            self.latch_safe_zero("configuration_out_of_order")
        if run_id < 0 or not full_hashes_match or not clocks_valid:
            self.latch_safe_zero("configuration_or_clock_failure")
        self.run_id, self.arm = run_id, arm
        self._transition(HostState.SELF_TEST, HostState.CONFIG_VERIFIED)

    def arm_zero(self, *, operator_confirmed: bool, all_heartbeats_valid: bool,
                 guards_closed: bool) -> None:
        if not (operator_confirmed and all_heartbeats_valid and guards_closed):
            self.latch_safe_zero("arm_guard_failure")
        self._transition(HostState.CONFIG_VERIFIED, HostState.ARMED_ZERO)

    def start(self, *, start_tick: int) -> None:
        if start_tick < 0:
            self.latch_safe_zero("invalid_start_tick")
        self.current_tick = start_tick
        self._transition(HostState.ARMED_ZERO, HostState.RUNNING)

    def advance_tick(self, tick: int) -> None:
        if self.state != HostState.RUNNING or tick != self.current_tick + 1:
            self.latch_safe_zero("non_consecutive_host_tick")
        self.current_tick = tick
        self._event("TICK_ADVANCED", tick=tick)

    def admit(self, frame: Frame, *, maximum_age_ticks: int = 0) -> None:
        try:
            if self.state != HostState.RUNNING or self.run_id is None or self.arm is None:
                raise ProtocolViolation("frame outside running state")
            validate_identity(frame, protocol_version=self.protocol_version,
                              experiment_prefix=self.experiment_prefix,
                              run_id=self.run_id, arm=self.arm)
            validate_acl(frame)
            key = (frame.sender_node, frame.message_type.value)
            if frame.sequence <= self._last_sequences.get(key, -1):
                raise ProtocolViolation("repeated or non-monotone sequence")
            if frame.sender_tick > self.current_tick:
                raise ProtocolViolation("future-tick frame")
            if self.current_tick - frame.sender_tick > maximum_age_ticks:
                raise ProtocolViolation("stale frame")
            self._last_sequences[key] = frame.sequence
            self._event("FRAME_ADMITTED", sender=frame.sender_node,
                        message_type=frame.message_type.value,
                        sequence=frame.sequence, sender_tick=frame.sender_tick)
        except ProtocolViolation as exc:
            self._event("FRAME_REJECTED", reason=str(exc), sender=frame.sender_node,
                        message_type=frame.message_type.value)
            self.latch_safe_zero("protocol_violation")

    def latch_safe_zero(self, reason: str) -> None:
        self.transport.force_safe_zero()
        self.state = HostState.SAFE_ZERO_LATCHED
        self._event("SAFE_ZERO_LATCHED", reason=reason)
        raise SafetyLatch(reason)

    def isolate_power(self) -> None:
        if self.state != HostState.SAFE_ZERO_LATCHED:
            self.latch_safe_zero("power_isolation_without_latch")
        self.state = HostState.POWER_ISOLATED
        self._event("POWER_ISOLATED")

    def _transition(self, expected: HostState, target: HostState) -> None:
        if self.state != expected:
            self.latch_safe_zero(f"invalid_transition_{self.state.value}_to_{target.value}")
        self.state = target
        self._event("STATE_TRANSITION", target=target.value)
