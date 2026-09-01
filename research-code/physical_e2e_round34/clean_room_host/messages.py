"""Versioned host-visible message identities and fail-closed ACL checks."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Tuple


class ProtocolViolation(RuntimeError):
    """A message cannot be admitted to the run."""


class Arm(str, Enum):
    OBSERVER_LOOP = "observer_loop"
    ORACLE_TARGET = "oracle_target"
    NO_TARGET = "no_target"


class MessageType(str, Enum):
    SYNC_BEACON = "SYNC_BEACON"
    LEADER_SAMPLE = "LEADER_SAMPLE"
    FOLLOWER_STATE = "FOLLOWER_STATE"
    OBSERVER_STATE = "OBSERVER_STATE"
    HEARTBEAT = "HEARTBEAT"
    RUN_CONTROL = "RUN_CONTROL"
    TELEMETRY_MIRROR = "TELEMETRY_MIRROR"
    FAULT_EVENT = "FAULT_EVENT"


HOST = 255
NODES = frozenset({0, 1, 2})


@dataclass(frozen=True)
class Frame:
    protocol_version: int
    message_type: MessageType
    experiment_prefix: str
    run_id: int
    sender_node: int
    receiver_nodes: Tuple[int, ...]
    arm: Arm
    sequence: int
    sender_tick: int
    sender_time_us: int
    payload: Mapping[str, object]
    crc_valid: bool = True


def _permitted(sender: int, receiver: int, kind: MessageType, arm: Arm) -> bool:
    if sender == HOST and receiver in NODES:
        return kind in {MessageType.SYNC_BEACON, MessageType.RUN_CONTROL}
    if sender in NODES and receiver == HOST:
        return kind in {
            MessageType.TELEMETRY_MIRROR,
            MessageType.HEARTBEAT,
            MessageType.FAULT_EVENT,
        }
    if sender == 0 and receiver == 1:
        if kind == MessageType.HEARTBEAT:
            return True
        return kind == MessageType.LEADER_SAMPLE and arm != Arm.NO_TARGET
    if sender == 0 and receiver == 2:
        return arm == Arm.ORACLE_TARGET and kind == MessageType.LEADER_SAMPLE
    if sender == 1 and receiver == 2:
        return kind in {
            MessageType.FOLLOWER_STATE,
            MessageType.OBSERVER_STATE,
            MessageType.HEARTBEAT,
        }
    return False


def validate_acl(frame: Frame) -> None:
    if not frame.receiver_nodes:
        raise ProtocolViolation("empty receiver set")
    if len(set(frame.receiver_nodes)) != len(frame.receiver_nodes):
        raise ProtocolViolation("duplicate receiver")
    for receiver in frame.receiver_nodes:
        if not _permitted(frame.sender_node, receiver, frame.message_type, frame.arm):
            raise ProtocolViolation(
                f"ACL denied {frame.sender_node}->{receiver} {frame.message_type.value}"
            )


def validate_identity(frame: Frame, *, protocol_version: int, experiment_prefix: str,
                      run_id: int, arm: Arm) -> None:
    if not frame.crc_valid:
        raise ProtocolViolation("invalid frame CRC")
    if frame.protocol_version != protocol_version:
        raise ProtocolViolation("protocol version mismatch")
    if frame.experiment_prefix != experiment_prefix:
        raise ProtocolViolation("experiment identity mismatch")
    if frame.run_id != run_id or frame.arm != arm:
        raise ProtocolViolation("run or arm mismatch")
    if frame.sequence < 0 or frame.sender_tick < 0 or frame.sender_time_us < 0:
        raise ProtocolViolation("negative causal field")
