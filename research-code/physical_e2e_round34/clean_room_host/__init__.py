"""Clean-room host orchestration skeleton; contains no acquisition results."""

from .core import HostOrchestrator, HostState, SafetyLatch
from .messages import Arm, MessageType, Frame, ProtocolViolation
from .transport import HardwareTransport, LoopbackTransport, EvidenceClass

__all__ = [
    "Arm",
    "EvidenceClass",
    "Frame",
    "HardwareTransport",
    "HostOrchestrator",
    "HostState",
    "LoopbackTransport",
    "MessageType",
    "ProtocolViolation",
    "SafetyLatch",
]
