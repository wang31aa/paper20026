"""Transport boundary. Only real adapters may ever claim evidentiary eligibility."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Iterable

from .messages import Frame


class EvidenceClass(str, Enum):
    PHYSICAL_CANDIDATE = "PHYSICAL_CANDIDATE_PENDING_VALIDATION"
    UNREVIEWED_NON_EVIDENTIARY = "NON_EVIDENTIARY_UNREVIEWED_TRANSPORT"
    NON_EVIDENTIARY = "NON_EVIDENTIARY_LOOPBACK_OR_MOCK"


class HardwareTransport(ABC):
    """Abstract deterministic hardware link; implementations require separate review."""

    # Fail closed: a new adapter must explicitly opt into physical-candidate
    # status and still pass every external acceptance gate.
    evidence_class = EvidenceClass.UNREVIEWED_NON_EVIDENTIARY

    @abstractmethod
    def send(self, frame: Frame) -> None:
        raise NotImplementedError

    @abstractmethod
    def receive(self) -> Iterable[Frame]:
        raise NotImplementedError

    @abstractmethod
    def force_safe_zero(self) -> None:
        """Invoke the transport's independently implemented safety path."""
        raise NotImplementedError


class LoopbackTransport(HardwareTransport):
    """Test-only transport. Its output is categorically non-evidentiary."""

    evidence_class = EvidenceClass.NON_EVIDENTIARY

    def __init__(self) -> None:
        self.frames = []
        self.safe_zero_calls = 0

    def send(self, frame: Frame) -> None:
        self.frames.append(frame)

    def receive(self) -> Iterable[Frame]:
        pending = tuple(self.frames)
        self.frames.clear()
        return pending

    def force_safe_zero(self) -> None:
        self.safe_zero_calls += 1
