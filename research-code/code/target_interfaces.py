"""Explicit target-information boundary for future closed-loop comparisons."""
from __future__ import annotations
from typing import Protocol
import numpy as np

class TargetProvider(Protocol):
    label: str
    comparable_to_manuscript_observer: bool
    def state(self, t: float, true_state: np.ndarray) -> np.ndarray: ...

class OracleTarget:
    label = "oracle-target"
    comparable_to_manuscript_observer = False
    def state(self, t: float, true_state: np.ndarray) -> np.ndarray:
        return np.array(true_state, copy=True)

class EstimatedTargetUnavailable:
    label = "estimated-target-unavailable"
    comparable_to_manuscript_observer = False
    def state(self, t: float, true_state: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Exact observer equations, all gains/LMI matrices, initial conditions, "
            "and solver records are required before this interface may be used."
        )
