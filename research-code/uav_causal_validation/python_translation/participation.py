"""Node-level residual participation and connectivity repair."""
from __future__ import annotations
import numpy as np


def residual_weights(adjacency: np.ndarray, residual: np.ndarray, threshold: float,
                     floor: float = 0.0) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=int)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if adjacency.shape != (len(residual), len(residual)):
        raise ValueError("adjacency and residual dimensions differ")
    if threshold <= 0 or not 0 <= floor <= 1:
        raise ValueError("invalid gate threshold or floor")
    trust = np.clip(threshold / np.maximum(residual, threshold), floor, 1.0)
    # Column j contains information entering agent j from source i.
    return adjacency.astype(float) * trust[:, None]


def is_strongly_connected(weights: np.ndarray) -> bool:
    graph = np.asarray(weights) > 0
    n = len(graph)
    def reachable(matrix):
        seen, stack = {0}, [0]
        while stack:
            node = stack.pop()
            for nxt in np.flatnonzero(matrix[node]):
                if int(nxt) not in seen:
                    seen.add(int(nxt)); stack.append(int(nxt))
        return len(seen) == n
    return reachable(graph) and reachable(graph.T)


def repair_connectivity(weights: np.ndarray, adjacency: np.ndarray,
                        residual: np.ndarray, minimum_weight: float) -> np.ndarray:
    """Restore lowest-residual missing edges until strong connectivity holds."""
    result = np.asarray(weights, dtype=float).copy()
    adjacency = np.asarray(adjacency, dtype=int)
    residual = np.asarray(residual, dtype=float).reshape(-1)
    if is_strongly_connected(result):
        return result
    candidates = [
        (residual[i], i, j) for i, j in zip(*np.nonzero(adjacency)) if result[i, j] == 0
    ]
    for _, source, receiver in sorted(candidates):
        result[source, receiver] = minimum_weight
        if is_strongly_connected(result):
            return result
    raise ValueError("declared adjacency cannot be repaired to strong connectivity")
