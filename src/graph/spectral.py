"""Spectral utilities for network diagnostics (GCC-focused).

Implements algebraic connectivity (lambda_2) and the Fiedler vector."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class SpectralResult:
    lambda2: float
    fiedler_vector: dict[str, float]  # node -> value


def compute_algebraic_connectivity_and_fiedler(
    G: nx.Graph,
    *,
    weight: str | None = None,
) -> SpectralResult:
    """Compute lambda_2 (algebraic connectivity) and the Fiedler vector.

    Notes:
    - The Fiedler vector sign is arbitrary. Callers may orient/flip the sign
      for interpretability (e.g., align with bank labels).
    - For disconnected graphs, lambda_2 = 0 and the Fiedler vector is not
      uniquely defined. This function expects a connected graph (e.g., GCC).
    """
    if G.number_of_nodes() < 2:
        raise ValueError("Graph must have at least 2 nodes for spectral metrics.")
    if not nx.is_connected(G):
        raise ValueError("Graph must be connected. Compute metrics on the GCC.")

    # Stable node ordering
    nodes_sorted = sorted(G.nodes(), key=lambda x: str(x))

    from scipy.sparse.linalg import eigsh  # type: ignore

    L = nx.laplacian_matrix(G, nodelist=nodes_sorted, weight=weight).astype(float)
    # Smallest eigenvalue of Laplacian is 0 (connected), we need the 2 smallest.
    vals, vecs = eigsh(L, k=2, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    lambda2 = float(vals[1])
    fiedler = vecs[:, 1]

    fiedler_map = {str(n): float(v) for n, v in zip(nodes_sorted, fiedler, strict=True)}
    return SpectralResult(lambda2=lambda2, fiedler_vector=fiedler_map)
