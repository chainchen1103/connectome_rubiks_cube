"""Validated directed, weighted graphs and explicit null-model controls."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Neuron:
    id: str
    neuron_type: str = "unknown"
    neurotransmitter: str = "unknown"
    role: str = "inter"
    region: str = "unknown"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Connectome:
    neurons: list[Neuron]
    source: np.ndarray
    target: np.ndarray
    synapse_count: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.neurons = list(self.neurons)
        ids = [n.id for n in self.neurons]
        if any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError("Neuron IDs must be nonempty strings (never convert root IDs to float).")
        if len(set(ids)) != len(ids):
            raise ValueError("Neuron IDs must be unique.")
        if any(not np.isfinite([n.x, n.y, n.z]).all() for n in self.neurons):
            raise ValueError("Neuron coordinates must be finite.")
        arrays = []
        for name, value in (("source", self.source), ("target", self.target)):
            raw = np.asarray(value)
            if raw.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional.")
            if raw.size and (raw.dtype.kind not in "iu" or np.any(raw < 0) or np.any(raw >= len(ids))):
                raise ValueError(f"{name} must contain valid integer neuron indices.")
            arrays.append(raw.astype(np.int64, copy=True))
        self.source, self.target = arrays
        self.synapse_count = np.asarray(self.synapse_count, dtype=np.float64).copy()
        if self.synapse_count.ndim != 1 or not (len(self.source) == len(self.target) == len(self.synapse_count)):
            raise ValueError("Edge arrays must be one-dimensional with equal length.")
        if not np.isfinite(self.synapse_count).all() or np.any(self.synapse_count <= 0):
            raise ValueError("Synapse counts must be finite and positive.")
        if len(np.unique(self.source * len(ids) + self.target)) != self.n_edges:
            raise ValueError("Duplicate directed edges must be aggregated before constructing a graph.")
        self.metadata = dict(self.metadata)

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return len(self.source)

    def degrees(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (out_degree, in_degree), counting edges rather than contacts."""
        return (np.bincount(self.source, minlength=self.n_neurons),
                np.bincount(self.target, minlength=self.n_neurons))


def subgraph(graph: Connectome, *, region: str | None = None,
             ids: Iterable[str] | None = None) -> Connectome:
    """Induced graph selected by neuron region and/or exact neuron IDs.

    Region describes neuron annotations, not an edge's synaptic neuropil.
    """
    wanted = None if ids is None else set(ids)
    known = {n.id for n in graph.neurons}
    if wanted is not None and not wanted <= known:
        raise ValueError(f"Unknown neuron IDs: {sorted(wanted - known)[:5]}")
    keep = np.array([i for i, n in enumerate(graph.neurons)
                     if (region is None or n.region == region)
                     and (wanted is None or n.id in wanted)], dtype=np.int64)
    remap = np.full(graph.n_neurons, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    edge_keep = (remap[graph.source] >= 0) & (remap[graph.target] >= 0)
    metadata = dict(graph.metadata)
    metadata["selection"] = {"region": region, "neuron_ids": [graph.neurons[i].id for i in keep],
                             "parent_neurons": graph.n_neurons, "parent_edges": graph.n_edges}
    return Connectome([graph.neurons[i] for i in keep], remap[graph.source[edge_keep]],
                      remap[graph.target[edge_keep]], graph.synapse_count[edge_keep], metadata)


def connectivity(graph: Connectome, seeds: Iterable[str], *,
                 direction: str = "downstream", hops: int = 1) -> list[str]:
    """IDs reachable within at most ``hops`` edges, including the seed IDs."""
    if direction not in {"upstream", "downstream", "both"}:
        raise ValueError("direction must be upstream, downstream, or both.")
    if isinstance(hops, bool) or not isinstance(hops, int) or hops < 0:
        raise ValueError("hops must be a nonnegative integer.")
    id_to_i = {n.id: i for i, n in enumerate(graph.neurons)}
    try:
        reached = {id_to_i[s] for s in seeds}
    except KeyError as exc:
        raise ValueError(f"Unknown seed neuron: {exc.args[0]}") from exc
    neighbors: list[set[int]] = [set() for _ in graph.neurons]
    for s, t in zip(graph.source, graph.target):
        if direction in {"downstream", "both"}:
            neighbors[s].add(int(t))
        if direction in {"upstream", "both"}:
            neighbors[t].add(int(s))
    frontier = reached.copy()
    for _ in range(hops):
        frontier = {j for i in frontier for j in neighbors[i]} - reached
        reached |= frontier
        if not frontier:
            break
    return [n.id for i, n in enumerate(graph.neurons) if i in reached]


def _control_metadata(graph: Connectome, control: str, seed: int) -> dict:
    metadata = dict(graph.metadata)
    metadata.update({"kind": "control", "control": control, "seed": seed,
                     "parent_kind": graph.metadata.get("kind", "unspecified"),
                     "biological_topology": False})
    return metadata


def degree_preserving_shuffle(graph: Connectome, seed: int = 0,
                              swaps: int | None = None) -> Connectome:
    """Directed double-edge swaps preserve each node's in/out degree.

    Counts remain attached to their source edge, preserving outgoing strength.
    Incoming strength is not preserved. Rejected swaps create neither duplicate
    edges nor self-loops. Limited attempts mean some rigid graphs cannot mix;
    achieved swap counts are always reported in metadata.
    """
    if np.any(graph.source == graph.target):
        raise ValueError("Remove self-loops explicitly before making loopless controls.")
    swaps = graph.n_edges * 10 if swaps is None else swaps
    if isinstance(swaps, bool) or not isinstance(swaps, int) or swaps < 0:
        raise ValueError("swaps must be a nonnegative integer.")
    rng = np.random.default_rng(seed)
    source, target = graph.source.copy(), graph.target.copy()
    edge_set = set(zip(source.tolist(), target.tolist()))
    completed = attempts = 0
    max_attempts = max(100, swaps * 30) if swaps else 0
    while completed < swaps and attempts < max_attempts and graph.n_edges > 1:
        attempts += 1
        a, b = rng.choice(graph.n_edges, 2, replace=False)
        s1, t1, s2, t2 = int(source[a]), int(target[a]), int(source[b]), int(target[b])
        if s1 == s2 or t1 == t2 or s1 == t2 or s2 == t1:
            continue
        if (s1, t2) in edge_set or (s2, t1) in edge_set:
            continue
        edge_set.remove((s1, t1))
        edge_set.remove((s2, t2))
        edge_set.update(((s1, t2), (s2, t1)))
        target[a], target[b] = t2, t1
        completed += 1
    metadata = _control_metadata(graph, "directed_degree_preserving", seed)
    metadata.update({"requested_swaps": swaps, "completed_swaps": completed,
                     "swap_attempts": attempts, "mixing_guaranteed": False,
                     "preserves": ["in_degree", "out_degree", "outgoing_strength", "count_multiset"],
                     "does_not_preserve": ["incoming_strength", "motifs", "edge_neuropil"]})
    return Connectome(graph.neurons, source, target, graph.synapse_count, metadata)


def random_control(graph: Connectome, seed: int = 0) -> Connectome:
    """Uniform directed G(n,m) graph, same nodes, edges and weight multiset."""
    if np.any(graph.source == graph.target):
        raise ValueError("Remove self-loops explicitly before making loopless controls.")
    rng = np.random.default_rng(seed)
    n, m = graph.n_neurons, graph.n_edges
    # Sampling indices avoids an n-by-n adjacency allocation.
    pairs = rng.choice(n * max(n - 1, 0), size=m, replace=False)
    source = pairs // max(n - 1, 1)
    target = pairs % max(n - 1, 1)
    target += target >= source
    metadata = _control_metadata(graph, "directed_gnm", seed)
    metadata.update({"preserves": ["neuron_count", "edge_count", "count_multiset"],
                     "does_not_preserve": ["degrees", "strengths", "motifs", "edge_neuropil"]})
    return Connectome(graph.neurons, source, target, rng.permutation(graph.synapse_count), metadata)


def synthetic_graph(seed: int = 0) -> Connectome:
    """68-neuron engineering fixture; this is NOT measured fly connectivity."""
    rng = np.random.default_rng(seed)
    neurons = []
    for i in range(68):
        role = "sensory" if i < 24 else "inter" if i < 56 else "motor"
        nt = "GABA" if role == "inter" and i % 5 == 0 else "acetylcholine"
        col = 0 if role == "sensory" else 1 if role == "inter" else 2
        neurons.append(Neuron(f"synthetic-{i:03d}", f"synthetic_{role}", nt, role,
                              "synthetic", float(col), float(i % 24), float(rng.uniform(-1, 1))))
    edges: dict[tuple[int, int], float] = {}
    for s in range(68):
        candidates = np.arange(24, 56) if s < 24 else np.arange(24, 68)
        candidates = candidates[candidates != s]
        for t in rng.choice(candidates, min(6, len(candidates)), replace=False):
            edges[s, int(t)] = float(rng.integers(1, 9))
    # Symmetric direct stimulus paths support learning without a prewired action.
    for s in range(24):
        for t in range(56, 68):
            edges[(s, t)] = 4.0
    pairs = sorted(edges)
    return Connectome(neurons, np.array([s for s, _ in pairs], dtype=np.int64),
                      np.array([t for _, t in pairs], dtype=np.int64),
                      np.array([edges[p] for p in pairs]),
                      {"kind": "synthetic", "seed": seed, "biological_topology": False,
                       "description": "Engineering fixture, not a reconstructed fly circuit.",
                       "coordinates": "synthetic layout", "neurotransmitters": "synthetic assignments"})
