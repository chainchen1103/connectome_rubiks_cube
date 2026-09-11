"""Sparse current-based LIF. Time is ms; voltage/current are dimensionless."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LIFConfig:
    dt_ms: float = 1.0
    tau_m_ms: float = 10.0
    tau_syn_ms: float = 12.0
    refractory_ms: float = 2.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    threshold: float = 1.0
    gain: float = 4.0
    noise_std: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("LIF parameters must be finite")
        if min(self.dt_ms, self.tau_m_ms, self.tau_syn_ms) <= 0:
            raise ValueError("Time constants must be positive")
        if self.dt_ms > min(self.tau_m_ms, self.tau_syn_ms):
            raise ValueError("dt must not exceed the shortest time constant")
        if self.refractory_ms < 0 or self.gain < 0 or self.noise_std < 0:
            raise ValueError("Refractory period, gain and noise must be nonnegative")
        if self.threshold <= max(self.v_rest, self.v_reset):
            raise ValueError("Threshold must exceed rest and reset")


def transmitter_signs(graph, unknown_sign=1.0, overrides=None):
    """Only GABA is negative by default; glutamate/unknown need assumptions.

    Drosophila glutamate can be inhibitory depending on receptor. No sign is
    inferred as measured physiology. Overrides are keyed by neuron ID.
    """
    if unknown_sign not in (-1, 1):
        raise ValueError("unknown_sign must be -1 or +1")
    overrides = overrides or {}
    if set(overrides) - {n.id for n in graph.neurons}:
        raise ValueError("Unknown neuron ID in transmitter overrides")
    signs = []
    for neuron in graph.neurons:
        nt = neuron.neurotransmitter.strip().lower()
        sign = -1.0 if nt in {"gaba", "gabaergic"} else 1.0 if nt in {
            "acetylcholine", "ach", "cholinergic"} else float(unknown_sign)
        sign = overrides.get(neuron.id, sign)
        if sign not in (-1, 1):
            raise ValueError("Transmitter signs must be -1 or +1")
        signs.append(sign)
    return np.asarray(signs)


class LIFNetwork:
    def __init__(self, graph, config=None, seed=0, sign_overrides=None):
        self.graph = graph
        self.config = config or LIFConfig()
        self.rng = np.random.default_rng(seed)
        self.signs = transmitter_signs(graph, overrides=sign_overrides)
        incoming = np.bincount(graph.target, weights=graph.synapse_count,
                               minlength=graph.n_neurons)
        self.weights = (self.config.gain * graph.synapse_count /
                        np.maximum(incoming[graph.target], 1.0)).astype(float)
        self.initial_weights = self.weights.copy()
        self.reset()

    def reset(self):
        n = self.graph.n_neurons
        self.voltage = np.full(n, self.config.v_rest)
        self.synaptic_current = np.zeros(n)
        self.refractory = np.zeros(n, dtype=np.int64)
        self.spikes = np.zeros(n, dtype=bool)
        self.time_ms = 0.0

    def step(self, external_current):
        c, g = self.config, self.graph
        external = np.asarray(external_current, dtype=float)
        if external.shape != (g.n_neurons,) or not np.isfinite(external).all():
            raise ValueError("Current must be a finite vector with one value per neuron")
        if not np.isfinite(self.weights).all() or (self.weights < 0).any():
            raise ValueError("Synaptic magnitudes must be finite and nonnegative")
        # One timestep axonal delay. E/I signs are fixed by source neuron.
        arrival = np.bincount(g.target, weights=self.weights * self.signs[g.source]
                              * self.spikes[g.source], minlength=g.n_neurons)
        self.synaptic_current *= math.exp(-c.dt_ms / c.tau_syn_ms)
        self.synaptic_current += arrival
        available = self.refractory == 0
        self.refractory = np.maximum(self.refractory - 1, 0)
        drive = external + self.synaptic_current
        if c.noise_std:
            drive = drive + self.rng.normal(0, c.noise_std, g.n_neurons)
        decay = math.exp(-c.dt_ms / c.tau_m_ms)
        candidate = c.v_rest + (self.voltage - c.v_rest) * decay + drive * (1 - decay)
        self.voltage = np.where(available, candidate, c.v_reset)
        self.spikes = available & (self.voltage >= c.threshold)
        self.voltage[self.spikes] = c.v_reset
        self.refractory[self.spikes] = math.ceil(c.refractory_ms / c.dt_ms)
        self.time_ms += c.dt_ms
        if not np.isfinite(self.voltage).all():
            raise FloatingPointError("Nonfinite neural activity")
        return self.spikes.copy()

    def run(self, currents, record_voltage=True):
        currents = np.asarray(currents, dtype=float)
        if currents.ndim != 2 or currents.shape[1] != self.graph.n_neurons:
            raise ValueError("Currents must have shape (steps, neurons)")
        spikes = np.zeros(currents.shape, dtype=bool)
        voltages = np.empty(currents.shape) if record_voltage else None
        for t, current in enumerate(currents):
            spikes[t] = self.step(current)
            if record_voltage:
                voltages[t] = self.voltage
        return {"spikes": spikes, "voltage": voltages,
                "dt_ms": self.config.dt_ms, "stability": activity_metrics(spikes, self.config.dt_ms)}


def activity_metrics(spikes, dt_ms=1.0):
    spikes = np.asarray(spikes)
    if spikes.ndim != 2 or not all(spikes.shape) or dt_ms <= 0:
        raise ValueError("Nonempty time-by-neuron activity and positive dt required")
    rates = spikes.mean(axis=0) * 1000.0 / dt_ms
    return {"mean_rate_hz": float(rates.mean()), "max_rate_hz": float(rates.max()),
            "silent_fraction": float(np.mean(rates == 0)),
            "active_fraction": float(np.mean(rates > 0)),
            "peak_population_fraction": float(spikes.mean(axis=1).max()),
            "all_finite": bool(np.isfinite(spikes).all())}
