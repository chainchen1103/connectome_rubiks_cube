"""Local Hebbian, pair STDP and reward-modulated STDP on existing edges."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class PlasticityConfig:
    rule: str = "reward_stdp"
    learning_rate: float = 0.002
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0
    tau_eligibility_ms: float = 300.0
    a_plus: float = 1.0
    a_minus: float = 1.05
    min_weight: float = 0.0
    max_weight: float = 8.0

    def __post_init__(self):
        if self.rule not in {"none", "hebbian", "stdp", "reward_stdp"}:
            raise ValueError("Unknown plasticity rule")
        values = [v for k, v in self.__dict__.items() if k != "rule"]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Plasticity parameters must be finite")
        if min(self.tau_pre_ms, self.tau_post_ms, self.tau_eligibility_ms) <= 0:
            raise ValueError("Time constants must be positive")
        if min(self.learning_rate, self.a_plus, self.a_minus, self.min_weight) < 0:
            raise ValueError("Rates, amplitudes and weight bounds must be nonnegative")
        if self.max_weight <= self.min_weight:
            raise ValueError("Invalid weight bounds")


class LocalPlasticity:
    def __init__(self, network, config=None, mask=None):
        self.network = network
        self.config = config or PlasticityConfig()
        self.mask = np.ones(network.graph.n_edges, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if self.mask.shape != network.weights.shape:
            raise ValueError("Plasticity mask must match edges")
        self.reset()

    def reset(self):
        self.pre = np.zeros(self.network.graph.n_neurons)
        self.post = np.zeros_like(self.pre)
        self.eligibility = np.zeros(self.network.graph.n_edges)

    def observe(self, spikes):
        spikes = np.asarray(spikes, dtype=float)
        if spikes.shape != self.pre.shape or not np.isin(spikes, [0, 1]).all():
            raise ValueError("Spikes must be a binary neuron vector")
        c, g, dt = self.config, self.network.graph, self.network.config.dt_ms
        self.pre *= math.exp(-dt / c.tau_pre_ms)
        self.post *= math.exp(-dt / c.tau_post_ms)
        self.eligibility *= math.exp(-dt / c.tau_eligibility_ms)
        if c.rule == "hebbian":
            delta = spikes[g.source] * spikes[g.target]
        else:
            # Read OLD traces before adding this step's spikes: simultaneous
            # isolated pairs contribute zero, pre-before-post potentiates.
            delta = (c.a_plus * self.pre[g.source] * spikes[g.target]
                     - c.a_minus * spikes[g.source] * self.post[g.target])
        self.pre += spikes
        self.post += spikes
        self.eligibility += delta * self.mask
        if c.rule in {"hebbian", "stdp"}:
            self._apply(delta)

    def reward(self, reward):
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite")
        if self.config.rule == "reward_stdp":
            self._apply(reward * self.eligibility)

    def _apply(self, delta):
        c = self.config
        weights = self.network.weights
        weights[self.mask] = np.clip(weights[self.mask] + c.learning_rate * delta[self.mask],
                                     c.min_weight, c.max_weight)
