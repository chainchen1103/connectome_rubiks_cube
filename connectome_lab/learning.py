"""Reward-driven local motor-synapse adaptation; no solver or target labels.

The three factors are a presynaptic spike-rate trace, a centered stochastic
postsynaptic action signal, and scalar reward. Only existing incoming motor
edges change. This is a local policy semi-gradient, not a claim that pair-STDP
alone learns the tasks or a backpropagated replacement policy network.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from .dynamics import LIFNetwork, LIFConfig, activity_metrics


@dataclass(frozen=True)
class AgentConfig:
    steps_per_action: int = 20
    input_current: float = 3.5
    learning_rate: float = 0.08
    temperature: float = 0.1
    activity_tau_ms: float = 120.0
    eligibility_decay: float = 0.95
    max_weight: float = 8.0
    memory: bool = True

    def __post_init__(self):
        if not isinstance(self.steps_per_action, int) or self.steps_per_action < 1:
            raise ValueError("steps_per_action must be a positive integer")
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Agent parameters must be finite")
        if min(self.temperature, self.activity_tau_ms, self.max_weight) <= 0:
            raise ValueError("Temperature, activity time constant, max_weight must be positive")
        if min(self.input_current, self.learning_rate) < 0 or not 0 <= self.eligibility_decay <= 1:
            raise ValueError("Invalid learning rate, current or eligibility decay")


def graph_fingerprint(graph):
    digest = hashlib.sha256()
    digest.update(json.dumps([n.__dict__ for n in graph.neurons], sort_keys=True).encode())
    for values, dtype in ((graph.source, '<i8'), (graph.target, '<i8'), (graph.synapse_count, '<f8')):
        digest.update(memoryview(np.ascontiguousarray(values, dtype=dtype)).cast('B'))
    return digest.hexdigest()


class NeuralAgent:
    def __init__(self, graph, observation_size, n_actions, seed=0, config=None, lif_config=None,
                 sensory_indices=None, motor_indices=None):
        self.config = config or AgentConfig()
        self.network = LIFNetwork(graph, lif_config or LIFConfig(gain=1.2), seed)
        self.rng = np.random.default_rng(seed)
        if observation_size < 1 or n_actions < 2:
            raise ValueError("Positive observation size and at least two actions required")
        self.observation_size, self.n_actions = int(observation_size), int(n_actions)
        # Biological annotations are used when available. Fallback assignments
        # are fixed, disclosed engineering interfaces, independent of reward.
        sensory = [i for i, n in enumerate(graph.neurons) if n.role.lower() in {'sensory', 'input'}]
        motor = [i for i, n in enumerate(graph.neurons) if n.role.lower() in {'motor', 'output', 'motor-related'}]
        if sensory_indices is not None:
            sensory = list(sensory_indices)
        if motor_indices is not None:
            motor = list(motor_indices)
        self.interface_notes = []
        if not sensory:
            sensory = list(range(max(1, graph.n_neurons // 4)))
            self.interface_notes.append("No sensory annotation: first quarter of neuron IDs used as input adapter")
        if len(motor) < n_actions:
            motor = [i for i in range(graph.n_neurons) if i not in sensory][-max(n_actions, 12):]
            self.interface_notes.append("Insufficient motor annotation: final non-input neuron IDs used as output adapter")
        self.sensory = np.asarray(sensory, dtype=np.int64)
        self.motor = np.asarray(motor, dtype=np.int64)
        for indices in (self.sensory, self.motor):
            if len(set(indices.tolist())) != len(indices) or (indices < 0).any() or (indices >= graph.n_neurons).any():
                raise ValueError("Invalid or duplicate interface neuron indices")
        if len(self.motor) < n_actions or np.intersect1d(self.sensory, self.motor).size:
            raise ValueError("Need distinct sensory and motor populations and one motor per action")
        self.groups = [self.motor[a::n_actions] for a in range(n_actions)]
        # Fixed balanced projection. For large observations (cube), each input
        # neuron receives a seeded random subset; never use a trained encoder.
        self.projection = np.zeros((len(self.sensory), observation_size))
        if observation_size <= len(self.sensory):
            self.projection[np.arange(len(self.sensory)), np.arange(len(self.sensory)) % observation_size] = 1
        else:
            for j in range(observation_size):
                chosen = self.rng.choice(len(self.sensory), min(3, len(self.sensory)), replace=False)
                self.projection[chosen, j] = 1
            self.projection /= np.maximum(self.projection.sum(axis=1, keepdims=True) / 6, 1)
        self.edge_action = np.full(graph.n_edges, -1, dtype=np.int64)
        self.edge_group_size = np.ones(graph.n_edges)
        for a, group in enumerate(self.groups):
            mask = np.isin(graph.target, group)
            self.edge_action[mask] = a
            self.edge_group_size[mask] = len(group)
        self.plastic_mask = self.edge_action >= 0
        self.reset()

    def reset(self):
        self.network.reset()
        self.activity = np.zeros(self.network.graph.n_neurons)
        self.eligibility = np.zeros(self.network.graph.n_edges)
        self.last_spikes = None
        self.last_probabilities = np.full(self.n_actions, 1 / self.n_actions)

    def act(self, observation, *, learn=True, greedy=False):
        obs = np.asarray(observation, dtype=float)
        if obs.shape != (self.observation_size,) or not np.isfinite(obs).all() or (obs < 0).any():
            raise ValueError("Observation must be finite, nonnegative and match the encoder")
        c, g, network = self.config, self.network.graph, self.network
        if not c.memory:
            network.reset()
            self.activity.fill(0)
        current = np.zeros(g.n_neurons)
        current[self.sensory] = c.input_current * (self.projection @ obs)
        spikes = np.array([network.step(current) for _ in range(c.steps_per_action)])
        rates = spikes.mean(axis=0)
        decay = math.exp(-c.steps_per_action * network.config.dt_ms / c.activity_tau_ms) if c.memory else 0.0
        self.activity = decay * self.activity + (1 - decay) * rates
        # Motor dendritic current is projected through actual graph edges.
        # Low-pass neuronal rates carry a finite decaying activity memory.
        edge_signal = network.signs[g.source] * self.activity[g.source]
        motor_drive = np.bincount(g.target, weights=network.weights * edge_signal, minlength=g.n_neurons)
        logits = np.array([motor_drive[group].mean() for group in self.groups]) / c.temperature
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        if greedy:
            winners = np.flatnonzero(np.isclose(logits, logits.max(), rtol=0, atol=1e-12))
            action = int(self.rng.choice(winners))
        else:
            action = int(self.rng.choice(self.n_actions, p=probabilities))
        self.last_probabilities = probabilities
        self.last_spikes = spikes
        if learn:
            self.eligibility *= c.eligibility_decay
            idx = self.plastic_mask
            post = (self.edge_action[idx] == action).astype(float) - probabilities[self.edge_action[idx]]
            self.eligibility[idx] += edge_signal[idx] * post / (self.edge_group_size[idx] * c.temperature)
        return action

    def reward(self, reward, *, learn=True):
        if not math.isfinite(reward):
            raise ValueError("Reward must be finite")
        if learn:
            idx = self.plastic_mask
            self.network.weights[idx] = np.clip(self.network.weights[idx] + self.config.learning_rate * reward
                                                 * self.eligibility[idx], 0, self.config.max_weight)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema_version": 1, "graph_sha256": graph_fingerprint(self.network.graph),
                    "config": asdict(self.config), "lif_config": asdict(self.network.config),
                    "observation_size": self.observation_size, "n_actions": self.n_actions,
                    "interface_notes": self.interface_notes}
        with path.open('wb') as stream:
            np.savez_compressed(stream, weights=self.network.weights, projection=self.projection,
                                sensory=self.sensory, motor=self.motor, metadata=json.dumps(metadata))
        return path

    @classmethod
    def load(cls, path, graph, seed=0):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            if meta['schema_version'] != 1 or meta['graph_sha256'] != graph_fingerprint(graph):
                raise ValueError("Checkpoint graph/version does not match")
            agent = cls(graph, meta['observation_size'], meta['n_actions'], seed,
                        AgentConfig(**meta['config']), LIFConfig(**meta['lif_config']),
                        data['sensory'], data['motor'])
            weights, projection = data['weights'], data['projection']
            if weights.shape != agent.network.weights.shape or not np.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Invalid checkpoint weights")
            if projection.shape != agent.projection.shape or not np.isfinite(projection).all():
                raise ValueError("Invalid checkpoint projection")
            agent.network.weights[:] = weights
            agent.projection[:] = projection
            agent.interface_notes = meta['interface_notes']
        return agent


def run_episode(agent, env, *, seed, learn=False, state=None, greedy=False, visited_states=None):
    agent.reset()
    kwargs = {"seed": int(seed)}
    if state is not None:
        kwargs['options'] = {'state': state}
    obs, _ = env.reset(**kwargs)
    if visited_states is not None:
        visited_states.add(env.state_key())
    total, interactions = 0.0, 0
    while True:
        action = agent.act(obs, learn=learn, greedy=greedy)
        obs, reward, terminated, truncated, info = env.step(action)
        if visited_states is not None:
            visited_states.add(env.state_key())
        agent.reward(reward, learn=learn)
        total += reward
        interactions += 1
        if terminated or truncated:
            return {"reward": total, "success": bool(info.get('success', terminated and reward > 0)),
                    "interactions": interactions, "terminated": terminated, "truncated": truncated}


def evaluate(agent, env_factory, episodes=100, seed=10000, states=None, greedy=False):
    """Frozen policy; preserve training RNG, all neural state and eligibility."""
    import copy
    if episodes < 1 or (states is not None and len(states) == 0):
        raise ValueError("Evaluation requires episodes and nonempty states")
    evaluator = copy.deepcopy(agent)
    evaluator.rng = np.random.default_rng(seed)
    evaluator.network.rng = np.random.default_rng(seed)
    rows = [run_episode(evaluator, env_factory(), seed=seed + i, learn=False,
                        state=None if states is None else states[i % len(states)], greedy=greedy)
            for i in range(episodes)]
    successes = sum(r['success'] for r in rows)
    p, n = successes / episodes, episodes
    z = 1.96
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {"success_rate": p, "episodes": episodes, "successes": successes,
            "wilson95": [center-half, center+half], "mean_return": float(np.mean([r['reward'] for r in rows])),
            "mean_steps": float(np.mean([r['interactions'] for r in rows])),
            "interval_note": "Descriptive Bernoulli interval; repeated states are not independent generalization samples"}
