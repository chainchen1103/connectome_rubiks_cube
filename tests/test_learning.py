import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from connectome_lab.dynamics import LIFConfig
from connectome_lab.environments import StimulusActionEnv
from connectome_lab.graph import synthetic_graph
from connectome_lab.learning import AgentConfig, NeuralAgent, evaluate, run_episode


class LearningTests(unittest.TestCase):
    def make_agent(self, *, seed=4, memory=True, noise_std=0.0):
        return NeuralAgent(
            synthetic_graph(seed=0), 2, 2, seed=seed,
            config=AgentConfig(memory=memory),
            lif_config=LIFConfig(gain=1.2, noise_std=noise_std),
        )

    def assert_agent_state_equal(self, first, second):
        for name in ("activity", "eligibility", "last_probabilities", "projection", "plastic_mask"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name), err_msg=name)
        if first.last_spikes is None:
            self.assertIsNone(second.last_spikes)
        else:
            np.testing.assert_array_equal(first.last_spikes, second.last_spikes)
        for name in ("voltage", "synaptic_current", "refractory", "spikes", "weights", "initial_weights"):
            np.testing.assert_array_equal(getattr(first.network, name), getattr(second.network, name), err_msg=name)
        self.assertEqual(first.network.time_ms, second.network.time_ms)
        self.assertEqual(first.rng.bit_generator.state, second.rng.bit_generator.state)
        self.assertEqual(first.network.rng.bit_generator.state, second.network.rng.bit_generator.state)

    def test_evaluation_preserves_training_state_and_future_randomness(self):
        agent = self.make_agent(noise_std=0.2)
        agent.act([1, 0], learn=True)
        agent.reward(1)
        agent.act([0, 1], learn=True)
        before = copy.deepcopy(agent)
        first = evaluate(agent, StimulusActionEnv, episodes=12, seed=91)
        second = evaluate(agent, StimulusActionEnv, episodes=12, seed=91)
        self.assertEqual(first, second)
        self.assert_agent_state_equal(agent, before)
        self.assertEqual(agent.act([1, 0]), before.act([1, 0]))
        agent.reward(-1)
        before.reward(-1)
        self.assert_agent_state_equal(agent, before)

    def test_learning_changes_only_existing_motor_incoming_edges(self):
        agent = self.make_agent(memory=False)
        graph = agent.network.graph
        before = agent.network.weights.copy()
        sources, targets = graph.source.copy(), graph.target.copy()
        signs = agent.network.signs.copy()
        for episode in range(30):
            run_episode(agent, StimulusActionEnv(), seed=episode, learn=True)
        changed = agent.network.weights != before
        allowed = np.isin(graph.target, agent.motor)
        self.assertTrue(changed.any(), "the learning smoke test must actually change weights")
        self.assertFalse(np.any(changed & ~allowed))
        np.testing.assert_array_equal(agent.plastic_mask, allowed)
        np.testing.assert_array_equal(graph.source, sources)
        np.testing.assert_array_equal(graph.target, targets)
        np.testing.assert_array_equal(agent.network.signs, signs)
        self.assertTrue(np.all(agent.network.weights >= 0))
        self.assertTrue(np.all(agent.network.weights[allowed] <= agent.config.max_weight))

    def test_checkpoint_roundtrip_and_graph_fingerprint_rejection(self):
        agent = self.make_agent(memory=False)
        for episode in range(15):
            run_episode(agent, StimulusActionEnv(), seed=episode, learn=True)
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "policy.npz"
            self.assertEqual(agent.save(checkpoint), checkpoint)
            loaded = NeuralAgent.load(checkpoint, agent.network.graph, seed=4)
            np.testing.assert_array_equal(loaded.network.weights, agent.network.weights)
            np.testing.assert_array_equal(loaded.projection, agent.projection)
            np.testing.assert_array_equal(loaded.sensory, agent.sensory)
            np.testing.assert_array_equal(loaded.motor, agent.motor)
            self.assertEqual(loaded.config, agent.config)
            for cue in ([1, 0], [0, 1]):
                agent.reset()
                loaded.reset()
                agent.act(cue, learn=False)
                loaded.act(cue, learn=False)
                np.testing.assert_array_equal(loaded.last_probabilities, agent.last_probabilities)
            altered = copy.deepcopy(agent.network.graph)
            altered.synapse_count[0] += 1
            with self.assertRaisesRegex(ValueError, "graph/version"):
                NeuralAgent.load(checkpoint, altered)

    def test_checkpoint_preserves_seeded_large_observation_encoder(self):
        graph = synthetic_graph(seed=0)
        agent = NeuralAgent(graph, observation_size=40, n_actions=2, seed=42)
        with tempfile.TemporaryDirectory() as folder:
            path = agent.save(Path(folder) / "encoder.npz")
            loaded = NeuralAgent.load(path, graph, seed=999)
            np.testing.assert_array_equal(loaded.projection, agent.projection)
            observation = np.zeros(40)
            observation[[0, 9, 25]] = 1
            agent.act(observation, learn=False)
            loaded.act(observation, learn=False)
            np.testing.assert_array_equal(loaded.last_probabilities, agent.last_probabilities)

    def test_no_memory_clears_activity_and_neural_state_between_observations(self):
        agent = self.make_agent(memory=False)
        fresh = copy.deepcopy(agent)
        agent.act([1, 0], learn=False)
        self.assertTrue(np.any(agent.activity > 0))
        agent.act([0, 0], learn=False)
        fresh.act([0, 0], learn=False)
        np.testing.assert_array_equal(agent.activity, fresh.activity)
        np.testing.assert_array_equal(agent.last_probabilities, fresh.last_probabilities)
        for field in ("voltage", "synaptic_current", "refractory", "spikes"):
            np.testing.assert_array_equal(getattr(agent.network, field), getattr(fresh.network, field))
        self.assertEqual(agent.network.time_ms, fresh.network.time_ms)
        # The same cue survives a blank observation in the memory condition.
        memory = self.make_agent(memory=True)
        memory.act([1, 0], learn=False)
        memory.act([0, 0], learn=False)
        self.assertTrue(np.any(memory.activity > 0))

    def test_invalid_observations_fail_before_changing_state(self):
        agent = self.make_agent()
        before = copy.deepcopy(agent)
        for observation in ([1], [1, 0, 0], [[1, 0]], [np.nan, 0], [np.inf, 0], [-1, 0]):
            with self.assertRaises(ValueError):
                agent.act(observation)
            self.assert_agent_state_equal(agent, before)

    def test_frozen_episode_does_not_change_weights_or_accumulate_eligibility(self):
        agent = self.make_agent()
        before = agent.network.weights.copy()
        for episode in range(10):
            run_episode(agent, StimulusActionEnv(), seed=episode, learn=False)
        np.testing.assert_array_equal(agent.network.weights, before)
        np.testing.assert_array_equal(agent.eligibility, np.zeros_like(agent.eligibility))

    def test_small_training_run_is_deterministic(self):
        first, second = self.make_agent(), self.make_agent()
        for episode in range(25):
            left = run_episode(first, StimulusActionEnv(cue_to_action=(1, 0)), seed=episode, learn=True)
            right = run_episode(second, StimulusActionEnv(cue_to_action=(1, 0)), seed=episode, learn=True)
            self.assertEqual(left, right)
        self.assert_agent_state_equal(first, second)

    def test_reversed_stimulus_mapping_learns_from_scalar_rewards(self):
        # This task needs no memory. A broad threshold avoids demanding an
        # exact stochastic score; 400 trials run in approximately one second.
        agent = self.make_agent(memory=False)
        factory = lambda: StimulusActionEnv(cue_to_action=(1, 0))
        before = evaluate(agent, factory, episodes=200, seed=400)["success_rate"]
        for episode in range(400):
            run_episode(agent, factory(), seed=episode, learn=True)
        after = evaluate(agent, factory, episodes=200, seed=400)["success_rate"]
        self.assertGreater(after, 0.75)
        self.assertGreater(after - before, 0.15)


if __name__ == "__main__":
    unittest.main()
