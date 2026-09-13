"""Replay rows must be synchronized with real simulator and neural transitions."""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from connectome_lab import dashboard
from connectome_lab.embodiment import FlyArenaEnv
from connectome_lab.environments import RubiksCubeEnv
from connectome_lab.graph import synthetic_graph
from connectome_lab.learning import NeuralAgent, graph_fingerprint


class DashboardTrackTests(unittest.TestCase):
    def setUp(self):
        self.graph = synthetic_graph(seed=4)

    def assert_neural_frame(self, frame, actions):
        self.assertEqual(len(frame["voltage"]), self.graph.n_neurons)
        self.assertEqual(len(frame["rates"]), self.graph.n_neurons)
        self.assertEqual(len(frame["probabilities"]), actions)
        self.assertTrue(np.isfinite(frame["voltage"]).all())
        self.assertTrue(np.isfinite(frame["rates"]).all())
        self.assertTrue(np.all(np.asarray(frame["rates"]) >= 0))
        self.assertTrue(np.all(np.asarray(frame["probabilities"]) >= 0))
        self.assertAlmostEqual(sum(frame["probabilities"]), 1., places=5)
        self.assertEqual(frame["spikes"], sorted(set(frame["spikes"])))
        self.assertTrue(all(0 <= index < self.graph.n_neurons for index in frame["spikes"]))

    def test_cube_recording_replays_exact_sticker_transitions_for_both_sizes(self):
        seed = 9
        for size in (2, 3):
            with self.subTest(size=size):
                track = dashboard.record_track(self.graph, "cube", frames=25, seed=seed, size=size)
                self.assertEqual(len(track["frames"]), 25)
                self.assertEqual(len(track["geometry"]), 6 * size * size)
                self.assertEqual(track["graph_sha256"], graph_fingerprint(self.graph))
                self.assertEqual(track["policy"], "untrained stochastic neural policy")
                cube = RubiksCubeEnv(size=size, scramble_depth=2, max_steps=18)
                for frame in track["frames"]:
                    self.assert_neural_frame(frame, actions=12)
                    if frame["action"] is None:
                        cube.reset(seed=seed + 10000 + frame["episode"])
                        self.assertEqual(frame["step"], 0)
                        self.assertEqual(frame["spikes"], [])
                    else:
                        _, reward, terminated, truncated, _ = cube.step(frame["action"])
                        self.assertEqual(frame["reward"], reward)
                        self.assertEqual(frame["terminated"], terminated)
                        self.assertEqual(frame["truncated"], truncated)
                    self.assertEqual(frame["state"], list(cube.state_key()))
                self.assertTrue(any(frame["spikes"] for frame in track["frames"]))
                self.assertTrue(any(np.any(frame["voltage"]) for frame in track["frames"]))
                self.assertEqual(track, dashboard.record_track(self.graph, "cube", frames=25, seed=seed, size=size))

    def test_recorded_neural_values_are_the_actual_decision_window(self):
        captured = []
        original_act = NeuralAgent.act

        def capture(agent, observation, **kwargs):
            action = original_act(agent, observation, **kwargs)
            captured.append({"voltage": np.round(agent.network.voltage, 5).tolist(),
                             "rates": np.round(agent.last_spikes.mean(axis=0) * 1000 / agent.network.config.dt_ms, 2).tolist(),
                             "spikes": np.flatnonzero(agent.last_spikes.any(axis=0)).tolist(),
                             "probabilities": np.round(agent.last_probabilities, 6).tolist(), "action": action})
            return action

        with patch.object(NeuralAgent, "act", new=capture):
            track = dashboard.record_track(self.graph, "cube", frames=8, seed=3)
        decisions = [frame for frame in track["frames"] if frame["action"] is not None]
        self.assertEqual(len(decisions), len(captured))
        for frame, expected in zip(decisions, captured):
            for field, values in expected.items():
                self.assertEqual(frame[field], values, field)

    def test_fly_recording_obeys_chosen_action_and_screen_heading(self):
        track = dashboard.record_track(self.graph, "fly", frames=30, seed=4)
        previous = None
        for frame in track["frames"]:
            self.assert_neural_frame(frame, actions=3)
            if frame["action"] is None:
                self.assertEqual((frame["x"], frame["y"]), (35., 50.))
                previous = frame
                continue
            action = frame["action"]
            expected_heading = previous["heading"]
            if action == 0:
                expected_x = np.clip(previous["x"] + 3 * math.cos(expected_heading), 3, 97)
                expected_y = np.clip(previous["y"] + 3 * math.sin(expected_heading), 3, 97)
                self.assertAlmostEqual(frame["x"], expected_x)
                self.assertAlmostEqual(frame["y"], expected_y)
            else:
                self.assertEqual((frame["x"], frame["y"]), (previous["x"], previous["y"]))
                expected_heading += (-1 if action == 1 else 1) * math.pi / 8
            expected_heading = (expected_heading + math.pi) % (2 * math.pi) - math.pi
            self.assertAlmostEqual(frame["heading"], expected_heading)
            self.assertEqual(frame["food"], previous["food"])
            before = math.hypot(previous["food"][0] - previous["x"], previous["food"][1] - previous["y"])
            after = math.hypot(frame["food"][0] - frame["x"], frame["food"][1] - frame["y"])
            self.assertAlmostEqual(frame["reward"], (before - after) * .01 + (1. if after < 6 else 0.))
            previous = frame
        self.assertEqual({frame["action"] for frame in track["frames"] if frame["action"] is not None}, {0, 1, 2})

    def test_zero_training_recordings_keep_weights_and_eligibility_frozen(self):
        agents = []

        def make_agent(*args, **kwargs):
            agent = NeuralAgent(*args, **kwargs)
            agents.append(agent)
            return agent

        source, target, contacts = self.graph.source.copy(), self.graph.target.copy(), self.graph.synapse_count.copy()
        with patch.object(dashboard, "NeuralAgent", side_effect=make_agent):
            for kind in ("cube", "fly"):
                track = dashboard.record_track(self.graph, kind, frames=12, train_episodes=0)
                self.assertEqual(track["training_episodes"], 0)
                self.assertEqual(track["policy"], "untrained stochastic neural policy")
        for agent in agents:
            np.testing.assert_array_equal(agent.network.weights, agent.network.initial_weights)
            np.testing.assert_array_equal(agent.eligibility, np.zeros(self.graph.n_edges))
        np.testing.assert_array_equal(self.graph.source, source)
        np.testing.assert_array_equal(self.graph.target, target)
        np.testing.assert_array_equal(self.graph.synapse_count, contacts)

    def test_checkpoint_replay_is_labeled_and_rejects_an_incompatible_task(self):
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "cube.npz"
            cube = RubiksCubeEnv(size=2)
            agent = NeuralAgent(self.graph, cube.observation_size, cube.n_actions, seed=7)
            agent.save(checkpoint)
            original = checkpoint.read_bytes()
            track = dashboard.record_track(self.graph, "cube", size=2, frames=4, checkpoint=checkpoint)
            self.assertEqual(track["policy"], "frozen checkpoint")
            self.assertEqual(checkpoint.read_bytes(), original)
            for kind, size in (("cube", 3), ("fly", 2)):
                with self.subTest(kind=kind, size=size), self.assertRaisesRegex(ValueError, "environment"):
                    dashboard.record_track(self.graph, kind, size=size, frames=4, checkpoint=checkpoint)

    def test_recording_rejects_invalid_request(self):
        for kwargs in ({"frames": 1}, {"train_episodes": -1}, {"kind": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                dashboard.record_track(self.graph, **kwargs)


class FlyArenaTests(unittest.TestCase):
    def test_forward_left_right_and_observation_are_consistent(self):
        arena = FlyArenaEnv(max_steps=3)
        observation, _ = arena.reset(seed=11)
        self.assertEqual(observation.shape, (6,))
        self.assertTrue(np.all((observation >= 0) & (observation <= 1)))
        initial = arena.pose()
        arena.step(1)
        self.assertAlmostEqual(arena.heading, initial["heading"] - math.pi / 8)
        self.assertEqual((arena.x, arena.y), (initial["x"], initial["y"]))
        arena.step(2)
        self.assertAlmostEqual(arena.heading, initial["heading"])
        observation, _, terminated, truncated, _ = arena.step(0)
        self.assertAlmostEqual(arena.x, initial["x"] + 3 * math.cos(initial["heading"]))
        self.assertAlmostEqual(arena.y, initial["y"] + 3 * math.sin(initial["heading"]))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            arena.step(0)
        for action in (-1, 3, True, 1.5):
            with self.subTest(action=action), self.assertRaises(ValueError):
                arena.step(action)


if __name__ == "__main__":
    unittest.main()
