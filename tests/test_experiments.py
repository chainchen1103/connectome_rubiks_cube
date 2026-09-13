"""Short experiment runs check artifacts and honest held-out accounting."""

import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from connectome_lab import experiments
from connectome_lab.environments import RubiksCubeEnv
from connectome_lab.graph import synthetic_graph
from connectome_lab.learning import NeuralAgent, graph_fingerprint, run_episode


def read_finite_json(path):
    def reject_constant(value):
        raise AssertionError(f"Non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.graph = synthetic_graph(seed=7)

    def assert_checkpoint_finite(self, path):
        with np.load(path, allow_pickle=False) as archive:
            for key in ("weights", "projection", "sensory", "motor"):
                self.assertTrue(np.isfinite(archive[key]).all(), key)
        agent = NeuralAgent.load(path, self.graph)
        self.assertEqual(agent.network.graph.n_edges, self.graph.n_edges)
        return agent

    def test_small_training_runs_write_finite_reusable_artifacts(self):
        for task in ("stimulus", "tmaze", "cube2", "cube3"):
            with self.subTest(task=task):
                output = self.output / task
                report = experiments.run_training(self.graph, output, task=task,
                                                  episodes=3, eval_episodes=3, seed=2)
                self.assertEqual(read_finite_json(output / "results.json"), report)
                self.assertEqual(report["metadata"]["graph_sha256"], graph_fingerprint(self.graph))
                self.assertTrue((output / "report.html").stat().st_size > 100)
                self.assert_checkpoint_finite(output / "agent.npz")
                result = report["runs"][0]
                self.assertEqual(len(result["training_rewards"]), 3)
                self.assertEqual(result["before"]["episodes"], 3)
                self.assertEqual(result["after"]["episodes"], 3)
                self.assertTrue(result["stability"]["all_finite"])
                self.assertGreaterEqual(result["changed_edges"], 0)
                self.assertLessEqual(result["changed_edges"], result["plastic_edges"])
                for field in ("before_success", "after_success"):
                    self.assertTrue(0 <= result[field] <= 1)
                with (output / "episodes.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row["episode"]) for row in rows], [1, 2, 3])
                interactions = np.array([int(row["interactions"]) for row in rows])
                self.assertEqual(result["interactions"], int(interactions.sum()))
                self.assertEqual([int(row["cumulative_interactions"]) for row in rows],
                                 interactions.cumsum().tolist())
                if task.startswith("cube"):
                    split = read_finite_json(output / "split.json")
                    train, test = set(map(tuple, split["train"])), set(map(tuple, split["test"]))
                    self.assertFalse(train & test)
                    self.assertEqual(result["split"]["initial_overlap"], 0)
                    self.assertFalse(result["split"]["symmetry_disjoint"])
                if task == "tmaze":
                    self.assertEqual(set(result["delay_generalization"]), {"3", "6", "12"})
                    self.assertEqual(result["post_training_memory_ablation"]["episodes"], 3)

    def test_frozen_training_keeps_original_weights_and_same_evaluation(self):
        report = experiments.run_training(self.graph, self.output, episodes=5,
                                          eval_episodes=5, seed=1, frozen=True)
        result = report["runs"][0]
        self.assertEqual(result["variant"], "frozen")
        self.assertEqual(result["before"], result["after"])
        self.assertEqual(result["weight_change_l1"], 0)
        self.assertEqual(result["changed_edges"], 0)
        loaded = self.assert_checkpoint_finite(self.output / "agent.npz")
        original = NeuralAgent(self.graph, 2, 2, seed=1)
        np.testing.assert_array_equal(loaded.network.weights, original.network.weights)

    def test_benchmark_covers_controls_and_ablations_with_finite_summary(self):
        report = experiments.run_benchmark(self.graph, self.output, tasks=("tmaze",),
                                           seeds=(0,), episodes=2, eval_episodes=2)
        expected = {(topology, variant) for topology in ("original", "degree_shuffled", "random")
                    for variant in ("plastic", "frozen", "memory_reset")}
        self.assertEqual({(run["topology"], run["variant"]) for run in report["runs"]}, expected)
        self.assertEqual(len(report["summary"]), 9)
        for row in report["summary"]:
            self.assertEqual(row["seeds"], 1)
            self.assertTrue(math.isfinite(row["after_mean"]))
            self.assertEqual(row["after_std"], 0)
        self.assertEqual(read_finite_json(self.output / "results.json"), report)
        self.assertTrue((self.output / "connectome.html").exists())

    def test_curriculum_global_holdout_is_disjoint_from_every_stage(self):
        observed = set()

        def record_training(*args, **kwargs):
            result = run_episode(*args, **kwargs)
            observed.update(kwargs["visited_states"])
            return result

        with patch.object(experiments, "run_episode", side_effect=record_training):
            report = experiments.run_curriculum(self.graph, self.output, depths=(1, 2),
                                                episodes=2, eval_episodes=2, seed=5)
        split = read_finite_json(self.output / "splits.json")
        heldout = set(map(tuple, split["heldout"]))
        for states in split["training_by_depth"].values():
            self.assertFalse(heldout & set(map(tuple, states)))
        unseen = heldout - observed
        audit = report["curriculum"]
        self.assertEqual(set(map(tuple, split["strict_unseen"])), unseen)
        self.assertEqual(audit["strict_unseen_count"], len(unseen))
        self.assertEqual(audit["heldout_visited_in_training_trajectories"], len(heldout & observed))
        self.assertEqual(audit["all_training_initial_overlap"], 0)
        self.assertFalse(audit["symmetry_disjoint"])
        self.assertTrue(all("not generalization" in run["evaluation_scope"] for run in report["runs"]))
        self.assertEqual(read_finite_json(self.output / "results.json"), report)
        self.assert_checkpoint_finite(self.output / "agent.npz")

    def test_curriculum_reports_a_real_trajectory_crossing_the_holdout(self):
        def state(*moves):
            cube = RubiksCubeEnv()
            for move in moves:
                cube.apply_move(move)
            return cube.state_key()

        start, leaked, unseen = state("U"), state("U'"), state("R'")
        # A depth-3 holdout may also contain one-turn configurations. The
        # earlier stage must exclude them, yet its trajectory U -> U2 -> U'
        # can still cross the holdout. Both events need separate accounting.
        def split_fixture(size, depth, *args, **kwargs):
            return ([start, leaked] if depth == 1 else [start]), [leaked, unseen]

        original_act = NeuralAgent.act

        def turn_u(agent, observation, **kwargs):
            original_act(agent, observation, **kwargs)
            return 0  # Execute U while retaining real neural activity traces.

        with patch.object(experiments, "build_cube_state_split", side_effect=split_fixture), \
                patch.object(NeuralAgent, "act", new=turn_u):
            report = experiments.run_curriculum(self.graph, self.output, depths=(1, 3),
                                                episodes=2, eval_episodes=2)
        split = read_finite_json(self.output / "splits.json")
        self.assertEqual(split["training_by_depth"]["1"], [list(start)])
        self.assertEqual(split["strict_unseen"], [list(unseen)])
        audit = report["curriculum"]
        self.assertEqual(audit["heldout_visited_in_training_trajectories"], 1)
        self.assertEqual(audit["strict_unseen_count"], 1)
        self.assertIsNotNone(audit["strict_unseen_evaluation"])

    def test_rejects_empty_or_invalid_experiment_requests(self):
        for kwargs in ({"episodes": 0}, {"eval_episodes": 0}, {"task": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                experiments.run_training(self.graph, self.output, **kwargs)
        for depths in ((), (2, 1), (1, 1)):
            with self.subTest(depths=depths), self.assertRaises(ValueError):
                experiments.run_curriculum(self.graph, self.output, depths=depths, episodes=2)
        with self.assertRaises(ValueError):
            experiments.run_benchmark(self.graph, self.output, seeds=())


if __name__ == "__main__":
    unittest.main()
