import copy
import json
import itertools
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from connectome_lab.environments import RubiksCubeEnv
from connectome_lab.graph import synthetic_graph
from connectome_lab.learning import NeuralAgent
from connectome_lab import training
from connectome_lab.training import train_timed


NEURAL_FIELDS = (
    'weights', 'initial_weights', 'projection', 'sensory', 'motor', 'voltage',
    'synaptic_current', 'refractory', 'spikes', 'neural_time_ms', 'activity',
    'eligibility', 'last_spikes', 'last_probabilities', 'rng', 'network_rng',
)


def read_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    return arrays, json.loads(str(arrays['progress']))


def run_decisions(graph, folder, count, **kwargs):
    """Bound tests by decisions, independent of machine and CI throughput."""
    stop = threading.Event()
    if count == 0:
        stop.set()
    frames = []

    def callback(status):
        frame = status.get('latest_frame')
        if status['state'] == 'running' and frame is not None and frame.get('action') is not None:
            frames.append(copy.deepcopy(frame))
        if status.get('interactions', 0) >= count:
            stop.set()

    result = train_timed(graph, folder, seconds=30, size=3, depth=4,
                         stop_event=stop, callback=callback, notify_interval=0,
                         **kwargs)
    return result, frames


def right_turn_start(env, seed=None, options=None):
    """A fixed R scramble cannot be solved by the U-only test trajectory."""
    if options is not None:
        return ORIGINAL_RESET(env, seed, options)
    ORIGINAL_RESET(env, seed)
    env._stickers = env._solved.copy()
    env.apply_move('R')
    return env._observe(), {}


ORIGINAL_RESET = RubiksCubeEnv.reset
ORIGINAL_ACT = NeuralAgent.act


def neural_u_turn(agent, observation, **kwargs):
    # Still advance actual spikes, eligibility and both RNGs in trajectory tests.
    ORIGINAL_ACT(agent, observation, **kwargs)
    return 0


class TimedTrainingTests(unittest.TestCase):
    def assert_neural_equal(self, expected, actual):
        for field in NEURAL_FIELDS:
            self.assertIn(field, actual)
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)

    def test_one_puzzle_continues_past_old_horizon_without_reset(self):
        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(RubiksCubeEnv, 'reset', right_turn_start), \
                patch.object(NeuralAgent, 'act', neural_u_turn):
            result, frames = run_decisions(graph, folder, 25)
            saved, progress = read_checkpoint(Path(folder) / 'checkpoint.npz')
        self.assertEqual(result['reason'], 'stopped')
        self.assertEqual(result['interactions'], 25)
        self.assertEqual(result['episode_steps'], 25)
        self.assertEqual(result['episodes'], 0)
        self.assertEqual(result['total_episodes'], 0)
        self.assertEqual(result['successes'], 0)
        self.assertEqual(progress['episode_steps'], 25)
        self.assertEqual([frame['step'] for frame in frames], list(range(1, 26)))
        expected = RubiksCubeEnv(size=3, max_steps=None)
        expected.reset(options={'state': result['target_state']})
        for frame in frames:
            expected.step(0)
            self.assertEqual(frame['state'], list(expected.state_key()))
            self.assertFalse(frame['terminated'])
            self.assertFalse(frame['truncated'])
            self.assertIn('reward_terms', frame)
        self.assertEqual(result['current_state'], list(expected.state_key()))
        self.assertNotEqual(result['target_state'], result['current_state'])
        self.assertAlmostEqual(float(saved['neural_time_ms']), 25 * 20)

    def test_stop_and_resume_preserves_current_cube_and_all_neural_state(self):
        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(RubiksCubeEnv, 'reset', right_turn_start), \
                patch.object(NeuralAgent, 'act', neural_u_turn):
            first, _ = run_decisions(graph, folder, 5)
            checkpoint = Path(folder) / 'checkpoint.npz'
            before, progress = read_checkpoint(checkpoint)
            second, _ = run_decisions(graph, folder, 0)
            after, next_progress = read_checkpoint(checkpoint)
            self.assert_neural_equal(before, after)
            for field in ('target_state', 'current_state', 'puzzle_id', 'episode_steps',
                          'total_interactions', 'cumulative_reward'):
                self.assertEqual(second[field], first[field], field)
            self.assertEqual(second['run_reward'], 0)
            self.assertEqual(second['run_number'], 2)
            self.assertTrue(second['resumed'])
            self.assertEqual(next_progress['current_state'], progress['current_state'])
            self.assertEqual(len(list((Path(folder) / 'history').glob('*-changes.npz'))), 2)
            np.testing.assert_array_equal(NeuralAgent.load(checkpoint, graph).network.weights,
                                          before['weights'])

    def test_split_execution_matches_uninterrupted_actions_rng_and_rewards(self):
        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as whole, tempfile.TemporaryDirectory() as split:
            full_result, full_frames = run_decisions(graph, whole, 29)
            first_result, first_frames = run_decisions(graph, split, 11)
            split_result, last_frames = run_decisions(graph, split, 18)
            full, full_progress = read_checkpoint(Path(whole) / 'checkpoint.npz')
            resumed, split_progress = read_checkpoint(Path(split) / 'checkpoint.npz')
        self.assertEqual(full_result['reason'], 'stopped')
        self.assertEqual(full_result['interactions'], 29)
        self.assertEqual(split_result['puzzle_id'], first_result['puzzle_id'])
        self.assertEqual(split_result['episode_steps'], 29)
        self.assert_neural_equal(full, resumed)
        for field in ('target_state', 'current_state', 'episode_steps', 'total_interactions',
                      'cumulative_reward'):
            self.assertEqual(full_result[field], split_result[field], field)
        def trajectory(frames):
            return [{key: value for key, value in frame.items() if key != 'puzzle_id'}
                    for frame in frames]
        self.assertEqual(trajectory(full_frames), trajectory(first_frames + last_frames))
        self.assertEqual(full_progress['reward_tracker'], split_progress['reward_tracker'])

    def test_periodic_checkpoint_contains_current_decision_boundary(self):
        graph = synthetic_graph()
        saves = []
        original_save = training._save
        clock = itertools.count(0, .001)

        def capture(path, agent, progress):
            original_save(path, agent, progress)
            saves.append(read_checkpoint(path))

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(training, '_save', capture), \
                patch.object(training.time, 'monotonic', side_effect=lambda: next(clock)), \
                patch.object(RubiksCubeEnv, 'reset', right_turn_start), \
                patch.object(NeuralAgent, 'act', neural_u_turn):
            result, frames = run_decisions(graph, folder, 7, checkpoint_interval=1e-9)
        periodic = [(arrays, progress) for arrays, progress in saves if progress['reason'] == 'running']
        self.assertTrue(periodic)
        for arrays, progress in periodic:
            step = progress['episode_steps']
            self.assertGreater(step, 0)
            self.assertLessEqual(step, 7)
            self.assertEqual(progress['current_state'], frames[step - 1]['state'])
            self.assertAlmostEqual(float(arrays['neural_time_ms']), step * 20)
        self.assertEqual(result['episode_steps'], 7)

    def test_deadline_stops_and_bad_config_does_not_overwrite_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            graph = synthetic_graph()
            result = train_timed(graph, folder, seconds=.001, size=3, depth=4)
            self.assertEqual(result['reason'], 'time_limit')
            checkpoint = Path(folder) / 'checkpoint.npz'
            original = checkpoint.read_bytes()
            invalid = ({'seconds': 0}, {'seconds': float('inf')}, {'size': 2, 'depth': 4},
                       {'size': 3, 'depth': 5}, {'checkpoint_interval': 0},
                       {'notify_interval': -1}, {'notify_interval': float('nan')})
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    train_timed(graph, folder, **kwargs)
                self.assertEqual(checkpoint.read_bytes(), original)
            other_graph = synthetic_graph(seed=987)
            with self.assertRaises(ValueError):
                train_timed(other_graph, folder, seconds=1, size=3, depth=4)
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_solution_ends_one_puzzle_then_new_puzzle_keeps_weights(self):
        # Fixed action is only a deterministic terminal-condition test fixture.
        def inverse_right(agent, observation, **kwargs):
            ORIGINAL_ACT(agent, observation, **kwargs)
            return 3

        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(RubiksCubeEnv, 'reset', right_turn_start), \
                patch.object(NeuralAgent, 'act', inverse_right):
            first, _ = run_decisions(graph, folder, 5)
            before, _ = read_checkpoint(Path(folder) / 'checkpoint.npz')
            self.assertEqual(first['reason'], 'solved')
            self.assertEqual(first['interactions'], 1)
            self.assertEqual(first['episodes'], 1)
            self.assertEqual(first['successes'], 1)
            second, _ = run_decisions(graph, folder, 0)
            after, _ = read_checkpoint(Path(folder) / 'checkpoint.npz')
            self.assertNotEqual(second['puzzle_id'], first['puzzle_id'])
            self.assertEqual(second['episode_steps'], 0)
            self.assertEqual(second['total_successes'], 1)
            self.assertEqual(second['total_episodes'], 1)
            np.testing.assert_array_equal(after['weights'], before['weights'])
            self.assertEqual(float(after['neural_time_ms']), 0)
            self.assertFalse(np.any(after['eligibility']))
            third, _ = run_decisions(graph, folder, 5)
            self.assertEqual(third['total_successes'], 2)
            self.assertEqual(third['total_episodes'], 2)

    def test_incomplete_continuous_checkpoint_is_rejected_without_replacement(self):
        graph = synthetic_graph()
        with tempfile.TemporaryDirectory() as folder:
            run_decisions(graph, folder, 3)
            checkpoint = Path(folder) / 'checkpoint.npz'
            original, progress = read_checkpoint(checkpoint)
            for missing in ('starting_state', 'current_state', 'reward_tracker', 'last_probabilities'):
                with self.subTest(missing=missing):
                    history_count = len(list((Path(folder) / 'history').glob('*.json')))
                    damaged = {key: value.copy() for key, value in original.items()}
                    if missing == 'last_probabilities':
                        damaged.pop(missing)
                    else:
                        damaged_progress = copy.deepcopy(progress)
                        damaged_progress.pop(missing)
                        damaged['progress'] = np.array(json.dumps(damaged_progress))
                    np.savez_compressed(checkpoint, **damaged)
                    before = checkpoint.read_bytes()
                    with self.assertRaises(ValueError):
                        run_decisions(graph, folder, 0)
                    self.assertEqual(checkpoint.read_bytes(), before)
                    self.assertEqual(len(list((Path(folder) / 'history').glob('*.json'))), history_count)

    def test_legacy_migration_backs_up_policy_and_restarts_known_initial_cube(self):
        graph = synthetic_graph()
        cube = RubiksCubeEnv(size=3, scramble_depth=4)
        cube.reset(seed=88)
        agent = NeuralAgent(graph, cube.observation_size, cube.n_actions, seed=88)
        agent.act(cube._observe(), learn=True)
        agent.reward(.25, learn=True)
        legacy = {'task': {'size': 3, 'depth': 4, 'max_steps': 16, 'failure_penalty': .1},
                  'total_episodes': 7, 'total_interactions': 115, 'total_successes': 0,
                  'run_count': 3, 'seed': 88, 'reason': 'time_limit',
                  'dataset_key': 'custom', 'starting_state': list(cube.state_key())}
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'checkpoint.npz'
            training._save(checkpoint, agent, legacy)
            # Recreate v1's missing boundary-output arrays, independently of v2 _save.
            old, _ = read_checkpoint(checkpoint)
            old.pop('last_spikes', None)
            old.pop('last_probabilities', None)
            np.savez_compressed(checkpoint, **old)
            original = checkpoint.read_bytes()
            result, _ = run_decisions(graph, folder, 0)
            saved, progress = read_checkpoint(checkpoint)
            backups = list(Path(folder).glob('checkpoint-v1-*.npz'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(result['current_state'], legacy['starting_state'])
            self.assertEqual(result['target_state'], legacy['starting_state'])
            self.assertEqual(result['episode_steps'], 0)
            self.assertEqual(result['total_interactions'], 115)
            self.assertEqual(result['total_episodes'], 0)
            self.assertEqual(result['run_number'], 4)
            np.testing.assert_array_equal(saved['weights'], old['weights'])
            self.assertEqual(float(saved['neural_time_ms']), 0)
            self.assertTrue(result.get('resume_notice'))
            self.assertEqual(progress['schema_version'], 2)
            self.assertEqual(progress['legacy_totals']['total_episodes'], 7)


if __name__ == '__main__':
    unittest.main()
