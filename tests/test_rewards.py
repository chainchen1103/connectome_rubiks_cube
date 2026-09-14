import copy
import json
import unittest

import numpy as np

from connectome_lab.environments import RubiksCubeEnv
from connectome_lab.rewards import CubeRewardConfig, CubeRewardTracker, cube_potential


class CubeRewardTests(unittest.TestCase):
    def cube(self, size=3):
        cube = RubiksCubeEnv(size=size, max_steps=None)
        cube.apply_move('R')
        cube.reset(options={'state': cube.state_key()})
        return cube

    def transition(self, tracker, cube, action):
        before = cube.state_key()
        _, _, solved, _, _ = cube.step(action)
        return tracker.step(before, action, cube.state_key(), solved)

    def test_pair_potential_known_values_and_orientation_invariance(self):
        for size, turned in ((2, 5 / 9), (3, 2 / 3)):
            solved = RubiksCubeEnv(size=size).stickers
            self.assertEqual(cube_potential(solved, size), 1)
            cube = self.cube(size)
            self.assertAlmostEqual(cube_potential(cube.state_key(), size), turned)
            state = cube.stickers[[3, 0, 5, 1, 2, 4], ::-1, ::-1]
            # Renaming colors and reorienting faces must not change coherence.
            self.assertEqual(cube_potential(5 - state, size), cube_potential(cube.stickers, size))

    def test_progress_and_solved_bonus_with_signed_diagnostics(self):
        cube = self.cube()
        tracker = CubeRewardTracker(3, initial_state=cube.state_key())
        terms = self.transition(tracker, cube, cube.action_names.index("R'"))
        self.assertGreater(terms['progress_reward'], 0)
        self.assertEqual(terms['solved_bonus'], 1)
        self.assertEqual(terms['potential_after'], 1)
        self.assertEqual(terms['step_penalty'], -.002)
        self.assertGreater(terms['reward'], 1)
        self.assertEqual(terms['reward'], sum(terms[key] for key in (
            'progress_reward', 'step_penalty', 'inverse_penalty', 'revisit_penalty', 'solved_bonus')))

    def test_unsolved_closed_cycles_cannot_farm_progress_reward(self):
        for moves, expected in (([0, 1], -.034), ([0, 0, 0, 0], -.028)):
            cube = self.cube()
            initial = cube.state_key()
            tracker = CubeRewardTracker(3, initial_state=initial)
            terms = [self.transition(tracker, cube, action) for action in moves]
            self.assertEqual(cube.state_key(), initial)
            self.assertAlmostEqual(sum(row['progress_reward'] for row in terms), 0)
            self.assertAlmostEqual(sum(row['reward'] for row in terms), expected)
            self.assertTrue(terms[-1]['is_revisit'])
            self.assertEqual(terms[-1]['revisit_penalty'], -.02)
            self.assertEqual(terms[-1]['is_inverse'], moves == [0, 1])

    def test_snapshot_resumes_inverse_and_recent_state_memory(self):
        cube = self.cube()
        tracker = CubeRewardTracker(3, initial_state=cube.state_key())
        self.transition(tracker, cube, 0)
        saved = json.loads(json.dumps(tracker.snapshot()))
        restored = CubeRewardTracker.from_snapshot(saved, size=3, config=CubeRewardConfig())
        before = cube.state_key()
        cube.step(1)
        expected = tracker.step(before, 1, cube.state_key(), False)
        actual = restored.step(before, 1, cube.state_key(), False)
        self.assertEqual(actual, expected)
        self.assertTrue(actual['is_inverse'] and actual['is_revisit'])
        self.assertEqual(restored.snapshot(), tracker.snapshot())

    def test_recent_history_is_bounded_and_duplicate_counts_survive_eviction(self):
        cube = self.cube()
        tracker = CubeRewardTracker(3, CubeRewardConfig(recent_window=3))
        for index in range(20):
            terms = self.transition(tracker, cube, index % 2)
            if index:
                self.assertTrue(terms['is_revisit'])
            self.assertLessEqual(len(tracker.snapshot()['recent_states']), 3)
        # A four-turn loop falls out of a two-state memory window.
        cube = self.cube()
        tracker = CubeRewardTracker(3, CubeRewardConfig(recent_window=2))
        for _ in range(4):
            terms = self.transition(tracker, cube, 0)
        self.assertFalse(terms['is_revisit'])

    def test_invalid_configuration_state_and_discontinuous_transition(self):
        for name, value in (('step_cost', -1), ('inverse_penalty', float('nan')),
                            ('solved_bonus', float('inf')), ('progress_scale', True),
                            ('recent_window', 0), ('recent_window', 1.5)):
            with self.assertRaises(ValueError):
                CubeRewardConfig(**{name: value})
        for state in ([0] * 54, [6] * 54, [0] * 24, [True] * 54):
            with self.assertRaises(ValueError):
                cube_potential(state, 3)
        cube = self.cube()
        tracker = CubeRewardTracker(3, initial_state=cube.state_key())
        self.transition(tracker, cube, 0)
        wrong_before = self.cube().state_key()
        with self.assertRaisesRegex(ValueError, 'continue'):
            tracker.step(wrong_before, 0, cube.state_key(), False)
        with self.assertRaisesRegex(ValueError, 'solved'):
            tracker.step(cube.state_key(), 0, cube.state_key(), True)

    def test_snapshot_rejects_mismatch_and_corruption(self):
        tracker = CubeRewardTracker(3, initial_state=self.cube().state_key())
        saved = tracker.snapshot()
        for kwargs in ({'size': 2}, {'config': CubeRewardConfig(step_cost=.5)}):
            with self.assertRaises(ValueError):
                CubeRewardTracker.from_snapshot(saved, **kwargs)
        for field, value in (('version', 999), ('previous_action', 12),
                             ('recent_states', [[0] * 54]), ('config', {})):
            bad = copy.deepcopy(saved)
            bad[field] = value
            with self.assertRaises(ValueError):
                CubeRewardTracker.from_snapshot(bad)
        saved['recent_states'] = saved['recent_states'] * 257
        with self.assertRaises(ValueError):
            CubeRewardTracker.from_snapshot(saved)


if __name__ == '__main__':
    unittest.main()
