import unittest

import numpy as np

from connectome_lab.environments import (
    CUBE_ACTIONS,
    RubiksCubeEnv,
    StimulusActionEnv,
    TMazeEnv,
    build_cube_state_split,
)


class StimulusTests(unittest.TestCase):
    def test_hidden_reversed_mapping(self):
        env = StimulusActionEnv(cue_to_action=(1, 0))
        seen = set()
        for seed in range(20):
            observation, info = env.reset(seed=seed)
            cue = int(np.argmax(observation))
            seen.add(cue)
            self.assertEqual(info, {})
            self.assertEqual(observation.dtype, np.float32)
            self.assertEqual(env.step(1 - cue)[1:4], (1.0, True, False))
            with self.assertRaises(RuntimeError):
                env.step(0)
        self.assertEqual(seen, {0, 1})

    def test_wrong_action_and_invalid_actions(self):
        env = StimulusActionEnv()
        observation, _ = env.reset(seed=4)
        for invalid in (-1, 2, 0.5, True, "0"):
            with self.assertRaises(ValueError):
                env.step(invalid)
        self.assertEqual(env.step(1 - int(np.argmax(observation)))[1], -1.0)


class TMazeTests(unittest.TestCase):
    def test_delayed_cue_no_leakage_and_terminal_reward(self):
        env = TMazeEnv(corridor_steps=4)
        paths = {}
        for seed in range(30):
            initial, reset_info = env.reset(seed=seed)
            cue = int(np.argmax(initial))
            self.assertEqual(reset_info, {})
            path = []
            for step in range(4):
                observation, reward, terminated, truncated, info = env.step(step % 2)
                self.assertTrue(np.all(observation[:2] == 0))
                self.assertEqual(reward, 0)
                self.assertFalse(terminated or truncated)
                self.assertEqual(info, {})
                path.append(observation.copy())
            self.assertEqual(path[-1].tolist(), [0, 0, 0, 1])
            paths[cue] = path
            self.assertEqual(env.step(cue)[1:4], (1.0, True, False))
            if len(paths) == 2:
                break
        np.testing.assert_array_equal(paths[0], paths[1])

    def test_truncation_and_wrong_choice(self):
        short = TMazeEnv(corridor_steps=3, max_steps=2)
        short.reset(seed=0)
        short.step(0)
        self.assertEqual(short.step(1)[1:4], (0.0, False, True))
        with self.assertRaises(RuntimeError):
            short.step(0)
        env = TMazeEnv(corridor_steps=1)
        cue, _ = env.reset(seed=1)
        env.step(0)
        self.assertEqual(env.step(1 - int(np.argmax(cue)))[1:4], (-1.0, True, False))


class CubeTests(unittest.TestCase):
    def test_quarter_turns_and_inverses(self):
        for size in (2, 3):
            cube = RubiksCubeEnv(size=size)
            solved = cube.state_key()
            for move in range(12):
                for _ in range(4):
                    cube.apply_move(move)
                self.assertEqual(cube.state_key(), solved, (size, move))
                cube.apply_move(move)
                self.assertFalse(cube.is_solved())
                cube.apply_move(move ^ 1)
                self.assertEqual(cube.state_key(), solved)

    def test_sequence_inverse_conservation_and_centers(self):
        moves = [0, 2, 4, 9, 10, 6, 3, 5, 1, 8, 7, 11]
        for size in (2, 3):
            cube = RubiksCubeEnv(size=size)
            solved = cube.state_key()
            for move in moves:
                cube.apply_move(move)
                np.testing.assert_array_equal(np.bincount(cube.stickers.reshape(-1)), np.full(6, size * size))
                if size == 3:
                    np.testing.assert_array_equal(cube.stickers[:, 1, 1], np.arange(6))
            for move in reversed(moves):
                cube.apply_move(move ^ 1)
            self.assertEqual(cube.state_key(), solved)

    def test_known_group_relations_and_opposite_faces(self):
        for size in (2, 3):
            cube = RubiksCubeEnv(size=size)
            original = cube.state_key()
            # The sexy move has order six on both supported physical cubes.
            for _ in range(6):
                for move in ("R", "U", "R'", "U'"):
                    cube.apply_move(move)
            self.assertEqual(cube.state_key(), original)
            for first, second in (("U", "D"), ("R", "L"), ("F", "B")):
                for move in (first, second, first + "'", second + "'"):
                    cube.apply_move(move)
                self.assertEqual(cube.state_key(), original)

    def test_u_direction_from_geometry(self):
        cube = RubiksCubeEnv(size=3)
        cube.apply_move("U")
        # U clockwise maps the upper front row to the left face.
        for index, (position, normal) in enumerate(cube._geometry):
            if position[1] == 2 and normal == (-1, 0, 0):
                self.assertEqual(cube.state_key()[index], 2)

    def test_episode_success_truncation_and_observation(self):
        cube = RubiksCubeEnv(size=2, max_steps=1, step_cost=0.01)
        cube.apply_move("R")
        state = cube.state_key()
        observation, info = cube.reset(options={"state": state})
        self.assertEqual(info, {})
        self.assertEqual(observation.shape, (cube.observation_size,))
        self.assertEqual(observation.dtype, np.float32)
        np.testing.assert_array_equal(observation.reshape(-1, 6).sum(axis=1), 1)
        self.assertEqual(cube.step(CUBE_ACTIONS.index("R'"))[1:4], (1.0, True, False))
        with self.assertRaises(RuntimeError):
            cube.step(0)
        cube.reset(options={"state": state})
        self.assertEqual(cube.step(CUBE_ACTIONS.index("U"))[1:4], (-0.01, False, True))

    def test_seed_curriculum_no_solved_starts_and_no_history(self):
        first, second = RubiksCubeEnv(seed=10), RubiksCubeEnv(seed=10)
        for depth in (1, 2, 3, 4, 8):
            first.set_scramble_depth(depth)
            second.set_scramble_depth(depth)
            for _ in range(10):
                first_observation, info = first.reset()
                second_observation, _ = second.reset()
                np.testing.assert_array_equal(first_observation, second_observation)
                self.assertFalse(first.is_solved())
                self.assertEqual(info, {"scramble_depth": depth})
            self.assertFalse(any("history" in key or "solution" in key for key in vars(first)))
        np.testing.assert_array_equal(first.reset(seed=77)[0], second.reset(seed=77)[0])

    def test_input_validation(self):
        for size in (1, 4, True, 2.5):
            with self.assertRaises(ValueError):
                RubiksCubeEnv(size=size)
        cube = RubiksCubeEnv()
        cube.reset(seed=1)
        for invalid in (-1, 12, 0.5, True, "U"):
            with self.assertRaises(ValueError):
                cube.step(invalid)
        for invalid_move in ("X", "R2", 12):
            with self.assertRaises(ValueError):
                cube.apply_move(invalid_move)
        with self.assertRaises(ValueError):
            cube.reset(options={"state": [0] * 24})
        with self.assertRaises(ValueError):
            cube.reset(options={"state": tuple(np.repeat(np.arange(6), 4))})
        for bad_cost in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RubiksCubeEnv(step_cost=bad_cost)

    def test_state_split_is_unique_disjoint_and_reproducible(self):
        for size in (2, 3):
            train, test = build_cube_state_split(size=size, scramble_depth=2, n_train=40, n_test=20, seed=11)
            self.assertEqual(len(set(train)), 40)
            self.assertEqual(len(set(test)), 20)
            self.assertFalse(set(train) & set(test))
            self.assertEqual((train, test), build_cube_state_split(size=size, scramble_depth=2, n_train=40, n_test=20, seed=11))
            cube = RubiksCubeEnv(size=size)
            for state in train + test:
                cube.reset(options={"state": state})
                self.assertFalse(cube.is_solved())
        train, test = build_cube_state_split(scramble_depth=1, n_train=8, n_test=4)
        self.assertEqual(len(set(train + test)), 12)

    def test_small_pool_and_bounded_sampling_exhaustion(self):
        with self.assertRaisesRegex(ValueError, "Only 12 distinct"):
            build_cube_state_split(scramble_depth=1, n_train=10, n_test=3)
        with self.assertRaisesRegex(ValueError, "finite pool was not enumerated"):
            build_cube_state_split(scramble_depth=4, n_train=4, n_test=4, max_attempts=1)
        train, test = build_cube_state_split(scramble_depth=4, n_train=5, n_test=3)
        self.assertEqual(len(set(train + test)), 8)


if __name__ == "__main__":
    unittest.main()
