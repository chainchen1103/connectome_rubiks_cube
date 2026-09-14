"""Small, dependency-light tasks with a Gym-style API.

Observations are one-dimensional float32 arrays. Task mappings and scramble
histories are never included in observations or ``info``. The cube is a sticker
simulator: permutations are derived from integer 3-D geometry, not a solver.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral
from typing import Sequence

import numpy as np


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _action(action: int, count: int) -> int:
    action = _integer(action, "action")
    if action >= count:
        raise ValueError(f"action must be in [0, {count})")
    return action


def _mapping(mapping: Sequence[int]) -> tuple[int, int]:
    if len(mapping) != 2:
        raise ValueError("cue_to_action must be a permutation of (0, 1)")
    result = tuple(_action(action, 2) for action in mapping)
    if set(result) != {0, 1}:
        raise ValueError("cue_to_action must be a permutation of (0, 1)")
    return result


class StimulusActionEnv:
    """One randomized A/B cue followed by one binary choice.

    ``cue_to_action=(1, 0)`` reverses the hidden contingency. A cue's identity
    is observed, but its correct action is not supplied to the agent.
    """

    n_actions = 2
    observation_size = 2
    action_names = ("left", "right")

    def __init__(self, cue_to_action: Sequence[int] = (0, 1), seed: int | None = None):
        self._cue_to_action = _mapping(cue_to_action)
        self._rng = np.random.default_rng(seed)
        self._done = True

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cue = int(self._rng.integers(2))
        self._done = False
        return np.eye(2, dtype=np.float32)[self._cue], {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = _action(action, self.n_actions)
        if self._done:
            raise RuntimeError("reset() is required before stepping a finished episode")
        success = action == self._cue_to_action[self._cue]
        self._done = True
        return np.zeros(2, dtype=np.float32), 1.0 if success else -1.0, True, False, {"success": success}


class TMazeEnv:
    """Remember a cue through a corridor, then choose left or right.

    Reset shows [A, B, corridor, choice]. Only A or B is active at reset;
    subsequent observations never contain the cue. Exactly ``corridor_steps``
    calls to step advance to the choice point, ignoring the binary action.
    The next call makes the choice and is the only rewarded transition.
    """

    n_actions = 2
    observation_size = 4
    action_names = ("left", "right")

    def __init__(
        self,
        corridor_steps: int = 3,
        cue_to_action: Sequence[int] = (0, 1),
        max_steps: int | None = None,
        seed: int | None = None,
    ):
        self.corridor_steps = _integer(corridor_steps, "corridor_steps", 1)
        self.max_steps = self.corridor_steps + 1 if max_steps is None else _integer(max_steps, "max_steps", 1)
        self._cue_to_action = _mapping(cue_to_action)
        self._rng = np.random.default_rng(seed)
        self._done = True

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cue = int(self._rng.integers(2))
        self._steps = 0
        self._done = False
        observation = np.zeros(self.observation_size, dtype=np.float32)
        observation[self._cue] = 1.0
        return observation, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = _action(action, self.n_actions)
        if self._done:
            raise RuntimeError("reset() is required before stepping a finished episode")
        at_choice = self._steps >= self.corridor_steps
        self._steps += 1
        observation = np.zeros(self.observation_size, dtype=np.float32)
        if at_choice:
            success = action == self._cue_to_action[self._cue]
            self._done = True
            return observation, 1.0 if success else -1.0, True, False, {"success": success}
        observation[3 if self._steps >= self.corridor_steps else 2] = 1.0
        truncated = self._steps >= self.max_steps
        self._done = truncated
        return observation, 0.0, False, truncated, {}


# Face order is also the order of color channels and sticker blocks.
_FACES = (("U", 1, 1), ("R", 0, 1), ("F", 2, 1), ("D", 1, -1), ("L", 0, -1), ("B", 2, -1))
CUBE_ACTIONS = tuple(name + suffix for name, _, _ in _FACES for suffix in ("", "'"))


def _rotate(vector: tuple[int, int, int], axis: int, quarter: int) -> tuple[int, int, int]:
    x, y, z = vector
    if axis == 0:
        return (x, -quarter * z, quarter * y)
    if axis == 1:
        return (quarter * z, y, -quarter * x)
    return (-quarter * y, quarter * x, z)


def _cube_geometry(size: int) -> tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]:
    extent = size - 1
    coordinates = range(-extent, extent + 1, 2)
    geometry = []
    for _, axis, sign in _FACES:
        varying = [dimension for dimension in range(3) if dimension != axis]
        for first, second in product(coordinates, repeat=2):
            position = [0, 0, 0]
            position[axis] = sign * extent
            position[varying[0]] = first
            position[varying[1]] = second
            normal = [0, 0, 0]
            normal[axis] = sign
            geometry.append((tuple(position), tuple(normal)))
    return tuple(geometry)


class RubiksCubeEnv:
    """Exact 2x2 or 3x3 cube, quarter turns, sparse terminal success.

    ``scramble_depth`` is the number of random turns, not optimal solution
    distance. Immediate inverse moves and solved starts are excluded. Use
    :func:`build_cube_state_split` when measuring held-out generalization;
    merely assigning different random seeds does not separate cube states.

    ``reset(options={"state": key})`` loads a supplied configuration, useful
    for a precomputed split. Include ``steps`` to resume its move count.
    ``max_steps=None`` keeps one episode running until solved; callers can
    impose a wall-time limit without resetting the cube. No scramble path is
    retained or returned.
    """

    action_names = CUBE_ACTIONS
    n_actions = len(CUBE_ACTIONS)

    def __init__(
        self,
        size: int = 2,
        scramble_depth: int = 1,
        max_steps: int | None = 20,
        step_cost: float = 0.0,
        seed: int | None = None,
    ):
        self.size = _integer(size, "size", 2)
        if self.size not in (2, 3):
            raise ValueError("size must be 2 or 3")
        self.scramble_depth = _integer(scramble_depth, "scramble_depth", 1)
        self.max_steps = None if max_steps is None else _integer(max_steps, "max_steps", 1)
        self.step_cost = float(step_cost)
        if not np.isfinite(self.step_cost) or self.step_cost < 0:
            raise ValueError("step_cost must be finite and nonnegative")
        self.observation_size = 36 * self.size * self.size
        self._rng = np.random.default_rng(seed)
        self._geometry = _cube_geometry(self.size)
        lookup = {sticker: index for index, sticker in enumerate(self._geometry)}
        permutations = []
        for _, axis, sign in _FACES:
            for inverse in (False, True):
                quarter = sign if inverse else -sign
                permutation = np.arange(len(self._geometry))
                for index, (position, normal) in enumerate(self._geometry):
                    if position[axis] == sign * (self.size - 1):
                        target = (_rotate(position, axis, quarter), _rotate(normal, axis, quarter))
                        permutation[index] = lookup[target]
                permutations.append(permutation)
        self._permutations = tuple(permutations)
        self._solved = np.repeat(np.arange(6, dtype=np.int8), self.size * self.size)
        self._stickers = self._solved.copy()
        self._steps = 0
        self._done = True

    @property
    def stickers(self) -> np.ndarray:
        """A copy, shaped (6, size, size), for inspection or rendering."""
        return self._stickers.reshape(6, self.size, self.size).copy()

    def state_key(self) -> tuple[int, ...]:
        """Exact orientation-sensitive state identifier for split auditing."""
        return tuple(int(color) for color in self._stickers)

    @property
    def steps(self) -> int:
        """Number of moves made in this episode, including resumed moves."""
        return self._steps

    def is_solved(self) -> bool:
        faces = self._stickers.reshape(6, -1)
        return bool(np.all(faces == faces[:, :1]))

    def _observe(self) -> np.ndarray:
        return np.eye(6, dtype=np.float32)[self._stickers].reshape(-1)

    def set_scramble_depth(self, depth: int) -> None:
        """Set the curriculum level for subsequent resets."""
        self.scramble_depth = _integer(depth, "scramble_depth", 1)

    def apply_move(self, move: int | str) -> None:
        """Apply a geometric move without advancing the episode clock.

        This simulator utility supports invariant tests and split generation;
        learners should use ``step``. It deliberately stores no move history.
        """
        if isinstance(move, str):
            try:
                move = self.action_names.index(move)
            except ValueError as error:
                raise ValueError(f"unknown move {move!r}") from error
        move = _action(move, self.n_actions)
        self._stickers[self._permutations[move]] = self._stickers.copy()

    def _load_state(self, state: Sequence[int]) -> None:
        candidate = np.asarray(state)
        if candidate.shape != self._solved.shape or candidate.dtype.kind not in "iu":
            raise ValueError("state must be a flat integer sticker configuration")
        if np.any(candidate < 0) or np.any(candidate > 5):
            raise ValueError("state colors must be in [0, 6)")
        if not np.all(np.bincount(candidate.astype(np.int64), minlength=6) == self.size * self.size):
            raise ValueError("state must preserve each color's sticker count")
        if self.size == 3 and not np.array_equal(candidate.reshape(6, 9)[:, 4], np.arange(6)):
            raise ValueError("3x3 state must preserve face centers")
        self._stickers = candidate.astype(np.int8, copy=True)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = {} if options is None else options
        unknown = set(options) - {"state", "scramble_depth", "steps"}
        if unknown:
            raise ValueError(f"unknown reset options: {sorted(unknown)}")
        steps = _integer(options.get("steps", 0), "steps")
        if "steps" in options and "state" not in options:
            raise ValueError("steps requires a supplied state")
        if self.max_steps is not None and steps >= self.max_steps:
            raise ValueError("steps must be less than max_steps for an unfinished episode")
        if "state" in options:
            if "scramble_depth" in options:
                raise ValueError("specify state or scramble_depth, not both")
            self._load_state(options["state"])
            if self.is_solved():
                raise ValueError("episodes must start from an unsolved state")
            info = {}
        else:
            depth = _integer(options.get("scramble_depth", self.scramble_depth), "scramble_depth", 1)
            # Bounded retries are relevant at longer depths, where a random
            # path can return to the solved cube without adjacent inverses.
            for _ in range(1000):
                self._stickers = self._solved.copy()
                previous = None
                for _ in range(depth):
                    legal = [action for action in range(self.n_actions) if previous is None or action != (previous ^ 1)]
                    move = int(self._rng.choice(legal))
                    self.apply_move(move)
                    previous = move
                if not self.is_solved():
                    break
            else:
                raise RuntimeError("failed to sample an unsolved cube after 1000 attempts")
            info = {"scramble_depth": depth}
        self._steps = steps
        self._done = False
        return self._observe(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = _action(action, self.n_actions)
        if self._done:
            raise RuntimeError("reset() is required before stepping a finished episode")
        self.apply_move(action)
        self._steps += 1
        terminated = self.is_solved()
        truncated = self.max_steps is not None and self._steps >= self.max_steps and not terminated
        self._done = terminated or truncated
        reward = 1.0 if terminated else -self.step_cost
        return self._observe(), reward, terminated, truncated, {"success": terminated, "steps": self._steps}


def build_cube_state_split(
    size: int = 2,
    scramble_depth: int = 2,
    n_train: int = 64,
    n_test: int = 16,
    seed: int = 0,
    max_attempts: int | None = None,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """Return unique, exactly state-disjoint unsolved train/test starts.

    For depths 1--3 the complete finite pool of non-cancelling paths is
    enumerated; impossible counts raise a descriptive ValueError. Larger
    depths use bounded rejection sampling and report sampling exhaustion,
    which is not evidence that the whole state space was exhausted.

    Distinct configurations may be related by symmetry or share transitions;
    this guarantees separation of *starting states*, not of trajectories.
    Load each key with ``env.reset(options={"state": key})``.
    """
    n_train = _integer(n_train, "n_train")
    n_test = _integer(n_test, "n_test")
    depth = _integer(scramble_depth, "scramble_depth", 1)
    total = n_train + n_test
    cube = RubiksCubeEnv(size=size, scramble_depth=depth, seed=seed)
    rng = np.random.default_rng(seed)
    pool: dict[tuple[int, ...], None] = {}
    if depth <= 3:
        def enumerate_paths(remaining: int, previous: int | None) -> None:
            if remaining == 0:
                if not cube.is_solved():
                    pool[cube.state_key()] = None
                return
            start = cube._stickers.copy()
            for move in range(cube.n_actions):
                if previous is not None and move == (previous ^ 1):
                    continue
                cube._stickers = start.copy()
                cube.apply_move(move)
                enumerate_paths(remaining - 1, move)
            cube._stickers = start

        enumerate_paths(depth, None)
        if total > len(pool):
            raise ValueError(
                f"Only {len(pool)} distinct unsolved states exist in the enumerated "
                f"depth-{depth} pool; requested {total}. Reduce counts or increase scramble_depth."
            )
    else:
        attempts = max(1000, total * 100) if max_attempts is None else _integer(max_attempts, "max_attempts", 1)
        for _ in range(attempts):
            if len(pool) >= total:
                break
            cube.reset()
            pool[cube.state_key()] = None
        if len(pool) < total:
            raise ValueError(
                f"Sampled only {len(pool)} unique states after {attempts} attempts; requested {total}. "
                "Increase max_attempts or scramble_depth, or reduce counts. The finite pool was not enumerated."
            )
    states = list(pool)
    rng.shuffle(states)
    return states[:n_train], states[n_train:total]
