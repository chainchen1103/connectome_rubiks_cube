"""Observable cube rewards for a continuous, resumable episode.

The potential measures how concentrated each face's colors are. It is not a
solution distance: useful solutions can temporarily reduce it. Its difference
telescopes over a path, so an unsolved closed cycle cannot earn net progress
reward. Action costs and the recent-state penalty discourage cycling further.
No scramble history, solver, or target action is consulted.
"""

from collections import Counter, deque
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np


def _integer(value, name, minimum, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} is outside its allowed range')
    return value


def _state_key(state, size):
    candidate = np.asarray(state)
    if candidate.shape not in ((6 * size * size,), (6, size, size)) or candidate.dtype.kind not in 'iu':
        raise ValueError('state must contain the cube integer stickers')
    candidate = candidate.reshape(-1)
    if np.any(candidate < 0) or np.any(candidate > 5):
        raise ValueError('state colors must be in [0, 6)')
    if not np.all(np.bincount(candidate.astype(np.int64), minlength=6) == size * size):
        raise ValueError('state must preserve each color sticker count')
    return tuple(int(color) for color in candidate)


def _potential(key, size):
    counts = np.stack([np.bincount(face, minlength=6) for face in np.asarray(key).reshape(6, -1)])
    # The factors of 1/2 in the unordered-pair numerator and denominator cancel.
    face_size = size * size
    return float(np.sum(counts * (counts - 1)) / (6 * face_size * (face_size - 1)))


def cube_potential(state, size):
    """Fraction of same-color unordered sticker pairs within all six faces.

    Invariant to color naming, face order, and within-face orientation. Equal
    to one exactly when all faces are uniform; defined for both 2x2 and 3x3.
    """
    size = _integer(size, 'size', 2, 3)
    return _potential(_state_key(state, size), size)


@dataclass(frozen=True)
class CubeRewardConfig:
    progress_scale: float = 1.0
    step_cost: float = 0.002
    inverse_penalty: float = 0.01
    revisit_penalty: float = 0.02
    solved_bonus: float = 1.0
    recent_window: int = 256

    def __post_init__(self):
        for name in ('progress_scale', 'step_cost', 'inverse_penalty', 'revisit_penalty', 'solved_bonus'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, 'recent_window', _integer(self.recent_window, 'recent_window', 1, 65536))


class CubeRewardTracker:
    """Stateful reward diagnostics with bounded, JSON-serializable history.

    ``step`` returns ``reward`` and signed contributions. Penalty values in
    that result are negative (configuration values are positive magnitudes).
    The first before-state is recorded automatically when no initial state is
    supplied. Subsequent transitions must begin at the last recorded state.
    """

    snapshot_version = 1

    def __init__(self, size, config=None, initial_state=None):
        self.size = _integer(size, 'size', 2, 3)
        self.config = CubeRewardConfig() if config is None else config
        if not isinstance(self.config, CubeRewardConfig):
            raise ValueError('config must be CubeRewardConfig')
        self.previous_action = None
        self._recent_states = deque()
        self._state_counts = Counter()
        if initial_state is not None:
            self._append(_state_key(initial_state, self.size))

    def _append(self, state):
        if len(self._recent_states) == self.config.recent_window:
            removed = self._recent_states.popleft()
            self._state_counts[removed] -= 1
            if not self._state_counts[removed]:
                del self._state_counts[removed]
        self._recent_states.append(state)
        self._state_counts[state] += 1

    def step(self, before, action, after, solved):
        before = _state_key(before, self.size)
        after = _state_key(after, self.size)
        action = _integer(action, 'action', 0, 11)
        if not isinstance(solved, (bool, np.bool_)):
            raise ValueError('solved must be a boolean')
        before_potential = _potential(before, self.size)
        after_potential = _potential(after, self.size)
        if bool(solved) != (after_potential == 1.0):
            raise ValueError('solved must agree with the resulting cube state')
        if before_potential == 1.0:
            raise ValueError('a solved episode cannot receive another transition reward')
        if self._recent_states and self._recent_states[-1] != before:
            raise ValueError('transition must continue from the last recorded state')
        if not self._recent_states:
            self._append(before)
        inverse = self.previous_action is not None and action == (self.previous_action ^ 1)
        revisit = after in self._state_counts
        delta = after_potential - before_potential
        terms = {
            'potential_before': before_potential,
            'potential_after': after_potential,
            'potential_delta': delta,
            'progress_reward': self.config.progress_scale * delta,
            'step_penalty': -self.config.step_cost,
            'inverse_penalty': -self.config.inverse_penalty if inverse else 0.0,
            'revisit_penalty': -self.config.revisit_penalty if revisit else 0.0,
            'solved_bonus': self.config.solved_bonus if solved else 0.0,
            'is_inverse': inverse,
            'is_revisit': revisit,
        }
        terms['reward'] = sum(terms[name] for name in (
            'progress_reward', 'step_penalty', 'inverse_penalty', 'revisit_penalty', 'solved_bonus'))
        self._append(after)
        self.previous_action = action
        return terms

    def snapshot(self):
        return {
            'version': self.snapshot_version,
            'size': self.size,
            'config': asdict(self.config),
            'previous_action': self.previous_action,
            'recent_states': [list(state) for state in self._recent_states],
        }

    @classmethod
    def from_snapshot(cls, snapshot, size=None, config=None):
        if not isinstance(snapshot, dict) or set(snapshot) != {'version', 'size', 'config', 'previous_action', 'recent_states'}:
            raise ValueError('invalid cube reward snapshot fields')
        if _integer(snapshot['version'], 'version', 1) != cls.snapshot_version:
            raise ValueError('unsupported cube reward snapshot version')
        saved_size = _integer(snapshot['size'], 'size', 2, 3)
        if size is not None and _integer(size, 'size', 2, 3) != saved_size:
            raise ValueError('reward snapshot cube size differs')
        try:
            if not isinstance(snapshot['config'], dict) or set(snapshot['config']) != set(asdict(CubeRewardConfig())):
                raise ValueError('invalid reward configuration fields')
            saved_config = CubeRewardConfig(**snapshot['config'])
        except TypeError as error:
            raise ValueError('invalid reward configuration') from error
        if config is not None and config != saved_config:
            raise ValueError('reward snapshot configuration differs')
        recent = snapshot['recent_states']
        if not isinstance(recent, list) or len(recent) > saved_config.recent_window:
            raise ValueError('invalid recent-state history length')
        previous = snapshot['previous_action']
        if previous is not None:
            previous = _integer(previous, 'previous_action', 0, 11)
        if not recent and previous is not None:
            raise ValueError('previous action requires a recorded state')
        if len(recent) > 1 and previous is None:
            raise ValueError('transition history requires a previous action')
        tracker = cls(saved_size, saved_config)
        for state in recent:
            tracker._append(_state_key(state, saved_size))
        tracker.previous_action = previous
        return tracker
