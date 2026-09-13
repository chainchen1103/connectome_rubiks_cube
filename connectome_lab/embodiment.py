"""An explicitly engineered virtual fly, not a biomechanical fly simulation."""
import math
import numpy as np


class FlyArenaEnv:
    n_actions = 3
    observation_size = 6
    action_names = ('forward', 'turn_left', 'turn_right')

    def __init__(self, max_steps=90, seed=0):
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError('max_steps must be positive')
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.done = True

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x, self.y = 35., 50.
        self.heading = float(self.rng.uniform(-.5, .5))
        self.food = (80., float(self.rng.uniform(25, 75)))
        self.steps = 0
        self.done = False
        return self._observe(), {}

    def _distance(self):
        return math.hypot(self.food[0]-self.x, self.food[1]-self.y)

    def _observe(self):
        angle = math.atan2(self.food[1]-self.y, self.food[0]-self.x) - self.heading
        wall = float(min(self.x, self.y, 100-self.x, 100-self.y) < 10)
        # Egocentric direction and contact cues, never the correct next action.
        return np.array([max(0, math.cos(angle)), max(0, -math.cos(angle)),
                         max(0, math.sin(angle)), max(0, -math.sin(angle)),
                         min(1, self._distance()/100), wall], dtype=np.float32)

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 3:
            raise ValueError('action must be 0, 1 or 2')
        if self.done:
            raise RuntimeError('reset required')
        before = self._distance()
        if action:
            # Screen y increases downwards: negative heading is visual left.
            self.heading += (-1 if action == 1 else 1) * math.pi / 8
        else:
            self.x = float(np.clip(self.x + 3 * math.cos(self.heading), 3, 97))
            self.y = float(np.clip(self.y + 3 * math.sin(self.heading), 3, 97))
        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        self.steps += 1
        distance = self._distance()
        success = distance < 6
        truncated = self.steps >= self.max_steps and not success
        self.done = success or truncated
        # Explicit distance shaping is part of this engineered arena task.
        reward = (before-distance)*.01 + (1. if success else 0.)
        return self._observe(), reward, success, truncated, {'success': success}

    def pose(self):
        return {'x': self.x, 'y': self.y, 'heading': self.heading,
                'food': list(self.food), 'arena_size': 100}
