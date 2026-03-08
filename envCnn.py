import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

from Game_logic import SnakeGame

MAX_STEPS  = 2_000
CH_HEAD    = 0
CH_BODY    = 1
CH_FOOD    = 2
CH_DIR     = 3
N_CHANNELS = 4

DIR_VALUES = {
    ( 1,  0): 64,
    (-1,  0): 128,
    ( 0,  1): 192,
    ( 0, -1): 255,
}


CURRICULUM_THRESHOLD = 8

CURRICULUM_RADIUS    = 2


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 10}

    def __init__(self, cols: int = 20, rows: int = 15, render_mode: str = "none"):
        super().__init__()
        self.cols        = cols
        self.rows        = rows
        self.render_mode = render_mode

        self.game = SnakeGame(cols, rows)
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(rows, cols, N_CHANNELS),
            dtype=np.uint8,
        )
        self.action_space      = spaces.Discrete(3)
        self._renderer         = None
        self._last_score       = 0
        self._steps_since_food = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        state = self.game.reset()

        start_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = start_dirs[self.np_random.integers(0, 4)]
        self.game.direction = (dx, dy)

        self._last_score       = 0
        self._steps_since_food = 0

        self._maybe_place_food_nearby()
        state = self.game._state()  

        return self._build_obs(state), {"score": 0}

    def step(self, action: int):
        dx, dy = self._resolve_action(action)
        state, reward, terminated = self.game.step(dx, dy)

        if terminated:
            reward = -1.0
        elif state["score"] > self._last_score:
            
            self._last_score       = state["score"]
            self._steps_since_food = 0
            reward = 1.0
            self._maybe_place_food_nearby()
            state = self.game._state()
        else:
            self._steps_since_food += 1
            reward = 0.0

            if self._steps_since_food >= self.cols * self.rows:
                return self._build_obs(state), -0.5, False, True, {"score": state["score"]}
        if not terminated:
            reachable  = self.game._flood_fill(tuple(state["body"][0]))
            board_size = self.cols * self.rows
            reward    += 0.01 * (reachable / board_size)

        truncated = state["steps"] >= MAX_STEPS
        obs       = self._build_obs(state)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {"score": state["score"]}

    def _maybe_place_food_nearby(self):
        if self.game.score >= CURRICULUM_THRESHOLD:
            return  

        hx, hy      = self.game.body[0]
        body_set    = set(map(tuple, self.game.body))
        candidates  = []

        for dx in range(-CURRICULUM_RADIUS, CURRICULUM_RADIUS + 1):
            for dy in range(-CURRICULUM_RADIUS, CURRICULUM_RADIUS + 1):
                dist = abs(dx) + abs(dy)
                if dist == 0 or dist > CURRICULUM_RADIUS:
                    continue
                nx, ny = hx + dx, hy + dy
                if 0 <= nx < self.cols and 0 <= ny < self.rows:
                    if (nx, ny) not in body_set:
                        candidates.append((nx, ny))

        if candidates:
            self.game.food = random.choice(candidates)
        

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _resolve_action(self, action: int) -> tuple[int, int]:
        dx, dy = self.game.direction
        if (dx, dy) == (0, 0):
            dx, dy = (1, 0)
        if action == 0:   return dx, dy
        elif action == 1: return -dy, dx
        else:             return dy, -dx

    def _build_obs(self, state: dict) -> np.ndarray:
        grid = np.zeros((self.rows, self.cols, N_CHANNELS), dtype=np.uint8)

        body = state["body"]
        hx, hy = body[0]
        grid[hy, hx, CH_HEAD] = 255
        for bx, by in body[1:]:
            grid[by, bx, CH_BODY] = 255

        fx, fy = state["food"]
        grid[fy, fx, CH_FOOD] = 255

        dx, dy = state["direction"]
        if (dx, dy) == (0, 0):
            dx, dy = (1, 0)
        grid[:, :, CH_DIR] = DIR_VALUES.get((dx, dy), 64)

        return grid