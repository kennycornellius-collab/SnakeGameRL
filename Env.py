import numpy as np
import gymnasium as gym
from gymnasium import spaces

from Game_logic import SnakeGame

MAX_STEPS = 2_000


class SnakeEnv(gym.Env):
    

    metadata = {"render_modes": ["human", "none"], "render_fps": 10}

    def __init__(self, cols: int = 20, rows: int = 15, render_mode: str = "none"):
        super().__init__()
        self.cols = cols
        self.rows = rows
        self.render_mode = render_mode

        self.game = SnakeGame(cols, rows)

        
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(3)

        
        self._renderer = None


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            import random, numpy as np
            random.seed(seed)
            np.random.seed(seed)

        state = self.game.reset()
        
        start_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        dx, dy = start_dirs[self.np_random.integers(0, 4)]
        self.game.direction = (dx, dy)

        obs = self._build_obs(state)
        info = {"score": 0}
        return obs, info

    def step(self, action: int):
        dx, dy = self._resolve_action(action)
        state, reward, terminated = self.game.step(dx, dy)

        truncated = state["steps"] >= MAX_STEPS

        obs = self._build_obs(state)
        info = {"score": state["score"]}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info


    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _resolve_action(self, action: int) -> tuple[int, int]:
        
        dx, dy = self.game.direction
        if (dx, dy) == (0, 0):
            dx, dy = (1, 0)  

        if action == 0:            
            return dx, dy
        elif action == 1:          
            return -dy, dx
        else:                      
            return dy, -dx

    def _build_obs(self, state: dict) -> np.ndarray:
        head = state["body"][0]
        dx, dy = state["direction"]
        if (dx, dy) == (0, 0):
            dx, dy = 1, 0           

        food = state["food"]
        body_set = set(map(tuple, state["body"]))

        def is_deadly(pos):
            x, y = pos
            return x < 0 or x >= self.cols or y < 0 or y >= self.rows or pos in body_set

        
        straight = (head[0] + dx,  head[1] + dy)
        right    = (head[0] + (-dy), head[1] + dx)
        left     = (head[0] + dy,  head[1] + (-dx))

        hx, hy = head
        fx, fy = food

        obs = np.array([
            float(is_deadly(straight)),      
            float(is_deadly(right)),          
            float(is_deadly(left)),           
            float(dx),                        
            float(dy),                        
            float(fx < hx),                   
            float(fx > hx),                   
            float(fy < hy),                   
            float(fy > hy),                   
        ], dtype=np.float32)

        return obs
