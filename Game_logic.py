import random

COLS = 20
ROWS = 15


class SnakeGame:

    def __init__(self, cols: int = COLS, rows: int = ROWS):
        self.cols = cols
        self.rows = rows
        self.reset()

    def reset(self):
        
        self.body = [(self.cols // 2, self.rows // 2)]
        self.direction = (0, 0)          
        self.food = self._spawn_food()
        self.score = 0
        self.steps = 0
        self.alive = True
        return self._state()

    def step(self, dx: int, dy: int) -> tuple[dict, float, bool]:
        
        assert self.alive, "Game is over — call reset() first."

        self.direction = (dx, dy)
        self.steps += 1

        head_x, head_y = self.body[0]
        new_head = (head_x + dx, head_y + dy)

       
        if self._is_wall(new_head) or new_head in set(self.body):
            self.alive = False
            return self._state(), -10.0, True

        
        ate_food = new_head == self.food
        if ate_food:
            self.body = [new_head] + self.body   
            self.score += 1
            self.food = self._spawn_food()
            reward = 10.0
        else:
            self.body = [new_head] + self.body[:-1]
            reward = -0.01                        

        return self._state(), reward, False


    def _state(self) -> dict:
        return {
            "body": list(self.body),
            "direction": self.direction,
            "food": self.food,
            "score": self.score,
            "steps": self.steps,
            "alive": self.alive,
        }

    def _is_wall(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return x < 0 or x >= self.cols or y < 0 or y >= self.rows

    def _spawn_food(self) -> tuple[int, int]:
        occupied = set(self.body)
        while True:
            pos = (random.randint(0, self.cols - 1), random.randint(0, self.rows - 1))
            if pos not in occupied:
                return pos
