# Snake RL

A Snake game refactored for reinforcement learning using [Gymnasium](https://gymnasium.farama.org/).

## Project structure

```
snake/
├── game.py        # Pure game logic — no pygame, fast for simulation
├── env.py         # Gymnasium environment wrapping the game
├── renderer.py    # Pygame rendering, fully isolated
└── smoke_test.py  # Sanity-check the env with a random agent
```

## Setup

```bash
pip install gymnasium stable-baselines3 pygame
```

## Smoke test

Verifies the environment contract is correct before plugging in an RL algo:

```bash
python smoke_test.py
```

## Environment details

| Property | Value |
|---|---|
| **Observation space** | `Box(9,)` — float32 |
| **Action space** | `Discrete(3)` — straight / turn-right / turn-left |
| **Max steps / episode** | 2 000 |

### Observation vector

| Index | Meaning |
|---|---|
| 0 | Danger straight |
| 1 | Danger right |
| 2 | Danger left |
| 3 | Current direction x |
| 4 | Current direction y |
| 5 | Food is to the left |
| 6 | Food is to the right |
| 7 | Food is above |
| 8 | Food is below |

### Reward function

| Event | Reward |
|---|---|
| Ate food | +10 |
| Died | −10 |
| Each step | −0.01 |

## Render the env (human mode)

```python
from env import SnakeEnv

env = SnakeEnv(render_mode="human")
obs, _ = env.reset()
while True:
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
```

## Next steps

- [ ] Train a DQN or PPO agent with Stable Baselines 3
- [ ] Add TensorBoard logging
- [ ] Experiment with raw grid observations + CNN policy
- [ ] Tune reward shaping
