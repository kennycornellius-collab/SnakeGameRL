# Snake RL

A Snake game built from the ground up for reinforcement learning using [Gymnasium](https://gymnasium.farama.org/) and [Stable Baselines 3](https://stable-baselines3.readthedocs.io/).

The project currently has a fully trained **MLP policy (PPO)** as the baseline. A **CNN policy** with a grid-based observation is in progress for comparison.

---

## Project structure

```
SnakeGameRL/
├── Game_logic.py        # Pure game logic — no pygame, fast for RL simulation
├── env.py               # Gymnasium env — MLP policy (vector obs, 12 features)
├── envCnn.py            # Gymnasium env — CNN policy (raw grid obs) 🚧 WIP
├── renderer.py          # Pygame rendering, fully isolated from game logic
├── Game.py              # Original pygame snake game
├── train.py             # Training script for MLP policy
├── trainCnn.py          # Training script for CNN policy 🚧 WIP
├── cnn_policy.py        # Custom CNN policy definition 🚧 WIP
├── watch.py             # Watch a trained agent play in real time
└── smoke_test.py        # Sanity-check the env with a random agent
```

---

## Setup

```bash
pip install gymnasium stable-baselines3 pygame rich tqdm "setuptools<70" tensorboard optuna
```

> `setuptools<70` is pinned due to a known compatibility issue with `tensorboard` and newer setuptools versions.

---

## Smoke test

Verifies the environment contract before plugging in any RL algorithm:

```bash
python smoke_test.py
```

---

## Environment — MLP policy (`env.py`)

### Spaces

| Property | Value |
|---|---|
| **Observation space** | `Box(12,)` — float32 |
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
| 9 | Flood-fill reachable cells going straight (normalised) |
| 10 | Flood-fill reachable cells going right (normalised) |
| 11 | Flood-fill reachable cells going left (normalised) |

Indices 9–11 are computed by running a flood-fill from each candidate next cell (capped at `MAX_SCAN` cells) and normalising the result. If a direction is immediately deadly the value is `0.0`.

## Training

```bash
python train.py
```

Checkpoints are saved under `models/checkpoints/`. TensorBoard logs are written to `logs/`.

```bash
tensorboard --logdir logs/
```

---

## Watch a trained agent play

```bash
python watch.py
```

Loads the saved model and renders the agent playing in real time via pygame.

---

## Results

> 🚧 Full results will be added here once both policies are trained and evaluated side by side so changes to the number may occur.

The result are done with 100 times episodes

| Policy | Observation | Avg. Score | Avg. Reward | Timesteps |
|---|---|---|---|---|
| MLP (PPO) | 12-dim vector(without tuning) | 71.48 | 692.85 | 1317.70 |
| MLP (PPO) | 12-dim vector(after tuning) | 60.69 | 587.18 | 1043.48 |
| CNN (PPO) | Raw grid(without tuning) | — | — | — |
| CNN (PPO) | Raw grid(after tuning) | — | — | — |

MLP tuning result:
Best params: {'learning_rate': 0.0006885124586537938, 'n_steps': 4096, 'batch_size': 128, 'n_epochs': 20, 'gamma': 0.9540035038350664}
Best reward: 678.3603422026866

MLP result are achieved on a model, trained on a 5 million total timesteps.

Notes MLP:
The hyperparameter tuning was conducted using 500k timesteps per trial due to computational constraints. The best parameters found by Optuna did not outperform the original parameters when trained for the full 5M timesteps. This is a known limitation — parameter combinations that converge quickly in short runs don't necessarily generalize to longer training runs. A more accurate tuning study would use a higher timestep budget per trial (ideally 2M+), but this was not feasible given available hardware.
---

## Roadmap

- [x] Game logic decoupled from rendering
- [x] Gymnasium environment (MLP / vector obs)
- [x] Renderer isolated into its own module
- [x] Smoke test
- [x] PPO training script with TensorBoard logging
- [x] Watch script to evaluate agent visually
- [x] Flood-fill reward shaping + extended observation (12 features)
- [ ] CNN policy with raw grid observation
- [ ] Train CNN policy
- [ ] Compare MLP vs CNN