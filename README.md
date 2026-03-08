# Snake RL

A Snake game built from the ground up for reinforcement learning using [Gymnasium](https://gymnasium.farama.org/) and [Stable Baselines 3](https://stable-baselines3.readthedocs.io/).

The project currently has a fully trained **MLP policy (PPO)** as the baseline and a **CNN policy** with a raw grid observation trained and compared against it.

---

## Project structure

```
SnakeGameRL/
├── Game_logic.py        # Pure game logic — no pygame, fast for RL simulation
├── env.py               # Gymnasium env — MLP policy (vector obs, 12 features)
├── envCnn.py            # Gymnasium env — CNN policy (raw grid obs, 4 channels)
├── renderer.py          # Pygame rendering, fully isolated from game logic
├── Game.py              # Original pygame snake game
├── train.py             # Training script for MLP policy
├── trainCnn.py          # Training script for CNN policy
├── cnn_policy.py        # Custom CNN features extractor (BaseFeaturesExtractor)
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

---

## Environment — CNN policy (`envCnn.py`)

### Spaces

| Property | Value |
|---|---|
| **Observation space** | `Box(0, 255, (15, 20, 4))` — uint8 |
| **Action space** | `Discrete(3)` — straight / turn-left / turn-right |
| **Max steps / episode** | 2 000 |

### Observation grid

The board is encoded as a `(15, 20, 4)` image with one channel per feature. SB3's `VecTransposeImage` wrapper automatically converts this to `(4, 15, 20)` channel-first format before passing it to the CNN.

| Channel | What it encodes | Pixel values |
|---|---|---|
| 0 — Head | Snake head position | 255 at head, 0 elsewhere |
| 1 — Body | Snake body positions | 255 at each body cell, 0 elsewhere |
| 2 — Food | Food position | 255 at food, 0 elsewhere |
| 3 — Direction | Current movement direction | Entire grid filled with a constant: 64=right, 128=left, 192=down, 255=up |

The direction channel fills the entire grid with a single constant value. A CNN learns a global constant far more easily than a sparse directional ray, which was an earlier approach that proved harder to train on.

### Reward structure

| Event | Reward |
|---|---|
| Eat food | `+1.0` |
| Die (wall or self-collision) | `-1.0` |
| Timeout (no food in 300 steps) | `-0.5` (truncate, not terminate) |
| Any other step | `0.0` |

The step reward is intentionally `0.0`. A non-zero living penalty (e.g. `-0.01`) was found to cause the agent to commit suicide after eating a few foods — after 100 steps the accumulated penalty equals the death cost, so dying becomes mathematically equivalent to surviving. Removing it means there is no incentive to die early.

Death and food are kept at a 1:1 ratio (`-1 / +1`) so the agent genuinely weighs survival against food-seeking. The MLP used a 1:10 ratio (`-1 / +10`) which made the agent comparatively reckless about dying.

### Curriculum learning

To address the sparse reward problem on a large 20×15 board, a curriculum phase is applied during early training. When `score < 5`, food is always spawned within Manhattan distance 2 of the snake's head instead of randomly. This guarantees the agent receives positive reward signal from the very first episodes and learns the food-seeking behaviour before generalising it to the full board. Once `score ≥ 5`, food placement reverts to normal random spawning.

Without this, the agent would die or time out hundreds of times before accidentally finding food, receiving only negative signals and never learning what it is optimising for.

### CNN architecture (`cnn_policy.py`)

```
Conv2d(4 → 32, 3×3, stride=1, padding=1)  →  ReLU
Conv2d(32 → 64, 3×3, stride=2, padding=1) →  ReLU
Conv2d(64 → 64, 3×3, stride=2, padding=1) →  ReLU
Flatten → Linear(n_flat → 256) → ReLU
```

The features extractor produces a 256-dim vector fed into a small `[128, 64]` MLP head for both the policy and value networks.

> **Note on `VecTransposeImage`:** SB3 automatically wraps CNN environments with `VecTransposeImage`, which converts observations from `(H, W, C)` to `(C, H, W)` before they reach the network. No manual `.permute()` is needed in the `forward()` method — adding one would double-transpose and corrupt the input.

---

## Training

```bash
# MLP
python train.py

# CNN
python trainCnn.py
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

> 🚧 CNN final results will be added once training is complete.

Results are averaged over 100 episodes.

| Policy | Observation | Avg. Score | Avg. Reward | Avg. Timesteps |
|---|---|---|---|---|
| MLP (PPO) | 12-dim vector (without tuning) | 71.48 | 692.85 | 1317.70 |
| MLP (PPO) | 12-dim vector (after tuning) | 60.69 | 587.18 | 1043.48 |
| CNN (PPO) | Raw grid, 4 channels | — | — | — |

### Reward scale difference between MLP and CNN

MLP and CNN reward values are **not directly comparable**. The MLP uses a food reward of `+10` while the CNN uses `+1`, so an MLP reward of `+100` and a CNN reward of `+10` represent identical gameplay (10 foods eaten). The rescaling was intentional — the CNN uses a tighter 1:1 death-to-food ratio for a cleaner training signal, whereas the MLP's 10:1 ratio made dying relatively cheap.

### MLP notes

The hyperparameter tuning was conducted using 500k timesteps per trial due to computational constraints. The best parameters found by Optuna did not outperform the original parameters when trained for the full 5M timesteps. This is a known limitation — parameter combinations that converge quickly in short runs don't necessarily generalise to longer training runs. A more accurate tuning study would use a higher timestep budget per trial (ideally 2M+), but this was not feasible given available hardware.

MLP best params from Optuna: `learning_rate=0.000689`, `n_steps=4096`, `batch_size=128`, `n_epochs=20`, `gamma=0.954`

---

## Roadmap

- [x] Game logic decoupled from rendering
- [x] Gymnasium environment (MLP / vector obs)
- [x] Renderer isolated into its own module
- [x] Smoke test
- [x] PPO training script with TensorBoard logging
- [x] Watch script to evaluate agent visually
- [x] Flood-fill reward shaping + extended observation (12 features)
- [x] CNN policy with raw grid observation
- [x] Curriculum learning for CNN (near-head food spawning in early training)
- [ ] Train CNN policy to completion
- [ ] Compare MLP vs CNN