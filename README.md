# Snake RL

A Snake game built from the ground up for reinforcement learning using [Gymnasium](https://gymnasium.farama.org/) and [Stable Baselines 3](https://stable-baselines3.readthedocs.io/).

The project trains and compares two policies on the same Snake environment — an **MLP policy** using hand-engineered features and a **CNN policy** learning directly from raw grid observations.

---
## Overview
A Snake game built from the ground up as a reinforcement learning testbed, comparing two fundamentally different approaches to the same problem: an MLP policy operating on hand-engineered features, and a CNN policy learning directly from raw grid observations.
The project goes beyond a standard RL tutorial. Training the CNN agent required solving a chain of non-obvious problems — sparse rewards on a large board, a circling exploit the agent discovered to avoid dying, suicide behaviour induced by a naive living penalty, and a score plateau caused by self-trapping. Each required a targeted fix: curriculum learning, reward restructuring, and a flood-fill survival bonus that teaches the agent to preserve escape routes before it gets trapped.
The MLP agent achieves an average score of 71 against the CNN's 9 over 100 episodes, not because CNNs are weaker, but because Snake is small and structured enough that knowing the right features matters more than raw representational power. The gap itself is the finding.
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
| Flood-fill survival bonus | `+0.01 × (reachable_cells / board_size)` per step |
| Timeout (no food in 300 steps) | `-0.5` (truncate, not terminate) |
| Any other step | `0.0` |

**Why step reward is `0.0`:** A living penalty (e.g. `-0.01`) was found to cause the agent to commit suicide after eating a few foods — after 100 steps the accumulated cost equals the death penalty, making dying and surviving mathematically equivalent. Removing it eliminates any incentive to die early.

**Why death and food are 1:1 (`-1 / +1`):** The MLP used a 1:10 ratio (`-1 / +10`) which made the agent reckless about dying since one food easily outweighed a death. The tighter ratio forces the CNN agent to genuinely weigh survival against food-seeking.

**Why the flood-fill survival bonus:** At around score 15–16 the agent plateaued, dying consistently from self-trapping — moving into corridors it couldn't escape. The flood-fill bonus rewards the agent proportionally to how much open space surrounds its head every step, teaching it to keep escape routes open before it gets trapped rather than reacting after.

### Curriculum learning

To address the sparse reward problem on a large 20×15 board, a curriculum phase is applied during early training. When `score < 8`, food is always spawned within Manhattan distance 2 of the snake's head instead of randomly. Once `score ≥ 8`, food placement reverts to normal random spawning.

Without this, the agent would die or time out for thousands of episodes before accidentally finding food, receiving only negative signals and never learning what it is optimising for. The curriculum guarantees early positive signal so the agent first learns *that* food exists and *that* moving toward it is rewarded — then generalises that behaviour to the full board.

### CNN architecture (`cnn_policy.py`)

```
Conv2d(4 → 32, 3×3, stride=1, padding=1)  →  ReLU
Conv2d(32 → 64, 3×3, stride=2, padding=1) →  ReLU
Conv2d(64 → 64, 3×3, stride=2, padding=1) →  ReLU
Flatten → Linear(n_flat → 256) → ReLU
```

The features extractor produces a 256-dim vector fed into a `[128, 64]` MLP head for both the policy and value networks.

> **Note on `VecTransposeImage`:** SB3 automatically wraps CNN environments with `VecTransposeImage`, which converts observations from `(H, W, C)` to `(C, H, W)` before they reach the network. No manual `.permute()` is needed in `forward()` — adding one would double-transpose and corrupt the input.

### CNN training hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO |
| Total timesteps | 15 000 000 |
| Parallel envs | 8 |
| Learning rate | 3e-4 |
| n_steps | 2048 |
| Batch size | 256 |
| n_epochs | 10 |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip range | 0.2 |
| Entropy coefficient | 0.01 |

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

Results are averaged over 100 episodes. The MLP was trained for 5 million timesteps, the CNN for 15 million.

| Policy | Observation | Avg. Score | Avg. Reward | Avg. Steps |
|---|---|---|---|---|
| MLP (PPO) | 12-dim vector (without tuning) | 71.48 | 692.85 | 1317.70 |
| MLP (PPO) | 12-dim vector (after tuning) | 60.69 | 587.18 | 1043.48 |
| CNN (PPO) | Raw grid, 4 channels | 8.74 | 8.42 | 267.16 |

### Reward scale difference between MLP and CNN

MLP and CNN reward values are **not directly comparable**. The MLP uses a food reward of `+10` while the CNN uses `+1`, so an MLP reward of `+100` and a CNN reward of `+10` represent identical gameplay (10 foods eaten). The rescaling was intentional — the CNN uses a tighter 1:1 death-to-food ratio for a cleaner training signal, whereas the MLP's 10:1 ratio made dying relatively cheap.

### CNN training challenges

Getting the CNN agent to learn at all required solving several problems that the MLP never encountered:

**Sparse reward problem.** On a 20×15 board with random food placement, an untrained agent dies or times out hundreds of times before stumbling on food by accident. With only negative signal, the agent has no idea what it is optimising for. Curriculum learning (near-head food spawning) solved this.

**Circling exploit.** Early reward structures were gamed by the agent circling in a tight 2×2 loop indefinitely — safe, predictable, and never dying. Every timeout-based fix was circumvented by making the loop smaller. The real fix was curriculum learning, which ensured the agent learned food-seeking before it could entrench a looping strategy.

**Suicide from living penalty.** A `-0.01` per-step cost caused the agent to deliberately die after eating a few foods. After 100 steps the accumulated penalty equalled the death cost, making early death mathematically rational. Removing the living penalty entirely fixed this.

**Score plateau at ~16.** The agent learned to chase food reliably but plateaued because it would trap itself with its own growing body. The flood-fill survival bonus — rewarding open space around the head every step — broke this plateau by teaching the agent to keep escape routes open proactively.

### MLP notes

The hyperparameter tuning was conducted using 500k timesteps per trial due to computational constraints. The best parameters found by Optuna did not outperform the original parameters when trained for the full 5M timesteps. This is a known limitation — parameter combinations that converge quickly in short runs don't necessarily generalise to longer training runs. A more accurate tuning study would use a higher timestep budget per trial (ideally 2M+), but this was not feasible given available hardware.

MLP best params from Optuna: `learning_rate=0.000689`, `n_steps=4096`, `batch_size=128`, `n_epochs=20`, `gamma=0.954`

---

## Conclusion

The MLP policy significantly outperforms the CNN at equivalent timesteps (avg score 71 vs 9). This is not because CNNs are inherently worse, but because the 20×15 Snake board is small and structured enough that hand-crafted features (danger detection, flood-fill reachability) give the MLP agent near-perfect domain knowledge from step one. The CNN must rediscover these same concepts from raw pixels, requiring far more training data and still producing a less consistent policy — scoring anywhere from 1 to 20 within the same 100-episode evaluation run.

On a visually complex environment where feature engineering is impossible, the CNN would likely win. Snake is not that environment — it is a case where knowing the right features matters more than having a powerful function approximator.

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
- [x] Flood-fill survival bonus for CNN
- [x] Train CNN policy to completion
- [x] Compare MLP vs CNN