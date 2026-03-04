"""
watch.py — load a trained model and watch it play Snake.

Usage:
    python watch.py
    python watch.py --model models/snake_ppo.zip   # specify a model path
    python watch.py --speed 10                      # control FPS (default 10)
"""

import argparse
import sys
from stable_baselines3 import PPO
from env import SnakeEnv
from renderer import Renderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/snake_ppo.zip", help="Path to trained model")
    parser.add_argument("--speed", type=int, default=60, help="Playback FPS")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to watch")
    args = parser.parse_args()

    # --- load model ---
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    # --- env + renderer ---
    env = SnakeEnv()
    renderer = Renderer(env.cols, env.rows, fps=args.speed)

    print(f"Watching {args.episodes} episodes at {args.speed} FPS. Press ESC to quit.\n")

    for episode in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += reward
            steps += 1

            renderer.draw(env.game._state())

            if terminated or truncated:
                print(f"Episode {episode + 1:>2} | steps={steps:>5} | "
                      f"score={info['score']:>3} | reward={total_reward:>8.2f}")
                break

    env.close()
    renderer.close()
    print("\nDone.")


if __name__ == "__main__":
    main()