"""
watch.py — load a trained model and watch it play Snake.

Usage:
    python watch.py
    python watch.py --model models/snake_ppo.zip   # specify a model path
    python watch.py --speed 60                      # control FPS (default 60)
"""

import argparse
import sys
from stable_baselines3 import PPO
#change env to envCnn to see the cnn policy agent
from env import SnakeEnv
from renderer import Renderer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/snake_ppo.zip", help="Path to trained model")
    parser.add_argument("--speed", type=int, default=300, help="Playback FPS")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to watch")
    args = parser.parse_args()

    
    print(f"Loading model from {args.model}...")
    model = PPO.load(args.model)

    
    env = SnakeEnv()
    renderer = Renderer(env.cols, env.rows, fps=args.speed)

    print(f"Watching {args.episodes} episodes at {args.speed} FPS. Press ESC to quit.\n")

    all_scores = []
    all_rewards = []
    all_timesteps = []

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
                all_scores.append(info['score'])
                all_rewards.append(total_reward)
                all_timesteps.append(steps)
                break

    env.close()
    renderer.close()

    print("\n--- Averages across all episodes ---")
    print(f"  Avg Score    : {sum(all_scores) / len(all_scores):.2f}")
    print(f"  Avg Reward   : {sum(all_rewards) / len(all_rewards):.2f}")
    print(f"  Avg Timesteps: {sum(all_timesteps) / len(all_timesteps):.2f}")
    print("\nDone.")


if __name__ == "__main__":
    main()