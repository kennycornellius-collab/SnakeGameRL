from env import SnakeEnv

def main():
    env = SnakeEnv(render_mode="none")

    print("observation_space:", env.observation_space)
    print("action_space     :", env.action_space)
    print()

    total_rewards = []

    for episode in range(5):
        obs, info = env.reset(seed=episode)
        ep_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()          
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        print(f"Episode {episode + 1:>2} | steps={steps:>5} | "
              f"score={info['score']:>3} | total_reward={ep_reward:>8.2f}")

    env.close()
    print()
    print(f"Mean reward over {len(total_rewards)} episodes: "
          f"{sum(total_rewards) / len(total_rewards):.2f}")
    print("\n✅  Env smoke test passed — ready for RL training.")


if __name__ == "__main__":
    main()
