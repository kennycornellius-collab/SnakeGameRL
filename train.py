from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from env import SnakeEnv


TIMESTEPS      = 1_000_000   
N_ENVS         = 4           
MODEL_PATH     = "models/snake_ppo"
LOG_PATH       = "logs/"
EVAL_FREQ      = 10_000      
SAVE_FREQ      = 50_000      


def main():
    
    env = make_vec_env(SnakeEnv, n_envs=N_ENVS)

    
    eval_env = SnakeEnv()

    
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=10,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path="models/checkpoints/",
        name_prefix="snake_ppo",
        verbose=1,
    )

    model = PPO.load("models/snake_ppo.zip", env=env)
    model.learn(total_timesteps=4_000_000, reset_num_timesteps=False)
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log=LOG_PATH,
    #     learning_rate=1e-3,
    #     n_steps=2048,
    #     batch_size=64,
    #     n_epochs=10,
    #     gamma=0.99,
    # )

    print(f"Training for {TIMESTEPS:,} timesteps across {N_ENVS} parallel envs...")
    print(f"TensorBoard: tensorboard --logdir {LOG_PATH}\n")

    # model.learn(
    #     total_timesteps=TIMESTEPS,
    #     callback=[eval_callback, checkpoint_callback],
    #     progress_bar=True,
    # )

    model.save(MODEL_PATH)
    print(f"\n✅  Training done. Model saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()