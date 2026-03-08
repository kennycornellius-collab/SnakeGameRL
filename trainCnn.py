from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envCnn import SnakeEnv
from cnn_policy import SnakeCNN

TIMESTEPS   = 10_000_000  
N_ENVS      = 8
MODEL_PATH  = "models/snake_cnn_ppo"
LOG_PATH    = "logs/"
EVAL_FREQ   = 50_000
SAVE_FREQ   = 200_000

LEARNING_RATE  = 3e-4     
N_STEPS        = 2048
BATCH_SIZE     = 256
N_EPOCHS       = 10
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_RANGE     = 0.2
ENT_COEF       = 0.01     
FEATURES_DIM   = 256      

def main():
    env      = make_vec_env(SnakeEnv, n_envs=N_ENVS)
    
    eval_env = VecTransposeImage(make_vec_env(SnakeEnv, n_envs=4))

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=max(EVAL_FREQ // N_ENVS, 1),
        n_eval_episodes=20,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1),
        save_path="models/checkpoints/",
        name_prefix="snake_cnn_ppo",
        verbose=0,
    )

    policy_kwargs = dict(
        features_extractor_class=SnakeCNN,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),  
    )

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        tensorboard_log=LOG_PATH,
        policy_kwargs=policy_kwargs,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
    )

    print(f"Policy architecture:\n{model.policy}\n")
    print(f"Training for {TIMESTEPS:,} timesteps across {N_ENVS} parallel envs...")
    print(f"TensorBoard: tensorboard --logdir {LOG_PATH}\n")

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    model.save(MODEL_PATH)
    print(f"\n✅  Training done. Model saved to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()