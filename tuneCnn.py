import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from envCnn import SnakeEnv
from cnn_policy import SnakeCNN


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.05)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])

    
    features_dim = trial.suggest_categorical("features_dim", [128, 256, 512])

    policy_kwargs = dict(
        features_extractor_class=SnakeCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    env = make_vec_env(SnakeEnv, n_envs=4)

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        device="cuda",
        verbose=0,
    )

    model.learn(total_timesteps=2_000_000)

    eval_env = SnakeEnv()
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)

    return mean_reward


study = optuna.create_study(
    direction="maximize",
    study_name="snake_cnn_ppo_tuning",
    storage="sqlite:///snake_cnn_tuning.db",  
    load_if_exists=True,
)

study.optimize(objective, n_trials=20)

print("Best params:", study.best_params)
print("Best reward:", study.best_value)