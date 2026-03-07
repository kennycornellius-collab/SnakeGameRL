import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from env import SnakeEnv

def objective(trial):
    
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_epochs = trial.suggest_int("n_epochs", 5, 20)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)

    env = make_vec_env(SnakeEnv, n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        device="cpu",
        verbose=0,  
    )

    
    model.learn(total_timesteps=500_000)

    
    eval_env = SnakeEnv()
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    return mean_reward


study = optuna.create_study(
    direction="maximize",
    study_name="snake_ppo_tuning",
    storage="sqlite:///snake_tuning.db",  
    load_if_exists=True,
)

study.optimize(objective, n_trials=20)

print("Best params:", study.best_params)
print("Best reward:", study.best_value)