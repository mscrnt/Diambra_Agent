import os
import optuna
from typing import Any, Dict
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
import torch
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule
import torch.nn as nn
import argparse
import logging
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Settings

env_settings = {
    'env_num': 1,
    'check_freq': 150000,
    'time_steps': 150000  
}

settings = {
    'game_id': "mvsc",
    'difficulty': 8,
    'characters': ("Spider Man", "Ryu"),  
    'frame_shape': (128, 128, 1)
}

wrappers_settings = {
    'stack_frames': 4,
    'dilation': 1,
    'no_attack_buttons_combinations': False,
    'normalize_reward': False,
    'normalization_factor': 0.5, 
    'scale': True,
    'exclude_image_scaling': True,
    'flatten': True,
    'process_discrete_binary': True,
    'role_relative': True,
    'add_last_action': True,
    'filter_keys': ['action', 'frame', 'opp_active_character', 'opp_character', 'opp_character_1', 
        'opp_character_2', 'opp_health_1', 'opp_health_2', 'opp_partner', 'opp_partner_attacks', 
        'opp_side', 'opp_super_bar', 'opp_super_count', 'opp_wins', 'own_active_character', 'own_character', 
        'own_character_1', 'own_character_2', 'own_health_1', 'own_health_2', 'own_partner', 'own_partner_attacks', 
        'own_side', 'own_super_bar', 'own_super_count', 'own_wins', 'stage', 'timer'
    ]
}

policy_kwargs = {
    'net_arch': {"pi": [128, 128], "vf": [64, 64]}
}

ppo_settings = {
    'gamma': 0.94,
    'model_checkpoint': "0",
    'batch_size': env_settings['env_num'] * 128,
    'learning_rate_start': 2.5e-4,
    'learning_rate_end': 2.5e-6,
    'clip_range_start': 0.15,
    'clip_range_end': 0.025,
    'seed': 42, 
    'policy_kwargs': policy_kwargs
}

# Optuna Hyperparameter Optimization

N_TRIALS = 200
N_STARTUP_TRIALS = 20
N_TIMESTEPS = int(15e4)  
DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
}

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate_start = trial.suggest_float("learning_rate_start", 1e-5, 2.5e-4, log=True)
    learning_rate_end = trial.suggest_float("learning_rate_end", 2.5e-6, 1e-4, log=True)
    clip_range_start = trial.suggest_float("clip_range_start", 0.05, 0.3)
    clip_range_end = trial.suggest_float("clip_range_end", 0.01, 0.1)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    net_arch = trial.suggest_categorical("net_arch", ["64_64", "128_128_64_64", "256_256_128_128"])

    net_arch_dict = {
        "64_64": {"pi": [64, 64], "vf": [64, 64]},
        "128_128_64_64": {"pi": [128, 128], "vf": [64, 64]},
        "256_256_128_128": {"pi": [256, 256], "vf": [128, 128]},
    }

    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    n_steps = trial.suggest_int("n_steps", 64, 1024)

    return {
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "batch_size": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": linear_schedule(learning_rate_start, learning_rate_end),
        "ent_coef": trial.suggest_float("ent_coef", 0.0, 0.1),
        "clip_range": linear_schedule(clip_range_start, clip_range_end),
        "vf_coef": trial.suggest_float("vf_coef", 0.5, 1.0),
        "max_grad_norm": trial.suggest_float("max_grad_norm", 0.3, 5.0),
        "policy_kwargs": {
            "net_arch": net_arch_dict[net_arch],
        }
    }

def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    params = sample_ppo_params(trial)
    kwargs.update(params)

    settings["action_space"] = trial.suggest_categorical("action_space", [SpaceTypes.DISCRETE, SpaceTypes.MULTI_DISCRETE])
    settings["step_ratio"] = trial.suggest_int("step_ratio", 1, 6)
    wrappers_settings["stack_actions"] = trial.suggest_int("stack_actions", 1, 48)

    env_settings_class = load_settings_flat_dict(EnvironmentSettings, settings)
    wrappers_settings_class = load_settings_flat_dict(WrappersSettings, wrappers_settings)

    env, num_envs = make_sb3_env(settings['game_id'], env_settings_class, wrappers_settings_class)

    kwargs.pop('policy', None)
    kwargs.pop('env', None)

    if 'seed' in ppo_settings:
        kwargs['seed'] = ppo_settings['seed']

    trial_log_dir = f"tensorboard_logs/trial_{trial.number}"
    os.makedirs(trial_log_dir, exist_ok=True)

    agent = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=trial_log_dir, **kwargs)


    agent.learn(total_timesteps=N_TIMESTEPS)

    total_rewards = []
    
    for eval_episode in range(10):
        eval_rewards = []
        observation = env.reset()
        done = False

        while not done:
            action, _states = agent.predict(observation, deterministic=False)
            observation, reward, done, info = env.step(action.tolist())
            eval_rewards.append(reward)

        episode_reward = sum(eval_rewards)
        total_rewards.append(episode_reward)
        logger.info(f"Episode {eval_episode + 1} Reward: {episode_reward}")

    mean_reward = sum(total_rewards) / len(total_rewards)
    logger.info(f"Mean Reward over 10 episodes: {mean_reward}")

    env.close()

    return mean_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-name', type=str, default='ppo_study')
    parser.add_argument('--storage', type=str, default='postgresql://optuna_user:your_password@localhost/optuna_db') 
    parser.add_argument('--process-id', type=int, default=0)  
    args = parser.parse_args()

    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS)

    study = None

    for attempt in range(3):
        try:
            study = optuna.create_study(
                study_name=args.study_name, 
                storage=args.storage, 
                sampler=sampler, 
                pruner=pruner, 
                direction="maximize",
                load_if_exists=True  
            )
            study.optimize(objective, n_trials=N_TRIALS)
            break  # Exit loop if successful
        except Exception as e:
            logger.error(f"Error creating or optimizing the study: {e}")
            if attempt < 2:  # Retry if not the last attempt
                logger.info("Retrying...")
                time.sleep(5)  # Wait for 5 seconds before retrying

        try:
            logger.info("Best trial:")
            trial = study.best_trial
            logger.info(f"  Value: {trial.value}")
            logger.info("  Params: ")
            for key, value in trial.params.items():
                logger.info(f"    {key}: {value}")
            logger.info("  User attrs:")
            for key, value in trial.user_attrs.items():
                logger.info(f"    {key}: {value}")
        except Exception as e:
            logger.error(f"Error accessing best trial: {e}")
