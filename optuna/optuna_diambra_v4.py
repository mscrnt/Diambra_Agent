import os
import optuna
from typing import Any, Dict
from optuna.pruners import PercentilePruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
import torch
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule
import torch.nn as nn
import argparse
import logging
import time
import traceback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Settings

env_settings = {
    'check_freq': 50000,
    'time_steps': 500000  
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

ppo_settings = {
    'model_checkpoint': "0",
    'seed': 42, 
}

# Optuna Hyperparameter Optimization
N_TRIALS = 1000
N_STARTUP_TRIALS = int(0.2 * N_TRIALS)
N_TIMESTEPS = env_settings['time_steps']
EVAL_FREQ = env_settings['check_freq']
DEFAULT_HYPERPARAMS = {
    "policy": "MultiInputPolicy",
}

# Advanced Pruner Class
class AdvancedPruner(PercentilePruner):
    def __init__(self, percentile: float, n_startup_trials: int = 5, n_warmup_steps: int = 0, interval_steps: int = 1, n_min_trials: int = 1, deviation_threshold: float = 0.05):
        super().__init__(percentile, n_startup_trials, n_warmup_steps, interval_steps, n_min_trials=n_min_trials)
        self.deviation_threshold = deviation_threshold  
        self.threshold_counter = 0

    def prune(self, study, trial):
        logger.info(f"Starting pruning evaluation for Trial {trial.number}")

        # Respect the n_startup_trials parameter
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        logger.info(f"Number of completed trials: {len(completed_trials)}")
        if len(completed_trials) < self._n_startup_trials:
            logger.info(f"Skipping pruning for Trial {trial.number} because fewer than {self._n_startup_trials} trials have completed.")
            return False

        # Apply the PercentilePruner logic
        logger.info(f"Applying PercentilePruner logic to Trial {trial.number}")
        if super().prune(study, trial):
            logger.info(f"Trial {trial.number} pruned by PercentilePruner at step {trial.last_step}")
            return True

        # Skip custom pruning logic if the trial has not completed 100 more trials than the n_startup_trials
        custom_startup_trials = self._n_startup_trials + 100
        if len(completed_trials) < custom_startup_trials:
            logger.info(f"Skipping custom pruning logic for Trial {trial.number} because fewer than {custom_startup_trials} trials have completed.")
            return False

        # Custom pruner logic
        logger.info(f"Applying custom pruning logic to Trial {trial.number}")

        step = trial.last_step
        if step is None:
            logger.info(f"Trial {trial.number} has no steps to evaluate for custom pruning")
            return False

        intermediate_values = trial.intermediate_values
        if len(intermediate_values) < 2:
            logger.info(f"Trial {trial.number} does not have enough intermediate values to prune (required at least 2, got {len(intermediate_values)})")
            return False

        values = list(intermediate_values.values())

        # Exclude the last value for the overall average calculation
        if len(values) > 1:
            overall_average = sum(values[:-1]) / len(values[:-1])
        else:
            overall_average = values[0]  # If only one value, use it as the average

        deviation = abs(overall_average) * self.deviation_threshold
        threshold = overall_average - deviation

        logger.info(f"Trial {trial.number} - Intermediate values (excluding last for average): {values[:-1]}")
        logger.info(f"Trial {trial.number} - Overall average (excluding last value): {overall_average}")
        logger.info(f"Trial {trial.number} - Deviation threshold: {threshold}")
        logger.info(f"Trial {trial.number} - Last value: {values[-1]}")

        # Increment counter if the last value is greater than the previous one but still below the threshold
        if len(values) >= 2 and values[-1] > values[-2] and values[-1] < threshold:
            self.threshold_counter += 1
            logger.info(f"Trial {trial.number} - Counter incremented to {self.threshold_counter}")
        else:
            self.threshold_counter = 0  # Reset counter if the condition is not met
            logger.info(f"Trial {trial.number} - Counter reset to {self.threshold_counter}")

        # Prune if the condition has been met 2 times in a row
        if step % 150000 == 0 and self.threshold_counter >= 2:
            logger.info(f"Trial {trial.number} pruned at step {step} due to performance below acceptable range for 2 consecutive evaluations")
            return True

        # Prune based on the most recent value and deviation threshold
        if values[-1] < threshold:
            logger.info(f"Trial {trial.number} pruned at step {step} due to performance below acceptable range (threshold: {threshold})")
            return True

        logger.info(f"Trial {trial.number} continues (not pruned)")
        return False

    
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""
    learning_rate_start = trial.suggest_float("learning_rate_start", 1e-5, 3.0e-4, log=True)
    learning_rate_end = trial.suggest_float("learning_rate_end", 1e-6, learning_rate_start, log=True)  
    clip_range_start = trial.suggest_float("clip_range_start", 0.1, 0.3, log=True)
    clip_range_end = trial.suggest_float("clip_range_end", 0.01, clip_range_start, log=True)  
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    net_arch = trial.suggest_categorical("net_arch", ["64_64", "128_128_64_64", "256_256_128_128"])

    net_arch_dict = {
        "64_64": {"pi": [64, 64], "vf": [64, 64]},
        "128_128_64_64": {"pi": [128, 128], "vf": [64, 64]},
        "256_256_128_128": {"pi": [256, 256], "vf": [128, 128]},
    }

    n_epochs = trial.suggest_int("n_epochs", 1, 10)
    n_steps = trial.suggest_int("n_steps", 64, 1024)

    # Optional parameters
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.99)
    vf_coef = trial.suggest_float("vf_coef", 0.5, 1.0)
    ent_coef = trial.suggest_float("ent_coef", 0.0, 0.1)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0)

    return {
        # Optional parameters
        "gae_lambda": gae_lambda,
        "vf_coef": vf_coef,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,

        # Required parameters
        "n_steps": n_steps,
        "n_epochs": n_epochs,
        "batch_size": n_steps,
        "gamma": gamma,
        "learning_rate_end": learning_rate_end,
        "clip_range_end": clip_range_end,
        "learning_rate_start": learning_rate_start,
        "clip_range_start": clip_range_start,
        "policy_kwargs": {
            "net_arch": net_arch_dict[net_arch],
        }
    }

def tflog2pandas(path: str) -> pd.DataFrame:
    """Convert single tensorflow log file to pandas DataFrame"""
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    except Exception:
        print(f"Event file possibly corrupt: {path}")
        # Print exception traceback
        traceback.print_exc()
    return runlog_data

def many_logs2pandas(event_paths):
    all_logs = []
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            all_logs.append(log)
    if all_logs:
        all_logs = pd.concat(all_logs, ignore_index=True)
    return all_logs

def get_latest_values_from_tensorboard(logdir):
    event_paths = glob.glob(os.path.join(logdir, "**", "event*"), recursive=True)
    all_log = many_logs2pandas(event_paths)
    logger.info(f"Found {len(all_log)} event entries")
    return all_log

class TrialEvalCallback(BaseCallback):
    """
    A callback for evaluating the model and stopping the trial if no improvement is observed.
    """
    def __init__(self, study, trial, eval_env, trial_log_dir, n_eval_episodes=10, eval_freq=10000, best_mean_reward=None, log_dir='./logs/', verbose=1):
        super(TrialEvalCallback, self).__init__(verbose)
        self.study = study
        self.trial = trial
        self.eval_env = eval_env
        self.trial_log_dir = trial_log_dir
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = best_mean_reward
        self.log_dir = log_dir
        self.last_mean_reward = -np.inf
        self.is_pruned = False

    def _on_step(self) -> bool:
        try:
            if self.n_calls % self.eval_freq == 0:
                # Evaluate the policy
                mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=False)
                
                # Report the mean reward to the trial
                self.trial.report(mean_reward, self.num_timesteps)
                
                # Get the latest values from TensorBoard logs
                log_df = get_latest_values_from_tensorboard(self.trial_log_dir)
                
                if not log_df.empty:
                    logger.info(f"All tags: {log_df['metric'].unique()}")
                    
                    # Get the explained variance
                    explained_variance_df = log_df[log_df['metric'] == 'train/explained_variance']
                    
                    if not explained_variance_df.empty:
                        explained_variance = explained_variance_df.iloc[-1]['value']
                        ev_string = f"Explained Variance at timestep {self.num_timesteps}"
                        self.trial.set_user_attr(ev_string, explained_variance)
                    else:
                        logger.warning(f"No explained variance data available at timestep {self.num_timesteps}")
                    

                # Update the best mean reward
                if self.best_mean_reward is None or mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                
                self.last_mean_reward = mean_reward
                
                if self.verbose > 0:
                    logger.info(f"Evaluation at timestep {self.num_timesteps}: mean reward {mean_reward} - best mean reward {self.best_mean_reward}")

                # Prune trial if the performance does not improve
                if self.trial.should_prune():
                    logger.info(f"Pruning trial at timestep {self.num_timesteps} with mean reward {mean_reward}")
                    
                    self.is_pruned = True
                    return False  # Stop training

                
        except Exception as e:
            logger.error(f"Error during _on_step at timestep {self.num_timesteps}: {str(e)}")
            traceback.print_exc()
        
        return True  # Continue training if not pruned

def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    params = sample_ppo_params(trial)
    kwargs.update(params)

    ending_lr = params['learning_rate_end']
    ending_clip = params['clip_range_end']
    start_lr = params['learning_rate_start']
    start_clip = params['clip_range_start']

    ## Uncomment the following line to the action space to the sampling
    # settings["action_space"] = trial.suggest_categorical("action_space", [SpaceTypes.DISCRETE, SpaceTypes.MULTI_DISCRETE])
    
    ## Uncomment the following line to the use the same action space for all trials
    settings["action_space"] = SpaceTypes.MULTI_DISCRETE

    # Step ratio is the same for all games. No need to change it.
    settings["step_ratio"] = trial.suggest_int("step_ratio", 1, 6)

    # Stack actions are specific to each game. Change the range according to the game.
    wrappers_settings["stack_actions"] = trial.suggest_int("stack_actions", 1, 48)

    env_settings_class = load_settings_flat_dict(EnvironmentSettings, settings)
    wrappers_settings_class = load_settings_flat_dict(WrappersSettings, wrappers_settings)

    env, num_envs = make_sb3_env(settings['game_id'], env_settings_class, wrappers_settings_class)

    kwargs.pop('policy', None)
    kwargs.pop('env', None)
    kwargs.pop('learning_rate_end', None)
    kwargs.pop('clip_range_end', None)
    kwargs.pop('learning_rate_start', None)
    kwargs.pop('clip_range_start', None)

    total_timesteps = env_settings['time_steps']
    eval_freq = env_settings['check_freq']
    n_eval_episodes = 10

    trial_log_dir = f"tensorboard_logs/trial_{trial.number}"
    os.makedirs(trial_log_dir, exist_ok=True)

    # Create schedule functions
    lr_schedule = linear_schedule(start_lr, ending_lr)
    clip_range_schedule = linear_schedule(start_clip, ending_clip)

    agent = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=trial_log_dir,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        **kwargs
    )

    callback = TrialEvalCallback(study, trial, eval_env=env, trial_log_dir=trial_log_dir, n_eval_episodes=n_eval_episodes, eval_freq=eval_freq)

    agent.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)

    if callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    # Use the last evaluation from the callback
    mean_reward = callback.last_mean_reward

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
    pruner = AdvancedPruner(
        percentile=50.0,  # Set this according to your needs
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=0,
        interval_steps=1,
        n_min_trials=1,
        deviation_threshold=0.05  # Allowable deviation from the average
    )

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

    if study:
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