from mscrnt_utils import RunIDGenerator, generate_parameters_report
import os
from custom_wrapper import custom_wrapper


# Environment Settings
env_settings = {
    'env_num': 1,
    'check_freq': 100000,
    'time_steps': 10000000
}

# Derived Settings
n_steps = 128
batch_size = env_settings['env_num'] * n_steps

# Game Settings
settings = {
    'game_id': "doapp",
    'difficulty': 4,
    'characters': "Ayane",
    'outfits': 4,
    'action_space': "multi_discrete",
    'step_ratio': 5,
    'frame_shape': (128, 128, 1)
}

# Wrappers Settings
wrappers_settings = {
    'wrappers': [[custom_wrapper, {}]],
    'stack_frames': 4,
    'dilation': 1,
    'no_attack_buttons_combinations': False,
    'normalize_reward': False,
    'normalization_factor': 0.3,
    'stack_actions': 12,
    'scale': False,
    'exclude_image_scaling': False,
    'flatten': True,
    'process_discrete_binary': True,
    'role_relative': True,
    'add_last_action': True,
    'filter_keys': ["stage", "timer", "own_character", "own_health", "own_side", "own_wins", "opp_character", "opp_health", "opp_side", "opp_wins", "frame", "action"]
}

# Policy Settings
policy_kwargs = {
    'net_arch': {"pi": [128, 128], "vf": [64, 64]}
}

# PPO Settings
ppo_settings = {
    'gamma': 0.94,
    'model_checkpoint': "0",
    'n_epochs': 4,
    'n_steps': n_steps,
    'batch_size': batch_size,
    'learning_rate_start': 2.5e-4,
    'learning_rate_end': 2.5e-6,
    'clip_range_start': 0.15,
    'clip_range_end': 0.025,
    'seed': 42,
    'policy_kwargs': policy_kwargs
}

# Generate run_id
run_id = RunIDGenerator.create(settings, wrappers_settings, ppo_settings)

# Folders
model_path = f"models/{settings['game_id']}/{settings['characters']}/{run_id}"
tensorboard_log_path = f"logs/{settings['game_id']}/{run_id}"

# Ensure directories are made
os.makedirs(model_path, exist_ok=True)
os.makedirs(tensorboard_log_path, exist_ok=True)

generate_parameters_report(run_id, settings, wrappers_settings, ppo_settings)
