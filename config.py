# config.py

from mscrnt_utils import RunIDGenerator, generate_parameters_report
import os

monitoring_settings = {
    'minimum_stage': 3,
    'maximum_stage': 8
}

# Environment Settings
env_settings = {
    'env_num': 32,
    'check_freq': 100000,
    'time_steps': 16000000
}

# Derived Settings
n_steps = 128
batch_size = env_settings['env_num'] * n_steps

# Game Settings
settings = {
    'game_id': "mvsc",
    'difficulty': 8,
    'characters': ("Spider Man", "Ryu"),  # Example: Using "Spider Man" and "Ryu" for two character slots
    'action_space': "MULTI_DISCRETE",
    'step_ratio': 1,
    'frame_shape': (128, 128, 1)
}

# Wrappers Settings
wrappers_settings = {
    'stack_frames': 4,
    'dilation': 1,
    'no_attack_buttons_combinations': False,
    'normalize_reward': False,
    'normalization_factor': 0.5, 
    'stack_actions': 8,
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
tensorboard_log_path = f"logs/{settings['game_id']}/{settings['characters']}/{run_id}"

# Ensure directories are made
os.makedirs(model_path, exist_ok=True)
os.makedirs(tensorboard_log_path, exist_ok=True)

generate_parameters_report(run_id, settings, wrappers_settings, ppo_settings, tensorboard_log_path)
