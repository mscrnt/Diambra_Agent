import numpy as np
import logging
from diambra.arena import SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from stable_baselines3 import PPO
import diambra
from logging.handlers import RotatingFileHandler


# Create handlers
file_handler = RotatingFileHandler('docker_logs/docker_monitor.log', maxBytes=1024*1024*5, backupCount=5)
file_handler.setLevel(logging.DEBUG)  # Log DEBUG and higher levels to the file
file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Log INFO and higher levels to the console
console_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s:%(message)s'))

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Setting to DEBUG will catch everything
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def main():
    # Settings
    settings = EnvironmentSettings()
    settings.step_ratio = 1
    settings.difficulty = 8
    settings.action_space = SpaceTypes.MULTI_DISCRETE
    settings.game_id = "mvsc"
    settings.characters = ("Spider Man", "Ryu")
    #settings.frame_shape = (0, 0, 0)

    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.frame_shape = (128, 128, 1)
    wrappers_settings.stack_frames = 4
    wrappers_settings.dilation = 1
    wrappers_settings.no_attack_buttons_combinations = False
    wrappers_settings.normalize_reward = False
    wrappers_settings.normalization_factor = 0.5
    wrappers_settings.stack_actions = 8
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.flatten = True
    wrappers_settings.process_discrete_binary = True
    wrappers_settings.role_relative = True
    wrappers_settings.add_last_action = True
    wrappers_settings.filter_keys = ['action', 'frame', 'opp_active_character', 'opp_character', 'opp_character_1', 
        'opp_character_2', 'opp_health_1', 'opp_health_2', 'opp_partner', 'opp_partner_attacks', 
        'opp_side', 'opp_super_bar', 'opp_super_count', 'opp_wins', 'own_active_character', 'own_character', 
        'own_character_1', 'own_character_2', 'own_health_1', 'own_health_2', 'own_partner', 'own_partner_attacks', 
        'own_side', 'own_super_bar', 'own_super_count', 'own_wins', 'stage', 'timer'
    ]
    env = diambra.arena.make("mvsc", settings, wrappers_settings, render_mode="human")

    checkpoint = "models/mvsc/('Spider Man', 'Ryu')/SR1-SA8/autosave_12400000.zip"

    agent = PPO.load(checkpoint, env, device="cpu")
    logger.info("Agent loaded successfully!")
    logger.info("Beginning Evaluation...")

    observation, info = env.reset()

    while True:
        env.render()
        action, _state = agent.predict(observation, deterministic=False)
        observation, reward, terminated, truncated, info = env.step(action.tolist())
        logger.debug(f"Action taken: {action} | Reward: {reward}")

        if terminated or truncated:
            observation, info = env.reset()
            logger.info("Environment reset due to termination.")
            if info["env_done"]:
                logger.info("Evaluation completed.")
                break    

    # Close the environment
    env.close()
    
    # Return success
    return 0

if __name__ == "__main__":
    main()