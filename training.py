import os
import glob
import config 
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule, AutoSave
from stable_baselines3 import PPO
import torch


def main():
    # Convert action_space from string to SpaceTypes enum
    config.settings["action_space"] = SpaceTypes.DISCRETE if config.settings["action_space"].lower() == "discrete" else SpaceTypes.MULTI_DISCRETE

    # Load settings from config into respective classes
    settings = load_settings_flat_dict(EnvironmentSettings, config.settings)
    wrappers_settings = load_settings_flat_dict(WrappersSettings, config.wrappers_settings)

    # Extract PPO settings from config
    hparams = config.ppo_settings
    model_path = config.model_path
    tensorboard_log_path = config.tensorboard_log_path

    os.makedirs(model_path, exist_ok=True)

    # Create environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
    #print("Activated {} environment(s)".format(num_envs))

    checkpoint_files = glob.glob(os.path.join(model_path, "*.zip"))
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        print(f"Loading latest checkpoint: {latest_checkpoint}")
        agent = PPO.load(latest_checkpoint, env,
                         learning_rate=linear_schedule(2.28e-4, 2.5e-6),
                         clip_range=linear_schedule(0.1388, 0.025),
                         ) 
    else:
        print("No checkpoint file found. Training a new model.")
        agent = PPO("MultiInputPolicy", env, verbose=1,
                    gamma=hparams['gamma'], batch_size=hparams['batch_size'],
                    n_epochs=hparams['n_epochs'], n_steps=hparams['n_steps'],
                    learning_rate=linear_schedule(hparams['learning_rate_start'], hparams['learning_rate_end']),
                    clip_range=linear_schedule(hparams['clip_range_start'], hparams['clip_range_end']),
                    policy_kwargs=hparams['policy_kwargs'],
                    tensorboard_log=tensorboard_log_path,
                    seed=hparams['seed'],
                    device="cuda" if torch.cuda.is_available() else "cpu")

    # Print policy network architecture
    #print("Policy architecture:")
    #print(agent.policy)

    # Create the callback: autosave every USER DEF steps
    autosave_freq = config.env_settings["check_freq"]
    auto_save_callback = AutoSave(check_freq=autosave_freq, num_envs=num_envs, save_path=model_path)

    # Train the agent

    time_steps = config.env_settings["time_steps"]
    print(f"Training for {time_steps} time steps...")
    agent.learn(total_timesteps=time_steps, callback=auto_save_callback)

    print("Training finished!")
    # Save the agent
    agent.save(model_path)

    print("Model saved!")

    # Close the environment
    env.close()

    # Return success
    return 0

if __name__ == "__main__":
    main()
