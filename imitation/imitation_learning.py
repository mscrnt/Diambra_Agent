import os
import tempfile
from stable_baselines3 import PPO
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data.types import Transitions
from imitation.util.util import make_vec_env
from diambra.arena import SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(CustomCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(22 * 22 * 64, features_dim),
            nn.ReLU(),
        )
        self.print_expected_input_shape()

    def print_expected_input_shape(self):
        # Assuming the first layer of your CNN is a Conv2d layer
        first_conv_layer = self.cnn[0]
        print("Expected input channels:", first_conv_layer.in_channels)
        print("Expected kernel size:", first_conv_layer.kernel_size)

    def forward(self, observations):
        print("Entering forward function... Observations:", observations.keys())
        frame_obs = observations["frame"].float()  # Ensure tensor is float for CNN
        # Ensure frame_obs is in the format (batch_size, channels, height, width)
        if frame_obs.dim() == 3:  # Add a batch dimension if it's missing
            frame_obs = frame_obs.unsqueeze(0)
        print("Shape before CNN:", frame_obs.shape)
        cnn_output = self.cnn(frame_obs)
        print("Shape after CNN:", cnn_output.shape)
        return cnn_output

    
def collect_transitions(data_loader):
    transitions = []

    n_loops = data_loader.reset()
    while n_loops == 0:
        obs, act, rew, terminated, truncated, info = data_loader.step()

        # Directly store the whole observation and action since they are complex
        transitions.append({
            "obs": obs,
            "acts": act,
            "rews": rew,
            "next_obs": None,  # Placeholder, to be filled in the next step
            "dones": terminated or truncated,
        })

        if terminated or truncated:
            n_loops = data_loader.reset()

    # Fill in the next_obs by shifting the observations
    for i in range(len(transitions) - 1):
        transitions[i]["next_obs"] = transitions[i + 1]["obs"]

    # Remove the last transition if it doesn't have next_obs
    if transitions and transitions[-1]["next_obs"] is None:
        transitions.pop()

    return transitions


def main():
    # Setup DIAMBRA environment settings
    settings = EnvironmentSettings()
    settings.game_id = "doapp"
    settings.characters = "Jann-Lee"
    settings.difficulty = 4
    settings.action_space = SpaceTypes.MULTI_DISCRETE
    settings.frame_shape = (256, 256, 1)
    settings.step_ratio = 1

    # Wrappers Settings
    wrappers_settings = WrappersSettings()
    wrappers_settings.stack_frames = 4
    wrappers_settings.dilation = 1
    wrappers_settings.no_attack_buttons_combinations = True
    wrappers_settings.normalize_reward = True
    wrappers_settings.normalization_factor = 0.5
    wrappers_settings.stack_actions = 7
    wrappers_settings.scale = True
    wrappers_settings.exclude_image_scaling = True
    wrappers_settings.flatten = True
    wrappers_settings.process_discrete_binary = True
    wrappers_settings.role_relative = True
    wrappers_settings.add_last_action = True
    wrappers_settings.filter_keys = ["stage", "timer", "own_character", "own_health", "own_side", "own_wins", "opp_character", "opp_health", "opp_side", "opp_wins", "frame", "action"]

    # Create DIAMBRA training environment
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)

    print("Observation space:", env.observation_space)
    print("Observation space shape:", env.observation_space.shape)
    print("Frame shape:", env.observation_space.spaces['frame'].shape)


    log_path = "DIAMBRA/episode_recording/doapp/logs/"
    os.makedirs(log_path, exist_ok=True)

    # Setup the PPO model
    expert_policy = PPO("MultiInputPolicy", env, verbose=1, gamma=0.94, batch_size=32, n_epochs=4, n_steps=128, learning_rate=2.5e-4, clip_range=0.15, 
                    policy_kwargs={
                        "features_extractor_class": CustomCNNFeatureExtractor,
                        "features_extractor_kwargs": {"features_dim": 512},
                        "net_arch": [dict(pi=[64, 64], vf=[32, 32])]  # Specify the architecture for both policy (pi) and value function (vf)
                    }, 
                    tensorboard_log=log_path, seed=42, device="cpu")

    # Initialize the dataset loader
    dataset_path = "DIAMBRA/episode_recording/doapp"
    data_loader = DiambraDataLoader(dataset_path)
    data_loader.reset()

    # Then use this function to collect transitions from your dataset
    transitions = collect_transitions(data_loader)

    # When initializing your BC trainer, make sure to use the environment's observation and action spaces
    bc_trainer = bc.BC(
        observation_space=env.observation_space,  # Ensure this matches your environment's observation space
        action_space=settings.action_space,  # Ensure this matches your environment's action space
        demonstrations=transitions,  # Use the transitions collected from your dataset
        rng=np.random.default_rng(42),
    )

    # Setup DAgger with the custom expert policy and BC trainer
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert_policy,  # Use the custom expert policy
            bc_trainer=bc_trainer,
        )

        dagger_trainer.train(8_000)

if __name__ == "__main__":
    main()