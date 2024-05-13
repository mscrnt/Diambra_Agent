import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import argparse
from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
from imitation.algorithms import bc
from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
import numpy as np
import config

class DiambraDataset(Dataset):
    def __init__(self, dataset_path):
        self.data_loader = DiambraDataLoader(dataset_path)
        self.data = []
        self.load_data()

    def load_data(self):
        n_loops = self.data_loader.reset()
        while n_loops == 0:
            obs, action, reward, terminated, truncated, info = self.data_loader.step()
            if terminated:
                self.data.append((obs, action, reward))
                n_loops = self.data_loader.reset()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        observation, action, reward = self.data[idx]
        frame = ToTensor()(observation['frame'])
        return frame, torch.tensor(action, dtype=torch.long), torch.tensor([reward], dtype=torch.float)

class SimpleCNN(nn.Module):
    def __init__(self, observation_shape, move_action_size, attack_action_size):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.feature_size = self._get_conv_output(observation_shape)
        self.fc_move = nn.Linear(self.feature_size, move_action_size)
        self.fc_attack = nn.Linear(self.feature_size, attack_action_size)

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))

    def forward(self, x):
        x = self.conv_layers(x)
        move_scores = self.fc_move(x)
        attack_scores = self.fc_attack(x)
        return move_scores, attack_scores

def get_observation_shape(observation_space):
    # This function assumes the observation_space is properly set. If not, it raises an error.
    if observation_space is None:
        raise ValueError("Observation space is not initialized.")
    return (observation_space.shape[0], observation_space.shape[1], observation_space.shape[2])

def setup_behavioral_cloning(env, dataset_path):
    dataset = DiambraDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Debug: print environment details to verify correct setup
    print(f"Environment details: Observation Space: {env.observation_space}, Action Space: {env.action_space}")

    if env.observation_space is None or env.action_space is None:
        raise ValueError("Environment spaces are not properly configured.")

    observation_shape = get_observation_shape(env.observation_space)
    move_action_size = env.action_space.nvec[0]  # Assuming the action space is MultiDiscrete
    attack_action_size = env.action_space.nvec[1]

    model = SimpleCNN(observation_shape, move_action_size, attack_action_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=model,
        optimizer=optimizer,
        demonstrations=dataloader
    )

    bc_trainer.train(n_epochs=10)
    torch.save(model.state_dict(), "behavior_cloning_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    args = parser.parse_args()
    if args.dataset_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        args.dataset_path = os.path.join(base_path, "DIAMBRA/episode_recording/doapp")
    print(f"Using dataset path: {args.dataset_path}")

    config.settings["action_space"] = SpaceTypes.MULTI_DISCRETE  # Adjust based on your config
    settings = load_settings_flat_dict(EnvironmentSettings, config.settings)
    wrappers_settings = load_settings_flat_dict(WrappersSettings, config.wrappers_settings)
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
    setup_behavioral_cloning(env, args.dataset_path)