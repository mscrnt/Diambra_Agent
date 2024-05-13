import os
import torch
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn

from diambra.arena import load_settings_flat_dict, SpaceTypes
from diambra.arena.stable_baselines3.make_sb3_env import make_sb3_env, EnvironmentSettings, WrappersSettings
from diambra.arena.utils.diambra_data_loader import DiambraDataLoader
from diambra.arena.stable_baselines3.sb3_utils import linear_schedule
from imitation.algorithms import bc
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import config

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Adjusted for 1 input channel
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fc_move = nn.Linear(64 * 64 * 64, 9)  # Output layer for moves
        self.fc_attack = nn.Linear(64 * 64 * 64, 8)  # Output layer for attacks

    def forward(self, x):
        x = self.conv_layers(x)
        move = self.fc_move(x)
        attack = self.fc_attack(x)
        return move, attack

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_extractor_class, features_dim):
        super().__init__(observation_space, action_space, lr_schedule=lr_schedule)
        self.features_extractor = features_extractor_class(observation_space, features_dim)

class ExpertDataset(Dataset):
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
        move, attack = action  # Assuming action is a tuple (move, attack)
        move = torch.tensor(move, dtype=torch.long)
        attack = torch.tensor(attack, dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        return frame, (move, attack), reward


def custom_collate_fn(batch):
    print("Collating batch of size:", len(batch))
    obs = torch.stack([item['obs']['frame'] for item in batch])
    acts = torch.stack([item['acts'] for item in batch])
    reward = torch.stack([item['reward'] for item in batch])
    terminated = torch.stack([item['terminated'] for item in batch])
    info = {key: torch.stack([item['info'][key] for item in batch]) for key in batch[0]['info']}
    print("Batch shapes - obs:", obs.shape, "acts:", acts.shape, "reward:", reward.shape, "terminated:", terminated.shape, "info:", {k: v.shape for k, v in info.items()})
    return {'obs': {'frame': obs}, 'acts': acts, 'reward': reward, 'terminated': terminated, 'info': info}

def setup_behavioral_cloning(env, features_extractor_class, features_dim):
    lr_schedule = linear_schedule(1e-3)
    policy = CustomPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lr_schedule,
        features_extractor_class=features_extractor_class,
        features_dim=features_dim
    )
    print("Policy setup completed.")
    return bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=policy,
        batch_size=3,  # Consistency with DataLoader
        rng=np.random.default_rng()
    )

def main():
    config.settings["action_space"] = SpaceTypes.DISCRETE if config.settings["action_space"].lower() == "discrete" else SpaceTypes.MULTI_DISCRETE
    settings = load_settings_flat_dict(EnvironmentSettings, config.settings)
    wrappers_settings = load_settings_flat_dict(WrappersSettings, config.wrappers_settings)
    env, num_envs = make_sb3_env(settings.game_id, settings, wrappers_settings)
    
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DIAMBRA/episode_recording/doapp")
    expert_dataset = ExpertDataset(dataset_path)
    print("Dataset Loaded. Number of entries:", len(expert_dataset))

    train_loader = DataLoader(expert_dataset, batch_size=3, shuffle=True, drop_last=True, collate_fn=custom_collate_fn)
    bc_trainer = setup_behavioral_cloning(env, CustomCNN, 512)

    # Function to simulate BC internal processing
    def simulate_bc_processing(batch, device='cpu'):
        try:
            # Convert to the device
            obs = batch['obs']['frame'].to(device)
            acts = batch['acts'].to(device)
            rewards = batch['reward'].to(device)
            done = batch['terminated'].to(device)
            info = {key: value.to(device) for key, value in batch['info'].items()}

            # Process with the policy
            with torch.no_grad():
                policy_output = bc_trainer.policy(obs)
                print("Policy output type:", type(policy_output))
                print("Policy output:", policy_output)

                # Check how to correctly use the policy output
                if isinstance(policy_output, dict) and 'log_prob' in policy_output:
                    loss = -torch.sum(policy_output['log_prob'](acts))
                else:
                    # Assuming policy_output directly gives the log probabilities
                    loss = -torch.sum(policy_output.log_prob(acts))
                    
            print("Simulated Loss calculated:", loss.item())
        except Exception as e:
            print("Error during BC processing simulation:", str(e))

    try:
        # Simulate processing as done in `bc.BC`
        for batch in train_loader:
            print("Batch Type:", type(batch))
            print("Batch keys:", list(batch.keys()))
            print("Observation structure:", type(batch['obs']), list(batch['obs'].keys()))
            print("Detailed batch content:", batch)
            simulate_bc_processing(batch)  # Simulate processing for each batch

            print("Setting demonstrations with the current batch...")
            bc_trainer.set_demonstrations(batch)
            print("Starting training...")
            bc_trainer.train(n_epochs=10)
            print("Training completed for one batch")
            break  # For debugging, process only one batch
    except Exception as e:
        print("Error during training:", e)
        print("Current batch causing error:", batch)

    model_path = os.path.join(config.model_path, "bc_trained_model.zip")
    bc_trainer.policy.save(model_path)
    print("Model saved.")
if __name__ == "__main__":
    main()
