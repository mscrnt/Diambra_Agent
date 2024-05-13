import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import os
import argparse
from diambra.arena.utils.diambra_data_loader import DiambraDataLoader

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
        move, attack = action 
        move = torch.tensor(move, dtype=torch.long)
        attack = torch.tensor(attack, dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        return frame, (move, attack), reward

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
        )
        self.fc_move = nn.Linear(64 * 64 * 64, 9)  
        self.fc_attack = nn.Linear(64 * 64 * 64, 8)  

    def forward(self, x):
        x = self.conv_layers(x)
        move = self.fc_move(x)
        attack = self.fc_attack(x)
        return move, attack


def train_bc(dataset_path):
    dataset = DiambraDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = SimpleCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for frames, (moves, attacks), _ in dataloader:
            optimizer.zero_grad()
            predicted_moves, predicted_attacks = model(frames)
            loss_moves = criterion(predicted_moves, moves)
            loss_attacks = criterion(predicted_attacks, attacks)
            loss = loss_moves + loss_attacks
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "behavior_cloning_model.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, help='Path to dataset')
    args = parser.parse_args()
    
    if args.dataset_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
        args.dataset_path = os.path.join(base_path, "DIAMBRA/episode_recording/doapp")
    
    print(f"Using dataset path: {args.dataset_path}")
    train_bc(args.dataset_path)
