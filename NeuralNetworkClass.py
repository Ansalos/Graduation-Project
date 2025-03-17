import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class GameNetwork(nn.Module):
    def __init__(self, board_size=9, in_channels=2):
        """
        A simple CNN with two heads:
          - Policy head: action probabilities
          - Value head: a scalar in [-1, +1]
        """
        super(GameNetwork, self).__init__()
        self.board_size = board_size
        self.in_channels = in_channels
        
        # Convolution trunk
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.flat_size = 64 * board_size * board_size
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Linear(256, board_size * board_size)  # one logit per cell
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # outputs in [-1, +1]
        )

    def forward(self, x):
        # x shape: (batch_size, in_channels, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.flat_size)
        
        policy_logits = self.policy_head(x)  # (batch_size, board_size*board_size)
        value = self.value_head(x)           # (batch_size, 1)
        return policy_logits, value

    def save_weights(self, path="network_weights.pt"):
        torch.save(self.state_dict(), path)
        print(f"Saved network weights to {path}")

    def load_weights(self, path="network_weights.pt"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Loaded network weights from {path}")
        else:
            print(f"No weights found at {path}; using random initialization.")
