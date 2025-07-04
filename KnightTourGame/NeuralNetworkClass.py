###################################################
# KnightAlphaZeroSelfPlay.py
# 
# One-file example of an AlphaZero-style self-play
# approach for the Knight’s Tour. 
#
# 1) KnightNetwork: Policy+Value Net
# 2) KnightPUCTPlayer: uses MCTS with the net
# 3) Self-play data generation 
# 4) Training routine (policy + value heads)
#
# You can integrate this into KnightGameStarter by
# adding a new mode that imports and calls 
# 'main_selfplay_training()'.
###################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class KnightNetwork(nn.Module):
    """
    Similar to your Gomoku "NeuralNetworkClass.py",
    but outputs policy over 8 knight moves + a scalar value.
    """
    def __init__(self, board_size=8, in_channels=2):
        super(KnightNetwork, self).__init__()
        self.board_size = board_size
        self.in_channels = in_channels

        # A small convolutional trunk
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.flat_size = 64 * board_size * board_size

        # Policy head => 8 knight-move logits
        self.policy_head = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 possible knight moves
        )

        # Value head => scalar in [-1, +1]
        self.value_head = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # x: (batch_size, in_channels, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.flat_size)

        policy_logits = self.policy_head(x)  # shape (batch, 8)
        value = self.value_head(x)           # shape (batch, 1)
        return policy_logits, value

    def save_weights(self, path="knight_network.pt"):
        torch.save(self.state_dict(), path)
        print(f"Saved network weights to {path}")

    def load_weights(self, path="knight_network.pt"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            print(f"Loaded network weights from {path}")
        else:
            print(f"No weights found at {path}; using random initialization.")


