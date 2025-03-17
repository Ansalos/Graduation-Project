import math
import random
import numpy as np
import torch

from Gomoku import Gomoku
from mcts_player import MCTSPlayer
from NeuralNetworkClass import GameNetwork

#######################################
# 1) GENERATE SELF-PLAY DATA (MCTS vs. MCTS)
#######################################

# def generate_self_play_data(mcts_player, num_games=1000):
#     """
#     Play 'num_games' MCTS-vs-itself games. 
#     For each move, record (state, root_visit_distribution, final_outcome).
#     Return a list of (state_tensor, policy_tensor, value_float).
#     """
#     dataset = []
#     for g in range(num_games):
#         game = Gomoku()
#         # We'll store records for each move in the current game
#         move_records = []  # each entry: (state_tensor, policy_distribution)
#         current_player = Gomoku.BLACK  # black always goes first

#         while game.status == game.ONGOING:
#             # Encode current state
#             state_tensor = encode_state(game)

#             # Run MCTS to get root node & visit counts
#             root, visit_counts = mcts_player.run_mcts(game.clone())
            
#             # Convert visit_counts -> policy vector
#             policy_vec = visit_counts_to_policy(game, visit_counts)

#             # Save partial record
#             move_records.append((state_tensor, policy_vec, current_player))

#             # Choose a move from visit_counts (e.g. argmax)
#             move = max(visit_counts.items(), key=lambda x: x[1])[0]
#             game.make(move)

#             # Switch player
#             current_player = game.player

#         # Now the game ended
#         outcome = game.status  # BLACK=+1, WHITE=-1, DRAW=0
#         if outcome == Gomoku.BLACK:
#             final_value = 1.0
#         elif outcome == Gomoku.WHITE:
#             final_value = -1.0
#         else:
#             final_value = 0.0

#         # Each move record is from the perspective of who moved. 
#         # If that mover was BLACK, the target value is final_value.
#         # If that mover was WHITE, the target value is -final_value.
#         for (st, pol, mover) in move_records:
#             if mover == Gomoku.BLACK:
#                 value = final_value
#             else:
#                 value = -final_value
#             dataset.append((st, pol, value))

#         if (g+1) % 100 == 0:
#             print(f"Generated {g+1}/{num_games} self-play games...")

#     return dataset

def generate_self_play_data_with_choose_move(mcts_player, num_games=1000):
    """
    Play 'num_games' using MCTS choose_move() (rather than run_mcts).
    For each move, record:
        - (state_tensor, 1-hot policy, final_value-from-this-player's-perspective)
    Return a list of (state_tensor, policy_tensor, value_float).
    """
    dataset = []
    for g in range(num_games):
        game = Gomoku()
        # We'll store records for each move in the current game
        move_records = []  # each entry: (state_tensor, 1-hot-policy-vector, mover_color)

        while game.status == game.ONGOING:
            # Encode current state as a (2, board_size, board_size) tensor
            state_tensor = encode_state(game)

            # Ask the MCTSPlayer for the best move
            move = mcts_player.choose_move(game.clone())

            # Build a 1-hot policy vector of length (size * size)
            policy_vec = torch.zeros(game.size * game.size, dtype=torch.float32)
            row_col_index = move[0] * game.size + move[1]
            policy_vec[row_col_index] = 1.0

            current_player = game.player  # This is whoever is about to move
            move_records.append((state_tensor, policy_vec, current_player))

            # Execute the move on the real game board
            game.make(move)

        # Once the game is over, determine final outcome (+1 = Black win, -1 = White win, 0 = draw)
        if game.status == Gomoku.BLACK:
            final_value = 1.0
        elif game.status == Gomoku.WHITE:
            final_value = -1.0
        else:
            final_value = 0.0

        # Assign the final_value perspective to each record
        for (st_tensor, pol_vec, mover) in move_records:
            # If the mover was BLACK, value = final_value
            # If the mover was WHITE, value = -final_value
            if mover == Gomoku.BLACK:
                value = final_value
            else:
                value = -final_value

            dataset.append((st_tensor, pol_vec, value))

        # if (g+1) % 100 == 0:
        print(f"Generated {g+1}/{num_games} self-play games...")

    return dataset

def encode_state(game):
    """
    Convert the game board into a (2, size, size) float tensor.
    """
    import torch
    size = game.size
    arr = np.zeros((2, size, size), dtype=np.float32)
    for r in range(size):
        for c in range(size):
            if game.board[r][c] == Gomoku.BLACK:
                arr[0, r, c] = 1.0
            elif game.board[r][c] == Gomoku.WHITE:
                arr[1, r, c] = 1.0
    return torch.from_numpy(arr)

# def visit_counts_to_policy(game, visit_counts):
#     """
#     Convert a dict {move: count} to a vector of length size*size,
#     then normalize to sum=1.
#     """
#     size = game.size
#     policy = np.zeros((size*size,), dtype=np.float32)
#     total = 1e-8
#     for (r,c), cnt in visit_counts.items():
#         idx = r*size + c
#         policy[idx] = cnt
#         total += cnt
#     policy /= total
#     return torch.from_numpy(policy)

#######################################
# 2) TRAIN THE NETWORK
#######################################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_network(network, dataset, epochs=5, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # Build big lists of states, policies, values
    states, policies, values = [], [], []
    for (s, p, v) in dataset:
        states.append(s)
        policies.append(p)
        values.append(v)

    states = torch.stack(states)                 # (N, 2, size, size)
    policies = torch.stack(policies)             # (N, size*size)
    values = torch.tensor(values).view(-1,1)     # (N,1)

    ds = TensorDataset(states, policies, values)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs+1):
        network.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0

        for batch_states, batch_policies, batch_values in loader:
            batch_states = batch_states.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)

            optimizer.zero_grad()

            # Forward
            logits, value_pred = network(batch_states)
            # Policy head => cross-entropy with 'batch_policies' distribution
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -torch.sum(batch_policies * log_probs, dim=1).mean()

            # Value head => MSE with batch_values
            value_loss = mse_loss(value_pred, batch_values)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            bsz = batch_states.size(0)
            total_policy_loss += policy_loss.item() * bsz
            total_value_loss += value_loss.item() * bsz
            total_samples += bsz

        avg_p_loss = total_policy_loss / total_samples
        avg_v_loss = total_value_loss / total_samples
        print(f"Epoch {epoch}/{epochs} | Policy Loss={avg_p_loss:.4f} | Value Loss={avg_v_loss:.4f}")

    network.save_weights("trained_network.pt")

#######################################
# 3) MAIN EXAMPLE
#######################################

if __name__ == "__main__":
    # 1) Generate self-play data with MCTS
    mcts_ai = MCTSPlayer(iterations=5000, exploration_constant=math.sqrt(2))
    dataset = generate_self_play_data_with_choose_move(mcts_ai, num_games=10)
    # dataset = generate_self_play_data(mcts_ai, num_games=10000)

    # 2) Create network and train
    game = Gomoku()  # Create an instance to access board size
    net = GameNetwork(board_size=game.size, in_channels=2)
    train_network(net, dataset, epochs=5, batch_size=64, lr=1e-3)

    # 3) Later, you can load and use in PUCT
    # net2 = GameNetwork(board_size=9, in_channels=2)
    # net2.load_weights("trained_network.pt")
    # puct_player = PUCTPlayer(network=net2, c_puct=1.0, simulations=400)
    # ...
