# SelfPlayAndTraining.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import random

from MCTSPlay import MCTSPlay
from MCTSNode import MCTSNode
from KnightTourGame import KnightGame
from NeuralNetworkClass import KnightNetwork


##################################################
# 1) Encode game state => (2,n,n) Tensor
##################################################
def encode_state_for_training(game):
    """
    Convert the KnightGame board to a (2, game.n, game.n) tensor:
      plane[0]: 1.0 for visited squares
      plane[1]: 1.0 at the knight’s current position
    """
    arr = np.zeros((2, game.n, game.n), dtype=np.float32)
    for r in range(game.n):
        for c in range(game.n):
            if game.board[r][c] != -1:  # visited squares
                arr[0, r, c] = 1.0
    arr[1, game.x, game.y] = 1.0
    return torch.from_numpy(arr)


##################################################
# 2) find_move_index
##################################################
def find_move_index(game, move):
    """
    Return which index i in 'game.moves' corresponds to 'move = (nx, ny)'
    from the knight's current position.
    """
    dx = move[0] - game.x
    dy = move[1] - game.y
    for i, (mx, my) in enumerate(game.moves):
        if (mx, my) == (dx, dy):
            return i
    return None


##################################################
# 3) KnightGame creation & cloning
##################################################
def create_knight_game(board_size, num_obstacles, random_start, fixed_start):
    """
    Returns a new KnightGame with either a random or fixed start position.
    Obstacles are placed randomly if num_obstacles > 0.
    """
    if random_start:
        rx = random.randint(0, board_size - 1)
        ry = random.randint(0, board_size - 1)
        return KnightGame(board_size, num_obstacles, rx, ry)
    else:
        sx, sy = fixed_start
        return KnightGame(board_size, num_obstacles, sx, sy)

def clone_knight_game(original_game):
    """
    Makes a new KnightGame with the same obstacle layout and step count
    as 'original_game'. This ensures consistent obstacles if the user
    wants the same arrangement each game.
    """
    new_g = KnightGame(
        original_game.n,
        original_game.num_obstacles,
        original_game.x,
        original_game.y
    )
    new_g.board = copy.deepcopy(original_game.board)
    new_g.step = original_game.step
    return new_g


##################################################
# 4) The main MCTS self-play + training function
##################################################
def run_mcts_selfplay_training(
    board_size,
    num_obstacles,
    random_start,       # bool
    fixed_start,        # (sx, sy)
    random_each_game,   # bool
    num_games,
    mcts_iterations,
    epochs,
    lr,
    print_each_move=False
):
    """
    Orchestrates self-play for 'num_games' using your MCTS snippet:
      - Create a KnightGame
      - On each move, reconstruct MCTS with the latest board state
      - Possibly print the board each move
      - Collect (state, 1-hot policy) => final outcome
      - Finally, train a KnightNetwork
    """

    dataset = []
    base_game = None

    for g in range(num_games):
        # (A) Create or clone a KnightGame
        if random_each_game:
            game = create_knight_game(board_size, num_obstacles, random_start, fixed_start)
        else:
            # same obstacles each game
            if g == 0:
                base_game = create_knight_game(board_size, num_obstacles, random_start, fixed_start)
            game = clone_knight_game(base_game)

        print(f"\n=== MCTS Play starting for Game {g+1}/{num_games}... ===")
        max_visits = 0
        move_records = []

        # (B) Repeatedly pick moves via MCTS,
        #     but re-build MCTS each iteration to see updated board:
        while True:
            # Re-create the board copy for MCTS
            board_copy = [row[:] for row in game.board]
            mcts = MCTSPlay(board_copy, game.moves, iterations=mcts_iterations)

            valid_moves = mcts.get_valid_moves(game.x, game.y, game.board)
            if not valid_moves:
                print("No more valid moves! Machine stopped.")
                break

            root = MCTSNode(game.x, game.y)
            nx, ny = mcts.select_best_move(root)

            # Build a training record
            state_tensor = encode_state_for_training(game)
            policy_vec = torch.zeros(8, dtype=torch.float32)
            idx = find_move_index(game, (nx, ny))
            if idx is not None:
                policy_vec[idx] = 1.0

            move_records.append((state_tensor, policy_vec))

            # Apply the move on the real KnightGame
            game.x, game.y = nx, ny
            game.board[game.x][game.y] = game.step
            game.step += 1

            if print_each_move:
                game.print_board()
                print()

            max_visits = max(max_visits, game.step)

        print(f"Max Visits Achieved: {max_visits}")
        remain = (board_size * board_size) - num_obstacles - max_visits
        print(f"Remaining blocks not Achieved: {remain}")

        # final outcome => +1 if all squares visited, else -1
        total_free = board_size * board_size - num_obstacles
        if game.step == total_free:
            final_val = 1.0
            print(f"Game {g+1} => COMPLETED the board!")
        else:
            final_val = -1.0
            print(f"Game {g+1} => stuck, visited {game.step}/{total_free} squares.")

        # Attach final outcome to all moves in this game
        for (st, pol) in move_records:
            dataset.append((st, pol, final_val))

    # (C) Train the network on the accumulated dataset
    net = KnightNetwork(board_size=board_size, in_channels=2)
    train_knight_network(net, dataset, epochs=epochs, batch_size=32, lr=lr)
    print("run_mcts_selfplay_training finished!")


##################################################
# 5) train_knight_network
##################################################
def train_knight_network(network, dataset, epochs=5, batch_size=32, lr=1e-3):
    from torch.utils.data import TensorDataset, DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    states, policies, values = [], [], []
    for (s, p, v) in dataset:
        states.append(s)
        policies.append(p)
        values.append(v)

    states = torch.stack(states)             # (N,2,n,n)
    policies = torch.stack(policies)         # (N,8)
    values = torch.tensor(values).view(-1,1) # (N,1)

    ds = TensorDataset(states, policies, values)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(network.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs+1):
        network.train()
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_samples     = 0

        for b_states, b_policies, b_values in dl:
            b_states   = b_states.to(device)
            b_policies = b_policies.to(device)
            b_values   = b_values.to(device)

            optimizer.zero_grad()

            logits, value_pred = network(b_states)
            # Policy => cross-entropy vs one-hot policy target
            log_probs = torch.log_softmax(logits, dim=1)
            policy_loss = -torch.sum(b_policies * log_probs, dim=1).mean()

            # Value => MSE
            value_loss = mse_loss(value_pred, b_values)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            bs = b_states.size(0)
            total_policy_loss += policy_loss.item() * bs
            total_value_loss  += value_loss.item()  * bs
            total_samples     += bs

        avg_p_loss = total_policy_loss / total_samples
        avg_v_loss = total_value_loss  / total_samples
        print(f"Epoch {epoch}/{epochs} | Policy Loss={avg_p_loss:.4f} | Value Loss={avg_v_loss:.4f}")

    network.save_weights("knight_network.pt")
    print("Network training done!")
