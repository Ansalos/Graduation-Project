# QNetwork.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        A simple Q-Learning agent using a Q-table.

        :param state_size: the total possible states (e.g. n*n for an n x n board)
        :param action_size: total possible actions (8 knight moves in standard chess, 
                            but you can keep it flexible if you have obstacles or want a different approach)
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: exploration probability (for epsilon-greedy)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((state_size, action_size))

    def get_state(self, game):
        # We want to encode the number of moves from each target square.
        # To have a manageable number of state, we have to dicretize aggressively.
        # Blocked or 0 steps --> index 0
        # 1 available move --> index 1
        # 2 or more available moves --> index 2.
        board = game.board
        n = game.n
        x = game.x
        y = game.y
        moves = game.moves
        # First, we encode the state as an array of 8 
        encode_arr = [0 for _ in range(8)]
        for i in range(8):
            move = moves[i]
            cur_x = x+move[0]
            cur_y = y+move[1]
            if 0<=cur_x<n and 0<=cur_y<n and board[cur_x][cur_y] == -1:
                for next_move in moves:
                    tmp_x = cur_x+next_move[0]
                    tmp_y = cur_y+next_move[1]
                    if 0<=tmp_x<n and 0<=tmp_y<n and board[tmp_x][tmp_y] == -1:
                        encode_arr[i] += 1
                        if encode_arr[i] == 2:
                            break
        ind = 0
        for i in reversed(range(8)):
            ind = ind*3 + encode_arr[i]

        return ind, encode_arr


        

    def choose_action(self, state, valid_action_indices, epsilon):
        """
        Choose an action (by index) based on an epsilon-greedy policy.

        :param state: current state index (integer)
        :param valid_action_indices: indices of valid actions from the current position
        :return: an index corresponding to the chosen action
        """
        # Exploration
        if random.random() < epsilon:
            return random.choice(valid_action_indices)
        # Exploitation
        else:
            # We pick the valid action with the highest Q-value
            q_values = [self.q_table[state, a] for a in valid_action_indices]
            best_action_index = valid_action_indices[np.argmax(q_values)]
            return best_action_index

    def update(self, state, action_index, reward, next_state, next_valid_action_indices):
        """
        Update the Q-table using the Q-Learning update rule.

        :param state: current state index
        :param action_index: which action we took (0 to action_size-1)
        :param reward: reward received after taking this action
        :param next_state: next state index (integer)
        :param next_valid_action_indices: valid actions in the next state
        """
        # Find the best possible action in the next state (greedy w.r.t. Q-table)
        if len(next_valid_action_indices) == 0:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state, a] for a in next_valid_action_indices)

        # Standard Q-Learning formula
        self.q_table[state, action_index] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[state, action_index])
