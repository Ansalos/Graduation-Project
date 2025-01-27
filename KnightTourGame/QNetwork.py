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

    def get_state(self, x, y, n):
        """
        Convert a 2D board coordinate (x, y) into a single integer state index.
        For instance, row-major indexing: state = x * n + y.
        """
        return x * n + y

    def choose_action(self, state, valid_action_indices):
        """
        Choose an action (by index) based on an epsilon-greedy policy.

        :param state: current state index (integer)
        :param valid_action_indices: indices of valid actions from the current position
        :return: an index corresponding to the chosen action
        """
        # Exploration
        if random.random() < self.epsilon:
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
        old_value = self.q_table[state, action_index]
        self.q_table[state, action_index] = old_value + self.alpha * (reward + self.gamma * max_next_q - old_value)
