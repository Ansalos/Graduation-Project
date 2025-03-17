import math
import torch
import numpy as np
from PUCTNode import PUCTNode
from Gomoku import Gomoku

class PUCTPlayer:
    def __init__(self, network, c_puct=1.0, simulations=400):
        """
        network: an instance of GameNetwork
        c_puct : the exploration constant
        simulations: how many PUCT iterations to run per move
        """
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations

    def choose_move(self, game):
        """
        Use PUCT to choose a move from the current 'game' state.
        """
        if game.status != game.ONGOING:
            return None  # no move if terminal

        # Create root node
        root = PUCTNode(game.clone(), parent=None, prior=1.0)

        # 1) Evaluate root with the network to get policy & value
        policy, value = self._evaluate(root.state)
        # Convert policy array -> dict {move: prob} for legal moves
        action_probs = self._policy_to_dict(root.state, policy)
        # Expand root
        root.expand(action_probs)
        # Backprop the root value
        # root.backpropagate(value)
        if root.state.player == Gomoku.BLACK:
            root.backpropagate(value)
        else:
            root.backpropagate(-value)

        # 2) Run multiple simulations
        for _ in range(self.simulations):
            node = root
            # (a) Selection
            while (not node.is_leaf()) and (not node.is_terminal()):
                action, node = node.select_child(self.c_puct)

            # (b) Expansion / Evaluation
            if (not node.is_terminal()) and node.is_leaf():
                policy_leaf, value_leaf = self._evaluate(node.state)
                child_probs = self._policy_to_dict(node.state, policy_leaf)
                node.expand(child_probs)
                # (c) Backprop
                # node.backpropagate(value_leaf)
                if node.state.player == Gomoku.BLACK:
                    node.backpropagate(value_leaf)
                else:
                    node.backpropagate(-value_leaf)
            else:
                # Terminal node: backprop final outcome
                val = self._game_result(node.state)
                # node.backpropagate(val)
                if node.state.player == Gomoku.BLACK:
                    node.backpropagate(val)
                else:
                    node.backpropagate(-val)

        # 3) Pick the child of root with the highest visit count
        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]
        return best_move

    def _evaluate(self, state):
        """
        Run the network to get (policy, value).
        policy: softmax distribution over all board cells
        value: single float in [-1, +1]
        """
        x = self._encode_state(state)  # shape (1, in_channels, size, size)
        with torch.no_grad():
            logits, val = self.network(x)
        logits = logits[0]  # shape (size*size,)
        value = val[0].item()

        # softmax for policy
        policy = torch.softmax(logits, dim=0).cpu().numpy()
        return policy, value

    def _policy_to_dict(self, state, policy):
        """
        Convert 'policy' vector into a dict {move: prob} only for legal moves.
        """
        size = state.size
        legal = state.legal_moves()
        move_dict = {}
        total_p = 1e-8
        for r in range(size):
            for c in range(size):
                idx = r * size + c
                if (r, c) in legal:
                    p = policy[idx]
                    move_dict[(r, c)] = p
                    total_p += p

        # Normalize
        for m in move_dict:
            move_dict[m] /= total_p
        return move_dict

    def _encode_state(self, state):
        """
        Example: 2-plane encoding (black=1 in plane0, white=1 in plane1).
        """
        import torch
        size = state.size
        arr = np.zeros((2, size, size), dtype=np.float32)
        for r in range(size):
            for c in range(size):
                if state.board[r][c] == Gomoku.BLACK:
                    arr[0, r, c] = 1.0
                elif state.board[r][c] == Gomoku.WHITE:
                    arr[1, r, c] = 1.0

        x = torch.from_numpy(arr).unsqueeze(0)  # (1,2,size,size)
        return x

    def _game_result(self, state):
        """
        Convert final state.status to numeric value from the perspective
        of the node's current player (if we do it that way).
        Here, we'll just do black=+1, white=-1, draw=0 for whichever node sees it.
        """
        if state.status == Gomoku.BLACK:
            return 1.0
        elif state.status == Gomoku.WHITE:
            return -1.0
        else:
            return 0.0
