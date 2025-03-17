import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        """
        state : a Gomoku game clone at this node
        parent: parent MCTSNode
        move  : the (row, col) that led to this state from the parent
        """
        self.state = state
        self.parent = parent
        self.move = move

        self.children = []
        self.visits = 0
        self.wins = 0    # from this node's perspective

        # Keep track of untried moves
        self.untried_moves = state.legal_moves()

    def expand(self):
        """
        Expand one of the untried moves, create a child node, and return it.
        """
        move = self.untried_moves.pop()
        next_state = self.state.clone()
        next_state.make(move)
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def best_child(self, exploration_constant):
        """
        Use UCB1 to select the best child:
          UCB = (wins / visits) + c * sqrt( ln(parent.visits) / visits )
        """
        best = None
        best_value = float('-inf')

        for child in self.children:
            # small epsilon to avoid zero visits
            child_visits = child.visits + 1e-9
            exploitation = child.wins / child_visits
            exploration = math.sqrt(math.log(self.visits + 1) / child_visits)
            ucb = exploitation + exploration_constant * exploration

            if ucb > best_value:
                best_value = ucb
                best = child

        return best

    def backpropagate(self, result):
        """
        'result' is +1 if this node's player eventually won, 0 if draw, -1 if lost.
        We store from the perspective of the node's current player. 
        The parent is from the opposite perspective, so pass -result upward.
        """
        self.visits += 1
        self.wins += result

        if self.parent is not None:
            self.parent.backpropagate(-result)

    def is_terminal(self):
        return self.state.status != self.state.ONGOING

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
