import math

class PUCTNode:
    def __init__(self, state, parent=None, prior=0.0):
        """
        A node for the PUCT algorithm (AlphaZero MCTS).
        state : current Gomoku state
        parent: parent node
        prior : prior probability P(s,a) from the parent's policy
        """
        self.state = state
        self.parent = parent
        self.prior = prior
        
        # Children: dict of action -> PUCTNode
        self.children = {}
        
        # PUCT statistics
        self.N = 0      # visit count
        self.Q = 0.0    # average value

    def expand(self, action_probs):
        """
        action_probs: dict {action: probability} from parent's policy
        Creates child nodes for each action with the given prior.
        """
        for action, prob in action_probs.items():
            if prob <= 1e-8:
                continue
            next_state = self.state.clone()
            next_state.make(action)
            child = PUCTNode(next_state, parent=self, prior=prob)
            self.children[action] = child

    def select_child(self, c_puct):
        """
        Select the action that maximizes:
          U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        Returns (action, child_node).
        """
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            U = child.Q + c_puct * child.prior * math.sqrt(self.N) / (1 + child.N)
            if U > best_score:
                best_score = U
                best_action = action
                best_child = child
        return best_action, best_child

    def backpropagate(self, value):
        """
        Update this node with 'value' in [-1, +1], from its perspective.
        Then propagate to parent with -value (since parent's perspective is opposite).
        """
        self.N += 1
        self.Q = (self.Q * (self.N - 1) + value) / self.N
        
        if self.parent is not None:
            self.parent.backpropagate(-value)

    def is_terminal(self):
        return (self.state.status != self.state.ONGOING)

    def is_leaf(self):
        return (len(self.children) == 0)
