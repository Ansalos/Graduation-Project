import math

class KnightPUCTNode:
    """
    MCTS Node storing:
      - state: a KnightGame clone
      - prior: prior probability from parent's policy
      - children: dict of move -> KnightPUCTNode
      - N, Q for PUCT stats
    """
    def __init__(self, state, parent=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.prior = prior

        self.children = {}  # move -> KnightPUCTNode
        self.N = 0
        self.Q = 0.0

    def expand(self, move_probs):
        """
        move_probs: dict {move: probability}
        For each move with prob>0, create a child node.
        """
        for move, prob in move_probs.items():
            if prob > 1e-8:
                next_state = self.state_clone_and_make(move)
                child = KnightPUCTNode(next_state, parent=self, prior=prob)
                self.children[move] = child

    def state_clone_and_make(self, move):
        # We'll import KnightGame inside the function to avoid circular references
        from KnightTourGame import KnightGame
        new_state = self.state.clone()  # We'll define clone() in KnightGame
        new_state.make_move(move)
        return new_state

    def select_child(self, c_puct):
        """
        UCB formula => child.Q + c_puct * child.prior * sqrt(self.N)/(1+child.N)
        """
        best_score = float('-inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            U = child.Q + c_puct * child.prior * math.sqrt(self.N) / (1 + child.N)
            if U > best_score:
                best_score = U
                best_move = move
                best_child = child
        return best_move, best_child

    def backpropagate(self, value):
        """
        Single-player puzzle perspective => just add the same 'value' upwards
        (instead of flipping sign).
        """
        self.N += 1
        self.Q = (self.Q * (self.N - 1) + value) / self.N

        if self.parent is not None:
            self.parent.backpropagate(value)

    def is_leaf(self):
        return len(self.children) == 0

    def is_terminal(self):
        # If there are no legal moves left in this game state, it's terminal
        return (not self.state.legal_moves())


