import math
from PUCTNode import KnightPUCTNode

class RuleBasedPUCTPlayer:
    def __init__(self, c_puct=1.0, simulations=100, show_debug=False):
        self.c_puct = c_puct
        self.simulations = simulations
        self.show_debug = show_debug

    def choose_move(self, game):
        if not game.legal_moves():
            return None

        root = KnightPUCTNode(game.clone(), parent=None, prior=1.0)

        policy = self._rule_based_policy(root.state)
        root.expand(policy)
        root.backpropagate(0.0)  # You can keep Q=0.0

        for _ in range(self.simulations):
            node = root
            while not node.is_terminal() and not node.is_leaf():
                move, node = node.select_child(self.c_puct)

            if not node.is_terminal() and node.is_leaf():
                p = self._rule_based_policy(node.state)
                node.expand(p)
                node.backpropagate(0.0)
            else:
                node.backpropagate(self._final_value(node.state))

        if not root.children:
            return None

        if self.show_debug:
            print("\n[PUCT Logic-only] Children Stats:")
            for move, child in root.children.items():
                print(f"Move {move} → N={child.N}, Q={round(child.Q, 3)}, Prior={round(child.prior, 3)}")

        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]
        return best_move

    def _rule_based_policy(self, state):
        moves = state.legal_moves()
        degrees = []
        for dx, dy in state.moves:
            nx, ny = state.x + dx, state.y + dy
            if (nx, ny) in moves:
                onward = 0
                for ddx, ddy in state.moves:
                    tx, ty = nx + ddx, ny + ddy
                    if state.is_valid_move(tx, ty):
                        onward += 1
                score = 1 / (onward + 1e-5)  # Lower degree → higher score
                degrees.append(((nx, ny), score))

        total = sum(score for _, score in degrees) + 1e-8
        return {move: score / total for move, score in degrees}

    def _final_value(self, state):
        total_free = state.n * state.n - state.num_obstacles
        if state.step == total_free:
            return 1.0
        elif not state.legal_moves():
            return -1.0
        else:
            return 0.0
