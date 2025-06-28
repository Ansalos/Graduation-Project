import math
import os
import csv
from PUCTNode import KnightPUCTNode

class RuleBasedPUCTPlayer:
    def __init__(self, c_puct=1.0, simulations=100, show_debug=False, log_to_csv=False):
        self.c_puct = c_puct
        self.simulations = simulations
        self.show_debug = show_debug
        self.log_to_csv = log_to_csv
        self.csv_path = "puct_logic_only.csv"

        if self.log_to_csv:
            self._init_csv()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "current_x", "current_y", "move_x", "move_y", "degree", "Q", "P", "N", "PUCT", "is_best"])

    def choose_move(self, game):
        if not game.legal_moves():
            return None

        root = KnightPUCTNode(game.clone(), parent=None, prior=1.0)
        policy = self._rule_based_policy(root.state)
        root.expand(policy)
        root.backpropagate(0.0)

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

        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]

        if self.log_to_csv:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for move, child in root.children.items():
                    degree = self._get_degree(game, move)
                    puct = child.Q + self.c_puct * child.prior * math.sqrt(root.N) / (1 + child.N)
                    is_best = int(move == best_move)
                    writer.writerow([game.step, game.x, game.y, move[0], move[1], degree, round(child.Q, 5), round(child.prior, 5), child.N, round(puct, 5), is_best])

        if self.show_debug:
            print("\n[PUCT Logic-only] Children Stats:")
            for move, child in root.children.items():
                print(f"Move {move} → N={child.N}, Q={round(child.Q, 3)}, Prior={round(child.prior, 3)}")

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

    def _get_degree(self, game, move):
        count = 0
        for dx, dy in game.moves:
            nx, ny = move[0] + dx, move[1] + dy
            if 0 <= nx < game.n and 0 <= ny < game.n and game.board[nx][ny] == -1:
                count += 1
        return count

    def _final_value(self, state):
        total_free = state.n * state.n - state.num_obstacles
        if state.step == total_free:
            return 1.0
        elif not state.legal_moves():
            return -1.0
        else:
            return 0.0
