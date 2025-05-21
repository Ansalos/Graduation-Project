import torch
import numpy as np
import math
import csv
import os
from PUCTNode import KnightPUCTNode
from WarnsdorffAlgo import WarnsdorffsAlgorithm

class KnightPUCTPlayer:
    def __init__(self, network, c_puct=1.0, simulations=100, show_debug=False, log_to_csv=True):
        self.network = network
        self.c_puct = c_puct
        self.simulations = simulations
        self.show_debug = show_debug
        self.log_to_csv = log_to_csv

        self.csv_path = "puct_decisions.csv"
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

        policy, value = self._evaluate(root.state)
        action_probs = self._policy_to_dict(root.state, policy)
        root.expand(action_probs)
        root.backpropagate(value)

        for _ in range(self.simulations):
            node = root
            while (not node.is_terminal()) and (not node.is_leaf()):
                move, node = node.select_child(self.c_puct)

            if (not node.is_terminal()) and node.is_leaf():
                p, v = self._evaluate(node.state)
                child_probs = self._policy_to_dict(node.state, p)
                node.expand(child_probs)
                node.backpropagate(v)
            else:
                val = self._final_value(node.state)
                node.backpropagate(val)

        if not root.children:
            return None

        best_move = max(root.children.items(), key=lambda x: x[1].N)[0]

        if self.log_to_csv:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                for move, child in root.children.items():
                    degree = self._get_warnsdorff_degree(game, move)
                    score = child.Q + self.c_puct * child.prior * math.sqrt(root.N) / (1 + child.N)
                    is_best = int(move == best_move)
                    writer.writerow([game.step, game.x, game.y, move[0], move[1], degree, round(child.Q, 5), round(child.prior, 5), child.N, round(score, 5), is_best])

        if self.show_debug:
            print(f"\n📝 [Step {game.step}] Knight at {game.x, game.y}")
            print("Evaluated Moves:")
            for move, child in root.children.items():
                degree = self._get_warnsdorff_degree(game, move)
                score = child.Q + self.c_puct * child.prior * math.sqrt(root.N) / (1 + child.N)
                print(f"  Move {move}: Degree={degree}, Q={child.Q:.3f}, P={child.prior:.3f}, N={child.N}, PUCT={score:.3f}")

            print(f"\n🏆 Best Move Selected: {best_move}")

            warn_game = game.clone()
            warn_algo = WarnsdorffsAlgorithm(warn_game)
            warn_algo.solve()
            warn_move = (warn_algo.x, warn_algo.y)
            print(f"⚔️ Warnsdorff's move would be: {warn_move}")

            policy_dict = self._policy_to_dict(game, policy)
            self.show_heatmap(policy_dict, game.n, "🧠 Policy Heatmap", knight_pos=(game.x, game.y), step=game.step, next_move=best_move)

            visit_dict = {move: child.N for move, child in root.children.items()}
            self.show_heatmap(visit_dict, game.n, "🔁 MCTS Visit Heatmap", knight_pos=(game.x, game.y), step=game.step, next_move=best_move)

        return best_move

    def _evaluate(self, state):
        x = self._encode_state(state)
        with torch.no_grad():
            logits, val = self.network(x)
        logits = logits[0]
        value = val[0].item()
        policy = torch.softmax(logits, dim=0).cpu().numpy()

        if self.show_debug:
            print(f"\n[Step {state.step}] Neural Network Evaluation:")
            print("Predicted Value (expected outcome):", round(value, 3))
            for i, move in enumerate(state.moves):
                print(f"  Move {move}: Policy={round(policy[i], 3)}")

        return policy, value

    def _policy_to_dict(self, state, policy):
        moves = state.legal_moves()
        if not moves:
            return {}

        offsets = state.moves
        total_p = 1e-8
        move_dict = {}

        for i, (dx, dy) in enumerate(offsets):
            nx = state.x + dx
            ny = state.y + dy
            if (nx, ny) in moves:
                p = policy[i]
                move_dict[(nx, ny)] = p
                total_p += p

        for m in move_dict:
            move_dict[m] /= total_p
        return move_dict

    def _encode_state(self, state):
        arr = np.zeros((2, state.n, state.n), dtype=np.float32)
        for r in range(state.n):
            for c in range(state.n):
                if state.board[r][c] != -1:
                    arr[0, r, c] = 1.0
        arr[1, state.x, state.y] = 1.0
        x = torch.from_numpy(arr).unsqueeze(0)
        return x

    def _final_value(self, state):
        total_free = state.n * state.n - state.num_obstacles
        if state.step == total_free:
            return 1.0
        elif not state.legal_moves():
            return -1.0
        else:
            return 0.0

    def _get_warnsdorff_degree(self, game, move):
        count = 0
        for dx, dy in game.moves:
            nx, ny = move[0] + dx, move[1] + dy
            if 0 <= nx < game.n and 0 <= ny < game.n and game.board[nx][ny] == -1:
                count += 1
        return count

    def show_heatmap(self, values_dict, n, title, knight_pos=None, step=None, next_move=None):
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox

        heat = np.zeros((n, n))
        for (x, y), v in values_dict.items():
            heat[x][y] = v

        fig, ax = plt.subplots()
        ax.imshow(heat, cmap='hot', interpolation='nearest')

        if knight_pos is not None:
            try:
                img = mpimg.imread("knight.png")
                knight_icon = OffsetImage(img, zoom=0.1)
                ab = AnnotationBbox(knight_icon, (knight_pos[1], knight_pos[0]), frameon=False)
                ax.add_artist(ab)
            except FileNotFoundError:
                ax.scatter(knight_pos[1], knight_pos[0], c='blue', s=120, edgecolors='white', label='Knight Position')

        if next_move is not None:
            ax.scatter(next_move[1], next_move[0], c='green', s=120, edgecolors='black', marker='*', label='Chosen Move')

        title_str = title
        if step is not None:
            title_str += f" (Step {step})"
        ax.set_title(title_str)
        if knight_pos is not None or next_move is not None:
            ax.legend(loc='upper right')
        plt.colorbar(ax.imshow(heat, cmap='hot', interpolation='nearest'))
        plt.show()
