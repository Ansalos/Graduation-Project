import joblib
import numpy as np
import pandas as pd  # Used to match feature names for sklearn warning suppression

class DecisionTreePlayer:
    def __init__(self, model_path="decision_tree_model.pkl"):
        self.tree = joblib.load(model_path)

    def extract_features(self, game, move, Q=0.0, P=1.0, N=1):
        """
        Extracts feature vector for a given move on the current game state.
        Feature names match the training CSV:
        ['degree', 'Q', 'P', 'N', 'PUCT']
        """
        degree = 0
        for dx, dy in game.moves:
            tx, ty = move[0] + dx, move[1] + dy
            if 0 <= tx < game.n and 0 <= ty < game.n and game.board[tx][ty] == -1:
                degree += 1

        total_visits = max(1, N)
        puct = Q + 1.0 * P * (np.sqrt(total_visits) / (1 + N))

        # Return a DataFrame with column names to match training input
        return pd.DataFrame([[degree, Q, P, N, puct]],
                            columns=["degree", "Q", "P", "N", "PUCT"])

    def choose_move(self, game):
        legal = game.legal_moves()
        if not legal:
            return None

        best = None
        for move in legal:
            features = self.extract_features(game, move)
            prediction = self.tree.predict(features)[0]
            if prediction == 1:
                return move  # found a move classified as "Best"

        # Fallback: choose move with the lowest degree (Warnsdorff-like)
        def move_degree(move):
            degree = 0
            for dx, dy in game.moves:
                tx, ty = move[0] + dx, move[1] + dy
                if 0 <= tx < game.n and 0 <= ty < game.n and game.board[tx][ty] == -1:
                    degree += 1
            return degree

        return min(legal, key=move_degree)

