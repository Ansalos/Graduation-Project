import random
import math
from mcts_node import MCTSNode
from Gomoku import Gomoku

class MCTSPlayer:
    def __init__(self, iterations=1000, exploration_constant=math.sqrt(2)):
        self.iterations = iterations
        self.exploration_constant = exploration_constant

    def choose_move(self, game):
        """
        Returns a (row, col) move using MCTS from 'game's current state.
        1) Check immediate AI win
        2) Check block threats
        3) Otherwise run MCTS
        """
        # If this player has made fewer than (line_length - 3) moves, take a random center move
        if game.size > 3 :
            if self._player_move_count(game, game.player) < (game.line_length - 3):
                center_move = self._random_center_move(game)
                if center_move is not None:
                    return center_move
        else: 
            if self._player_move_count(game, game.player) == 0 and (game.player == game.BLACK):
                legal = game.legal_moves()
                if legal:
                    return random.choice(legal)
        # If for some reason that fails (e.g. no open squares near center),
        # we'll fall through to the rest of the logic.

        # 1) Check for an immediate winning move
        for move in game.legal_moves():
            temp_state = game.clone()
            temp_state.make(move)
            # If placing this move ends the game with our current player as the winner, do it immediately
            if temp_state.status == game.player:  
                return move
        # 2) if there was line_length-1 on the boared make sure you block it because its a winning move for the oponent view.
        block_immediate = self._find_opponent_immediate_win(game)
        if block_immediate is not None:
            return block_immediate
        # 3) If the opponent has a 3-in-a-row that is open on both sides, block it.
        if game.line_length < game.size:
            block_move = self._find_open_seq_block_move(game, game.line_length - 2)
            if block_move is not None:
                return block_move
        # 4) MCTS search
        root = MCTSNode(game.clone(), parent=None, move=None)
        for _ in range(self.iterations):
            node = root

            # (a) Selection
            while (not node.is_terminal()) and node.is_fully_expanded():
                node = node.best_child(self.exploration_constant)

            # (b) Expansion
            if (not node.is_terminal()) and (not node.is_fully_expanded()):
                node = node.expand()

            # (c) Simulation (random rollout)
            result = self._simulate(node.state.clone())

            # (d) Backpropagation
            # If this node’s state is for black, a result=+1 means black wins
            # If the node’s state is for white, result=+1 means white wins
            # We store from the node’s perspective, so we might invert if needed.
            if node.state.player == Gomoku.BLACK:
                node.backpropagate(result)
            else:
                node.backpropagate(-result)

        # Choose child with highest visits
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.move

    # def run_mcts(self, game_clone):
    #     """
    #     (Optional helper for self-play data generation.)
    #     Runs MCTS from 'game_clone' as the root, returning
    #       (root_node, visit_counts dict).
    #     """
    #     root = MCTSNode(game_clone, parent=None, move=None)
    #     for _ in range(self.iterations):
    #         node = root

    #         while (not node.is_terminal()) and node.is_fully_expanded():
    #             node = node.best_child(self.exploration_constant)
    #         if not node.is_terminal():
    #             node = node.expand()

    #         result = self._simulate(node.state.clone())
    #         if node.state.player == Gomoku.BLACK:
    #             node.backpropagate(result)
    #         else:
    #             node.backpropagate(-result)

    #     # Gather visit counts at the root
    #     visit_counts = {}
    #     for child in root.children:
    #         visit_counts[child.move] = child.visits
    #     return root, visit_counts

    def _simulate(self, state):
        """
        Random rollout until terminal. Return +1 if BLACK wins, -1 if WHITE wins, 0 if draw.
        """
        while state.status == state.ONGOING:
            moves = state.legal_moves()
            m = random.choice(moves)
            state.make(m)

        if state.status == Gomoku.BLACK:
            return 1
        elif state.status == Gomoku.WHITE:
            return -1
        return 0

    def _count_sequence(self, game, r, c, dr, dc, player):
        board = game.board
        size = game.size
        # move backward
        br, bc = r, c
        while True:
            nr, nc = br - dr, bc - dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                br, bc = nr, nc
            else:
                break
        # move forward
        fr, fc = r, c
        while True:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                fr, fc = nr, nc
            else:
                break
        length = max(abs(fr - br), abs(fc - bc)) + 1
        return (length, br, bc, fr, fc)

    def _block_ends(self, game, br, bc, fr, fc, dr, dc, legal):
        """
        If there's an open cell at either end that might block a longer chain, use it.
        """
        candidates = [
            (br - dr, bc - dc),
            (fr + dr, fc + dc)
        ]
        for (rr, cc) in candidates:
            if 0 <= rr < game.size and 0 <= cc < game.size:
                if (rr, cc) in legal:
                    return (rr, cc)
        return None
    
    def _find_open_seq_block_move(self, game, threat_length):
        """
        Finds a move that blocks the opponent's open 'threat_length'-in-a-row.
        Returns (row, col) if such a blocking move is found, else None.

        An 'open threat_length-in-a-row' means:
        - Exactly 'threat_length' consecutive stones of the opponent
        - Both ends are within bounds and EMPTY
        """
        board = game.board
        size = game.size
        opponent = game.other(game.player)

        # Directions: horizontal, vertical, diagonal-down, diagonal-up
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for r in range(size):
            for c in range(size):
                if board[r][c] == opponent:
                    # For each of the 4 main directions
                    for dr, dc in directions:
                        length, (start_r, start_c), (end_r, end_c) = \
                            self._count_sequence(game, r, c, dr, dc, opponent)
                        # If exactly 'threat_length' consecutive opponent stones found
                        if length == threat_length:
                            # Look at the cells immediately before and after this sequence
                            before_r = start_r - dr
                            before_c = start_c - dc
                            after_r = end_r + dr
                            after_c = end_c + dc

                            # Check if BOTH ends are open
                            if self._on_board(game, before_r, before_c) \
                            and board[before_r][before_c] == game.EMPTY \
                            and self._on_board(game, after_r, after_c) \
                            and board[after_r][after_c] == game.EMPTY:
                                # We can block at either end.
                                # Return the first one we find (or you can choose which end).
                                return (before_r, before_c)
        return None


    def _count_sequence(self, game, row, col, dr, dc, player):
        """
        From (row, col), move backward in direction (-dr, -dc) and forward in (dr, dc)
        to find how many consecutive 'player' stones exist in a line.
        
        Returns:
        (length_of_sequence, (start_r, start_c), (end_r, end_c))
        """
        board = game.board
        size = game.size

        # Move backward
        br, bc = row, col
        while True:
            nr, nc = br - dr, bc - dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                br, bc = nr, nc
            else:
                break

        # Move forward
        fr, fc = row, col
        while True:
            nr, nc = fr + dr, fc + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == player:
                fr, fc = nr, nc
            else:
                break

        length = max(abs(fr - br), abs(fc - bc)) + 1
        return (length, (br, bc), (fr, fc))

    def _on_board(self, game, r, c):
        """
        Checks if (r, c) is a valid board coordinate.
        """
        return 0 <= r < game.size and 0 <= c < game.size
    
    def _find_opponent_immediate_win(self, game):
        """
        Checks if the opponent can win immediately on their next move.
        Returns (row, col) if we should block that spot, else None.
        """
        opponent = game.other(game.player)
        for move in game.legal_moves():
            temp_state = game.clone()
            
            # Force it to be opponent's turn
            temp_state.player = opponent
            temp_state.make(move)
            
            # If that results in the opponent winning,
            # then we have to place a stone there to block them.
            if temp_state.status == opponent:
                return move
        return None
    
    def _player_move_count(self, game, player):
        """
        Counts how many stones of 'player' are currently on the board.
        """
        count = 0
        for r in range(game.size):
            for c in range(game.size):
                if game.board[r][c] == player:
                    count += 1
        return count

    def _random_center_move(self, game):
        """
        Chooses a random legal move that is NOT on the board's boundary.
        If no such moves exist, it picks any random legal move.
        """
        legal = game.legal_moves()
        if not legal:
            return None  # No moves available

        non_boundary_moves = []
        for (r, c) in legal:
            # Skip boundary cells: row=0, row=size-1, col=0, col=size-1
            if r > 0 and r < game.size - 1 and c > 0 and c < game.size - 1:
                non_boundary_moves.append((r, c))

        if non_boundary_moves:
            return random.choice(non_boundary_moves)
        else:
            # If no non-boundary moves remain, fall back to any random legal move
            return random.choice(legal)