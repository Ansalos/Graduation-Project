class WarnsdorffsAlgorithm:
    def __init__(self, game):
        self.game = game
        self.is_valid_move = game.is_valid_move
        self.moves = game.moves
        self.board = game.board
        self.x, self.y = game.x, game.y
        self.max_path_length = 0

    def get_degree(self, x, y):
        return sum(
            self.is_valid_move(x + dx, y + dy) for dx, dy in self.moves
        )

    def solve(self):
        total_free = self.game.n * self.game.n - self.game.num_obstacles

        for step in range(1, total_free):          # same loop range as before
            self.max_path_length = step
            min_degree = float('inf')
            next_move = None

            for dx, dy in self.moves:
                nx, ny = self.x + dx, self.y + dy

                # ────────────────────────────────
                # Accept the move if either:
                #  (a) it passes the usual validity check, or
                #  (b) it is the *very last* square still unvisited
                #      (degree-0 squares allowed only on the final step)
                # ────────────────────────────────
                usual_ok   = self.is_valid_move(nx, ny)
                last_step  = (step == total_free - 1)
                empty_here = (
                    0 <= nx < self.game.n and
                    0 <= ny < self.game.n and
                    self.board[nx][ny] == -1
                )

                if not (usual_ok or (last_step and empty_here)):
                    continue

                degree = self.get_degree(nx, ny)

                # exclude early dead-ends (degree 0) unless it’s the last step
                if degree == 0 and not last_step:
                    continue

                if degree < min_degree:
                    min_degree = degree
                    next_move  = (nx, ny)

            if not next_move:
                return self.board          # stuck early → return board as before

            # make the move
            self.x, self.y = next_move
            self.board[self.x][self.y] = step

        return self.board
