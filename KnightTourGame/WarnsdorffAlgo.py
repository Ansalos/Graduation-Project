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
        for step in range(1, self.game.n * self.game.n - self.game.num_obstacles):
            self.max_path_length = step  
            min_degree = float('inf')
            next_move = None

            for move in self.moves:
                nx, ny = self.x + move[0], self.y + move[1]
                if self.is_valid_move(nx, ny):
                    degree = self.get_degree(nx, ny)
                    if degree < min_degree:
                        min_degree = degree
                        next_move = (nx, ny)

            if not next_move:
                return self.board

            self.x, self.y = next_move
            self.board[self.x][self.y] = step

        return self.board
