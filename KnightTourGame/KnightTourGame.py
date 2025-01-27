import random

class KnightGame:
    def __init__(self, n, num_obstacles=0, start_x=0, start_y=0):
        self.n = n
        self.num_obstacles = num_obstacles
        self.start_x = start_x
        self.start_y = start_y
        self.board = [[-1 for _ in range(n)] for _ in range(n)]
        self.moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1),
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        self.x, self.y = start_x, start_y
        self.board[self.x][self.y] = 0
        self.step = 1
        self.place_obstacles()
    
    def is_valid_move(self, x, y):
        if 0 <= x < self.n and 0 <= y < self.n and self.board[x][y] == -1:
            remaining_moves = [
                (x + dx, y + dy) for dx, dy in self.moves
                if 0 <= x + dx < self.n and 0 <= y + dy < self.n and self.board[x + dx][y + dy] == -1
            ]
            return len(remaining_moves) > 0 
        return False

    def place_obstacles(self):
        count = 0
        while count < self.num_obstacles:
            x, y = random.randint(0, self.n - 1), random.randint(0, self.n - 1)
            if self.board[x][y] == -1:  # Ensure the spot is not already an obstacle
                self.board[x][y] = 'X'
                count += 1
    # def place_obstacles(self):
    #     fixed_obstacles = []
    #     # [(0, 1), (1, 3), (2, 2), (3, 0), (4, 4), (5, 1), (6, 3), (7, 2), (8, 0)]
    #     for x, y in fixed_obstacles:
    #         self.board[x][y] = 'X'

    def print_board(self):
        for row in self.board:
            print(" ".join(f"{col:2}" if isinstance(col, int) and
                           col >= 0 else (" X" if col == 'X' else " *")
                           for col in row))

    def play(self):
        if self.board is None:
            return

        while True:
            self.print_board()
            print("Current position:", (self.x, self.y))
            print("Enter your next move (row and column):")
            try:
                nx, ny = map(int, input().split())
                if (nx, ny) in [(self.x + dx, self.y + dy) for dx, dy in self.moves] and self.is_valid_move(nx, ny):
                    self.x, self.y = nx, ny
                    self.board[self.x][self.y] = self.step
                    self.step += 1
                else:
                    print("Invalid move! Try again.")
            except ValueError:
                print("Invalid input! Enter two integers.")