class Gomoku:
    # Player/constants
    BLACK = 1
    WHITE = -1
    EMPTY = 0

    # Game status
    BLACK_WIN = 1
    WHITE_WIN = -1
    DRAW = 0
    ONGOING = -17  # Arbitrary ongoing status

    # Fixed board size
    FIXED_SIZE = 9
    line_length = 5

    def __init__(self):
        """
        Initializes a 15×15 Gomoku board (fixed size).
        Black goes first by convention.
        """
        self.size = self.FIXED_SIZE
        self.board = [[self.EMPTY for _ in range(self.size)]
                      for _ in range(self.size)]
        self.player = self.BLACK
        self.status = self.ONGOING

    def legal_moves(self):
        """Returns a list of all (row, col) positions that are still EMPTY."""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == self.EMPTY:
                    moves.append((r, c))
        return moves

    def make(self, move):
        """
        Places a stone for the current player at 'move' = (row, col).
        Raises ValueError if move is out of bounds or cell is occupied.
        """
        (row, col) = move
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Illegal move: ({row}, {col}) is out of bounds.")
        if self.board[row][col] != self.EMPTY:
            raise ValueError(f"Illegal move: Cell ({row}, {col}) is occupied.")

        self.board[row][col] = self.player

        # Check for win or draw
        if self.winning_move(move):
            self.status = self.player
        elif len(self.legal_moves()) == 0:
            self.status = self.DRAW
        else:
            self.player = self.other(self.player)

    def other(self, player):
        return self.BLACK if player == self.WHITE else self.WHITE

    def unmake(self, move):
        """
        Removes the stone from (row, col) and resets status to ONGOING.
        This can be used by a search algorithm to backtrack.
        """
        (row, col) = move
        if self.board[row][col] == self.EMPTY:
            raise ValueError(f"Cannot unmake an empty cell ({row}, {col}).")
        self.board[row][col] = self.EMPTY
        self.player = self.other(self.player)
        self.status = self.ONGOING

    def clone(self):
        """Returns a deep copy of the current game state."""
        new_game = Gomoku()
        new_game.board = [row[:] for row in self.board]
        new_game.player = self.player
        new_game.status = self.status
        return new_game

    def winning_move(self, move):
        """
        Checks if placing a stone at 'move' = (row, col) results
        in 5 in a row for the current player.
        """
        (row, col) = move
        player = self.board[row][col]
        if player == self.EMPTY:
            return False

        directions = [
            (1, 0),   # horizontal
            (0, 1),   # vertical
            (1, 1),   # diagonal down-right
            (1, -1)   # diagonal up-right
        ]

        for dx, dy in directions:
            count = 0

            # Forward direction
            x, y = row + dx, col + dy
            while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                count += 1
                x += dx
                y += dy

            # Backward direction
            x, y = row - dx, col - dy
            while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                count += 1
                x -= dx
                y -= dy

            if count >= self.line_length - 1:  # total 5 in a row (including current stone)
                return True

        return False

    def __str__(self):
        """
        Returns a string representation of the 15×15 board.
        'B' = BLACK, 'W' = WHITE, '.' = EMPTY
        """
        header_labels = [self.index_to_label(col) for col in range(self.size)]
        header = "   " + " ".join(header_labels)

        rows_str = []
        for r in range(self.size):
            row_label = f"{r+1:2d}"
            row_symbols = []
            for c in range(self.size):
                if self.board[r][c] == self.BLACK:
                    row_symbols.append("B")
                elif self.board[r][c] == self.WHITE:
                    row_symbols.append("W")
                else:
                    row_symbols.append(".")
            row_line = row_label + " " + " ".join(row_symbols)
            rows_str.append(row_line)

        return header + "\n" + "\n".join(rows_str)

    @staticmethod
    def index_to_label(index):
        """
        Converts 0->A, 1->B, ... 14->O
        """
        result = ""
        i = index
        while True:
            remainder = i % 26
            letter = chr(ord('A') + remainder)
            result = letter + result
            i = i // 26
            if i == 0:
                break
            i -= 1
        return result

    @staticmethod
    def label_to_index(label):
        """
        Converts 'A'..'O' to 0..14
        """
        label = label.upper()
        result = 0
        for c in label:
            result = result * 26 + (ord(c) - ord('A') + 1)
        return result - 1

    @staticmethod
    def parse_position(pos_str):
        """
        Parse strings like 'A1', 'C10' -> (row, col)
        """
        pos_str = pos_str.strip().upper()
        alpha_part = ""
        numeric_part = ""

        for char in pos_str:
            if char.isalpha():
                if numeric_part:
                    raise ValueError("Letters after digits not allowed.")
                alpha_part += char
            elif char.isdigit():
                numeric_part += char
            else:
                raise ValueError("Invalid character in input.")

        if not alpha_part or not numeric_part:
            raise ValueError("Missing letter or digit (e.g. A1).")

        col = Gomoku.label_to_index(alpha_part)
        row = int(numeric_part) - 1

        if not (0 <= col < Gomoku.FIXED_SIZE):
            raise ValueError(f"Column '{alpha_part}' out of range.")
        if not (0 <= row < Gomoku.FIXED_SIZE):
            raise ValueError(f"Row '{row+1}' out of range.")

        return (row, col)
