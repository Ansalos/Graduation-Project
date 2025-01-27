import random
import copy

class MCTSPlay:
    def __init__(self, board, moves, iterations=1000):
        self.board = board
        self.moves = moves
        self.iterations = iterations

    def get_valid_moves(self, x, y, board):
        n = len(board)
        valid_moves = []
        for dx, dy in self.moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < n and 0 <= ny < n and board[nx][ny] == -1:
                valid_moves.append((nx, ny))
        return valid_moves

    def simulate(self, x, y, cur_board):
        index = cur_board[x][y] + 1
        path = [] 
        steps = 0
        n = len(self.board)

        while True:
            valid_moves = self.get_valid_moves(x, y, cur_board)
            if not valid_moves or steps >= n * n:
                break
            valid_moves.sort(key=lambda m: len(self.get_valid_moves(m[0], m[1], cur_board)))

            x, y = valid_moves[0]
            path.append((x, y))
            cur_board[x][y] = index
            index += 1

            steps += 1
        return len(path)
        
    def select_best_move(self, root):
        start_x = root.x
        start_y = root.y
        index = self.board[start_x][start_y] + 1
        for _ in range(self.iterations):
            cur_board = copy.deepcopy(self.board)
            node = root
            path = []
            path.append(root)

            # Selection phase - prioritize future options
            while not node.is_leaf() and node.is_fully_expanded(self.get_valid_moves(node.x, node.y, cur_board)):
                node = node.best_child()
                cur_board[node.x][node.y] = index
                index += 1
                path.append(node)

            # Expansion phase
            valid_moves = self.get_valid_moves(node.x, node.y, cur_board)
            unexplored_moves = [move for move in valid_moves if move not in [(child.x, child.y) for child in node.children]]
            if unexplored_moves:
                move = random.choice(unexplored_moves)
                node = node.expand(move)
                cur_board[node.x][node.y] = index
                index += 1
                path.append(node)

            # Simulation phase
            path_length = self.simulate(node.x, node.y, cur_board)

            # Backpropagation phase
            for n in path:
                n.update(path_length)

        best_node = max(root.children, key=lambda c: c.visits)
        return best_node.x, best_node.y
