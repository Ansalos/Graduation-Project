import math

class MCTSNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0

    def is_fully_expanded(self, valid_moves):
        explored_moves = {(child.x, child.y) for child in self.children}
        return all(move in explored_moves for move in valid_moves)

    def best_child(self, exploration_weight=1.41):
            def uct_value(node):
                if node.visits == 0:
                    return float('inf')
                exploitation = node.wins / node.visits
                exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / node.visits)
                return exploitation + exploration

            return max(self.children, key=uct_value)
   
    def expand(self, move):
        child = MCTSNode(move[0], move[1], parent=self)
        self.children.append(child)
        return child

    def update(self, path_length):
        self.visits += 1
        self.wins += path_length


    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0
