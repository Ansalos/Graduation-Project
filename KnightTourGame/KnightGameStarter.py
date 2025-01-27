from KnightTourGame import KnightGame
from MCTSNode import MCTSNode
from MCTSPlay import MCTSPlay
from WarnsdorffAlgo import WarnsdorffsAlgorithm
import copy



def main():
    print("Welcome to Knight's Tour Game!")
    print("Select Game Mode:")
    print("1. Play the Knight's Tour")
    print("2. MCTS Play (Machine-Controlled Tour)")
    print("3. Warnsdorff's Algorithm (Optimized Tour)")
    print("4. MCTS vs Warnsdorff's Algorithm on Same Board")
    print("5. MCTS with Q-Learning Integration (AI-Enhanced Tour)")
    print("6. Brute Force (Backtracking) Mode")

    mode = input("Enter the mode number: ")

    if mode == '1':
        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        start_x = int(input("Enter starting X position: "))
        start_y = int(input("Enter starting Y position: "))

        game = KnightGame(n, num_obstacles, start_x, start_y)

        move_history = [(start_x, start_y)]
        max_visits = 0  

        while True:
            moves = []
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            temp_board = [row[:] for row in game.board]

            valid_move_count = 0  

            for dx, dy in game.moves:
                nx, ny = game.x + dx, game.y + dy
                if game.is_valid_move(nx, ny):
                    moves.append((nx, ny))
                    temp_board[nx][ny] = labels[valid_move_count]
                    valid_move_count += 1  

            for row in temp_board:
                print(" ".join(f"{col:2}" if isinstance(col, int) and col >= 0 else (" X" if col == 'X' else (f" {col}" if isinstance(col, str) else " *")) for col in row))
            print() 

            max_visits = max(max_visits, game.step)  

            if not moves:
                print("No more valid moves! Game over.")
                break

            print("Legal moves:")
            for i, move in enumerate(moves):
                print(f"{labels[i]}. {move}")

            try:
                choice = input("Choose your move (alphabet or 'U' to undo): ").upper()
                if choice == 'U':
                    if len(move_history) > 1: 
                        prev_x, prev_y = move_history.pop()
                        game.board[prev_x][prev_y] = -1  
                        game.x, game.y = move_history[-1]  
                        game.step -= 1
                    else:
                        print("Cannot undo further!")
                elif choice in labels[:len(moves)]:
                    nx, ny = moves[labels.index(choice)]
                    game.x, game.y = nx, ny
                    game.board[game.x][game.y] = game.step
                    game.step += 1
                    move_history.append((game.x, game.y))
                else:
                    print("Invalid choice! Try again.")
            except ValueError:
                print("Invalid input! Enter an alphabet or 'U' for undo.")

        print(f"Max Visits Achieved: {max_visits}")
        print(f"Remaining blocks not Achieved: {(n*n) - num_obstacles - max_visits}")


    elif mode == '2':
        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        start_x = int(input("Enter starting X position: "))
        start_y = int(input("Enter starting Y position: "))

        game = KnightGame(n, num_obstacles, start_x, start_y)
        mcts = MCTSPlay(game.board, game.moves, iterations=1000)

        print("MCTS Play starting...")
        max_visits = 0  

        while True:
            valid_moves = mcts.get_valid_moves(game.x, game.y, game.board)
            if not valid_moves:
                print("No more valid moves! Machine stopped.")
                break

            root = MCTSNode(game.x, game.y)
            nx, ny = mcts.select_best_move(root)

            game.x, game.y = nx, ny
            game.board[game.x][game.y] = game.step
            game.step += 1

            game.print_board()
            print()  

            max_visits = max(max_visits, game.step)  

        print(f"Max Visits Achieved: {max_visits}")
        print(f"Remaining blocks not Achieved: {(n*n) - num_obstacles - max_visits}")


    elif mode == '3':
        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        start_x = int(input("Enter starting X position: "))
        start_y = int(input("Enter starting Y position: "))
        game = KnightGame(n, num_obstacles, start_x, start_y)
        algo = WarnsdorffsAlgorithm(game)
        result = algo.solve()
        max_visits = 0
        for row in result:
            max_visits = max(max_visits, max([col for col in row if isinstance(col, int)]))
        game.print_board()
        print(f"Maximum Visits Achieved: {max_visits+1}")
        print(f"Remaining blocks not Achieved: {(n*n) - num_obstacles - max_visits - 1}")


    elif mode == '4':
        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        start_x = int(input("Enter starting X position: "))
        start_y = int(input("Enter starting Y position: "))
        game = KnightGame(n, num_obstacles, start_x, start_y)
        game2 = copy.deepcopy(game)

        mcts = MCTSPlay(game.board, game.moves, iterations=1000)

        print("MCTS Play starting...")
        max_visits = 0  

        while True:
            valid_moves = mcts.get_valid_moves(game.x, game.y, game.board)
            if not valid_moves:
                print("No more valid moves! Machine stopped.")
                break
            root = MCTSNode(game.x, game.y)
            nx, ny = mcts.select_best_move(root)

            game.x, game.y = nx, ny
            game.board[game.x][game.y] = game.step
            game.step += 1

            game.print_board()
            print()  

            max_visits = max(max_visits, game.step)  

        print(f"Max Visits Achieved: {max_visits}")
        print(f"Remaining blocks not Achieved: {(n*n) - num_obstacles - max_visits}")

        algo = WarnsdorffsAlgorithm(game2)
        result = algo.solve()
        max_visits = 0
        for row in result:
            max_visits = max(max_visits, max([col for col in row if isinstance(col, int)]))
        game2.print_board()
        print(f"Maximum Visits Achieved: {max_visits+1}")
        print(f"Remaining blocks not Achieved: {(n*n) - num_obstacles - max_visits - 1}")


    elif mode == '5':
        from QNetwork import QLearningAgent  # Ensure you define or import correctly

        # Ask user for board parameters
        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        start_x = int(input("Enter starting X position: "))
        start_y = int(input("Enter starting Y position: "))

        # Create a reference KnightGame environment (though we'll recreate each episode)
        game = KnightGame(n, num_obstacles, start_x, start_y)

        # Create a Q-Learning agent
        agent = QLearningAgent(
            state_size = n * n,
            action_size = len(game.moves),
            alpha = 0.1,      # learning rate
            gamma = 0.9,      # discount factor
            epsilon = 1.0     # initial epsilon (will be decayed below)
        )

        # Training hyperparameters
        num_episodes = 1000
        max_steps_per_episode = n * n * 2  # a bit larger than the board size

        # Epsilon decay parameters
        initial_epsilon = 1.0
        final_epsilon   = 0.01
        decay_rate      = 0.99

        print("=== Q-Learning training starts... ===")

        for episode in range(num_episodes):
            # Decay epsilon over episodes
            agent.epsilon = max(final_epsilon, initial_epsilon * (decay_rate ** episode))
            
            # Re-initialize the environment each episode
            game = KnightGame(n, num_obstacles, start_x, start_y)

            # Track visited squares for reward shaping
            visited_squares = set()
            visited_squares.add((game.x, game.y))

            total_reward = 0

            for step_count in range(max_steps_per_episode):
                # Current state
                state = agent.get_state(game.x, game.y, n)

                # Collect valid moves
                valid_moves = []
                valid_indices = []
                for i, (dx, dy) in enumerate(game.moves):
                    nx, ny = game.x + dx, game.y + dy
                    if 0 <= nx < n and 0 <= ny < n and game.board[nx][ny] == -1:
                        valid_moves.append((nx, ny))
                        valid_indices.append(i)

                # If no valid moves, knight is stuck -> negative reward, end episode
                if not valid_moves:
                    reward = -10
                    agent.update(state, 0, reward, state, [])
                    total_reward += reward
                    break

                # Choose an action via epsilon-greedy
                action_index = agent.choose_action(state, valid_indices)
                dx, dy = game.moves[action_index]

                # Execute the move
                nx, ny = game.x + dx, game.y + dy
                game.x, game.y = nx, ny
                game.board[nx][ny] = game.step
                game.step += 1

                # Reward shaping: bonus for new squares
                if (nx, ny) not in visited_squares:
                    visited_squares.add((nx, ny))
                    reward = 3  # for example, +3 total
                # else:
                #     reward = 1  # for revisited squares, or you could use 0

                # Next state
                next_state = agent.get_state(nx, ny, n)

                # Next valid moves
                next_valid_indices = []
                for i2, (dx2, dy2) in enumerate(game.moves):
                    nx2, ny2 = nx + dx2, ny + dy2
                    if 0 <= nx2 < n and 0 <= ny2 < n and game.board[nx2][ny2] == -1:
                        next_valid_indices.append(i2)

                # Update Q-table
                agent.update(state, action_index, reward, next_state, next_valid_indices)
                total_reward += reward

            # (Optional) print progress every X episodes
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Epsilon={agent.epsilon:.3f}, Total Reward={total_reward}")

        print("=== Q-Learning training finished! ===")

        # B. Final Test Run with epsilon=0 (pure exploitation)
        agent.epsilon = 0
        test_game = KnightGame(n, num_obstacles, start_x, start_y)

        print("\n=== Final Test Episode (Exploitation Only) ===")
        step_count = 0
        while True:
            state = agent.get_state(test_game.x, test_game.y, n)

            # Gather valid moves
            valid_moves = []
            valid_indices = []
            for i, (dx, dy) in enumerate(test_game.moves):
                nx, ny = test_game.x + dx, test_game.y + dy
                if 0 <= nx < n and 0 <= ny < n and test_game.board[nx][ny] == -1:
                    valid_moves.append((nx, ny))
                    valid_indices.append(i)

            if not valid_moves:
                print("No more valid moves! Knight is stuck.")
                break

            # Choose best action (epsilon=0 => no random exploration)
            action_index = agent.choose_action(state, valid_indices)
            dx, dy = test_game.moves[action_index]
            nx, ny = test_game.x + dx, test_game.y + dy

            # Execute move
            test_game.x, test_game.y = nx, ny
            test_game.board[nx][ny] = test_game.step
            test_game.step += 1
            step_count += 1

        print(f"Final test episode ended after {step_count} moves.")
        test_game.print_board()

        # C. Examine Q-Table
        # print("\n=== Q-Table Snapshot (Raw) ===")
        # print(agent.q_table)

        # print("\n=== Q-Table Detailed View (State -> Q-Values) ===")
        # for s in range(n*n):
        #     x = s // n
        #     y = s % n
        #     print(f"State (x={x}, y={y}): {agent.q_table[s]}")
        
    elif mode == '6':
        # BRUTE FORCE / BACKTRACKING MODE
        import sys
        sys.setrecursionlimit(10**7)  # sometimes needed for deep recursion

        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        start_x = int(input("Enter starting X position: "))
        start_y = int(input("Enter starting Y position: "))

        game = KnightGame(n, num_obstacles, start_x, start_y)

        # Overwrite the board with -1 (unvisited) except obstacles
        # Because KnightGame might have set the starting position to 0,
        # let's reset everything to -1 except 'X' for obstacles
        for i in range(n):
            for j in range(n):
                if game.board[i][j] != 'X':
                    game.board[i][j] = -1

        # Place the knight at the starting position
        game.board[start_x][start_y] = 0

        # We'll define the brute force backtracking function inside Mode 6
        def is_valid_move_bf(x, y):
            """
            Check if (x, y) is a valid cell: in bounds, not an obstacle, not visited.
            """
            return (
                0 <= x < n and
                0 <= y < n and
                game.board[x][y] == -1
            )

        def backtrack(step, x, y):
            """
            Backtracking function:
            step: which move number we're on (0-based or 1-based)
            x, y: current knight position
            """
            # If we've made step moves, see if that covers all free squares
            # (step is index in the path, so if step == total_free_squares, we're done)
            # total_free_squares = n*n - num_obstacles
            if step == (n*n - num_obstacles):
                # All free squares visited
                return True

            # Try all knight moves
            for (dx, dy) in game.moves:
                nx, ny = x + dx, y + dy
                if is_valid_move_bf(nx, ny):
                    game.board[nx][ny] = step  # mark visited with current step
                    if backtrack(step+1, nx, ny):
                        return True
                    # Otherwise backtrack
                    game.board[nx][ny] = -1

            return False

        print("Brute Force (Backtracking) starting...")
        total_free = (n*n - num_obstacles)

        # Start from step=1 because we've already placed the knight with step=0
        success = backtrack(1, start_x, start_y)

        if success:
            print("Brute force found a complete tour!")
        else:
            print("No tour found with brute force.")

        # Print final board
        print("Final Board (Brute Force):")
        game.print_board()

    else:
        print("Invalid mode selected. Exiting the game.")


if __name__ == "__main__":
    main()
