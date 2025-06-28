from KnightTourGame import KnightGame
from MCTSNode import MCTSNode
from MCTSPlay import MCTSPlay
from WarnsdorffAlgo import WarnsdorffsAlgorithm
import copy
import random
import numpy as np



def main():
    print("Welcome to Knight's Tour Game!")
    print("Select Game Mode:")
    print("1. Play the Knight's Tour")
    print("2. MCTS Play (Machine-Controlled Tour)")
    print("3. Warnsdorff's Algorithm (Optimized Tour)")
    print("4. MCTS vs Warnsdorff's Algorithm on Same Board")
    print("5. Q_learning")
    print("6. Brute Force (Backtracking) Mode")
    print("7. MCTS Self-Play + Training")
    print("8. Play with Trained AlphaZero")
    print("9. PUCT with No Neural Net (Logic Only)")
    print("10. Elo Match: Warnsdorff vs PUCT Logic-Only")    
    print("11. Play with Decision Tree (Mimics Logic-based AI)")
    print("12. Elo Match: Decision Tree vs Warnsdorff Only")
    print("13. Elo Arena: Warnsdorff vs PUCT Logic vs Decision Tree")



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
            state_size = 3 ** n,
            action_size = 8,
            alpha = 0.1,      # learning rate
            gamma = 0.9,      # discount factor
            epsilon = 1.0     # initial epsilon (will be decayed below)
        )

        # Training hyperparameters
        num_episodes = 10000
        max_steps_per_episode = n * n * 2  # a bit larger than the board size

        # Epsilon decay parameters
        initial_epsilon = 1.0
        final_epsilon   = 0.03
        decay_rate      = 0.99995

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
            all_twos_and_zeros = [0,0,0]
            ones_exist = [0,0,0]

            for step_count in range(max_steps_per_episode):
                # Current state
                state, encode_arr = agent.get_state(game)

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
                if step_count<5:
                    action_index = agent.choose_action(state, valid_indices, 1)
                else:
                    action_index = agent.choose_action(state, valid_indices, agent.epsilon)

                # For our statistics:
                q_values = [agent.q_table[state, a] for a in valid_indices]
                best_action_index = valid_indices[np.argmax(q_values)]
                if 1 in encode_arr:
                    ones_exist[encode_arr[best_action_index]] += 1
                else:
                    all_twos_and_zeros[encode_arr[best_action_index]] += 1

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

                # Next state
                next_state, _ = agent.get_state(game)

                # Next valid moves
                next_valid_indices = []
                for i2, (dx2, dy2) in enumerate(game.moves):
                    nx2, ny2 = nx + dx2, ny + dy2
                    if 0 <= nx2 < n and 0 <= ny2 < n and game.board[nx2][ny2] == -1:
                        next_valid_indices.append(i2)

                # Update Q-table
                agent.update(state, action_index, reward, next_state, next_valid_indices)
                total_reward += reward

            # Print progress every X episodes
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode+1}/{num_episodes} | Epsilon={agent.epsilon:.3f}, Total Reward={total_reward}")

        print("=== Q-Learning training finished! ===")

        # B. Final Test Run with epsilon=0 (pure exploitation)
        test_game = KnightGame(n, num_obstacles, start_x, start_y)

        print("\n=== Final Test Episode (Exploitation Only) ===")
        step_count = 0
        while True:
            state, _ = agent.get_state(test_game)

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
            action_index = agent.choose_action(state, valid_indices, 0)
            dx, dy = test_game.moves[action_index]
            nx, ny = test_game.x + dx, test_game.y + dy

            # Execute move
            test_game.x, test_game.y = nx, ny
            test_game.board[nx][ny] = test_game.step
            test_game.step += 1
            step_count += 1

        print(f"Final test episode ended after {step_count} moves.")
        test_game.print_board()
        print("Move stats:")
        arr_twos_zeros = np.array(all_twos_and_zeros)
        print("All twos and zeros:", arr_twos_zeros / arr_twos_zeros.sum())

        arr_ones_exist = np.array(ones_exist)
        print("Ones Exist:", arr_ones_exist / arr_ones_exist.sum())
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

    elif mode == '7':
        print("=== MCTS Self-Play + NN Training ===")
        n = int(input("Enter the board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        ans_start = input("Randomize the knight's starting position each game? (y/n): ").strip().lower()
        random_start_pos = (ans_start == 'y')
        if not random_start_pos:
            sx = int(input("Enter starting X position: "))
            sy = int(input("Enter starting Y position: "))
        else:
            sx, sy = 0, 0

        ans_obstacles = input("Place obstacles randomly EACH training game? (y/n): ").strip().lower()
        random_each_game = (ans_obstacles == 'y')

        num_games = int(input("How many self-play games to train on? ex(10000) "))
        mcts_iters = int(input("MCTS iterations per move? ex(1000) "))
        ep = int(input("Training epochs? ex(5) "))
        lr = float(input("Learning rate? (e.g. 1e-3): "))
        print_moves_ans = input("Print the board each move while training? (y/n): ").strip().lower()
        print_moves = (print_moves_ans == 'y')

        from SelfPlayAndTraining import run_mcts_selfplay_training
        run_mcts_selfplay_training(
            board_size = n,
            num_obstacles = num_obstacles,
            random_start = random_start_pos,
            fixed_start = (sx, sy),
            random_each_game = random_each_game,
            num_games = num_games,
            mcts_iterations = mcts_iters,
            epochs = ep,
            lr = lr,
            print_each_move = print_moves
        )
        print("MCTS Self-Play + NN training complete!")

    elif mode == '8':
            print("\n=== Knight Tour [PUCT-based AI] ===")
            # 1) Ask user for board parameters
            n = int(input("Enter board size (n x n): "))
            num_obstacles = int(input("Enter number of obstacles: "))
            start_x = int(input("Enter starting X position: "))
            start_y = int(input("Enter starting Y position: "))
            show_debug_input = input("Show AI transparency (NN/MCTS/Warnsdorff/heatmaps)? (y/n): ").strip().lower()
            show_debug = (show_debug_input == 'y')
            log_csv_input = input("Log AI decisions to CSV file? (y/n): ").strip().lower()
            log_to_csv = (log_csv_input == 'y')

            # 2) Create the KnightGame
            game = KnightGame(n, num_obstacles, start_x, start_y)

            # 3) Load the neural net
            from NeuralNetworkClass import KnightNetwork
            net = KnightNetwork(board_size=n, in_channels=2)
            net.load_weights("knight_network.pt")  # Make sure knight_network.pt is present

            # 4) Create your PUCT-based AI
            from PUCTPlay import KnightPUCTPlayer
            puct_ai = KnightPUCTPlayer(network=net, c_puct=1.0, simulations=1000, show_debug=show_debug, log_to_csv=log_to_csv)

            print("\nThe AI will move the knight until no valid moves remain.")
            input("Press Enter to begin...")

            while True:
                # 5) Check if there are valid moves
                valid_moves = []
                for (dx, dy) in game.moves:
                    nx = game.x + dx
                    ny = game.y + dy
                    if game.is_valid_move(nx, ny):
                        valid_moves.append((nx, ny))

                if not valid_moves:
                    print("No more valid moves! Knight is stuck or board is filled.")
                    break

                # 6) Use the PUCT AI to pick a move
                move = puct_ai.choose_move(game.clone())
                if move is None:
                    print("PUCT returned None (no moves?). Stopping.")
                    break

                # 7) Apply the move
                if not game.is_valid_move(move[0], move[1]):
                    print(f"PUCT picked invalid move {move} - stopping.")
                    break

                game.x, game.y = move
                game.board[game.x][game.y] = game.step
                game.step += 1

                # Print the board so we can see the knight's progress
                print(f"\nKnight just moved to: {move}   (Step={game.step-1})")
                game.print_board()

            # 8) End
            print("\nFinal board:")
            game.print_board()
            print("PUCT-based knight tour ended.")
            print(f"Squares visited: {game.step} / {(n*n)-num_obstacles}")

    elif mode == '9':
        print("\n=== Knight Tour [PUCT with Logic Only] ===")

        num_games = int(input("Enter number of games to play: "))
        n = int(input("Enter board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))

        manual_start = input("Set starting position manually? (y/n): ").strip().lower() == 'y'
        if manual_start:
            start_x = int(input("Enter starting X position: "))
            start_y = int(input("Enter starting Y position: "))

        show_debug_input = input("Show logic details for each step? (y/n): ").strip().lower()
        show_debug = (show_debug_input == 'y')

        from RuleBasedPUCTPlayer import RuleBasedPUCTPlayer

        def ask_csv_logging():
            return input("Save AI decisions to CSV file? (y/n): ").strip().lower() == 'y'

        log_to_csv = ask_csv_logging()
        logic_ai = RuleBasedPUCTPlayer(c_puct=1.0, simulations=1000, show_debug=show_debug, log_to_csv=log_to_csv)

        def get_warnsdorff_next_move(game):
            legal = game.legal_moves()
            best_deg = float('inf')
            best_move = None
            for nx, ny in legal:
                onward = 0
                for dx, dy in game.moves:
                    tx, ty = nx + dx, ny + dy
                    if 0 <= tx < game.n and 0 <= ty < game.n and game.board[tx][ty] == -1:
                        onward += 1
                if onward < best_deg:
                    best_deg = onward
                    best_move = (nx, ny)
            return best_move

        for game_num in range(1, num_games + 1):
            print(f"\n--- Game {game_num} ---")

            if manual_start:
                sx, sy = start_x, start_y
            else:
                while True:
                    sx = random.randint(0, n - 1)
                    sy = random.randint(0, n - 1)
                    if (sx, sy) not in [(i, j) for i in range(n) for j in range(n)][:num_obstacles]:
                        break

            game = KnightGame(n, num_obstacles, sx, sy)

            print("\nThe logic-only AI will now play the Knight's Tour...")
            while True:
                valid_moves = game.legal_moves()
                if not valid_moves:
                    print("No more valid moves. Tour ended.")
                    break

                warnsdorff_game = game.clone()

                move = logic_ai.choose_move(game.clone())
                if move is None:
                    print("AI could not find a move. Tour ended.")
                    break

                game.x, game.y = move
                game.board[game.x][game.y] = game.step
                game.step += 1

                print(f"\n[Step {game.step}] Knight moved to: {move}")

                warn_move = get_warnsdorff_next_move(warnsdorff_game)
                if warn_move:
                    print(f"   → Warnsdorff would pick: {warn_move}")
                else:
                    print("   → Warnsdorff has no valid moves.")

                print(f"   → Logic-PUCT chose:     {move}")
                game.print_board()

            print("\nFinal board:")
            game.print_board()
            print(f"Logic-based PUCT Tour finished with {game.step} squares visited out of {n * n - num_obstacles}.")

    elif mode == '10':
        print("\n=== Mode 10: Warnsdorff vs PUCT Logic-Only (Elo Rating Match) ===")
        n_games = int(input("Enter number of games per agent: "))
        board_size = int(input("Enter board size: "))
        num_obstacles = int(input("Enter number of obstacles: "))
        fairness = input("Should both agents play on the same board with same start? (y/n): ").strip().lower()
        show_boards = input("Show final boards for each game? (y/n): ").strip().lower()
        k_factor = 32

        from RuleBasedPUCTPlayer import RuleBasedPUCTPlayer

        rating_warn = 1200
        rating_logic = 1200

        def run_game_on(game, agent_type):
            if agent_type == 'warnsdorff':
                algo = WarnsdorffsAlgorithm(game)
                result = algo.solve()
                score = sum(isinstance(cell, int) and cell >= 0 for row in result for cell in row)
            elif agent_type == 'logic':
                logic_ai = RuleBasedPUCTPlayer(c_puct=1.0, simulations=1000, show_debug=False)
                while True:
                    valid = game.legal_moves()
                    if not valid:
                        break
                    move = logic_ai.choose_move(game.clone())
                    if move is None:
                        break
                    game.x, game.y = move
                    game.board[game.x][game.y] = game.step
                    game.step += 1
                score = sum(isinstance(cell, int) and cell >= 0 for row in game.board for cell in row)
            return score, game

        for i in range(n_games):
            print(f"\nGame {i+1}:")

            if fairness == 'y':
                sx = random.randint(0, board_size - 1)
                sy = random.randint(0, board_size - 1)
                base_game = KnightGame(board_size, num_obstacles, sx, sy)
                game_for_warn = copy.deepcopy(base_game)
                game_for_logic = copy.deepcopy(base_game)
            else:
                sx1, sy1 = random.randint(0, board_size - 1), random.randint(0, board_size - 1)
                sx2, sy2 = random.randint(0, board_size - 1), random.randint(0, board_size - 1)
                game_for_warn = KnightGame(board_size, num_obstacles, sx1, sy1)
                game_for_logic = KnightGame(board_size, num_obstacles, sx2, sy2)

            score_warn, final_warn = run_game_on(game_for_warn, 'warnsdorff')
            score_logic, final_logic = run_game_on(game_for_logic, 'logic')

            print(f"  Warnsdorff visited: {score_warn}")
            print(f"  PUCT Logic-only visited: {score_logic}")

            if show_boards == 'y':
                print("\n  Warnsdorff Final Board:")
                final_warn.print_board()
                print("\n  PUCT Logic-only Final Board:")
                final_logic.print_board()

            # Elo rating update
            if score_warn > score_logic:
                s_warn, s_logic = 1, 0
            elif score_warn < score_logic:
                s_warn, s_logic = 0, 1
            else:
                s_warn = s_logic = 0.5

            expected_warn = 1 / (1 + 10 ** ((rating_logic - rating_warn) / 400))
            expected_logic = 1 / (1 + 10 ** ((rating_warn - rating_logic) / 400))

            rating_warn += k_factor * (s_warn - expected_warn)
            rating_logic += k_factor * (s_logic - expected_logic)

            print(f"  Updated Ratings → Warnsdorff: {round(rating_warn, 2)}, LogicPUCT: {round(rating_logic, 2)}")

        print("\n=== Final Ratings ===")
        print(f"  Warnsdorff: {round(rating_warn, 2)}")
        print(f"  PUCT Logic-Only: {round(rating_logic, 2)}")

    elif mode == '11':
        print("\n=== Knight Tour [Decision Tree-Based AI] ===")

        import pandas as pd
        import joblib
        from sklearn.tree import DecisionTreeClassifier

        # === STEP 1: Train the Decision Tree ===
        try:
            df = pd.read_csv("puct_logic_only.csv") # pick which csv file to make DecisionTree puct_decisions.csv/ puct_logic_only.csv
            df.columns = df.columns.str.strip()

            features = ["degree", "Q", "P", "N", "PUCT"]
            for feature in features:
                if feature not in df.columns:
                    raise KeyError(f"Missing expected column: {feature}")

            X = df[features]
            y = df["is_best"]
            clf = DecisionTreeClassifier(max_depth=4, random_state=42)
            clf.fit(X, y)
            joblib.dump(clf, "decision_tree_model.pkl")
            print("Trained Decision Tree saved as decision_tree_model.pkl")
            from sklearn.tree import export_text
            tree_rules = export_text(clf, feature_names=["degree", "Q", "P", "N", "PUCT"])
            print("\n Decision Tree Rules:\n")
            print(tree_rules)

        except Exception as e:
            print("Failed to train decision tree:", e)
            return

        # === STEP 2: Setup game ===
        n = int(input("Enter board size (n x n): "))
        num_obstacles = int(input("Enter number of obstacles: "))
        sx = int(input("Enter starting X position: "))
        sy = int(input("Enter starting Y position: "))

        from DecisionTreePlayer import DecisionTreePlayer
        game = KnightGame(n, num_obstacles, sx, sy)
        tree_ai = DecisionTreePlayer("decision_tree_model.pkl")


        # === STEP 3: Play using the decision tree ===
        while True:
            legal = game.legal_moves()
            if not legal:
                print("No valid moves left. Game Over.")
                break

            move = tree_ai.choose_move(game)
            if not move:
                print("Decision Tree returned no move. Stopping.")
                break

            game.x, game.y = move
            game.board[game.x][game.y] = game.step
            game.step += 1
            print(f"\n[Step {game.step-1}] Knight moved to: {move}")
            # game.print_board()

        print("\nFinal board:")
        game.print_board()
        print(f"Squares visited: {game.step} / {(n * n) - num_obstacles}")

    elif mode == '12':
        print("\n=== Mode 12: Decision Tree vs Warnsdorff (Elo Match) ===")
        n_games = int(input("Enter number of games: "))
        board_size = int(input("Enter board size: "))
        num_obstacles = int(input("Enter number of obstacles: "))
        fairness = input("Should both agents play the same board and start? (y/n): ").strip().lower()
        show_boards = input("Show final boards? (y/n): ").strip().lower()
        k_factor = 32

        from DecisionTreePlayer import DecisionTreePlayer

        rating_warn = 1200
        rating_tree = 1200

        def run_warnsdorff(game):
            algo = WarnsdorffsAlgorithm(game)
            algo.solve()
            return game

        def run_tree_ai(game):
            tree_ai = DecisionTreePlayer("decision_tree_model.pkl")
            while True:
                valid = game.legal_moves()
                if not valid:
                    break
                move = tree_ai.choose_move(game.clone())
                if move is None:
                    break
                game.x, game.y = move
                game.board[game.x][game.y] = game.step
                game.step += 1
            return game

        def count_score(game):
            return sum(isinstance(cell, int) and cell >= 0 for row in game.board for cell in row)

        def update_elo(r1, r2, s1, s2):
            e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
            return (
                r1 + k_factor * (s1 - e1),
                r2 + k_factor * (s2 - e2)
            )

        for i in range(n_games):
            print(f"\n Game {i+1}/{n_games}")
            if fairness == 'y':
                sx = random.randint(0, board_size - 1)
                sy = random.randint(0, board_size - 1)
                base_game = KnightGame(board_size, num_obstacles, sx, sy)
                game_warn = copy.deepcopy(base_game)
                game_tree = copy.deepcopy(base_game)
            else:
                game_warn = KnightGame(board_size, num_obstacles, random.randint(0, board_size - 1), random.randint(0, board_size - 1))
                game_tree = KnightGame(board_size, num_obstacles, random.randint(0, board_size - 1), random.randint(0, board_size - 1))

            game_warn = run_warnsdorff(game_warn)
            game_tree = run_tree_ai(game_tree)

            score_warn = count_score(game_warn)
            score_tree = count_score(game_tree)

            print(f"  Warnsdorff visited: {score_warn}")
            print(f"  Decision Tree AI visited: {score_tree}")

            if show_boards == 'y':
                print("\n  Warnsdorff Final Board:")
                game_warn.print_board()
                print("\n  Decision Tree AI Final Board:")
                game_tree.print_board()

            if score_warn > score_tree:
                s_warn, s_tree = 1, 0
            elif score_warn < score_tree:
                s_warn, s_tree = 0, 1
            else:
                s_warn = s_tree = 0.5

            rating_warn, rating_tree = update_elo(rating_warn, rating_tree, s_warn, s_tree)

            print(f" Updated Elo → Warnsdorff: {round(rating_warn, 2)} | Tree AI: {round(rating_tree, 2)}")

        print("\n=== Final Elo Ratings ===")
        print(f"Warnsdorff: {round(rating_warn, 2)}")
        print(f"Decision Tree AI: {round(rating_tree, 2)}")

    elif mode == '13':
        print("\n=== Mode 10: Elo Match – Warnsdorff vs PUCT Logic-Only vs Decision Tree ===")
        n_games = int(input("Enter number of games per agent: "))
        board_size = int(input("Enter board size: "))
        num_obstacles = int(input("Enter number of obstacles: "))
        fairness = input("Should all agents play on the same board with same start? (y/n): ").strip().lower()
        show_boards = input("Show final boards for each game? (y/n): ").strip().lower()

        from RuleBasedPUCTPlayer import RuleBasedPUCTPlayer
        from DecisionTreePlayer import DecisionTreePlayer
    
        tree_player = DecisionTreePlayer("decision_tree_model.pkl")

        k_factor = 32
        ratings = {
            "warnsdorff": 1200,
            "logic": 1200,
            "tree": 1200
        }

        def run_game(agent_type, game):
            game = copy.deepcopy(game)
            if agent_type == 'warnsdorff':
                algo = WarnsdorffsAlgorithm(game)
                result = algo.solve()
                score = sum(isinstance(cell, int) and cell >= 0 for row in result for cell in row)
            elif agent_type == 'logic':
                logic_ai = RuleBasedPUCTPlayer(c_puct=1.0, simulations=1000, show_debug=False)
                while True:
                    valid = game.legal_moves()
                    if not valid:
                        break
                    move = logic_ai.choose_move(game.clone())
                    if move is None:
                        break
                    game.x, game.y = move
                    game.board[game.x][game.y] = game.step
                    game.step += 1
                score = sum(isinstance(cell, int) and cell >= 0 for row in game.board for cell in row)
            elif agent_type == 'tree':
                while True:
                    legal = game.legal_moves()
                    if not legal:
                        break
                    move = tree_player.choose_move(game)
                    if not move:
                        break
                    game.x, game.y = move
                    game.board[game.x][game.y] = game.step
                    game.step += 1
                score = sum(isinstance(cell, int) and cell >= 0 for row in game.board for cell in row)
            return score, game

        agents = ['warnsdorff', 'logic', 'tree']

        for i in range(n_games):
            print(f"\n Game {i+1} of {n_games}")

            if fairness == 'y':
                sx = random.randint(0, board_size - 1)
                sy = random.randint(0, board_size - 1)
                base_game = KnightGame(board_size, num_obstacles, sx, sy)
                boards = {agent: copy.deepcopy(base_game) for agent in agents}
            else:
                boards = {
                    agent: KnightGame(board_size, num_obstacles,
                                    random.randint(0, board_size - 1),
                                    random.randint(0, board_size - 1))
                    for agent in agents
                }

            scores = {}
            for agent in agents:
                score, board = run_game(agent, boards[agent])
                scores[agent] = score
                if show_boards == 'y':
                    print(f"\n {agent.upper()} Final Board (Visited: {score}):")
                    board.print_board()

            print("\nScores:")
            for a in agents:
                print(f"  {a.capitalize()}: {scores[a]}")

            # Pairwise Elo updates
            for a1 in agents:
                for a2 in agents:
                    if a1 == a2:
                        continue
                    s1 = scores[a1]
                    s2 = scores[a2]
                    r1 = ratings[a1]
                    r2 = ratings[a2]

                    if s1 == s2:
                        actual1, actual2 = 0.5, 0.5
                    elif s1 > s2:
                        actual1, actual2 = 1, 0
                    else:
                        actual1, actual2 = 0, 1

                    expected1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
                    expected2 = 1 / (1 + 10 ** ((r1 - r2) / 400))

                    ratings[a1] += k_factor * (actual1 - expected1)
                    ratings[a2] += k_factor * (actual2 - expected2)

        print("\n Final Elo Ratings:")
        for agent in agents:
            print(f"  {agent.capitalize()}: {round(ratings[agent], 2)}")

    else:
        print("Invalid mode selected. Exiting the game.")


if __name__ == "__main__":
    main()
