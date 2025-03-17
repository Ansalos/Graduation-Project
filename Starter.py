import math
import os
import sys
from Gomoku import Gomoku
from mcts_player import MCTSPlayer
from PUCTPlayer import PUCTPlayer
from NeuralNetworkClass import GameNetwork

class GameStarter:
    @staticmethod
    def check_network_file():
        network_file = "trained_network.pt"
        if not os.path.exists(network_file):
            print(f"Error: Network file '{network_file}' not found. Exiting the program.")
            sys.exit(1)
    
    @staticmethod
    def main():
        while True:
            print("\n=== GAME STARTER ===")
            print("1) Play Gomoku [Human vs. Human]")
            print("2) Play Gomoku [Human vs. MCTS (UCB)]")
            print("3) Play Gomoku [MCTS (UCB) vs. MCTS (UCB)]")
            print("4) Play MCTS vs MCTS multiple times")
            print("5) Play Gomoku [Human vs. PUCT (Neural Net)]")
            print("6) Play Gomoku [PUCT vs. MCTS]")
            print("7) Play MCTS vs. PUCT multiple times")
            print("8) Exit")

            choice = input("Enter your choice: ")
            if choice == "1":
                GameStarter.play_gomoku_human()
            elif choice == "2":
                GameStarter.play_gomoku_mcts()
            elif choice == "3":
                GameStarter.play_gomoku_mcts_mcts()
            elif choice == "4":
                GameStarter.play_multiple_mcts_vs_mcts()
            elif choice == "5":
                GameStarter.play_gomoku_puct()
            elif choice == "6":
                GameStarter.play_gomoku_puct_mcts()
            elif choice == "7":
                GameStarter.play_multiple_puct_vs_mcts()
            elif choice == "8":
                print("Goodbye!")
                break
            else:
                print("Invalid choice.")

    #
    # NEW ELO HELPER
    #
    @staticmethod
    def update_elo(ratingA, ratingB, winner=None, k=32):
        """
        Returns (newRatingA, newRatingB) after a single game between A and B.

        :param ratingA: Elo of player A
        :param ratingB: Elo of player B
        :param winner:  'A' if A won, 'B' if B won, or None (draw)
        :param k:       K-factor (commonly 32)
        """
        # Expected score for A
        expectedA = 1.0 / (1.0 + 10 ** ((ratingB - ratingA) / 400))

        # Actual score for A
        if winner == 'A':
            scoreA = 1.0
        elif winner == 'B':
            scoreA = 0.0
        else:
            # Treat draws as 0.5–0.5
            scoreA = 0.5

        newRatingA = ratingA + k * (scoreA - expectedA)

        # B’s score is 1 - A’s score
        scoreB = 1.0 - scoreA
        expectedB = 1.0 - expectedA
        newRatingB = ratingB + k * (scoreB - expectedB)

        return (newRatingA, newRatingB)

    @staticmethod
    def choose_first_player():
        print("Choose who goes first:")
        print("1) Human")
        print("2) AI")
        first = input("Enter your choice: ")
        if first == "1":
            return Gomoku.BLACK
        elif first == "2":
            return Gomoku.WHITE
        else:
            print("Invalid choice, defaulting to Human (BLACK).")
            return Gomoku.BLACK

    @staticmethod
    def play_gomoku_human():
        game = Gomoku()
        print("\nGomoku [Human vs. Human]")

        while game.status == game.ONGOING:
            # Show board each turn
            print("\nCurrent Board:")
            print(game)

            # Indicate who is thinking/playing
            if game.player == Gomoku.BLACK:
                print("Human (BLACK) is thinking...")
            else:
                print("Human (WHITE) is thinking...")

            move_str = input("Enter your move (e.g. A1): ")
            try:
                move = Gomoku.parse_position(move_str)
                if move not in game.legal_moves():
                    print("Illegal move.")
                    continue
                game.make(move)
            except ValueError as e:
                print("Invalid move:", e)
                continue

        GameStarter.print_result(game)

    @staticmethod
    def play_gomoku_mcts():
        game = Gomoku()
        print("\nGomoku [Human vs MCTS]")
        mcts_ai = MCTSPlayer(iterations=50000, exploration_constant=math.sqrt(2))
        first_player = GameStarter.choose_first_player()

        while game.status == game.ONGOING:
            # Show board each turn
            print("\nCurrent Board:")
            print(game)

            if game.player == first_player:
                print("Human is thinking...")
                move_str = input("Enter your move (e.g. A1): ")
                try:
                    move = Gomoku.parse_position(move_str)
                    if move not in game.legal_moves():
                        print("Illegal move.")
                        continue
                    game.make(move)
                except ValueError as e:
                    print("Invalid:", e)
                    continue
            else:
                print("MCTS is thinking...")
                move = mcts_ai.choose_move(game)
                game.make(move)
                move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                print("MCTS played:", move_str)

        GameStarter.print_result(game)

    #
    # MODIFIED to RETURN the winner: "BLACK", "WHITE", or "DRAW"
    #
    @staticmethod
    def play_gomoku_mcts_mcts(show_board=1):
        game = Gomoku()
        print("\nGomoku [MCTS (B) vs. MCTS (W)]")
        mcts_ai_b = MCTSPlayer(iterations=50000, exploration_constant=math.sqrt(2))
        mcts_ai_w = MCTSPlayer(iterations=50000, exploration_constant=math.sqrt(2))

        while game.status == game.ONGOING:
            if show_board == 1:
                print("\nCurrent Board:")
                print(game)

            if game.player == game.BLACK:
                print("MCTS (BLACK) is thinking...")
                move = mcts_ai_b.choose_move(game)
                game.make(move)
                move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                print("MCTS (B) played:", move_str)
            else:
                print("MCTS (WHITE) is thinking...")
                move = mcts_ai_w.choose_move(game)
                game.make(move)
                move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                print("MCTS (W) played:", move_str)

        # This prints final board & “<Color> wins!” or “It's a draw!”
        GameStarter.print_result(game)

        if game.status == Gomoku.BLACK:
            return "BLACK"
        elif game.status == Gomoku.WHITE:
            return "WHITE"
        else:
            return "DRAW"

    @staticmethod
    def play_gomoku_puct():
        game = Gomoku()
        print("\nGomoku [Human vs PUCT]")

        GameStarter.check_network_file()
        # Load or create the network
        net = GameNetwork(board_size=game.size, in_channels=2)
        net.load_weights("trained_network.pt")

        # Create PUCT
        puct_ai = PUCTPlayer(network=net, c_puct=1.0, simulations=50000)
        first_player = GameStarter.choose_first_player()

        while game.status == game.ONGOING:
            # Show board each turn
            print("\nCurrent Board:")
            print(game)

            if game.player == first_player:
                print("Human is thinking...")
                move_str = input("Enter your move (e.g. A1): ")
                try:
                    move = Gomoku.parse_position(move_str)
                    if move not in game.legal_moves():
                        print("Illegal move.")
                        continue
                    game.make(move)
                except ValueError as e:
                    print("Invalid:", e)
                    continue
            else:
                print("PUCT is thinking...")
                move = puct_ai.choose_move(game)
                game.make(move)
                move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                print("PUCT played:", move_str)

        GameStarter.print_result(game)

    #
    # MULTIPLE MCTS vs MCTS with ELO + end-of-game announcement
    #
    @staticmethod
    def play_multiple_mcts_vs_mcts():
        num_games = int(input("Enter the number of games to play: "))
        show_board = int(input("Show Boards? Enter your choice (1 = yes , 0 = no): "))

        # Initialize Elo for MCTS(Black) and MCTS(White)
        mcts_black_rating = 1600
        mcts_white_rating = 1600

        black_wins = 0
        white_wins = 0
        draws = 0

        for i in range(num_games):
            print(f"\n=== Game {i + 1} of {num_games} ===")
            result = GameStarter.play_gomoku_mcts_mcts(show_board)

            # Show who won or if it's a draw
            print(f"Game {i+1} result: {result}")

            # Update tallies and Elo
            if result == "BLACK":
                black_wins += 1
                # 'A' is black, 'B' is white
                mcts_black_rating, mcts_white_rating = GameStarter.update_elo(
                    mcts_black_rating, mcts_white_rating, winner='A'
                )
            elif result == "WHITE":
                white_wins += 1
                mcts_black_rating, mcts_white_rating = GameStarter.update_elo(
                    mcts_black_rating, mcts_white_rating, winner='B'
                )
            else:
                draws += 1
                # Elo update for a draw
                mcts_black_rating, mcts_white_rating = GameStarter.update_elo(
                    mcts_black_rating, mcts_white_rating, winner=None
                )

            print(f"Current Elo => MCTS(Black): {int(mcts_black_rating)} | MCTS(White): {int(mcts_white_rating)}")

        # Final results
        print("\nResults after", num_games, "games:")
        print("BLACK wins:", black_wins)
        print("WHITE wins:", white_wins)
        print("Draws:", draws)
        print(f"Final Elo => MCTS(Black): {int(mcts_black_rating)}, MCTS(White): {int(mcts_white_rating)}")

    #
    # SINGLE PUCT vs MCTS (one game)
    #
    @staticmethod
    def play_gomoku_puct_mcts():
        game = Gomoku()
        print("\nGomoku [PUCT vs. MCTS]")

        # Who goes first?
        print("Who goes first?")
        print("1) PUCT")
        print("2) MCTS")
        choice = input("Enter your choice: ")
        if choice == "1":
            first_is_puct = True
        elif choice == "2":
            first_is_puct = False
        else:
            print("Invalid choice, defaulting to PUCT going first.")
            first_is_puct = True

        GameStarter.check_network_file()
        # Load network for PUCT
        net = GameNetwork(board_size=game.size, in_channels=2)
        net.load_weights("trained_network.pt")

        # Create AIs
        puct_ai = PUCTPlayer(network=net, c_puct=1.0, simulations=50000)
        mcts_ai = MCTSPlayer(iterations=50000, exploration_constant=math.sqrt(2))

        while game.status == game.ONGOING:
            print("\nCurrent Board:")
            print(game)

            if game.player == Gomoku.BLACK:
                if first_is_puct:
                    print("PUCT (BLACK) is thinking...")
                    move = puct_ai.choose_move(game)
                    game.make(move)
                    move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                    print("PUCT (B) played:", move_str)
                else:
                    print("MCTS (BLACK) is thinking...")
                    move = mcts_ai.choose_move(game)
                    game.make(move)
                    move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                    print("MCTS (B) played:", move_str)
            else:  # White's turn
                if first_is_puct:
                    print("MCTS (WHITE) is thinking...")
                    move = mcts_ai.choose_move(game)
                    game.make(move)
                    move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                    print("MCTS (W) played:", move_str)
                else:
                    print("PUCT (WHITE) is thinking...")
                    move = puct_ai.choose_move(game)
                    game.make(move)
                    move_str = f"{chr(move[1] + ord('A'))}{move[0] + 1}"
                    print("PUCT (W) played:", move_str)

        GameStarter.print_result(game)

    #
    # MULTIPLE PUCT vs MCTS WITH ELO + explicit result print
    #
    @staticmethod
    def play_multiple_puct_vs_mcts():
        """
        Play MCTS vs PUCT for N games in a row, updating Elo each time,
        and printing the winner or draw for every game.
        """
        num_games = int(input("Enter the number of games to play (MCTS vs PUCT): "))

        print("\nDo you want to choose who goes first for each game or only once?")
        print("1) For each game")
        print("2) Only once for all games")
        print("3) Ping Pong (1-1) each of them starts first")

        choice_mode = input("Enter your choice: ").strip()

        if choice_mode == "1":
            choose_each_game = True
            only_once = False
            ping_pong_mode = False
        elif choice_mode == "2":
            choose_each_game = False
            only_once = True
            ping_pong_mode = False
        elif choice_mode == "3":
            choose_each_game = False
            only_once = False
            ping_pong_mode = True
        else:
            print("Invalid choice; defaulting to 'choose for each game'.")
            choose_each_game = True
            only_once = False
            ping_pong_mode = False

        # If we're only choosing once, pick who starts
        puct_is_black_for_all = True
        if only_once:
            print("\nWho goes first for ALL games?")
            print("1) PUCT")
            print("2) MCTS")
            once_choice = input("Enter your choice: ").strip()
            if once_choice == "1":
                puct_is_black_for_all = True
            elif once_choice == "2":
                puct_is_black_for_all = False
            else:
                print("Invalid choice, defaulting to PUCT going first.")
                puct_is_black_for_all = True

        show_board = int(input("\nShow board each turn? (1 = yes, 0 = no): "))

        # Initialize Elo
        puct_rating = 1600
        mcts_rating = 1600

        puct_wins = 0
        mcts_wins = 0
        draws = 0
        
        GameStarter.check_network_file()
        # Create network & AIs once
        temp_game = Gomoku()
        net = GameNetwork(board_size=temp_game.size, in_channels=2)
        net.load_weights("trained_network.pt")
        puct_ai = PUCTPlayer(network=net, c_puct=1.0, simulations=50000)
        mcts_ai = MCTSPlayer(iterations=50000, exploration_constant=math.sqrt(2))

        for i in range(num_games):
            print(f"\n=== Game {i+1} of {num_games} ===")

            # Decide who goes first this game
            if choose_each_game:
                # Pick each time
                print("Who goes first this game?")
                print("1) PUCT")
                print("2) MCTS")
                each_choice = input("Enter your choice: ").strip()
                if each_choice == "1":
                    puct_is_black = True
                elif each_choice == "2":
                    puct_is_black = False
                else:
                    print("Invalid choice, defaulting to PUCT going first.")
                    puct_is_black = True

            elif only_once:
                # Use the same choice for all
                puct_is_black = puct_is_black_for_all

            elif ping_pong_mode:
                # Alternate every game (e.g., even index => PUCT black, odd => MCTS black)
                # You can invert this logic if you want the other side to start first
                if i % 2 == 0:
                    puct_is_black = True
                else:
                    puct_is_black = False

            # Start a new Gomoku game
            game = Gomoku()
            while game.status == game.ONGOING:
                if show_board == 1:
                    print("\nCurrent Board:")
                    print(game)

                if game.player == Gomoku.BLACK:
                    if puct_is_black:
                        print("PUCT (BLACK) is thinking...")
                        move = puct_ai.choose_move(game)
                    else:
                        print("MCTS (BLACK) is thinking...")
                        move = mcts_ai.choose_move(game)
                else:  # White
                    if puct_is_black:
                        print("MCTS (WHITE) is thinking...")
                        move = mcts_ai.choose_move(game)
                    else:
                        print("PUCT (WHITE) is thinking...")
                        move = puct_ai.choose_move(game)

                game.make(move)

            # Print final board if requested
            if show_board == 1:
                print("\nFinal Board:")
                print(game)

            # Identify winner
            if game.status == Gomoku.BLACK:
                if puct_is_black:
                    puct_wins += 1
                    print("Game over, BLACK (PUCT) wins!")
                    # Elo update: PUCT is A, MCTS is B
                    puct_rating, mcts_rating = GameStarter.update_elo(puct_rating, mcts_rating, winner='A')
                else:
                    mcts_wins += 1
                    print("Game over, BLACK (MCTS) wins!")
                    # Elo update: MCTS is A, PUCT is B
                    mcts_rating, puct_rating = GameStarter.update_elo(mcts_rating, puct_rating, winner='A')
            elif game.status == Gomoku.WHITE:
                if puct_is_black:
                    mcts_wins += 1
                    print("Game over, WHITE (MCTS) wins!")
                    # MCTS is B, PUCT is A
                    puct_rating, mcts_rating = GameStarter.update_elo(puct_rating, mcts_rating, winner='B')
                else:
                    puct_wins += 1
                    print("Game over, WHITE (PUCT) wins!")
                    # PUCT is B, MCTS is A
                    mcts_rating, puct_rating = GameStarter.update_elo(mcts_rating, puct_rating, winner='B')
            else:
                draws += 1
                print("Game over, it's a draw!")
                # Draw => 0.5 each
                if puct_is_black:
                    puct_rating, mcts_rating = GameStarter.update_elo(puct_rating, mcts_rating, winner=None)
                else:
                    mcts_rating, puct_rating = GameStarter.update_elo(mcts_rating, puct_rating, winner=None)

            # Show updated Elo
            print(f"Current Elo => PUCT: {int(puct_rating)} | MCTS: {int(mcts_rating)}")

        # Final summary
        print("\n=== Final Results ===")
        print("PUCT wins:", puct_wins)
        print("MCTS wins:", mcts_wins)
        print("Draws:", draws)
        print(f"Final Elo => PUCT: {int(puct_rating)}, MCTS: {int(mcts_rating)}")

    @staticmethod
    def print_result(game):
        print("\nFinal Board:")
        print(game)
        if game.status == Gomoku.BLACK:
            print("BLACK wins!")
        elif game.status == Gomoku.WHITE:
            print("WHITE wins!")
        else:
            print("It's a draw!")

if __name__ == "__main__":
    GameStarter.main()
