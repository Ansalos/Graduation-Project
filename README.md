# Knight’s Tour Game

This repository contains a **Knight’s Tour** program with multiple **game modes** and **AI methods** to solve or explore the Knight’s Tour problem on an nXn chessboard. Modes include:

1. **Manual Play**  
2. **Monte Carlo Tree Search (MCTS)**  
3. **Warnsdorff's Algorithm**  
4. **MCTS vs Warnsdorff**  
5. **Q-Learning**  
6. **Brute Force / Backtracking** (optional extra)
7. **Self-Play Reinforcement Training** (Neural Network)  
8. **AlphaZero-Inspired PUCT Agent with Neural Network**

## What Is the Knight’s Tour?

The Knight’s Tour is a classic puzzle where a **knight** (as in chess) must visit every square on the board **exactly once**, following the knight’s \(L\)-shaped movement rules. For an nXn board, the puzzle is to find the longest path possible in a spacific boared where Obstacles can be introduced to complicate the path further.

## Table of Contents

1. [Features and Modes](#features-and-modes)  
2. [Repository Structure](#repository-structure)  
3. [Running the Program](#running-the-program)  
4. [Modes in Detail](#modes-in-detail)  
   - [Mode 1: Manual Play](#mode-1-manual-play)  
   - [Mode 2: MCTS Play](#mode-2-mcts-play)  
   - [Mode 3: Warnsdorffs Algorithm](#mode-3-warnsdorffs-algorithm)  
   - [Mode 4: MCTS vs. Warnsdorff](#mode-4-mcts-vs-warnsdorff)  
   - [Mode 5: Q-Learning](#mode-5-q-learning)  
   - [Mode 6: Brute Force (Backtracking)](#mode-6-brute-force-backtracking)  
   - [Mode 7: Self-Play Reinforcement Training](#mode-7-self-play-reinforcement-training)  
   - [Mode 8: AlphaZero-Style PUCT Agent](#mode-8-alphazero-style-puct-agent-neural-network--mcts)  
5. [Future Enhancements](#future-enhancements)
---

## Features and Modes

- **Manual Play**: Human player moves the knight, can place obstacles.  
- **MCTS**: Monte Carlo Tree Search that picks moves by simulating possible paths and choosing the best.  
- **Warnsdorff**: Classic heuristic algorithm that tends to pick next moves with the fewest onward options.  
- **Q-Learning**: Reinforcement learning approach where the knight learns a policy to maximize coverage by trial and error.  
- **Brute Force**: A backtracking solution that systematically tries all paths.  
- **Self-Play and Training**: Trains a policy/value neural network through self-play and stores model weights.  
- **PUCT + Neural Network**: AlphaZero-inspired MCTS guided by a neural net, with full transparency and interpretability.

- Supports optional logging of all decision data (Q, P, N, PUCT score, and Warnsdorff degree) to a CSV file.  

- Includes the `Analyze_Decisions.py` script to extract interpretable heuristics from move logs using a decision tree classifier.

## Repository Structure

```
┌──── KnightTourGame                 # Main Game file with Core game logic and AI agents
├── Analyze_Decisions.py             # Analyzes PUCT move logs and extracts heuristics
├── KnightGameStarter.py             # Entry point script with all game modes
├── KnightTourGame.py                # Knight movement logic and game board
├── MCTSNode.py                      # Node class for MCTS tree
├── MCTSPlay.py                      # Standard MCTS agent logic
├── PUCTNode.py                      # PUCT-enhanced tree node with priors
├── PUCTPlay.py                      # AlphaZero-style MCTS agent using neural network
├── QNetwork.py                      # Q-learning table agent
├── NeuralNetworkClass.py            # PyTorch-based neural network policy/value model
├── SelfPlayAndTraining.py           # Self-play training routine
├── WarnsdorffAlgo.py                # Warnsdorff's algorithm implementation
└── puct_decisions.csv               # CSV log generated during Mode 8 runs
── Interpretable Neural Networks דוח הצעה.pdf
── Interpretable Neural Networks for Tackling Alpha Report.pdf
── README.md                         # This file
```

### Main Scripts

1. **KnightGameStarter.py**  
   - This is the **primary** script you run.  
   - Prompts you to pick a mode.

2. **KnightTourGame.py**  
   - Defines the `KnightGame` class: board initialization, obstacle placement, moves, etc.

3. **MCTSNode.py & MCTSPlay.py**  
   - Contains classes and methods implementing **Monte Carlo Tree Search**.

4. **WarnsdorffAlgo.py**  
   - Implements **Warnsdorff’s Algorithm** to solve the Knight’s Tour.

5. **QNetwork.py**  
   - Contains `QLearningAgent`, a Q-table approach to reinforcement learning.

6. **PUCTNode.py & PUCTPlay.py**  
   - Contains AlphaZero-style MCTS with neural policy guidance and full interpretability.

7. **NeuralNetworkClass.py**  
   - The neural network class used for training and inference.

8. **SelfPlayAndTraining.py**  
   - Automates the self-play reinforcement learning process to train the network.

9. **Analyze_Decisions.py**  
   - Script for loading `puct_decisions.csv` and extracting decision rules using decision trees.

---

## Running the Program

1. **Clone** or **download** this repository.
2. Open a terminal or command prompt in the project folder.
3. Run:
   ```bash
   python KnightGameStarter.py
   ```
4. You’ll see a menu:
   ```
   Welcome to Knight's Tour Game!
   Select Game Mode:
   1. Play the Knight's Tour
   2. MCTS Play (Machine-Controlled Tour)
   3. Warnsdorff's Algorithm (Optimized Tour)
   4. MCTS vs Warnsdorff's Algorithm on Same Board
   5. Q-Learning Integration (AI-Enhanced Tour)
   6. Brute Force (Backtracking) Mode
   7. Self-Play Training (Reinforcement Learning)
   8. Play with Trained AlphaZero-like PUCT AI
   Enter the mode number:
   ```
5. **Enter** a mode number. The program will prompt you for **board size**, **number of obstacles**, and **start position**.  
6. Depending on the mode:
   - You may see step-by-step movements, or an overall solution showing the path taken
   - The AI will attempt to find a path automatically.

---

## Modes in Detail

### Mode 1: Manual Play

1. **Prompt**: asks for \(n\), obstacles, and starting position.  
2. **Gameplay**:  
   - You move the knight by entering letters (like A, B, C, …) corresponding to valid moves displayed on the board.  
   - **Undo** is available by pressing `U`.

This is ideal if you want to **manually** practice or demonstrate the Knight’s Tour logic.

### Mode 2: MCTS Play

- Uses **Monte Carlo Tree Search** to decide the knight’s next move.  
- **Simulations** ran to pick moves with the highest potential coverage.  
- Prints the board after each move until no valid moves remain.

### Mode 3: Warnsdorff’s Algorithm

- Implements the classic **Warnsdorff** strategy:  
- Always move the knight to the square from which the knight has the **fewest** onward moves.  
- Often finds complete tours on many boards **quickly**, mainly efficiently with smaller boards.  
- Results in a near-perfect or perfect coverage in many cases.

### Mode 4: MCTS vs. Warnsdorff

- Compares how many squares each **MCTS** and **Warnsdorff** can visit on **the same initial board**.  
- Runs MCTS first, then a **copy** of the same board with Warnsdorff, printing each approach’s results.

### Mode 5: Q-Learning

- The agent trains over **multiple episodes** to improve policy.  
- **Epsilon-greedy** exploration and a **Q-table** are used to find better paths.  
- Prints a final exploitation run (epsilon=0) to show how well it covers the board after training.

### Mode 6: Brute Force (Backtracking)

- Mode that uses a **depth-first search (DFS)** or **backtracking** method.  
- Tries **all possible** moves from each position.  
- If the knight visits all free squares, you get a **complete** tour.  
- Very **time-consuming** for large boards but guarantees a correct solution if one exists.

### Mode 7: Self-Play Reinforcement Training

- Launches automated training episodes where the AI plays against itself using policy/value network predictions.  
- After training, the model is saved and can be used by Mode 8.

### Mode 8: AlphaZero-Style PUCT Agent (Neural Network + MCTS)

- Uses a trained neural network (`knight_network.pt`) to guide move selection via PUCT.  
- Provides detailed logging of Q, P, N, and PUCT scores.  
- Optionally logs this data to a CSV for interpretability.  
- Includes heatmaps for decision transparency.  
- You can analyze logs using `Analyze_Decisions.py` to extract interpretable rules.
  
---

## Future Enhancements

- Expand interpretability features to the Q-learning and self-play systems  
- Train on larger boards (e.g., 10x10+)  
- Apply the approach to other NP problems like the Traveling Salesman Problem (TSP)  
- Export learned heuristics to a symbolic rule engine

---

**Enjoy exploring the Knight’s Tour!**
