# alphazero-simple

This repository contains an implementation of the AlphaZero algorithm, inspired by the original work from the following GitHub repository:

- [AlphaZero-General](https://github.com/suragnair/alpha-zero-general)

## Overview

AlphaZero is a state-of-the-art reinforcement learning algorithm developed by DeepMind. It combines deep neural networks with Monte Carlo Tree Search (MCTS) to achieve superhuman performance in board games like Gomoku. The algorithm learns purely from self-play, without any prior knowledge or human data.

This repository provides a general implementation of AlphaZero that can be applied to different board games, with the ability to train, evaluate, and test models.

## Play with AI

To play against the AI, simply run the following command:

Run the Game
Ensure all dependencies are installed.

Open a terminal and navigate to the root directory of this project.

Execute the command:

```
python -m test.gomoku_test.PureMCTS_Playground
```
Game Objective
The user can select amongst any 4 of the AI players or play against human opponents. 
You (black) and the AI (white) take turns placing pieces on a 15x15 board.
The first player to get five consecutive pieces in a row (horizontally, vertically, or diagonally) wins.
Enjoy the game!

## Trained Models for DQN and MCTS+NN:

https://www.dropbox.com/scl/fo/8riivrfbsuxdepbuactys/AN7sntY4xukWFvvwk6ap2xQ?rlkey=n196036kvz0cboeojoc0nst3v&st=nsqvftp2&dl=0

## Run Evaluation Matches:

The user can run Arena matches between any two players using this code. The code generates the results of win, loss and draw. The number of games played can be customized. Both the players alternate as black and white in equal number of games.

Open a terminal and navigate to the root directory of this project.

Execute the command:

```
python -m plotting.MCTS+NN_vs_greedy
```



