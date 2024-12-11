### This code can be used to play games against any one of the four players the user selects. The user can also play games against another person.

import sys
import os

sys.path.append('../..')
from gomoku.GomokuGame import GomokuGame

def main():
    print("Select the type of player:\n1. Greedy Player\n2. MCTS Player\n3. MCTS + NN Player\n4. DQN Player")
    choices = {1: 'GreedyPlayer', 2: 'PureMCTSPlayer', 3: 'MCTSNNPlayer', 4:'DQNPlayer'} 

    while (choice := int(input("Enter your choice (1/2/3/4): "))) not in choices:
        print("Invalid choice. Please select 1, 2, 3 or 4.") 

    game = GomokuGame(AI_type=choices[choice]) 
    game.run()

if __name__ == "__main__":
    main() 

