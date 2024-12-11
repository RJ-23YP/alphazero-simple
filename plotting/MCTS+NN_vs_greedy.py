from Arena import Arena

from gomoku.GomokuGame import GomokuGame as Game
from gomoku.GomokuAIPlayer import *
from gomoku.GomokuEnv import GomokuEnv
from MCTS import MCTS
from gomoku.GomokuGameVars import *

from nnet_models.NNet import NNetWrapper as nn

# back to root dir and run:
# python -m plotting.MCTS+NN_vs_greedy


game = GomokuEnv()

nnet = nn(Game())

checkpoint_path = "/media/rj/New Volume/Northeastern University/Semester-3 (Fall 2024)/CS 5180 - RL/FInal Project/Submission/alphazero-simple/saved_models/TRAIN_50SP_10EPOCH_100SIM.pth.tar" 

if checkpoint_path and os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    folder, filename = os.path.split(checkpoint_path)  
    nnet.load_checkpoint(folder, filename) 
else:
    print("Warning: No checkpoint specified; using randomly initialized network.") 


# # Add DQNPlayer with model weights
# dqn_weights_path = "/media/rj/New Volume/Northeastern University/Semester-3 (Fall 2024)/CS 5180 - RL/FInal Project/Submission/alphazero-simple/saved_models/dqn_final_weights.pth"  # Path to DQN weights
# dqn_player1 = DQNPlayer(game)  # Initialize DQN player 

# if dqn_weights_path and os.path.exists(dqn_weights_path):
#     print(f"Loading DQN model from {dqn_weights_path}...")
#     dqn_player1.load_model(dqn_weights_path)
# else:
#     print(f"No DQN weights found at {dqn_weights_path}. Using randomly initialized DQN.") 


# player1 = dqn_player1  # Use DQN player as Player 1 

player1 = MCTSNNPlayer(game, MCTS(game=game, nnet=nnet, args=args)) 
# player2 = PureMCTSPlayer(game, MCTS(game=game, nnet=None, args=args))

player2 = GreedyPlayer(game) 

arena = Arena(player1, player2, game) 

result = arena.plays(play_num=5) 
print(result) 