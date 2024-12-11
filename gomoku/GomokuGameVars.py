## THIS CODE CAN BE USED TO CHANGE THE NO. OF SELF-PLAY GAMES, NO. OF MCTS SIMULATIONS AND OTHER GAME AND ALPHAZERO PARAMETERS. 

from enum import Enum
import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

# Game parameters and visual configurations for Gomoku
GameParam = {
    'BOARD_SIZE': 15,          # Size of the board (15x15 grid)
    'WIN_LENGTH': 5,           # Number of consecutive stones needed to win
    'PIXEL_SIZE': 670,         # Size of the board in pixels
    'MARGIN': 27,              # Margin around the board
    'WINDOW_SIZE': (870, 670), # Total size of the game window (including margins)
    'LINE_COLOR': [0, 0, 0],   # Color of the grid lines (black)
    'POINT_COLOR': [0, 0, 0],  # Color of points on the board (black)
    'BOARD_COLOR': [238, 154, 73], # Color of the board background (light brown)
    'BUTTON_COLOR': [224, 183, 70], # Color of buttons (light yellow-brown)
    'FOCUS_COLOR': [92, 4, 224],    # Highlight color for the current focus
    'BLACK': [0, 0, 0],         # Color for black stones
    'WHITE': [255, 255, 255]    # Color for white stones
}

# Enumeration for the game status
class GameStatus(Enum):
    Init = 0    # Game is in the initialization phase
    Start = 1   # Game has started
    End = 2     # Game has ended
    
args = dotdict({
    'numIters': 3,                       # Total number of training iterations
    'numEps': 50,                        # Number of self-play games per iteration
    'tempThreshold': 15,                 # Temperature threshold for exploration in MCTS
    'updateThreshold': 0.6,              # Threshold for accepting a new model in arena playoffs
    'maxlenOfQueue': 200000,             # Maximum length of the training example queue
    'numMCTSSims': 30,                   # Number of simulations to perform in MCTS
    'arenaCompare': 40,                  # Number of games in the arena to compare models
    'cpuct': 1,                          # Exploration-exploitation tradeoff constant in MCTS

    'checkpoint': './temp/', # Directory for saving checkpoints
    'load_model': True, # Select true if you want to laod a pretrained model. If you want to train from scratch, you can select False here.
    'load_folder_file': ('saved_models','TRAIN_50SP_10EPOCH_100SIM.pth.tar'), # The first entry is the foler location where model is stored. The second entry is the name of the model file. Change these as per requirements. 
    'numItersForTrainExamplesHistory': 20,  # Number of iterations to store training examples
})