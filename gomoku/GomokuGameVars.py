from enum import Enum
import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

GameParam = {
    'BOARD_SIZE': 15,
    'WIN_LENGTH': 5,
    'PIXEL_SIZE': 670,
    'MARGIN': 27,
    'WINDOW_SIZE': (870, 670),
    'LINE_COLOR': [0, 0, 0],
    'POINT_COLOR': [0, 0, 0],
    'BOARD_COLOR': [238, 154, 73],
    'BUTTON_COLOR': [224, 183, 70],
    'FOCUS_COLOR': [92, 4, 224],
    'BLACK': [0, 0, 0],
    'WHITE': [255, 255, 255]
}

class GameStatus(Enum):
    Init = 0
    Start = 1
    End = 2
    
args = dotdict({
    'numIters': 3,
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration. 
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 30,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': True,            # Select true if you want to laod a pretrained model. If you want to train from scratch, you can select False here.
    'load_folder_file': ('saved_models','TRAIN_50SP_10EPOCH_100SIM.pth.tar'), # The first entry is the foler location where model is stored. The second entry is the name of the model file. Change these as per requirements. 
    'numItersForTrainExamplesHistory': 20,  
})