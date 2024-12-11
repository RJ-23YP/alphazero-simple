#### This is the main file to launch the AlphaZero-based model training process. 

import logging
import coloredlogs

from Coach import Coach
from gomoku.GomokuGame import GomokuGame as Game
from nnet_models.NNet import NNetWrapper as nn
from utils import *
from gomoku.GomokuGameVars import * 
import torch 

# Set up logging
log = logging.getLogger(__name__)

# Configure colored logging for better readability
coloredlogs.install(level='INFO') 

def main():
    """
    Main function to initialize the game, neural network, and coach,
    and then start the learning process.
    """
    # Set the device (CPU or GPU) for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the game environment
    log.info('Loading %s...', Game.__name__)
    g = Game()

    # Initialize the neural network wrapper
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    # Optionally load a pre-trained model checkpoint
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    # Initialize the Coach (manages self-play, training, and evaluation)
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    # Optionally load training examples from a previous run
    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    # Start the learning process
    log.info('Starting the learning process: ')
    c.learn()

# Entry point of the program
if __name__ == "__main__":
    main()
