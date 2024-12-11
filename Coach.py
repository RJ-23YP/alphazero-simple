### This code contains the main function for self-play using MCTS and Neural Network model training. This is the core algorithm inspired by the AlphaZero approach.



import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from gomoku.GomokuAIPlayer import MCTSNNPlayer
from gomoku.GomokuEnv import GomokuEnv 

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Arena import Arena
from MCTS import MCTS
from gomoku.GomokuGameVars import * 


log = logging.getLogger(__name__)


class Coach():
    """
    Manages the training process, including self-play, neural network training, 
    and evaluation against previous models.
    """
    def __init__(self, game, nnet, args):
        """
        Initialize the Coach with the game, neural network, and configuration arguments.
        """
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples() 

    def executeEpisode(self):
        """
        Executes a single episode of self-play using MCTS and the current neural network.
        
        Returns:
            A list of training examples generated during the episode.
        """
        trainExamples = []
        board = self.game.getInitBoard()
        self.curPlayer = 1
        episodeStep = 0

        # Self-play loop
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
            temp = int(episodeStep < self.args.tempThreshold)

            # Get action probabilities using MCTS
            pi = self.mcts.getActionProb(canonicalBoard, temperature=temp)
            sym = self.game.getSymmetries(canonicalBoard, pi)

            # Generate symmetric examples for data augmentation
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            # Choose an action based on probabilities
            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)
            
            # Check if the game has ended
            r = self.game.getGameEnded(board, self.curPlayer)

            if r != 0:
                # Assign outcomes to training examples
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Main training loop for AlphaZero. Alternates between self-play, training, 
        and evaluation of the new model against the previous one.
        """

        for i in range(1, self.args.numIters + 1):
            # Logging iteration progress
            log.info(f'Starting Iter #{i} ...')

            # Generate training examples via self-play
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)
                
                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = MCTS(self.game, nnet=self.nnet, args=self.args)  # Standard MCTS with NN 
                    iterationTrainExamples += self.executeEpisode() 

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

             # Maintain a limited history of training examples
            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)

            # Save training examples to a file 
            self.saveTrainExamples(i - 1)

            # Combine and shuffle all training examples
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples) 

            # Save the current neural network and train a new one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar') 
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar') 
            pmcts = MCTS(self.game, self.pnet, self.args) 
            
            # Train the new model and get epoch-wise losses
            epoch_policy_losses, epoch_value_losses, epoch_total_losses = self.nnet.train(trainExamples) 

            # Plot the losses for the current iteration
            self.plot_epoch_losses(i, epoch_policy_losses, epoch_value_losses, epoch_total_losses) 

            nmcts = MCTS(self.game, self.nnet, self.args) 

            # Evaluate the new model against the previous one
            game1 = GomokuEnv()
            player1 = MCTSNNPlayer(game1, nmcts) 
            player2 = MCTSNNPlayer(game1, pmcts) 

            # Create the new Arena object
            arena = Arena(player1, player2, game1) 

            # Play games using the modified plays method
            results = arena.plays(play_num=self.args.arenaCompare // 2) 

            # Extract results: [draws, player1 wins, player2 wins]
            draws, nwins, pwins = results[0], results[1], results[2]

            log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws)) 

            # Decide whether to accept the new model
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar') 


    def plot_epoch_losses(self, iteration, policy_losses, value_losses, total_losses):
        """
        Plots the epoch-wise losses for a single iteration with a fixed y-axis range
        and adjusted plot size.
        """
        # Set a smaller figure size
        plt.figure(figsize=(8, 5))  # Adjust the width and height to make the plot smaller
        epochs = list(range(1, len(policy_losses) + 1)) 
        
        # Plot the losses
        plt.plot(epochs, policy_losses, label="Policy Loss", marker='o')
        plt.plot(epochs, value_losses, label="Value Loss", marker='o')
        plt.plot(epochs, total_losses, label="Total Loss", marker='o')

        # Set axis labels and title
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training Progress")
        
        # Add legend and grid
        plt.legend()
        plt.grid()
        
        # Set consistent y-axis limits
        plt.ylim(0, 6)
        
        # Show the plot
        plt.tight_layout()  # Ensures everything fits within the smaller size
        plt.show() 


    def getCheckpointFile(self, iteration):
        """
        Generate a checkpoint filename for a specific iteration.
        """
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        """
        Save the training examples to a file for later use.
        """
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        """
        Load training examples from a file, if available.
        """
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit() 
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # Skip the first self-play iteration if examples are loaded
            self.skipFirstSelfPlay = True
