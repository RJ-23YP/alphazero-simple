### THIS CODE DEFINES THE HYPERPARAMETERS AND THE MODEL TRAINING LOOP FOR THE ALPHAZERO NEURAL NETWORK.


import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append('../../')
from utils import *
from gomoku.GomokuGameVars import * 

import torch
import torch.optim as optim

from .GomokuNNet import GomokuNNet as Gnnet
from gomoku.GomokuEnv import GomokuEnv

args = dotdict({
    'lr': 0.001,               # Learning rate
    'dropout': 0.3,            # Dropout rate for regularization
    'epochs': 10,              # Number of training epochs
    'batch_size': 64,          # Size of each training batch
    'cuda': torch.cuda.is_available(),  # Use GPU if available
    'num_channels': 512,       # Number of channels for convolutional layers
})


class NNetWrapper:
    """
    A wrapper for the Gomoku neural network, providing utility functions for training, 
    prediction, and saving/loading model checkpoints.
    """

    def __init__(self, game):
        """
        Initializes the neural network and configures GPU usage.

        Args:
            game: The Gomoku game object, used to retrieve board dimensions and action size.
        """
        self.nnet = Gnnet(game, args)  # Initialize the Gomoku neural network
        self.board_x, self.board_y = game.getBoardSize()  # Get board dimensions
        self.action_size = game.getActionSize()  # Get the total number of possible actions

        if args.cuda:
            self.nnet.cuda()  # Move the network to GPU if CUDA is available

    def train(self, examples):
        """
        Trains the neural network using the provided training examples.

        Args:
            examples: A list of training examples, where each example is a tuple
                      (board, policy, value).

        Returns:
            policy_losses: List of policy loss values for each epoch.
            value_losses: List of value loss values for each epoch.
            total_losses: List of total loss values for each epoch.
        """
        optimizer = optim.Adam(self.nnet.parameters())  # Use Adam optimizer

        # Initialize lists to track losses for each epoch
        policy_losses = []
        value_losses = []
        total_losses = []

        # Training loop over epochs
        for epoch in range(args.epochs):
            print('EPOCH ::: ' + str(epoch + 1))
            self.nnet.train()  # Set the network to training mode
            pi_losses = MetricTracker()  # Tracks policy losses for this epoch
            v_losses = MetricTracker()  # Tracks value losses for this epoch

            batch_count = int(len(examples) / args.batch_size)  # Number of batches per epoch

            # Process each batch
            t = tqdm(range(batch_count), desc='Training Net')  # Progress bar for batches
            for _ in t:
                # Sample a batch of examples
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # Move data to GPU if available
                if args.cuda:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

                # Compute network outputs and losses
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v  # Total loss is the sum of policy and value losses

                # Record losses for tracking
                pi_losses.update(l_pi.item(), boards.size(0)) 
                v_losses.update(l_v.item(), boards.size(0)) 
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses) 

                # Perform backpropagation and update model weights
                optimizer.zero_grad() 
                total_loss.backward() 
                optimizer.step()

            # Record average losses for this epoch
            policy_losses.append(pi_losses.average)
            value_losses.append(v_losses.average) 
            total_losses.append(pi_losses.average + v_losses.average)

        return policy_losses, value_losses, total_losses 

    def predict(self, board):
        """
        Predicts the policy (action probabilities) and value (game outcome) for a given board state.

        Args:
            board: A numpy array representing the board state.

        Returns:
            pi: Action probabilities for each possible action.
            v: Predicted value of the board state.
        """
        # Prepare the input tensor
        board = torch.FloatTensor(board.astype(np.float64))
        if args.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)  # Reshape to match the network's input shape
        self.nnet.eval()  # Set the network to evaluation mode
        with torch.no_grad(): 
            pi, v = self.nnet(board)  # Predict policy and value

        # Convert predictions to numpy arrays
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def loss_pi(self, targets, outputs):
        """
        Computes the policy loss using cross-entropy.

        Args:
            targets: Ground truth policy probabilities.
            outputs: Predicted policy probabilities.

        Returns:
            The policy loss.
        """
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        """
        Computes the value loss using mean squared error.

        Args:
            targets: Ground truth values.
            outputs: Predicted values.

        Returns:
            The value loss.
        """
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Saves the model checkpoint to the specified folder and filename.

        Args:
            folder: Directory to save the checkpoint.
            filename: Name of the checkpoint file.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        """
        Loads a model checkpoint from the specified folder and filename.

        Args:
            folder: Directory containing the checkpoint.
            filename: Name of the checkpoint file.
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}") 
        map_location = None if args.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict']) 
