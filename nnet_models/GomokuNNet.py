### THIS CODE DEFINES THE NEURAL NETWORK ARCHITECTURE FOR THE ALPHAZERO APPRAOCH. 

import sys
sys.path.append('..')
from utils import *

import argparse
import torch    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gomoku.GomokuEnv import GomokuEnv


import torchvision
import matplotlib.pyplot as plt
from torchviz import make_dot


class GomokuNNet(nn.Module):
    """
    A neural network for the Gomoku game. This network predicts:
    1. The policy: probabilities of taking each possible action.
    2. The value: an estimate of the game's outcome from the current state.
    """
    def __init__(self, game, args):
        """
        Initializes the neural network with convolutional and fully connected layers.

        Args:
            game: The Gomoku game object, used to retrieve board dimensions and action size.
            args: Configuration arguments containing hyperparameters such as num_channels and dropout.
        """
        # Retrieve board dimensions and action space size from the game object
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args  # Store additional configuration arguments

        # Initialize the parent class
        super(GomokuNNet, self).__init__()

        # Convolutional layers to extract spatial features from the board
        self.conv1 = nn.Conv2d(1, args.num_channels, kernel_size=3, stride=1, padding=1)  # First convolution
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1, padding=1)  # Second convolution
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1)  # Third convolution
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, kernel_size=3, stride=1)  # Fourth convolution

        # Batch normalization layers to stabilize training and improve convergence
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        # Fully connected layers for policy and value prediction
        # Input dimension of the first fully connected layer is determined by the flattened size of the convolutional output
        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 4) * (self.board_y - 4), 1024)  # Hidden layer with 1024 units
        self.fc_bn1 = nn.BatchNorm1d(1024)  # Batch normalization for the first FC layer

        self.fc2 = nn.Linear(1024, 512)  # Second hidden layer with 512 units
        self.fc_bn2 = nn.BatchNorm1d(512)  # Batch normalization for the second FC layer

        # Output layers
        self.fc3 = nn.Linear(512, self.action_size)  # Policy output (probabilities of actions)
        self.fc4 = nn.Linear(512, 1)  # Value output (game outcome)


    def forward(self, s):
        """
        Forward pass through the network.

        Args:
            s (torch.Tensor): Input tensor representing the game state, shape (batch_size, board_x, board_y).

        Returns:
            pi (torch.Tensor): Log probabilities of actions, shape (batch_size, action_size).
            v (torch.Tensor): Predicted value of the state, shape (batch_size, 1).
        """
        # Reshape input for convolutional layers
        s = s.view(-1, 1, self.board_x, self.board_y)  # Shape: (batch_size, 1, board_x, board_y)

        # Pass through convolutional layers with batch normalization and ReLU activation
        s = F.relu(self.bn1(self.conv1(s)))  # Shape: (batch_size, num_channels, board_x, board_y)
        s = F.relu(self.bn2(self.conv2(s)))  # Shape: (batch_size, num_channels, board_x, board_y)
        s = F.relu(self.bn3(self.conv3(s)))  # Shape: (batch_size, num_channels, board_x-2, board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))  # Shape: (batch_size, num_channels, board_x-4, board_y-4)

        # Flatten the output from convolutional layers for fully connected layers
        s = s.view(-1, self.args.num_channels * (self.board_x - 4) * (self.board_y - 4))

        # Pass through the first fully connected layer with dropout
        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)

        # Pass through the second fully connected layer with dropout
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)

        # Policy head (action probabilities)
        pi = self.fc3(s)  # Shape: (batch_size, action_size)

        # Value head (state value)
        v = self.fc4(s)  # Shape: (batch_size, 1)

        # Apply log softmax to the policy output and tanh to the value output
        return F.log_softmax(pi, dim=1), torch.tanh(v)
    
### Optional main function to visualize the architecture via Torchviz network graph. 
if __name__ == "__main__":
    from gomoku.GomokuGame import GomokuGame
    from nnet_models.NNet import NNetWrapper, args

    # Initialize a dummy GomokuGame and model
    game = GomokuGame()
    model = NNetWrapper(game)

    # Create an input tensor with a batch size greater than 1
    device = torch.device("cuda" if args.cuda else "cpu") 
    input_tensor = torch.randn(2, game.getBoardSize()[0], game.getBoardSize()[1]).to(device)  # Batch size = 2

    # Visualize the network
    output = model.nnet(input_tensor)
    make_dot(output[0], params=dict(model.nnet.named_parameters())).render("network_graph", format="png")

