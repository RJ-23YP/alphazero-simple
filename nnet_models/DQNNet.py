### This code defines the neural network architecture for the Target and Policy Networks. 


import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNet(nn.Module):
    """
    A Deep Q-Network (DQN) architecture designed for reinforcement learning tasks on a Gomoku board.
    It uses convolutional layers to process the board state and fully connected layers to produce Q-values
    for each possible action.
    """
    def __init__(self, board_size=15):
        """
        Initializes the DQNNet with convolutional layers for spatial feature extraction
        and fully connected layers for Q-value estimation.

        Args:
            board_size (int): The size of the Gomoku board (default: 15x15).
        """
        super(DQNNet, self).__init__()
        self.board_size = board_size

        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Input: (1, board_size, board_size)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: (128, board_size, board_size)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)  # Output: (128, board_size, board_size)

        # Flatten the convolutional output for the fully connected layers
        self.fc_input_dim = 128 * board_size * board_size  # Calculate input dimensions for the first FC layer

        # Fully connected layers for Q-value prediction
        self.fc1 = nn.Linear(self.fc_input_dim, 512)  # Hidden layer with 512 units
        self.fc2 = nn.Linear(512, board_size * board_size)  # Output layer with Q-values for all board positions

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, board_size, board_size).

        Returns:
            q (torch.Tensor): Q-values for each possible action, shape (batch_size, board_size * board_size).
        """
        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the convolutional layers
        x = x.view(-1, self.fc_input_dim)

        # Pass through fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))

        # Final Q-value predictions
        q = self.fc2(x)
        return q


