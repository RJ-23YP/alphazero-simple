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

# # Initialize the environment
# game = GomokuEnv(board_size=15)


class GomokuNNet(nn.Module):
    def __init__(self, game, args):
        # # game params
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args    

        super(GomokuNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, args.num_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 3, stride=1)

        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)

        self.fc1 = nn.Linear(args.num_channels*(self.board_x-4)*(self.board_y-4), 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, self.action_size)

        self.fc4 = nn.Linear(512, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)                # batch_size x 1 x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x board_x x board_y
        s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = s.view(-1, self.args.num_channels*(self.board_x-4)*(self.board_y-4))

        s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.args.dropout, training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.args.dropout, training=self.training)  # batch_size x 512

        pi = self.fc3(s)                                                                         # batch_size x action_size
        v = self.fc4(s)                                                                          # batch_size x 1

        return F.log_softmax(pi, dim=1), torch.tanh(v)
    

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

