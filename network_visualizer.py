from torch.utils.tensorboard import SummaryWriter
from torchviz import make_dot
import torch    
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gomoku.GomokuEnv import GomokuEnv

if __name__ == "__main__":
    from gomoku.GomokuGame import GomokuGame
    from nnet_models.NNet import NNetWrapper, args 

    # Initialize a dummy GomokuGame and model
    game = GomokuGame()
    model = NNetWrapper(game)

    # Create an input tensor with a batch size greater than 1
    device = torch.device("cuda" if args.cuda else "cpu") 
    input_tensor = torch.randn(2, game.getBoardSize()[0], game.getBoardSize()[1]).to(device)  # Batch size = 2

    # TensorBoard writer
    writer = SummaryWriter()

    # Log the computational graph to TensorBoard
    writer.add_graph(model.nnet, input_tensor)
    writer.close()

    # # Visualize the network with torchviz
    # output = model.nnet(input_tensor)
    # make_dot(output[0], params=dict(model.nnet.named_parameters())).render("network_graph", format="png")
