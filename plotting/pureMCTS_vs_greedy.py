from Arena import Arena

from gomoku.GomokuAIPlayer import GreedyPlayer
from gomoku.GomokuAIPlayer import PureMCTSPlayer
from gomoku.GomokuEnv import GomokuEnv
from MCTS import MCTS
from gomoku.GomokuGameVars import *

# back to root dir and run:
# python -m plotting.pureMCTS_vs_greedy

game = GomokuEnv()
player1 = GreedyPlayer(game)
player2 = PureMCTSPlayer(game, MCTS(game=game, nnet=None, args=args))

arena = Arena(player1, player2, game)

result = arena.plays(play_num=3)
print(result)