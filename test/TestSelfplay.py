from Arena import Arena

from gomoku.GomokuAIPlayer import GreedyPlayer
from gomoku.GomokuAIPlayer import PureMCTSPlayer
from gomoku.GomokuEnv import GomokuEnv

# python -m test.TestSelfplay

game = GomokuEnv()
player1 = GreedyPlayer(game)
player2 = GreedyPlayer(game)

arena = Arena(player1, player2, game)

arena.plays(play_num=500)