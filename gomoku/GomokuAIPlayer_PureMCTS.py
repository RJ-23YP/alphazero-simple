import numpy as np

# no need
class PureMCTSPlayer:
    def __init__(self, env, mcts):
        self.env = env
        self.mcts = mcts

    def get_move(self):
        board = self.env.board
        current_player = self.env.current_player
        canonical_board = self.env.get_canonical_form(board, current_player)
        action_probs = self.mcts.getActionProb(canonical_board, temp=0)  # temp=0 for greedy move
        best_action = np.argmax(action_probs)
        return divmod(best_action, self.env.board_size)

