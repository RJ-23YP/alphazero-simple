import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    def __init__(self, player1, player2, game, display=None):
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        
    def play(self, firstPlayer, secondPlayer):
        self.game.reset()
        players = [firstPlayer, secondPlayer]
        currentPlayerIndex = 0
        while not self.game.done:
            players[currentPlayerIndex].move()
            currentPlayerIndex ^= 1
        return self.game.winner
    
    def plays(self, play_num):
        # draws, p1 wins, p2 wins
        results = [0] * 3
        for i in tqdm(range(play_num), desc="Arena.plays (1)"):
            result = self.play(self.player1, self.player2)
            results[result] += 1
        for i in tqdm(range(play_num), desc="Arena.plays (2)"):
            result = self.play(self.player2, self.player1)
            if result != 0:
                result = 3 - result
            results[result] += 1 
        # print(results)
        return results