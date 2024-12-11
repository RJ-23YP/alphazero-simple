### This code contains the Arena class which facilates custom number of matches between any 2 players

import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


class Arena():
    """
    A class to facilitate matches between two players in a given game.
    Handles both single games and multiple games in a series, with players alternating roles.
    """
    def __init__(self, player1, player2, game, display=None):
        """
        Initializes the Arena.

        Args:
            player1: The first player.
            player2: The second player.
            game: The game environment to be played.
            display: Optional display function for visualizing the game state.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        
    def play(self, firstPlayer, secondPlayer):
        """
        Plays a single game between two players.

        Args:
            firstPlayer: The player who makes the first move.
            secondPlayer: The player who makes the second move.

        Returns:
            The winner of the game (0 = draw, 1 = firstPlayer wins, 2 = secondPlayer wins).
        """
        self.game.reset()  # Reset the game to its initial state
        players = [firstPlayer, secondPlayer]
        currentPlayerIndex = 0
        
        # Main game loop: players take turns making moves until the game ends
        while not self.game.done:
            players[currentPlayerIndex].move()
            currentPlayerIndex ^= 1  # Switch to the other player
        
        return self.game.winner  # Return the result of the game
    
    def plays(self, play_num):
        """
        Plays a series of games between the two players, alternating roles.

        Args:
            play_num: The number of games to play as each role.

        Returns:
            results: A list containing the cumulative results:
                     - results[0]: Number of draws
                     - results[1]: Number of wins by player1
                     - results[2]: Number of wins by player2
        """

        # draws, p1 wins, p2 wins
        results = [0] * 3

        # First series: player1 plays first, player2 plays second
        for i in tqdm(range(play_num), desc="Arena.plays (1)"):
            result = self.play(self.player1, self.player2)
            results[result] += 1

        # Second series: player2 plays first, player1 plays second
        for i in tqdm(range(play_num), desc="Arena.plays (2)"):
            result = self.play(self.player2, self.player1)
            
            # Adjust result to account for role reversal
            if result != 0:
                result = 3 - result  # Swap player1 and player2 results
            
            results[result] += 1 

        # Return cumulative results of the series
        return results