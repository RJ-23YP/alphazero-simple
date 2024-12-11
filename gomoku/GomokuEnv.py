### THIS CODE DEFINES THE GOMOKU GAME ENVIRONEMNT 

import numpy as np

class GomokuEnv:
    """
    Environment for the Gomoku game. Handles the game board, moves, and rules.
    """
    def __init__(self, board_size=15, win_length=5):
        """
        Initialize the environment.

        Args:
            board_size (int): The size of the square board (default: 15x15).
            win_length (int): Number of consecutive stones needed to win (default: 5).
        """
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)  # Initialize an empty board
        self.current_player = 1  # Player 1 starts the game
        self.done = False        # Flag to indicate if the game has ended
        self.winner = 0          # Winner of the game (0 if none yet)
        self.action_space = list(range(self.board_size * self.board_size))  # All possible actions

    def get_action_space_size(self):
        """
        Get the total number of possible actions.

        Returns:
            int: Size of the action space.
        """
        return len(self.action_space)

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.array: The initial board state.
        """
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board

    def step(self, action):
        """
        Apply an action to the board.

        Args:
            action (int): The linear index of the move.

        Returns:
            tuple: Updated board, reward for the move, and whether the game has ended.
        """
        row, col = divmod(action, self.board_size)

        if self.board[row, col] != 0:  # Invalid move
            return self.board, float('-inf'), True
        
        # Place the player's stone and check for winner
        self.board[row, col] = self.current_player
        reward, done = self.check_winner(row, col)
        
        if not done:
            self.current_player = 3 - self.current_player  # Switch to the other player
        else:
            self.winner = self.current_player if reward != 0 else 0
        
        self.done = done
        return self.board, reward, done

    def check_winner(self, row, col):
        """
        Check if the last move resulted in a win or draw.

        Args:
            row (int): Row index of the move.
            col (int): Column index of the move.

        Returns:
            tuple: Reward (1 for win, 0 for draw) and game status (True if game ended).
        """
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, vertical, diagonal directions

        for dr, dc in directions:
            count = 1

            # Count stones in the positive direction
            for step in range(1, self.win_length):
                r, c = row + dr * step, col + dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            # Count stones in the negative direction
            for step in range(1, self.win_length):
                r, c = row - dr * step, col - dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_length:
                return 1, True  # Player wins
        
        if np.all(self.board != 0):  # Check for draw
            return 0, True
        
        return 0, False

    def render(self):
        """
        Render the board for human-readable display.
        """
        symbols = {0: '.', 1: 'X', 2: 'O'}  # Map board values to symbols
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print("\n")

    def legal_moves(self):
        """
        Get a list of all legal moves.

        Returns:
            list: Indices of all legal moves.
        """
        return [i for i in self.action_space if self.board[i // self.board_size, i % self.board_size] == 0]

    def board_legal_moves(self, board, player):
        """
        Get legal moves for a specific board state and player.

        Args:
            board (np.array): The current board state.
            player (int): The current player.

        Returns:
            list: Legal moves for the given board state.
        """
        legal_moves = [i for i in self.action_space if board[i // len(board[0])][i % len(board[0])] == 0]
        return legal_moves

    def board_valid_moves(self, board, player):
        """
        Get a binary vector indicating valid moves.

        Args:
            board (np.array): The current board state.
            player (int): The current player.

        Returns:
            np.array: A binary vector indicating valid moves.
        """
        valids = np.zeros(len(board) * len(board[0]), dtype=int)
        legal_moves = self.board_legal_moves(board, player)
        assert len(legal_moves) != 0
        for move in legal_moves:
            valids[move] = 1
        return valids

    def pos_to_action(self, pos):
        """
        Convert a (row, col) position to a linear action index.

        Args:
            pos (tuple): (row, col) position.

        Returns:
            int: Linear action index.
        """
        x, y = pos
        action = x * self.board_size + y
        return action
    
    def get_canonical_form(self, board, player):
        """
        Get the canonical form of the board (flipped for player 2).

        Args:
            board (np.array): The current board state.
            player (int): The current player.

        Returns:
            np.array: Canonical board state.
        """
        if player == 1:
            return board
        else:
            return np.array([[3 - x if x != 0 else 0 for x in row] for row in board])

    def board_tostring(self, board):
        """
        Convert the board to a string representation.

        Args:
            board (np.array): The current board state.

        Returns:
            str: String representation of the board.
        """
        return board.tostring()

    def get_result(self, board):
        """
        Evaluate the board to determine the game result.

        Args:
            board (np.array): The current board state.

        Returns:
            float: 1 for player 1 win, -1 for player 2 win, small value for draw, or 0 for ongoing game.
        """
        # should be canonicalBoard
        # if not terminate, 0
        # if draw, return very little number
        # and if player 1 is winner, return 1. Else -1
        # Helper function to check win in one direction
        def check_win_length(x, y, dx, dy):
            player = board[x][y]
            if player == 0:
                return False
            for i in range(1, self.win_length):
                nx, ny = x + i * dx, y + i * dy
                if not (0 <= nx < len(board) and 0 <= ny < len(board[0])) or board[nx][ny] != player:
                    return False
            return True

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(len(board)):
            for y in range(len(board[0])):
                for dx, dy in directions:
                    if check_win_length(x, y, dx, dy):
                        return 1 if board[x][y] == 1 else -1
        
        # TODO: here simply check empty pos
        for row in board:
            if 0 in row:
                return 0
        # print('draw')
        return 1e-8
    
    def get_next_state(self, board, player, action):
        """
        Get the next state after a move.

        Args:
            board (np.array): The current board state.
            player (int): The current player.
            action (int): The action taken.

        Returns:
            tuple: Next board state and next player.
        """
        next_board = np.copy(board)
        x, y = divmod(action, self.board_size)
        # assert board[x][y] == 0
        next_board[x][y] = player
        return next_board, -player
    



    def check_winner_player(self, board):
        """
        Check if a specific player has won.

        Args:
            board (np.array): The current board state.

        Returns:
            int: Winner (1, -1, or 0 for no winner).
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for x in range(self.board_size):
            for y in range(self.board_size):
                player = board[x][y]
                if player != 0:
                    for dx, dy in directions:
                        count = 1
                        nx, ny = x, y
                        while True:
                            nx += dx
                            ny += dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx][ny] == player:
                                count += 1
                                if count == self.win_length:
                                    return player
                            else:
                                break
        return 0 
    

    def get_result_player(self, board):
        """
        Get the game result for a specific board.

        Args:
            board (np.array): The current board state.

        Returns:
            float: 1 for win, -1 for loss, 1e-8 for draw, or 0 for ongoing game.
        """
        result = self.check_winner_player(board)
        if result != 0:
            return result
        elif np.all(board != 0):
            return 1e-8  # Draw
        else:
            return 0  # Game not ended
        
    def board_valid_moves_player(self, board):
        """
        Get valid moves as a binary vector for a specific board state.

        Args:
            board (np.array): The current board state.

        Returns:
            np.array: Binary vector indicating valid moves.
        """
        valids = np.zeros(self.board_size * self.board_size, dtype=int)
        for idx in range(self.board_size * self.board_size):
            x, y = divmod(idx, self.board_size)
            if board[x][y] == 0:
                valids[idx] = 1
        return valids


