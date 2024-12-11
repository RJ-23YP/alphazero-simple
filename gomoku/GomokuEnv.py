import numpy as np

class GomokuEnv:
    def __init__(self, board_size=15, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1 # 1 is BLACK and 2 is WHITE
        self.done = False   
        self.winner = 0
        self.action_space = list(range(self.board_size * self.board_size))
        
    def get_action_space_size(self):
        return len(self.action_space)           
        
    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board

    def step(self, action):
        row, col = divmod(action, self.board_size)
        # may be no need
        if self.board[row, col] != 0:
            return self.board, float('-inf'), True
        
        self.board[row, col] = self.current_player
        reward, done = self.check_winner(row, col)
        
        if not done:
            self.current_player = 3 - self.current_player
        else:
            if reward != 0:
                self.winner = self.current_player
            else:
                self.winner = 0
        
        self.done = done
        return self.board, reward, done
    
    # Just a checker for one move.    
    def check_winner(self, row, col):
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            count = 1
            for step in range(1, self.win_length):
                r, c = row + dr * step, col + dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            for step in range(1, self.win_length):
                r, c = row - dr * step, col - dc * step
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                else:
                    break
            
            if count >= self.win_length:
                return 1, True
        
        if np.all(self.board != 0):
            return 0, True
        
        return 0, False
    
    def render(self):
        symbols = {0: '.', 1: 'X', 2: 'O'}
        for row in self.board:
            print(" ".join(symbols[cell] for cell in row))
        print("\n")
    
    def legal_moves(self):
        return [i for i in self.action_space if self.board[i // self.board_size, i % self.board_size] == 0]

    def board_legal_moves(self, board, player):
        # TODO: Consider banning moves?
        legal_moves = [i for i in self.action_space if board[i // len(board[0])][i % len(board[0])] == 0]
        return legal_moves

    
    def board_valid_moves(self, board, player):
        valids = np.zeros(len(board) * len(board[0]), dtype=int)
        legal_moves = self.board_legal_moves(board, player)
        assert len(legal_moves) != 0
        for move in legal_moves:
            valids[move] = 1
        return valids

    def pos_to_action(self, pos):
        x, y = pos
        action = x * self.board_size + y
        return action
    
    def get_canonical_form(self, board, player):
        if player == 1:
            return board
        else:
            return np.array([[3 - x if x != 0 else 0 for x in row] for row in board])
        
    def board_tostring(self, board):
        return board.tostring()

    def get_result(self, board):
        # should be canonicalBoard
        # if not terminate, 0
        # if draw, return very little number
        # and if player 1 is winner, return 1. Else -1
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
        next_board = np.copy(board)
        x, y = divmod(action, self.board_size)
        # assert board[x][y] == 0
        next_board[x][y] = player
        return next_board, -player
    



    def check_winner_player(self, board):
        # Board contains 0 for empty, 1 for player 1, -1 for player -1
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
        return 0  # No winner
    

    def get_result_player(self, board):
        result = self.check_winner_player(board)
        if result != 0:
            return result
        elif np.all(board != 0):
            return 1e-8  # Draw
        else:
            return 0  # Game not ended
        
    def board_valid_moves_player(self, board):
        valids = np.zeros(self.board_size * self.board_size, dtype=int)
        for idx in range(self.board_size * self.board_size):
            x, y = divmod(idx, self.board_size)
            if board[x][y] == 0:
                valids[idx] = 1
        return valids


