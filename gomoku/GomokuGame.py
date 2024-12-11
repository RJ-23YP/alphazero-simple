#### THIS CODE CONTAINS THE FUNCTIONS REQUIRED TO VISUALIZE AND RUN THE GOMOKU ENVIRONMENT AND SELECT AI PLAYERS

import pygame
import pygame.gfxdraw
import sys
import os
from pygame.locals import QUIT, KEYDOWN, MOUSEBUTTONDOWN

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from GomokuGameVars import *
from GomokuEnv import GomokuEnv 
from GomokuAIPlayer import GreedyPlayer
from GomokuAIPlayer import PureMCTSPlayer
from GomokuAIPlayer import MCTSNNPlayer, DQNPlayer

# from gomoku.GomokuGame import GomokuGame as Game
from nnet_models.NNet import NNetWrapper as nn

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MCTS import MCTS

class GomokuGame:
    """ THIS CLASS CONTAINS ALL THE NECESSARY VARIABLES AND FUCNTIONS 
    FOR GOMOKU GAME PLAY"""
    def __init__(self, AI_type='GreedyPlayer'):
        ## Loading the Game parameters for visualziation 
        pygame.init()
        self.screen = pygame.display.set_mode(GameParam['WINDOW_SIZE'])
        self.screen_color = GameParam['BOARD_COLOR']
        self.board_line_color = GameParam["LINE_COLOR"]
        self.board_size = GameParam["BOARD_SIZE"]
        self.win_length = GameParam["WIN_LENGTH"]
        self.margin = GameParam['MARGIN']
        self.grid_size = (GameParam['PIXEL_SIZE'] - 2 * self.margin) // (self.board_size - 1)
        self.board_pixel_size = GameParam["PIXEL_SIZE"]
        self.color_black = GameParam['BLACK'] 
        self.color_white = GameParam['WHITE']

        self.game_status = GameStatus.Init
        self.human_color = 0  # only works in PvA mode 
        self.selected_mode = None

        ## Initializing the environment 
        self.env = GomokuEnv(board_size=self.board_size, win_length=self.win_length)
        # TODO
        ## AI Player Selection 
        if AI_type == 'GreedyPlayer':
            self.AI = GreedyPlayer(self.env) 
        elif AI_type == 'PureMCTSPlayer':
            self.AI = PureMCTSPlayer(self.env, MCTS(game=self.env, nnet=None, args=args))
        elif AI_type == 'MCTSNNPlayer':
            nnet = nn(self)  # Initialize the neural network wrapper 
            checkpoint_path = "/media/rj/New Volume/Northeastern University/Semester-3 (Fall 2024)/CS 5180 - RL/FInal Project/Submission/alphazero-simple/saved_models/TRAIN_50SP_10EPOCH_100SIM.pth.tar"
            # checkpoint_path = None 

            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                folder, filename = os.path.split(checkpoint_path)
                nnet.load_checkpoint(folder, filename)  
            else:
                print("Warning: No checkpoint specified; using randomly initialized network.")

            self.AI = MCTSNNPlayer(self.env, MCTS(game=self.env, nnet=nnet, args=args)) 

        elif AI_type == 'DQNPlayer':
            # Add logic for DQNPlayer initialization
            self.AI = DQNPlayer(self.env)  # Use default DQNPlayer configuration
            # Path to the DQN model checkpoint 
            checkpoint_path = "/media/rj/New Volume/Northeastern University/Semester-3 (Fall 2024)/CS 5180 - RL/FInal Project/Submission/alphazero-simple/saved_models/dqn_final_weights.pth"  # Update to your model path as needed

            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"Loading DQN model from {checkpoint_path}...")
                self.AI.load_model(checkpoint_path)  # Load the pre-trained DQN model
            else:
                print(f"No DQN model checkpoint found at {checkpoint_path}. Using randomly initialized model.") 

        else: 
            raise ValueError(f"Unsupported AI type: {AI_type}") 

    
            
    ### THE FOLLOWING FUCNTIONS FACILIATE THE VISUALIZATION OF THE GOMOKU GAME BOARD AS PER THE GIVEN BOARD SIZE 

    def draw_board(self):
        spacing = (self.board_pixel_size - 2 * self.margin) / (self.board_size - 1)
        
        for i in range(self.board_size):
            pos = self.margin + i * spacing

            # bound tougher
            if i == 0 or i == self.board_size - 1:
                pygame.draw.line(self.screen, self.board_line_color, [pos, self.margin], [pos, self.board_pixel_size - self.margin], 4)
                pygame.draw.line(self.screen, self.board_line_color, [self.margin, pos], [self.board_pixel_size - self.margin, pos], 4)
            else:
                pygame.draw.line(self.screen, self.board_line_color, [pos, self.margin], [pos, self.board_pixel_size - self.margin], 2)
                pygame.draw.line(self.screen, self.board_line_color, [self.margin, pos], [self.board_pixel_size - self.margin, pos], 2)

    def draw_special_points(self):
        offset = self.margin
        center = self.board_size // 2

        pygame.draw.circle(self.screen, GameParam['POINT_COLOR'], 
                        [offset + self.grid_size * center, offset + self.grid_size * center], 8, 0)

        margin = max(2, self.board_size // 4)
        corner_points = [
            (margin, margin),
            (margin, self.board_size - margin - 1),
            (self.board_size - margin - 1, margin),
            (self.board_size - margin - 1, self.board_size - margin - 1)
        ]
        
        for x, y in corner_points:
            pygame.draw.circle(self.screen, GameParam['POINT_COLOR'], 
                            [offset + self.grid_size * x, offset + self.grid_size * y], 5, 0)

    def draw_buttons(self):
        font = pygame.font.Font(None, 36)
        button_width, button_height = 160, 50
        button_spacing = 30
        button_offset = 20
        button_color = GameParam['BUTTON_COLOR']
        text_color = self.color_black
        
        pvp_button = pygame.Rect(self.board_pixel_size + button_offset, 100, button_width, button_height)
        pygame.draw.rect(self.screen, button_color, pvp_button, border_radius=10)
        pvp_text = font.render('PVP', True, text_color)
        self.screen.blit(pvp_text, (pvp_button.centerx - pvp_text.get_width() // 2, 
                            pvp_button.centery - pvp_text.get_height() // 2))
        
        pva_button = pygame.Rect(self.board_pixel_size + button_offset, 100 + button_height + button_spacing, button_width, button_height)
        pygame.draw.rect(self.screen, button_color, pva_button, border_radius=10)
        pva_text = font.render('PVA', True, text_color)
        self.screen.blit(pva_text, (pva_button.centerx - pva_text.get_width() // 2, 
                            pva_button.centery - pva_text.get_height() // 2))
        
        return pvp_button, pva_button

    def within_point(self, mouse_pos, grid_x, grid_y):
        return (abs(mouse_pos[0] - grid_x) < self.grid_size // 2 and 
                abs(mouse_pos[1] - grid_y) < self.grid_size // 2)

    def draw_focus_square(self, mouse_pos):
        offset = self.margin
        focus_square_size = self.grid_size * 0.65

        for i in range(self.board_size):
            for j in range(self.board_size):
                grid_x = offset + i * self.grid_size
                grid_y = offset + j * self.grid_size
                if self.within_point(mouse_pos, grid_x, grid_y):
                    pygame.draw.rect(self.screen, GameParam['FOCUS_COLOR'], 
                                    (grid_x - focus_square_size // 2, 
                                    grid_y - focus_square_size // 2, 
                                    focus_square_size, 
                                    focus_square_size))
                    return

    def draw_pieces(self, board):
        offset = self.margin

        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] == 1:
                    pygame.gfxdraw.aacircle(self.screen, 
                                            offset + i * self.grid_size, 
                                            offset + j * self.grid_size, 
                                            self.grid_size // 2 - 4, 
                                            self.color_black)
                    pygame.gfxdraw.filled_circle(self.screen, 
                                                offset + i * self.grid_size, 
                                                offset + j * self.grid_size, 
                                                self.grid_size // 2 - 4, 
                                                self.color_black)
                elif board[i, j] == 2:
                    pygame.gfxdraw.aacircle(self.screen, 
                                            offset + i * self.grid_size, 
                                            offset + j * self.grid_size, 
                                            self.grid_size // 2 - 4, 
                                            self.color_white)
                    pygame.gfxdraw.filled_circle(self.screen, 
                                                offset + i * self.grid_size, 
                                                offset + j * self.grid_size, 
                                                self.grid_size // 2 - 4, 
                                                self.color_white)

    def draw_current_player(self, current_player):
        font = pygame.font.Font(None, 40)
        current_player_text = ""
        if current_player == 1:
            current_player_text = "Black's turn"
        elif current_player == 2:
            current_player_text = "White's turn"

        if current_player_text:
            text_surface = font.render(current_player_text, True, self.color_black if current_player == 1 else self.color_white)
            text_x = self.board_pixel_size + 15
            text_y = 20
            self.screen.blit(text_surface, (text_x, text_y))

    def draw_result(self, env):
        font = pygame.font.Font(None, 48)
        winner_text = ""
        if env.winner == 1:
            winner_text = "Black wins!"
        elif env.winner == 2:
            winner_text = "White wins!"
        elif env.winner == 0:
            winner_text = "Draw!"

        if winner_text:
            text_surface = font.render(winner_text, True, self.color_black if env.winner == 1 else self.color_white)
            text_x = self.board_pixel_size
            text_y = 300
            self.screen.blit(text_surface, (text_x, text_y))
                
    def get_grid_position(self, mouse_pos):
        offset = self.margin
        for i in range(self.board_size):
            for j in range(self.board_size):
                grid_x = offset + i * self.grid_size
                grid_y = offset + j * self.grid_size
                if self.within_point(mouse_pos, grid_x, grid_y):
                    return (i, j)
        return None

    def handle_PvP_button(self):
        self.game_status = GameStatus.Start
        self.env.reset()
        
    def handle_PvA_button(self):
        font = pygame.font.Font(None, 36)
        dialog_width, dialog_height = 300, 150
        dialog_x = (self.screen.get_width() - dialog_width) // 2
        dialog_y = (self.screen.get_height() - dialog_height) // 2
        dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        dialog_color = (200, 200, 200)
        
        button_width, button_height = 120, 40
        button_spacing = 20
        first_button = pygame.Rect(dialog_x + button_spacing, dialog_y + dialog_height - button_height - 10, button_width, button_height)
        second_button = pygame.Rect(dialog_x + button_spacing + button_width + button_spacing, dialog_y + dialog_height - button_height - 10, button_width, button_height)
        
        pygame.draw.rect(self.screen, dialog_color, dialog_rect, border_radius=10)
        pygame.draw.rect(self.screen, GameParam['BUTTON_COLOR'], first_button, border_radius=5)
        pygame.draw.rect(self.screen, GameParam['BUTTON_COLOR'], second_button, border_radius=5)
        
        select_text = font.render("Select Player", True, self.color_black)
        self.screen.blit(select_text, (dialog_x + (dialog_width - select_text.get_width()) // 2, dialog_y + 20))
        first_text = font.render("Black", True, self.color_black)
        second_text = font.render("White", True, self.color_white)
        self.screen.blit(first_text, (first_button.centerx - first_text.get_width() // 2, first_button.centery - first_text.get_height() // 2))
        self.screen.blit(second_text, (second_button.centerx - second_text.get_width() // 2, second_button.centery - second_text.get_height() // 2))
    
        pygame.display.update()
    
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if first_button.collidepoint(event.pos):
                        self.game_status = GameStatus.Start
                        self.env.reset()
                        self.human_color = 1
                        return
                    elif second_button.collidepoint(event.pos):
                        self.game_status = GameStatus.Start
                        self.env.reset()
                        self.human_color = 2
                        return
    
    def check_valid_action(self, action):
        valid_moves = self.env.legal_moves()
        # print(action)
        if action in valid_moves:
            return True
        return False

    def try_place_chess(self, pos):
        env = self.env
        if self.game_status != GameStatus.Start:
            return
        if self.selected_mode == 'PvA' and env.current_player != self.human_color:
            return
        action = env.pos_to_action(pos)
        # print(action)
        if self.check_valid_action(action):
            env.step(action)
            # print(env.board)
            
    def AI_place_chess(self, pos):
        self.env.step(self.env.pos_to_action(pos))
    
    def run(self):
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.WINDOWFOCUSLOST:
                    pass
                if event.type in (QUIT, KEYDOWN):
                    sys.exit()
                elif event.type == MOUSEBUTTONDOWN:
                    if self.pvp_button.collidepoint(event.pos):
                        self.selected_mode = 'PvP'
                        self.handle_PvP_button()
                    elif self.pva_button.collidepoint(event.pos):
                        self.selected_mode = 'PvA'
                        self.handle_PvA_button()
                    else:
                        # For human move
                        grid_pos = self.get_grid_position(event.pos)
                        if grid_pos is not None:
                            self.try_place_chess(grid_pos)

            # Drawing updates
            self.screen.fill(self.screen_color)
            self.pvp_button, self.pva_button = self.draw_buttons()
            self.draw_board()
            self.draw_special_points()
            
            # Game End check
            if self.env.done:
                self.game_status = GameStatus.End
            
            if self.game_status == GameStatus.Start:
                self.draw_pieces(self.env.board)
                self.draw_current_player(self.env.current_player)
                
                # For human get focus
                mouse_pos = pygame.mouse.get_pos()
                if self.margin <= mouse_pos[0] <= self.board_pixel_size - self.margin and self.margin <= mouse_pos[1] <= self.board_pixel_size - self.margin:
                    self.draw_focus_square(mouse_pos)
                
                # For AI move
                if self.selected_mode == 'PvA' and self.env.current_player != self.human_color:
                    grid_pos = self.AI.get_move()
                    self.AI_place_chess(grid_pos)
            elif self.game_status == GameStatus.End:
                self.draw_pieces(self.env.board)
                self.draw_result(self.env)
            
            pygame.display.update()




 
    def getInitBoard(self):
        """
        Get the initial board state.

        Returns:
            np.array: A copy of the initial board state as a numpy array.
        """
        self.env.reset()  # Reset the environment to its initial state
        return np.copy(self.env.board)

    def getBoardSize(self):
        """
        Get the dimensions of the board.

        Returns:
            tuple: Dimensions of the board as (rows, columns).
        """
        return (self.board_size, self.board_size)

    def getActionSize(self):
        """
        Get the total number of possible actions.

        Returns:
            int: Total number of cells on the board.
        """
        return self.board_size * self.board_size

    def getNextState(self, board, player, action):
        """
        Get the next state after a given action.

        Args:
            board (np.array): Current board state.
            player (int): Current player (1 or -1).
            action (int): Linear index of the action.

        Returns:
            tuple: The next board state and the next player.
        """
        next_board = np.copy(board)  # Create a copy of the board to modify
        x, y = divmod(action, self.board_size)  # Convert action to 2D coordinates
        next_board[x][y] = player  # Place the player's piece on the board
        return (next_board, -player)  # Switch to the other player

    def getValidMoves(self, board, player):
        """
        Get a binary vector of valid moves for the current board state.

        Args:
            board (np.array): Current board state.
            player (int): Current player (1 or -1).

        Returns:
            np.array: Binary vector indicating valid moves.
        """
        valids = np.zeros(self.getActionSize(), dtype=int)  # Initialize vector
        for idx in range(self.getActionSize()):
            x, y = divmod(idx, self.board_size)  # Convert linear index to 2D coordinates
            if board[x][y] == 0:  # Check if the cell is empty
                valids[idx] = 1  # Mark as a valid move
        return valids 

    def getGameEnded(self, board, player):
        """
        Check if the game has ended.

        Args:
            board (np.array): Current board state.
            player (int): Current player (1 or -1).

        Returns:
            float: 0 if not ended, 1 if player 1 wins, -1 if player -1 wins,
                   a small value (1e-8) for a draw.
        """
        result = self.check_winner(board)  # Check for a winner
        if result == 0:  # No winner
            if np.all(board != 0):  # Check for a draw
                return 1e-8  # Small value for draw
            else:
                return 0  # Game not ended
        else:
            return result if result == player else -result  # Return result from the current player's perspective

    def check_winner(self, board):
        """
        Check for a winner in the current board state.

        Args:
            board (np.array): Current board state.

        Returns:
            int: 1 for player 1 win, -1 for player -1 win, 0 for no winner.
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Possible win directions
        for x in range(self.board_size):
            for y in range(self.board_size):
                player = board[x][y]
                if player != 0:  # Skip empty cells
                    for dx, dy in directions:
                        count = 1
                        nx, ny = x, y
                        while True:  # Check consecutive stones in one direction
                            nx += dx
                            ny += dy
                            if 0 <= nx < self.board_size and 0 <= ny < self.board_size and board[nx][ny] == player:
                                count += 1
                                if count == self.env.win_length:  # Win condition met
                                    return player
                            else:
                                break
        return 0  # No winner

    def getCanonicalForm(self, board, player):
        """
        Get the canonical form of the board from the current player's perspective.

        Args:
            board (np.array): Current board state.
            player (int): Current player (1 or -1).

        Returns:
            np.array: Canonical board state.
        """
        return board * player  # Flip perspective for player -1

    def getSymmetries(self, board, pi):
        """
        Generate symmetric board states and policies for data augmentation.

        Args:
            board (np.array): Current board state.
            pi (np.array): Policy vector for the current board.

        Returns:
            list: List of tuples, where each tuple contains a symmetric board and corresponding policy.
        """
        assert(len(pi) == self.getActionSize())  # Ensure policy vector matches board size
        pi_board = np.reshape(pi, (self.board_size, self.board_size))  # Reshape policy to board dimensions
        l = []

        for i in range(4):  # Rotate the board 0, 90, 180, and 270 degrees
            for j in [False, True]:  # Optionally flip the board horizontally
                newB = np.rot90(board, i)  # Rotate the board
                newPi = np.rot90(pi_board, i)  # Rotate the policy
                if j:
                    newB = np.fliplr(newB)  # Flip the board
                    newPi = np.fliplr(newPi)  # Flip the policy
                l.append((newB, list(newPi.ravel())))  # Flatten the policy and append
        return l
   
    
    def board_valid_moves(self, board, player):
        """
        Delegates the call to GomokuEnv's board_valid_moves method.
        """
        return self.env.board_valid_moves(board, player)

    def board_tostring(self, board):
        """
        Converts the board to a string representation.
        Delegates to the GomokuEnv's board_tostring method.
        """
        return self.env.board_tostring(board)
    
    def get_result(self, board):
        """
        Delegates the call to GomokuEnv's get_result method.
        """
        return self.env.get_result(board)
    
    def get_action_space_size(self):
        """
        Delegates the call to GomokuEnv's get_action_space_size method.
        """
        return self.env.get_action_space_size()
    
    def get_action_space_size(self):
        """
        Delegates the call to GomokuEnv's get_action_space_size method.
        """
        return self.env.get_action_space_size()
    
    def get_next_state(self, board, player, action):
        """
        Get the next board state and the next player after applying an action.

        Args:
            board (np.array): The current board state.
            player (int): The current player (1 or -1).
            action (int): The linear index of the action.

        Returns:
            tuple: Next board state and next player.
        """
        return self.env.get_next_state(board, player, action) 
    
    def get_canonical_form(self, board, player):
        """
        Convert the board to its canonical form, flipping perspective for player -1.

        Args:
            board (np.array): The current board state.
            player (int): The current player (1 or -1).

        Returns:
            np.array: Canonical board state.
        """
        return self.env.get_canonical_form(board, player)  

    
    @staticmethod
    def display(board):
        """
        Display the board in a human-readable format.

        Args:
            board (np.array): The current board state.
        """
        # Mapping board values to symbols for display
        symbols = {0: '.', 1: 'X', -1: 'O'}  # Empty: '.', Player 1: 'X', Player -1: 'O'

        # Print column indices
        for y in range(board.shape[1]):
            print("{0:2}".format(y), end="")
        print("")
        
        # Print each row with corresponding symbols
        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                piece = board[x][y]
                symbol = symbols.get(piece, '?')  # Default to '?' if symbol not found
                print("{0:2}".format(symbol), end="")
            print("")  # Newline after each row
        
        print("")  # Additional newline for spacing


def main():
    # game = GomokuGame('PureMCTSPlayer')
    # MCTSNNPlayer
    game = GomokuGame('MCTSNNPlayer')
    
    # game = GomokuGame('GreedyPlayer')
    game.run()

if __name__ == "__main__":
    main()