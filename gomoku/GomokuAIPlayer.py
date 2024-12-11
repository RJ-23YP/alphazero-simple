import numpy as np
import math
import random

import torch
import torch.optim as optim
import torch.nn.functional as F
from nnet_models.DQNNet import DQNNet
from nnet_models.dqn_utils import * 


class Player:
    def get_move(self):
        raise NotImplementedError("Subclasses should implement get_move!")
    
    def move(self):
        raise NotImplementedError("Subclasses should implement move!")

class GreedyPlayer(Player):
    def __init__(self, env):
        self.env = env

    def get_move(self):
        
        board = self.env.board
        current_player = self.env.current_player
        opponent = 1 if current_player == 2 else 2
        board_size = self.env.board_size
        legal_moves = self.env.legal_moves()

        def calculate_score(x, y, player):
            score = 0
            directions = [(1, 0), (0, 1), (1, 1), (1, -1),
                        (-1, 0), (0, -1), (-1, -1), (-1, 1)]
    
            for dx, dy in directions:
                player_count = 0
                opponent_count = 0

                nx, ny = x + dx, y + dy
                cur_player = board[nx, ny] if 0 <= nx < board_size and 0 <= ny < board_size else 0
                
                if cur_player != 0:
                    while 0 <= nx < board_size and 0 <= ny < board_size:
                        if board[nx, ny] == cur_player:
                            if cur_player == player:
                                player_count += 1
                            else:
                                opponent_count += 1
                        else:
                            break
                        nx += dx
                        ny += dy
                score += player_count ** 2.5
                score += opponent_count ** 2.5

            centre_x, centre_y = self.env.board_size // 2, self.env.board_size // 2
            score -= math.sqrt(abs(x - centre_x) ** 2 + abs(y - centre_y) ** 2) / (self.env.board_size)

            return score

        best_move = None
        max_score = -float('inf')

        for move in legal_moves:
            x, y = move // board_size, move % board_size
            score = calculate_score(x, y, current_player)
            if score > max_score:
                max_score = score
                best_move = move

        return divmod(best_move, board_size)

    def move(self):
        x, y = self.get_move()
        self.env.step(self.env.pos_to_action((x, y)))
    
class PureMCTSPlayer(Player):
    def __init__(self, env, mcts, num_train_steps=5):
        self.env = env
        self.mcts = mcts
        self.num_train_steps = num_train_steps
        
        board = self.env.board
        current_player = self.env.current_player
        canonical_board = self.env.get_canonical_form(board, current_player)
        for _ in range(self.num_train_steps):
            self.mcts.getActionProb(canonical_board, temperature=1)

    def get_move(self):
        board = self.env.board
        current_player = self.env.current_player
        canonical_board = self.env.get_canonical_form(board, current_player)
        
        # print(len(self.mcts.Qsa))
        for _ in range(self.num_train_steps):
            self.mcts.getActionProb(canonical_board, temperature=1)
        # print(canonical_board)
        action_probs = self.mcts.getActionProb(canonical_board, temperature=0)  # temperature=0 for greedy move
        best_action = np.argmax(action_probs)
        # print(best_action)
        return divmod(best_action, self.env.board_size)
    
    def move(self):
        x, y = self.get_move()
        self.env.step(self.env.pos_to_action((x, y)))
        
class MCTSNNPlayer(Player):
    def __init__(self, env, mcts, num_train_steps=5):
        self.env = env
        self.mcts = mcts
        self.num_train_steps = num_train_steps
        
        board = self.env.board
        current_player = self.env.current_player
        canonical_board = self.env.get_canonical_form(board, current_player)
        for _ in range(self.num_train_steps):
            self.mcts.getActionProb(canonical_board, temperature=1)

    def get_move(self):
        board = self.env.board
        current_player = self.env.current_player
        canonical_board = self.env.get_canonical_form(board, current_player)
        
        for _ in range(self.num_train_steps):
            self.mcts.getActionProb(canonical_board, temperature=1)
        action_probs = self.mcts.getActionProb(canonical_board, temperature=0)  # temperature=0 for greedy move
        best_action = np.argmax(action_probs)
        return divmod(best_action, self.env.board_size)
    
    def move(self):
        x, y = self.get_move()
        self.env.step(self.env.pos_to_action((x, y)))

    


class DQNPlayer(Player):
    def __init__(self, env, gamma=0.99, lr=5e-5, batch_size=256, buffer_size=50000, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995, target_update=1000):
        self.env = env
        self.board_size = env.board_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update 
        self.train_steps = 0

        self.policy_net = DQNNet(board_size=self.board_size)
        self.target_net = DQNNet(board_size=self.board_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size) 

        # Track last state-action for training
        self.last_state = None
        self.last_action = None

    def get_state_tensor(self, board):
        # board: (board_size, board_size)
        # Convert to float, add batch dim and channel dim
        state = torch.FloatTensor(board.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        return state

    def select_action(self, board, legal_moves):
        # Epsilon-greedy selection
        if np.random.rand() < self.epsilon:
            # random legal action
            return np.random.choice(legal_moves)
        else:
            state = self.get_state_tensor(board)
            with torch.no_grad():
                q_values = self.policy_net(state)
                # Mask illegal moves by setting them to -inf
                q_values = q_values.detach().cpu().numpy()[0]
                mask = np.ones_like(q_values, dtype=bool)
                mask[legal_moves] = False
                q_values[mask] = -np.inf
                best_action = np.argmax(q_values) 
                return best_action

    def get_move(self):
        # Return x,y for the chosen action
        legal_moves = self.env.legal_moves()
        board = self.env.board
        action = self.select_action(board, legal_moves)
        return divmod(action, self.board_size)

    def move(self):
        # Perform selected action in env and store transition
        current_board = np.copy(self.env.board)
        action_pos = self.get_move()
        action = self.env.pos_to_action(action_pos)

        # Step the environment
        next_board, reward, done = self.env.step(action) 

        # Store the transition in replay buffer
        if self.last_state is not None and self.last_action is not None:
            self.replay_buffer.push(self.last_state, self.last_action, reward, next_board, done)

        # Update last_state and last_action
        self.last_state = current_board
        self.last_action = action

        # After each move, train the DQN if we have enough samples
        if len(self.replay_buffer) > self.batch_size:
            self.train_dqn()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Update target network
        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def train_dqn(self):
        states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).unsqueeze(1)  # [batch,1,board_size,board_size]
        next_states_t = torch.FloatTensor(next_states).unsqueeze(1)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards)
        dones_t = torch.FloatTensor(np.array(dones, dtype=float))  # Convert tuple to NumPy array and then to Tensor
        weights_t = torch.FloatTensor(weights)  # Importance-sampling weights 

        # Compute Q(s,a)
        q_values = self.policy_net(states_t)
        q_values = q_values.gather(1, actions_t).squeeze(1)

        # Compute Q target
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t)
            max_next_q_values, _ = next_q_values.max(dim=1)
            target = rewards_t + (1 - dones_t) * self.gamma * max_next_q_values

        # Compute loss with importance-sampling weights
        td_errors = target - q_values
        loss = (weights_t * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping here
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)

        self.optimizer.step()

        # Update priorities in the replay buffer
        td_errors_np = td_errors.detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors_np) 

    
    def save_model(self, filepath):
        """Save the Q-network weights."""
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved at {filepath}")


    def load_model(self, filepath):
        """Load Q-network weights from file."""
        self.policy_net.load_state_dict(torch.load(filepath))
        self.policy_net.eval()  # Set to evaluation mode 
        print(f"Model loaded from {filepath}")



class RandomPlayer(Player):
    def __init__(self, env):
        self.env = env

    def get_move(self):
        legal_moves = self.env.legal_moves()
        return random.choice(legal_moves)   

    def move(self):
        action = self.get_move()
        self.env.step(action) 