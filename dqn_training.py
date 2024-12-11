### This is the code to start the training process for the DQN agent. 


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Add tqdm for progress tracking
import random
import torch 
from gomoku.GomokuEnv import GomokuEnv
from gomoku.GomokuAIPlayer import DQNPlayer, GreedyPlayer, RandomPlayer

def plot_rewards(episode_rewards, total_episodes, plot_file):
    """
    Plots the rewards obtained during training over episodes.
    Includes a moving average to smooth out fluctuations.
    """
    # Smooth rewards
    window = 20
    if total_episodes >= window:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window) / window, mode="full")
        smoothed_rewards = smoothed_rewards[:total_episodes]
    else:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(total_episodes) / total_episodes, mode="full")
        smoothed_rewards = smoothed_rewards[:total_episodes]

    plt.figure(figsize=(10, 6))
    plt.plot(range(total_episodes), episode_rewards, label="Episode Reward", marker='o')
    plt.plot(range(total_episodes), smoothed_rewards, label=f"Moving Avg(20)", marker='o')

    plt.xlabel("Episode") 
    plt.ylabel("Reward")
    plt.title("DQN Training Performance (All Iterations)")
    plt.legend()
    plt.grid()
    plt.savefig(plot_file)
    plt.show()


def plot_win_rates(iteration_points, black_win_rates, white_win_rates, plot_file="win_rates.png"):
    """
    Plots the win rates of the DQN player (Black) and opponent (White)
    over training iterations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iteration_points, black_win_rates, label="Black (DQN) Win Rate", marker='o')
    plt.plot(iteration_points, white_win_rates, label="White (Opponent) Win Rate", marker='o')

    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Cumulative Win Rates over Iterations")
    plt.ylim([0, 1]) 
    plt.legend()
    plt.grid()
    plt.savefig(plot_file)
    plt.show()


def get_opponent_for_iteration(i, env):
    """
    Dynamically selects an opponent for each iteration.
    Uses modular arithmetic to alternate between RandomPlayer,
    GreedyPlayer, and DQNPlayer as opponents.
    """
    mod_val = i % 3 ### Use this to include Greedy Player
    # mod_val = i % 2  ### Uncomment this and comment above line if no greedy player is required for training. 
    if mod_val == 1:
        return RandomPlayer(env)
    # else: 
    #     return DQNPlayer(env) #### Uncomment this and comment below 2 statements if no greedy player is required for training.
    elif mod_val == 2:
        return DQNPlayer(env)
    else:
        return GreedyPlayer(env) 


def run_dqn_training(num_iterations=30, episodes_per_iteration=50, plot_file="dqn_performance.png"):
    """
    The main DQN training loop. Alternates between self-play with opponents
    and updates the DQN model. Tracks rewards and win rates for analysis.

    Args:
        num_iterations: Number of iterations (epochs) for training.
        episodes_per_iteration: Number of games played per iteration.
        plot_file: Path to save the reward plot.
    """
    # Initialize the Gomoku environment and DQN player
    env = GomokuEnv(board_size=15, win_length=5)
    dqn_player = DQNPlayer(env)
    
    # Load pre-trained weights if available
    weights_path = "dqn_final_weights.pth"
    if os.path.exists(weights_path):
        print(f"Pre-trained weights found at {weights_path}. Loading model...")
        dqn_player.load_model(weights_path)
    else:
        print("No pre-trained weights found. Starting training from scratch.")

    total_episodes = num_iterations * episodes_per_iteration
    episode_rewards = []  # Tracks rewards for each episode
    win_counts = {0: 0, 1: 0, 2: 0}  # Tracks wins/draws for each player

    # Initialize arrays for win rate analysis
    iteration_points = []
    black_win_rates = []
    white_win_rates = []

    # Main training loop
    for i in range(1, num_iterations + 1):
        # Select the opponent dynamically
        opponent = get_opponent_for_iteration(i, env)

        # Temporary win counts for the current iteration
        win_counts_iteration = {0: 0, 1: 0, 2: 0}

        # Track episodes in the current iteration with tqdm
        iteration_progress = tqdm(range(episodes_per_iteration), 
                                  desc=f"Iteration {i}/{num_iterations} - Episodes",
                                  unit="episode")

        for ep in iteration_progress:

            # Reset the environment and initialize the players
            env.reset()
            current_player_idx = 0
            players = [dqn_player, opponent] 
            # # Comment above line and uncomment below line to Assign players: opponent (black) and DQN (white). 
            # players = [dqn_player, opponent] 

            # Reset the DQN player's state-action memory
            dqn_player.last_state = None
            dqn_player.last_action = None

            # Play the game until it ends
            while not env.done:
                players[current_player_idx].move()
                current_player_idx = 1 - current_player_idx  # Alternate turns

            # Calculate rewards based on the game outcome
            result = env.winner
            if result == 1:
                ep_reward = 1.0  # DQN win
            elif result == 2:
                ep_reward = -1.0  # Opponent win
            else:
                ep_reward = 0.0  # Draw

            # Update rewards and win counts
            episode_rewards.append(ep_reward)
            win_counts[result] += 1
            win_counts_iteration[result] += 1  # Update the iteration-specific win counts

        # Print iteration-specific results
        print(f"Iteration {i}/{num_iterations}: "
            f"Black Wins={win_counts_iteration[1]}, "
            f"White Wins={win_counts_iteration[2]}, "
            f"Draws={win_counts_iteration[0]}") 
        
        # Save a checkpoint of the DQN model after each iteration
        dqn_checkpoint_path = f"dqn_checkpoint_iter_{i}.pth"
        dqn_player.save_model(dqn_checkpoint_path)
        print(f"Model saved at {dqn_checkpoint_path}")

        # Calculate cumulative win rates
        black_wins_so_far = win_counts[1]
        white_wins_so_far = win_counts[2]
        draws_so_far = win_counts[0]
        total_episodes_so_far = i * episodes_per_iteration

        print(f"Iteration {i}/{num_iterations}, Episodes so far: {total_episodes_so_far}, "
              f"Wins(Black={black_wins_so_far},White={white_wins_so_far},Draws={draws_so_far})")

        # Calculate non-draw win rates
        total_non_draw = black_wins_so_far + white_wins_so_far
        if total_non_draw > 0:
            black_rate = black_wins_so_far / total_non_draw
            white_rate = white_wins_so_far / total_non_draw
        else:
            black_rate = 0.0
            white_rate = 0.0 

        # Update win rate tracking arrays
        iteration_points.append(total_episodes_so_far)
        black_win_rates.append(black_rate)
        white_win_rates.append(white_rate)

    # Save the final model
    final_model_path = "dqn_final_weights.pth"
    dqn_player.save_model(final_model_path)
    print(f"Final model saved: {final_model_path}")

    # Print overall win rates
    win_rate_black = win_counts[1] / total_episodes
    win_rate_white = win_counts[2] / total_episodes
    print(f"Final Overall Win Rate (Black/DQN): {win_rate_black * 100:.2f}%")
    print(f"Final Overall Win Rate (White/Opponent): {win_rate_white * 100:.2f}%")
    ## Uncomment below 2 lines and comment above 2 lines if DQN is second player
    # print(f"Final Overall Win Rate (Black/Opponent): {win_rate_black * 100:.2f}%") 
    # print(f"Final Overall Win Rate (White/DQN): {win_rate_white * 100:.2f}%")

    # Plot rewards and win rates
    plot_rewards(episode_rewards, total_episodes, plot_file)
    plot_win_rates(iteration_points, black_win_rates, white_win_rates, "win_rates.png") 


if __name__ == "__main__": 
    # Run the training loop with specified parameters
    run_dqn_training(num_iterations=10, episodes_per_iteration=10, plot_file="dqn_performance.png") 