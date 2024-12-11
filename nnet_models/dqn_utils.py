### This code defines the prioritized replay buffer for the DQN agent. 


import random
import numpy as np

class ReplayBuffer:
    """
    A simple replay buffer for storing and sampling experience tuples
    (state, action, reward, next_state, done) for reinforcement learning.
    """
    def __init__(self, capacity=10000):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): The maximum number of experiences to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0 # Tracks the current position for overwriting old experiences

    def push(self, state, action, reward, next_state, done):
        """
        Store a new experience in the buffer.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state after taking the action.
            done: Whether the episode is finished.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Add a placeholder if buffer is not full
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Circular buffer logic

    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            A tuple containing batches of states, actions, rewards, next_states, and dones.
        """
        batch = random.sample(self.buffer, batch_size) # Randomly sample from the buffer
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: The number of experiences stored in the buffer.
        """
        return len(self.buffer)     
    

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    A prioritized replay buffer that samples experiences based on their TD-error priorities.
    Implements proportional prioritization with importance sampling corrections.
    """
    def __init__(self, capacity=10000, alpha=0.6):
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity (int): The maximum number of experiences to store.
            alpha (float): Degree of prioritization (0 = uniform sampling, 1 = full prioritization).
        """
        super().__init__(capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)  # Stores priorities for each experience
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done, td_error=1.0):
        """
        Store a new experience with a priority.

        Args:
            state: The current state.
            action: The action taken.
            reward: The reward received.
            next_state: The next state after taking the action.
            done: Whether the episode is finished.
            td_error (float): The temporal-difference error for prioritization.
        """
        max_priority = self.priorities.max() if self.buffer else td_error  # Use max priority for new experiences
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Add a placeholder if buffer is not full
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity  # Circular buffer logic

    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on priorities.

        Args:
            batch_size (int): The number of experiences to sample.
            beta (float): Importance sampling exponent (0 = no correction, 1 = full correction).

        Returns:
            A tuple containing batches of states, actions, rewards, next_states, dones,
            indices of sampled experiences, and their importance-sampling weights.
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty.")

        # Compute sampling probabilities proportional to priorities
        priorities = self.priorities[:len(self.buffer)]
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Retrieve the sampled experiences
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize weights

        return (
            np.array(states),
            actions,
            rewards,
            np.array(next_states),
            dones,
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        """
        Update priorities of sampled experiences based on new TD-errors.

        Args:
            indices (list[int]): Indices of the sampled experiences.
            td_errors (list[float]): The updated TD-errors for the sampled experiences.
        """
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha  # Update priority with smoothing factor


