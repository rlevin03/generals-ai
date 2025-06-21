"""
Deep Q-Network (DQN) Agent for Generals Game

This module implements a DQN agent for playing the Generals game using PyTorch.
The agent uses a convolutional neural network with dueling architecture to learn
optimal strategies through reinforcement learning.

Most of this code is AI generated.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import List, Tuple, Optional
from generals_rl_env_gpu import GeneralsEnv
from generals import GRID_WIDTH, GRID_HEIGHT


# Constants
DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
PASS_ACTION = None


def action_to_coords(action: int) -> Tuple[int, int]:
    """
    Convert a flat action index to grid coordinates.
    
    Args:
        action (int): Flat action index
        
    Returns:
        Tuple[int, int]: (x, y) coordinates on the grid
    """
    idx = action // 4
    return idx % GRID_WIDTH, idx // GRID_WIDTH


def get_valid_actions(state_flat: np.ndarray, player_id: int) -> List[int]:
    """
    Get all valid actions for the current player based on the game state.
    
    Args:
        state_flat (np.ndarray): Flattened game state
        player_id (int): ID of the current player
        
    Returns:
        List[int]: List of valid action indices
    """
    valid = []
    state = state_flat.reshape(GRID_HEIGHT, GRID_WIDTH, 4)
    
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            owner, army, _, _ = state[y, x]
            # Check if this cell belongs to the player and has enough army
            if int(owner) == player_id and army >= 2:
                # Check all four directions
                for d, (dx, dy) in enumerate(DIRS):
                    nx, ny = x + dx, y + dy
                    # Ensure the target is within grid bounds
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        valid.append((y * GRID_WIDTH + x) * 4 + d)
    return valid


class QNetwork(nn.Module):
    """
    Deep Q-Network with dueling architecture for the Generals game.
    
    This network uses convolutional layers to process the game state and
    dueling streams to separate state value and action advantages.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
        """
        super(QNetwork, self).__init__()
        
        # Convolutional layers for spatial feature extraction
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Fully connected layers for feature processing
        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 25, 512),  # Match environment dimensions
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
        
        # Dueling streams: value and advantage
        self.value_stream = nn.Linear(256, 1)      # State value
        self.advantage_stream = nn.Linear(256, action_dim)  # Action advantages
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        # Reshape input if needed (from flattened to 2D)
        if len(x.shape) == 2:
            x = x.view(-1, 6, 20, 25)  # Match environment dimensions
        
        # Convolutional feature extraction
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected feature processing
        x = self.fc(x)
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    This buffer stores (state, action, reward, next_state, done) tuples
    and provides random sampling for training.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity (int): Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                              torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): Number of transitions to sample
            
        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        """Return the current number of transitions in the buffer."""
        return len(self.buffer)


def select_action(q_net: QNetwork, state: torch.Tensor, epsilon: float, player_id: int) -> Optional[int]:
    """
    Select an action using epsilon-greedy policy.
    
    Args:
        q_net (QNetwork): The Q-network for action selection
        state (torch.Tensor): Current game state
        epsilon (float): Exploration rate
        player_id (int): ID of the current player
        
    Returns:
        Optional[int]: Selected action index, or None if no valid actions
    """
    valid_actions = get_valid_actions(state.numpy(), player_id)
    
    if not valid_actions:
        return PASS_ACTION

    # Epsilon-greedy action selection
    if random.random() < epsilon:
        # Exploration: random action
        return random.choice(valid_actions)
    else:
        # Exploitation: best action according to Q-network
        with torch.no_grad():
            q_values = q_net(state.unsqueeze(0)).squeeze(0)
            # Mask invalid actions with negative infinity
            mask = torch.full_like(q_values, float('-inf'))
            mask[valid_actions] = 0.0
            q_masked = q_values + mask
            return int(q_masked.argmax().item())


def train_dqn(env: GeneralsEnv, num_episodes: int = 1000, batch_size: int = 64,
              gamma: float = 0.99, lr: float = 1e-4, buffer_capacity: int = 100000,
              min_buffer_size: int = 1000, target_update_freq: int = 1000,
              epsilon_start: float = 1.0, epsilon_end: float = 0.1,
              epsilon_decay: float = 1e-5) -> None:
    """
    Train a DQN agent on the Generals environment.
    
    Args:
        env (GeneralsEnv): The game environment
        num_episodes (int): Number of training episodes
        batch_size (int): Batch size for training
        gamma (float): Discount factor for future rewards
        lr (float): Learning rate for the optimizer
        buffer_capacity (int): Maximum size of replay buffer
        min_buffer_size (int): Minimum buffer size before training starts
        target_update_freq (int): Frequency of target network updates
        epsilon_start (float): Initial exploration rate
        epsilon_end (float): Final exploration rate
        epsilon_decay (float): Rate of epsilon decay
    """
    # Initialize environment and get dimensions
    initial_state = env.reset()
    state_dim = len(initial_state)
    action_dim = env.action_space.n

    # Initialize networks and optimizer
    online_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(online_network.state_dict())
    optimizer = optim.Adam(online_network.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)

    # Training parameters
    epsilon = epsilon_start
    total_steps = 0

    print(f"Starting DQN training for {num_episodes} episodes...")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    for episode in range(num_episodes):
        # Reset environment for new episode
        raw_state = env.reset()
        state = torch.tensor(raw_state, dtype=torch.float32)
        last_base_reward = env.compute_reward()

        episode_reward = 0.0
        step_count = 0

        while True:
            # Select action using epsilon-greedy policy
            action = select_action(online_network, state, epsilon, env.player_id)

            if action is PASS_ACTION:
                # No valid actions, pass turn
                env.game.update()
                next_raw_state = env.extract_state()
                done = env.game.game_over
                base_reward = env.compute_reward()
                reward = base_reward - last_base_reward
                last_base_reward = base_reward
            else:
                # Execute action
                from_x, from_y = action_to_coords(action)
                dx, dy = DIRS[action % 4]
                to_x, to_y = from_x + dx, from_y + dy
                
                print(f"Step {step_count}: Move from ({from_x},{from_y}) → ({to_x},{to_y})")
                
                next_raw_state, _, done, _ = env.step(action)
                base_reward = env.compute_reward()
                reward = base_reward - last_base_reward
                last_base_reward = base_reward
                episode_reward += reward
                step_count += 1

            next_state = torch.tensor(next_raw_state, dtype=torch.float32)

            # Store transition in replay buffer (only for valid actions)
            if action is not PASS_ACTION:
                replay_buffer.push(state.numpy(), action, reward, next_state.numpy(), done)

            state, raw_state = next_state, next_raw_state

            # Train the network if buffer has enough samples
            if len(replay_buffer) >= min_buffer_size:
                # Sample batch from replay buffer
                states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                    replay_buffer.sample(batch_size)
                
                # Compute current Q-values
                current_q_values = online_network(states_batch).gather(1, actions_batch.unsqueeze(1)).squeeze(1)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_network(next_states_batch).max(1)[0]
                    target_q_values = rewards_batch + gamma * next_q_values * (1 - dones_batch)
                
                # Compute loss and update network
                loss = nn.MSELoss()(current_q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update target network periodically
                if total_steps % target_update_freq == 0:
                    target_network.load_state_dict(online_network.state_dict())

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon - epsilon_decay)
            total_steps += 1

            if done:
                break

        # Episode summary
        if episode % 100 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, "
                  f"Epsilon: {epsilon:.3f}, Steps: {step_count}")


def main():
    """Main function to run DQN training."""
    print("Initializing Generals environment and DQN agent...")
    env = GeneralsEnv(player_id=0)
    train_dqn(env)
    print("Training completed!")


if __name__ == "__main__":
    main()

