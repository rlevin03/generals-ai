"""
GPU-accelerated Deep Q-Network (DQN) training for the Generals game.

This module implements an optimized DQN agent with parallel environment training,
prioritized experience replay, and self-play capabilities for the Generals game.
The agent uses convolutional neural networks and is designed to run efficiently
on GPU hardware.

Key Features:
- Parallel environment training for faster data collection
- Prioritized experience replay for better sample efficiency
- Reduced action space for improved training stability
- Self-play training with historical model opponents
- Comprehensive training statistics and visualization
- Multi-GPU support with DataParallel

Most of this code is AI generated.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, defaultdict
import random
import time
from torch.utils.tensorboard import SummaryWriter
from generals_rl_env_gpu import GeneralsEnv
from generals import CellType
import multiprocessing as mp
from torch.nn.parallel import DataParallel
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")

class ConvDQN(nn.Module):
    """
    Enhanced Convolutional Deep Q-Network with reduced action space support.
    
    This network uses a dueling architecture with separate value and advantage streams
    to improve training stability and performance. It supports both full and reduced
    action spaces for efficient training.
    
    Args:
        input_shape (tuple): Shape of input observations (height, width, channels)
        n_actions (int): Number of possible actions
        use_reduced_actions (bool): Whether to use reduced action space for efficiency
    """
    
    def __init__(self, input_shape, n_actions, use_reduced_actions=True):
        super(ConvDQN, self).__init__()
        
        h, w, c = input_shape
        self.use_reduced_actions = use_reduced_actions
        
        # Convolutional feature extraction layers with batch normalization
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)  # Reduce spatial dimensions by half
        )
        
        # Calculate output size after convolutions
        conv_h = h // 2
        conv_w = w // 2
        conv_out_size = 128 * conv_h * conv_w
        
        # Fully connected layers for feature processing
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Regularization
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Dueling DQN architecture: separate value and advantage streams
        self.value_stream = nn.Linear(256, 1)  # State value function
        
        if use_reduced_actions:
            # Output Q-values for each cell and each direction (4 directions)
            # This allows dynamic action space based on valid moves
            self.advantage_stream = nn.Linear(256, h * w * 4)
        else:
            # Standard fixed action space
            self.advantage_stream = nn.Linear(256, n_actions)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights using Xavier uniform initialization."""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, valid_actions_mask=None):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input observations of shape (batch_size, h, w, c)
            valid_actions_mask (torch.Tensor, optional): Boolean mask for valid actions
            
        Returns:
            torch.Tensor: Q-values for all actions
        """
        batch_size = x.size(0)
        
        # Convert from (batch, h, w, c) to (batch, c, h, w) for convolutions
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine value and advantage streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Apply valid actions mask if provided (set invalid actions to -inf)
        if valid_actions_mask is not None:
            q_values = q_values.masked_fill(~valid_actions_mask, float('-inf'))
        
        return q_values

class OptimizedGeneralsEnv(GeneralsEnv):
    """
    Optimized environment wrapper with reduced action space for efficient training.
    
    This class extends the base GeneralsEnv to provide a more efficient action space
    by only considering valid moves rather than all possible moves. This significantly
    reduces the action space size and improves training efficiency.
    
    Attributes:
        valid_source_cells (list): List of cells that can make moves
        action_mapping (dict): Mapping from action indices to actual moves
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.valid_source_cells = []
        self.action_mapping = {}
        
    def get_valid_source_cells(self):
        """
        Get cells that can make moves (have at least 1 army).
        
        Returns:
            list: List of (x, y) coordinates of cells that can make moves
        """
        valid_sources = []
        owned_cells = 0
        cells_with_armies = 0
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner == self.player_id:
                    owned_cells += 1
                    if cell.army >= 1:  # Allow moves with 1 army
                        cells_with_armies += 1
                        valid_sources.append((x, y))
        
        # Debug output for troubleshooting
        if len(valid_sources) == 0 and owned_cells > 0:
            print(f"Debug: Player {self.player_id} owns {owned_cells} cells, "
                  f"{cells_with_armies} have >=1 army, but no valid moves!")
            if cells_with_armies > 0:
                # Check why no valid moves exist
                for y in range(self.grid_height):
                    for x in range(self.grid_width):
                        cell = self.game.grid[y][x]
                        if cell.owner == self.player_id and cell.army >= 1:
                            print(f"  Cell ({x},{y}) has {cell.army} armies")
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                    to_cell = self.game.grid[ny][nx]
                                    print(f"    -> ({nx},{ny}): type={to_cell.type}, "
                                          f"visible={self.player_id in to_cell.visible_to}")
        
        return valid_sources
    
    def get_reduced_action_space(self):
        """
        Get only valid actions, not all possible actions.
        
        This method creates a reduced action space by only including moves that
        are actually possible given the current game state.
        
        Returns:
            tuple: (list of valid action indices, dict mapping indices to moves)
        """
        actions = []
        self.action_mapping.clear()
        
        valid_sources = self.get_valid_source_cells()
        
        action_idx = 0
        for x, y in valid_sources:
            # Check all four directions: up, down, left, right
            for direction, (dx, dy) in enumerate([(0, -1), (0, 1), (-1, 0), (1, 0)]):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    to_cell = self.game.grid[ny][nx]
                    # Allow moves into fog or non-mountain cells
                    if (self.player_id not in to_cell.visible_to or 
                        to_cell.type != CellType.MOUNTAIN):
                        self.action_mapping[action_idx] = (x, y, nx, ny)
                        actions.append(action_idx)
                        action_idx += 1
        
        return actions, self.action_mapping
    
    def step_with_reduced_action(self, action_idx):
        """
        Execute action using reduced action space.
        
        Args:
            action_idx (int): Index of action in the reduced action space
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Debug: Check army count before step
        if hasattr(self, 'game') and hasattr(self.game, 'grid'):
            general_pos = self.game.players[self.player_id].general_pos
            if general_pos:
                gx, gy = general_pos
                general_cell = self.game.grid[gy][gx]
                if general_cell.army <= 1 and self.step_count % 100 == 0:
                    print(f"Debug: General at ({gx},{gy}) has {general_cell.army} armies")
        
        # Special case: if no valid actions, just wait (no-op)
        if not self.action_mapping:
            # Important: We still need to call the parent step to update the game!
            # Use a dummy action that won't do anything
            dummy_action = 0  # This will likely be invalid but that's ok
            result = self.step(dummy_action)
            # Override the invalid move penalty if it was just because no valid moves
            obs, reward, done, info = result
            if 'invalid_move' in info and info['invalid_move']:
                reward = -0.01  # Just time penalty, not invalid move penalty
                info['no_valid_actions'] = True
                info['invalid_move'] = False
            return obs, reward, done, info
        
        # Check if action is valid
        if action_idx not in self.action_mapping:
            return self._get_observation(), self.invalid_move_penalty, False, {"invalid_move": True}
        
        # Extract move coordinates from action mapping
        from_x, from_y, to_x, to_y = self.action_mapping[action_idx]
        
        # Encode as original action for parent class
        original_action = from_y * self.grid_width * 4 + from_x * 4
        # Add direction
        if to_y < from_y:
            original_action += 0  # Up
        elif to_y > from_y:
            original_action += 1  # Down
        elif to_x < from_x:
            original_action += 2  # Left
        else:
            original_action += 3  # Right
        
        return self.step(original_action)

class ParallelEnvPool:
    """
    Pool of parallel environments for faster data collection.
    
    This class manages multiple environment instances to collect experience
    in parallel, significantly speeding up the training process.
    
    Args:
        num_envs (int): Number of parallel environments
        env_params (dict): Parameters to initialize each environment
    """
    
    def __init__(self, num_envs, env_params):
        self.num_envs = num_envs
        self.env_params = env_params
        self.envs = [OptimizedGeneralsEnv(**env_params) for _ in range(num_envs)]
        self.dones = [False] * num_envs
        
    def reset(self, env_idx=None):
        """
        Reset specific environment or all environments.
        
        Args:
            env_idx (int, optional): Index of specific environment to reset.
                                   If None, resets all environments.
        
        Returns:
            list or object: Observation(s) from reset environment(s)
        """
        if env_idx is not None:
            obs = self.envs[env_idx].reset()
            self.dones[env_idx] = False
            return obs
        else:
            observations = []
            for i in range(self.num_envs):
                obs = self.envs[i].reset()
                self.dones[i] = False
                observations.append(obs)
            return observations
    
    def step(self, actions):
        """
        Step all environments in parallel.
        
        Args:
            actions (list): List of actions to take in each environment
            
        Returns:
            tuple: (observations, rewards, dones, infos) for all environments
        """
        observations = []
        rewards = []
        dones = []
        infos = []
        
        for i, action in enumerate(actions):
            if self.dones[i]:
                # Reset done environments automatically
                obs = self.reset(i)
                observations.append(obs)
                rewards.append(0)
                dones.append(False)
                infos.append({})
            else:
                obs, reward, done, info = self.envs[i].step_with_reduced_action(action)
                observations.append(obs)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                self.dones[i] = done
        
        return observations, rewards, dones, infos
    
    def get_reduced_actions(self):
        """
        Get reduced action spaces for all environments.
        
        Returns:
            tuple: (list of action lists, list of action mappings) for all environments
        """
        all_actions = []
        all_mappings = []
        
        for env in self.envs:
            actions, mapping = env.get_reduced_action_space()
            all_actions.append(actions)
            all_mappings.append(mapping)
        
        return all_actions, all_mappings

class PrioritizedReplayBuffer:
    """
    Enhanced replay buffer with prioritized experience replay.
    
    This replay buffer implements prioritized experience replay to improve
    sample efficiency by sampling transitions with higher TD-error more frequently.
    
    Args:
        capacity (int): Maximum number of transitions to store
        alpha (float): Priority exponent (0 = uniform sampling, 1 = pure priority)
    """
    
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.stats = defaultdict(list)
        
    def push_batch(self, transitions):
        """
        Push multiple transitions at once for efficiency.
        
        Args:
            transitions (list): List of (state, action, reward, next_state, done, info) tuples
        """
        for state, action, reward, next_state, done, info in transitions:
            self.push(state, action, reward, next_state, done, info)
    
    def push(self, state, action, reward, next_state, done, info=None):
        """
        Add a single transition to the replay buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode ended
            info: Additional information (optional)
        """
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done, info))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done, info)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
        
        # Track statistics for monitoring
        self.stats['rewards'].append(reward)
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.stats[key].append(value)
    
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of transitions with importance sampling weights.
        
        Args:
            batch_size (int): Number of transitions to sample
            beta (float): Importance sampling exponent for bias correction
            
        Returns:
            tuple: (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # Convert priorities to probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights for bias correction
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        # Unpack samples into tensors
        batch = list(zip(*samples))
        states = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=device)
        actions = torch.tensor(batch[1], dtype=torch.long, device=device)
        rewards = torch.tensor(batch[2], dtype=torch.float32, device=device)
        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=device)
        dones = torch.tensor(batch[4], dtype=torch.float32, device=device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices (list): Indices of transitions to update
            priorities (list): New priority values
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6  # Small constant to avoid zero priority
    
    def get_stats_summary(self):
        """
        Get summary statistics from buffer for monitoring.
        
        Returns:
            dict: Summary statistics for recent transitions
        """
        summary = {}
        for key, values in self.stats.items():
            if values and len(values) > 0:
                recent = values[-1000:] if len(values) > 1000 else values
                summary[f"{key}_mean"] = np.mean(recent)
                summary[f"{key}_std"] = np.std(recent) if len(recent) > 1 else 0
                summary[f"{key}_max"] = np.max(recent)
                summary[f"{key}_min"] = np.min(recent)
        return summary
    
    def __len__(self):
        """Return current number of transitions in buffer."""
        return len(self.buffer)

class TrainingStatistics:
    """
    Comprehensive statistics tracking for training progress.
    
    This class handles logging, visualization, and analysis of training metrics
    including episode statistics, training losses, and evaluation results.
    
    Args:
        save_dir (str): Directory to save statistics and plots
    """
    
    def __init__(self, save_dir="training_stats"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Statistics storage
        self.episode_stats = defaultdict(list)
        self.training_stats = defaultdict(list)
        self.evaluation_stats = defaultdict(list)
        
        # File paths for CSV logging
        self.episode_csv = os.path.join(save_dir, "episode_stats.csv")
        self.training_csv = os.path.join(save_dir, "training_stats.csv")
        self.eval_csv = os.path.join(save_dir, "evaluation_stats.csv")
        
        # Performance tracking
        self.last_log_time = time.time()
        self.last_episode_count = 0
        
    def log_episode(self, episode, stats):
        """
        Log episode statistics to memory and CSV file.
        
        Args:
            episode (int): Episode number
            stats (dict): Episode statistics to log
        """
        stats['episode'] = episode
        stats['timestamp'] = time.time()
        
        # Store in memory
        for key, value in stats.items():
            self.episode_stats[key].append(value)
        
        # Append to CSV for persistence
        df = pd.DataFrame([stats])
        df.to_csv(self.episode_csv, mode='a', 
                  header=not os.path.exists(self.episode_csv), index=False)
    
    def log_training(self, step, stats):
        """
        Log training statistics to memory and CSV file.
        
        Args:
            step (int): Training step number
            stats (dict): Training statistics to log
        """
        stats['step'] = step
        stats['timestamp'] = time.time()
        
        # Store in memory
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        # Batch CSV writes for efficiency
        if len(self.training_stats['step']) % 100 == 0:
            df = pd.DataFrame({k: v[-100:] for k, v in self.training_stats.items()})
            df.to_csv(self.training_csv, mode='a', 
                      header=not os.path.exists(self.training_csv), index=False)
    
    def log_evaluation(self, episode, stats):
        """
        Log evaluation statistics to memory and CSV file.
        
        Args:
            episode (int): Episode number
            stats (dict): Evaluation statistics to log
        """
        stats['episode'] = episode
        stats['timestamp'] = time.time()
        
        # Store in memory
        for key, value in stats.items():
            self.evaluation_stats[key].append(value)
        
        # Append to CSV
        df = pd.DataFrame([stats])
        df.to_csv(self.eval_csv, mode='a', 
                  header=not os.path.exists(self.eval_csv), index=False)
    
    def get_episodes_per_hour(self, current_episode):
        """
        Calculate episodes per hour for performance monitoring.
        
        Args:
            current_episode (int): Current episode number
            
        Returns:
            float: Episodes per hour rate
        """
        current_time = time.time()
        time_diff = current_time - self.last_log_time
        episode_diff = current_episode - self.last_episode_count
        
        if time_diff > 0:
            eps_per_hour = (episode_diff / time_diff) * 3600
            self.last_log_time = current_time
            self.last_episode_count = current_episode
            return eps_per_hour
        return 0
    
    def plot_training_curves(self):
        """
        Generate comprehensive training curve plots.
        
        Creates a 2x3 grid of plots showing:
        - Episode rewards with moving average
        - Training loss over time
        - Win rate progression
        - Territory control over time
        - Episode length distribution
        - Q-value evolution
        """
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Training Progress', fontsize=16)
            
            # Episode rewards
            if 'total_reward' in self.episode_stats and len(self.episode_stats['total_reward']) > 0:
                ax = axes[0, 0]
                rewards = self.episode_stats['total_reward']
                episodes = self.episode_stats['episode']
                ax.plot(episodes, rewards, alpha=0.3)
                
                # Moving average for trend visualization
                window = min(100, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    ma = pd.Series(rewards).rolling(window, min_periods=1).mean()
                    ax.plot(episodes, ma, 'r-', linewidth=2, label=f'{window}-ep MA')
                
                ax.set_xlabel('Episode')
                ax.set_ylabel('Total Reward')
                ax.set_title('Episode Rewards')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Training loss
            if 'loss' in self.training_stats and len(self.training_stats['loss']) > 0:
                ax = axes[0, 1]
                losses = self.training_stats['loss']
                steps = self.training_stats['step']
                
                # Downsample for plotting efficiency
                if len(losses) > 1000:
                    idx = np.linspace(0, len(losses)-1, 1000, dtype=int)
                    losses = [losses[i] for i in idx]
                    steps = [steps[i] for i in idx]
                
                ax.plot(steps, losses)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss')
                ax.set_yscale('log')  # Log scale for better visualization
                ax.grid(True, alpha=0.3)
            
            # Win rate
            if 'game_won' in self.episode_stats and len(self.episode_stats['game_won']) > 0:
                ax = axes[0, 2]
                wins = self.episode_stats['game_won']
                episodes = self.episode_stats['episode']
                
                # Calculate rolling win rate
                window = min(100, len(wins) // 10) if len(wins) > 10 else 1
                if window > 1:
                    win_rate = pd.Series(wins).rolling(window, min_periods=1).mean() * 100
                    ax.plot(episodes, win_rate)
                    ax.set_ylabel('Win Rate (%)')
                    ax.set_ylim(0, 100)
                
                ax.set_xlabel('Episode')
                ax.set_title(f'Win Rate ({window}-ep window)')
                ax.grid(True, alpha=0.3)
            
            # Territory control
            if 'final_territory' in self.episode_stats and len(self.episode_stats['final_territory']) > 0:
                ax = axes[1, 0]
                territory = self.episode_stats['final_territory']
                episodes = self.episode_stats['episode']
                ax.plot(episodes, territory, alpha=0.3)
                
                # Moving average
                window = min(100, len(territory) // 10) if len(territory) > 10 else 1
                if window > 1:
                    ma = pd.Series(territory).rolling(window, min_periods=1).mean()
                    ax.plot(episodes, ma, 'g-', linewidth=2, label=f'{window}-ep MA')
                
                ax.set_xlabel('Episode')
                ax.set_ylabel('Final Territory Count')
                ax.set_title('Territory Control')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Episode length
            if 'episode_length' in self.episode_stats and len(self.episode_stats['episode_length']) > 0:
                ax = axes[1, 1]
                lengths = self.episode_stats['episode_length']
                episodes = self.episode_stats['episode']
                ax.plot(episodes, lengths, alpha=0.3)
                
                # Moving average
                window = min(100, len(lengths) // 10) if len(lengths) > 10 else 1
                if window > 1:
                    ma = pd.Series(lengths).rolling(window, min_periods=1).mean()
                    ax.plot(episodes, ma, 'b-', linewidth=2, label=f'{window}-ep MA')
                
                ax.set_xlabel('Episode')
                ax.set_ylabel('Episode Length')
                ax.set_title('Episode Duration')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Q-values
            if 'mean_q_value' in self.training_stats and len(self.training_stats['mean_q_value']) > 0:
                ax = axes[1, 2]
                q_values = self.training_stats['mean_q_value']
                steps = self.training_stats['step']
                
                # Downsample for efficiency
                if len(q_values) > 1000:
                    idx = np.linspace(0, len(q_values)-1, 1000, dtype=int)
                    q_values = [q_values[i] for i in idx]
                    steps = [steps[i] for i in idx]
                
                ax.plot(steps, q_values)
                ax.set_xlabel('Training Step')
                ax.set_ylabel('Mean Q-Value')
                ax.set_title('Q-Value Evolution')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f"Error plotting training curves: {e}")
    
    def save_summary(self):
        """
        Save training summary statistics to JSON file.
        
        Returns:
            dict: Summary statistics including final metrics
        """
        summary = {
            'total_episodes': len(self.episode_stats['episode']) if 'episode' in self.episode_stats else 0,
            'total_training_steps': len(self.training_stats['step']) if 'step' in self.training_stats else 0,
            'final_metrics': {}
        }
        
        # Calculate final metrics from recent episodes
        if 'total_reward' in self.episode_stats and len(self.episode_stats['total_reward']) > 0:
            last_100 = self.episode_stats['total_reward'][-100:]
            summary['final_metrics']['avg_reward_last_100'] = np.mean(last_100)
            summary['final_metrics']['std_reward_last_100'] = np.std(last_100)
        
        if 'game_won' in self.episode_stats and len(self.episode_stats['game_won']) > 0:
            last_100 = self.episode_stats['game_won'][-100:]
            summary['final_metrics']['win_rate_last_100'] = np.mean(last_100) * 100
        
        if 'final_territory' in self.episode_stats and len(self.episode_stats['final_territory']) > 0:
            last_100 = self.episode_stats['final_territory'][-100:]
            summary['final_metrics']['avg_territory_last_100'] = np.mean(last_100)
        
        if 'loss' in self.training_stats and len(self.training_stats['loss']) > 0:
            last_1000 = self.training_stats['loss'][-1000:]
            summary['final_metrics']['avg_loss_last_1000'] = np.mean(last_1000)
        
        # Save to JSON file
        with open(os.path.join(self.save_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

class OptimizedSelfPlayDQN:
    """
    Optimized DQN agent with parallel environments and reduced action space.
    
    This class implements a comprehensive DQN training system with the following
    advanced features:
    - Parallel environment training for faster data collection
    - Prioritized experience replay for better sample efficiency
    - Reduced action space for improved training stability
    - Self-play training with historical model opponents
    - Multi-GPU support with DataParallel
    - Comprehensive logging and visualization
    
    Args:
        env_params (dict): Environment initialization parameters
        lr (float): Learning rate for optimizer
        batch_size (int): Batch size for training
        num_parallel_envs (int): Number of parallel environments
        save_dir (str): Directory to save checkpoints and logs
    """
    
    def __init__(self, env_params, lr=5e-4, batch_size=256, num_parallel_envs=4, save_dir="checkpoints"):
        self.env_params = env_params
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Create sample environment to get dimensions
        sample_env = OptimizedGeneralsEnv(**env_params)
        self.n_actions = sample_env.action_space.n
        self.obs_shape = sample_env.observation_space.shape
        
        # Initialize neural networks
        self.q_network = ConvDQN(self.obs_shape, self.n_actions, use_reduced_actions=True).to(device)
        self.target_network = ConvDQN(self.obs_shape, self.n_actions, use_reduced_actions=True).to(device)
        
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.q_network = DataParallel(self.q_network)
            self.target_network = DataParallel(self.target_network)
        
        self.update_target_network()
        
        # Optimizer with learning rate scheduling
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10000
        )
        
        # Experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(200000)  # Large buffer for stability
        self.batch_size = batch_size
        
        # Training hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.update_target_every = 1000  # Target network update frequency
        
        # Parallel environment management
        self.num_parallel_envs = num_parallel_envs
        self.env_pool = ParallelEnvPool(num_parallel_envs, env_params)
        
        # Self-play configuration
        self.past_models = deque(maxlen=20)  # Keep historical models for self-play
        self.self_play_update_interval = 200  # Update opponents every N episodes
        
        # Training state tracking
        self.step_count = 0
        self.episode_count = 0
        
        # Logging and visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f'runs/generals_optimized_{timestamp}')
        self.stats = TrainingStatistics()
        
        # Training metrics storage
        self.training_metrics = defaultdict(list)
    
    def update_target_network(self):
        """Update target network weights from main network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_checkpoint(self, episode):
        """
        Save model checkpoint with training state.
        
        Args:
            episode (int): Current episode number for checkpoint naming
        """
        # Handle DataParallel wrapper
        model_state = (self.q_network.module.state_dict() 
                      if hasattr(self.q_network, 'module') 
                      else self.q_network.state_dict())
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }
        path = os.path.join(self.save_dir, f'checkpoint_ep{episode}.pth')
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Also save as latest for easy loading
        torch.save(checkpoint, os.path.join(self.save_dir, 'latest_checkpoint.pth'))
    
    @torch.no_grad()
    def act_batch(self, states, valid_actions_list):
        """
        Select actions for multiple states efficiently.
        
        Args:
            states (list): List of state observations
            valid_actions_list (list): List of valid action lists for each state
            
        Returns:
            list: Selected actions for each state
        """
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        
        actions = []
        for i, (state, valid_actions) in enumerate(zip(states, valid_actions_list)):
            if random.random() < self.epsilon:
                # Epsilon-greedy: random valid action
                if valid_actions:
                    action = random.choice(valid_actions)
                else:
                    action = 0
            else:
                # Greedy: best valid action
                q_values = self.q_network(states_tensor[i:i+1]).squeeze(0)
                
                if valid_actions:
                    # Create mask for valid actions
                    valid_q_values = {a: q_values[a].item() for a in valid_actions}
                    action = max(valid_q_values, key=valid_q_values.get)
                else:
                    action = 0
            
            actions.append(action)
        
        return actions
    
    def train_step(self, beta=0.4):
        """
        Perform a single training step with prioritized experience replay.
        
        Args:
            beta (float): Importance sampling exponent for bias correction
            
        Returns:
            dict or None: Training statistics if training occurred, None otherwise
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones, indices, weights = \
            self.replay_buffer.sample(self.batch_size, beta)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network for action selection, target for evaluation
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards + self.gamma * next_q_values.squeeze() * (1 - dones)
        
        # Compute TD error and loss
        td_errors = nn.SmoothL1Loss(reduction='none')(
            current_q_values.squeeze(), target_q_values)
        loss = (weights * td_errors).mean()
        
        # Update priorities in replay buffer
        priorities = td_errors.detach().cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)
        
        # Optimize network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)  # Gradient clipping
        self.optimizer.step()
        
        # Compute training statistics
        with torch.no_grad():
            mean_q = current_q_values.mean().item()
            max_q = current_q_values.max().item()
            mean_target_q = target_q_values.mean().item()
            grad_norm = sum(p.grad.norm().item() for p in self.q_network.parameters() 
                          if p.grad is not None)
        
        stats = {
            'loss': loss.item(),
            'mean_q_value': mean_q,
            'max_q_value': max_q,
            'mean_target_q': mean_target_q,
            'td_error': td_errors.mean().item(),
            'grad_norm': grad_norm,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        # Log statistics periodically
        if self.step_count % 100 == 0:
            self.stats.log_training(self.step_count, stats)
            for key, value in stats.items():
                self.writer.add_scalar(f'Training/{key}', value, self.step_count)
        
        return stats
    
    def collect_parallel_experience(self, num_steps=100):
        """
        Collect experience from parallel environments.
        
        This method runs multiple environments in parallel to collect training
        data efficiently. It handles episode termination, statistics tracking,
        and automatic environment resetting.
        
        Args:
            num_steps (int): Number of steps to collect from each environment
        """
        # Initialize episode tracking if not exists
        if not hasattr(self, 'env_episode_lengths'):
            self.env_episode_lengths = [0] * self.num_parallel_envs
            self.episode_rewards = [0] * self.num_parallel_envs
            self.episode_stats = [defaultdict(float) for _ in range(self.num_parallel_envs)]
            self.invalid_move_counts = [0] * self.num_parallel_envs
            self.states = self.env_pool.reset()
            print(f"Initialized {self.num_parallel_envs} environments")
            
            # Debug: Check initial game state
            for i, env in enumerate(self.env_pool.envs):
                if hasattr(env, 'game'):
                    game = env.game
                    player = game.players[self.env_params['player_id']]
                    print(f"Env {i}: Player {self.env_params['player_id']} alive={player.is_alive}, "
                          f"general at {player.general_pos}")
                    
                    # Count initial territory
                    owned = sum(1 for row in game.grid 
                              for cell in row if cell.owner == self.env_params['player_id'])
                    print(f"  Initial territory: {owned} cells")
        
        MAX_EPISODE_LENGTH = 5000  # Force episode to end after this many steps
        
        # Debug: Print active episodes at start
        active_episodes = self.num_parallel_envs - sum(self.env_pool.dones)
        if self.step_count % 1000 == 0:
            print(f"Step {self.step_count}: {active_episodes} active episodes")
        
        states = self.states
        
        for step in range(num_steps):
            # Get valid actions for all environments
            valid_actions_list, _ = self.env_pool.get_reduced_actions()
            
            # Debug: Check if we have any valid actions
            if self.step_count % 1000 == 0:
                valid_counts = [len(actions) for actions in valid_actions_list]
                print(f"Valid actions per env: {valid_counts}")
                if all(count == 0 for count in valid_counts):
                    print("WARNING: No valid actions in any environment!")
            
            # Select actions using epsilon-greedy policy
            actions = self.act_batch(states, valid_actions_list)
            
            # Step all environments
            next_states, rewards, dones, infos = self.env_pool.step(actions)
            
            # Store transitions and handle episode tracking
            transitions = []
            for i in range(self.num_parallel_envs):
                # Track episode length
                self.env_episode_lengths[i] += 1
                
                # Debug: Check if we're getting invalid moves
                if 'invalid_move' in infos[i] and infos[i]['invalid_move']:
                    if not hasattr(self, 'invalid_move_counts'):
                        self.invalid_move_counts = [0] * self.num_parallel_envs
                    self.invalid_move_counts[i] += 1
                
                # Force episode end if too long or too many invalid moves
                force_end = False
                if self.env_episode_lengths[i] >= MAX_EPISODE_LENGTH:
                    force_end = True
                    reason = f"max_length ({MAX_EPISODE_LENGTH} steps)"
                elif (hasattr(self, 'invalid_move_counts') and 
                      self.invalid_move_counts[i] > 500):
                    force_end = True
                    reason = f"too many invalid moves ({self.invalid_move_counts[i]})"
                
                if force_end and not dones[i]:
                    dones[i] = True
                    infos[i]['timeout'] = True
                    infos[i]['forced_end'] = True
                    print(f"Force ending episode in env {i} due to {reason}")
                    print(f"  Total reward: {self.episode_rewards[i]:.2f}")
                    print(f"  Invalid moves: {getattr(self, 'invalid_move_counts', [0])[i]}")
                    
                    # Check game state
                    if hasattr(self.env_pool.envs[i], 'game'):
                        game = self.env_pool.envs[i].game
                        print(f"  Game over: {game.game_over}, Winner: {getattr(game, 'winner', 'N/A')}")
                        print(f"  Player alive: {game.players[self.env_params['player_id']].is_alive}")
                
                transitions.append((
                    states[i], actions[i], rewards[i],
                    next_states[i], dones[i], infos[i]
                ))
                
                self.episode_rewards[i] += rewards[i]
                
                # Track episode stats
                if 'territory' in infos[i]:
                    self.episode_stats[i]['final_territory'] = infos[i]['territory']
                if 'army' in infos[i]:
                    self.episode_stats[i]['final_army'] = infos[i]['army']
                
                # Handle episode end
                if dones[i]:
                    # Log episode stats
                    self.episode_count += 1
                    
                    # Determine win status
                    game_won = False
                    if 'winner' in infos[i]:
                        game_won = float(infos[i]['winner'] == self.env_params['player_id'])
                    elif (hasattr(self.env_pool.envs[i], 'game') and 
                          hasattr(self.env_pool.envs[i].game, 'winner')):
                        game_won = float(self.env_pool.envs[i].game.winner == self.env_params['player_id'])
                    
                    stats = {
                        'total_reward': self.episode_rewards[i],
                        'episode_length': self.env_episode_lengths[i],
                        'epsilon': self.epsilon,
                        'game_won': game_won,
                        'timeout': infos[i].get('timeout', False),
                        **self.episode_stats[i]
                    }
                    
                    self.stats.log_episode(self.episode_count, stats)
                    
                    # Debug print for episode completion
                    if self.episode_count % 10 == 0:
                        print(f"Episode {self.episode_count} completed! "
                              f"Reward: {self.episode_rewards[i]:.2f}, "
                              f"Length: {self.env_episode_lengths[i]}, "
                              f"Won: {game_won}")
                    
                    # Reset counters for next episode
                    self.episode_rewards[i] = 0
                    self.env_episode_lengths[i] = 0
                    self.episode_stats[i] = defaultdict(float)
                    if hasattr(self, 'invalid_move_counts'):
                        self.invalid_move_counts[i] = 0
            
            # Push transitions to replay buffer
            self.replay_buffer.push_batch(transitions)
            
            # Update states for next iteration
            states = next_states
            
            # Train network periodically
            if self.step_count % 4 == 0 and len(self.replay_buffer) > self.batch_size:
                beta = min(1.0, 0.4 + 0.6 * (self.step_count / 100000))
                self.train_step(beta)
            
            # Update target network periodically
            if self.step_count % self.update_target_every == 0:
                self.update_target_network()
            
            self.step_count += self.num_parallel_envs
        
        # Save current states for next collection
        self.states = states
        
        # Update learning rate based on recent loss
        if 'loss' in self.stats.training_stats and len(self.stats.training_stats['loss']) > 0:
            recent_losses = self.stats.training_stats['loss'][-100:]
            if recent_losses:
                avg_loss = np.mean(recent_losses)
                self.scheduler.step(avg_loss)
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def update_self_play_opponents(self):
        """
        Update opponents for self-play training.
        
        This method manages the pool of historical models used as opponents
        in self-play training. It saves the current model and updates the
        opponent selection strategy based on training progress.
        """
        # Save current model to history
        model_copy = ConvDQN(self.obs_shape, self.n_actions, use_reduced_actions=True).to(device)
        
        # Handle DataParallel wrapper
        if hasattr(self.q_network, 'module'):
            model_copy.load_state_dict(self.q_network.module.state_dict())
        else:
            model_copy.load_state_dict(self.q_network.state_dict())
            
        model_copy.eval()
        self.past_models.append(model_copy)
        
        # Update opponents in environment pool
        for env in self.env_pool.envs:
            if hasattr(env, 'set_opponent_models'):
                opponents = []
                epsilons = []
                
                for i in range(env.num_opponents):
                    if self.episode_count < 500:
                        # Early training: mostly random opponents
                        if random.random() < 0.7:
                            opponents.append(None)
                            epsilons.append(1.0)
                        else:
                            opponents.append(self.q_network)
                            epsilons.append(0.3)
                    elif self.episode_count < 2000:
                        # Mid training: mix of random and historical models
                        if random.random() < 0.3 and len(self.past_models) > 0:
                            opponents.append(random.choice(self.past_models))
                            epsilons.append(0.15)
                        else:
                            opponents.append(self.q_network)
                            epsilons.append(0.2)
                    else:
                        # Late training: strong opponents
                        if random.random() < 0.5 and len(self.past_models) > 0:
                            opponents.append(self.past_models[-1])
                            epsilons.append(0.05)
                        else:
                            opponents.append(self.q_network)
                            epsilons.append(0.1)
                
                env.set_opponent_models(opponents, epsilons)

def train_optimized():
    """
    Optimized training function with all improvements.
    
    This function implements a comprehensive training pipeline with the following
    features:
    - Multi-phase training with increasing difficulty
    - Parallel environment training for efficiency
    - Self-play with historical model opponents
    - Comprehensive logging and checkpointing
    - Performance monitoring and visualization
    
    Returns:
        tuple: (trained agent, training summary statistics)
    """
    
    # Training configuration
    total_episodes = 5000
    steps_per_collection = 200  # Steps before updating
    checkpoint_interval = 500
    evaluation_interval = 250
    
    # Phase 1: Start with 1 opponent
    print("="*60)
    print("OPTIMIZED TRAINING WITH PARALLEL ENVIRONMENTS")
    print("="*60)
    
    # Environment parameters
    env_params = {
        'player_id': 0,
        'num_opponents': 1,
        'training_mode': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Use more parallel environments for faster data collection
    num_parallel_envs = min(8, mp.cpu_count() // 2)  # Use half of CPU cores
    print(f"Using {num_parallel_envs} parallel environments")
    
    # Initialize trainer
    trainer = OptimizedSelfPlayDQN(
        env_params, 
        lr=5e-4, 
        batch_size=256,  # Larger batch size for stability
        num_parallel_envs=num_parallel_envs
    )
    
    # Training state tracking
    start_time = time.time()
    last_checkpoint_episode = 0
    
    print("\nPhase 1: Training with 1 opponent")
    phase_episodes = [2000]  # Episode targets for each phase
    phase_opponents = [1]  # Number of opponents for each phase
    
    current_phase = 0
    phase_start_episode = 0
    
    # Main training loop
    while trainer.episode_count < total_episodes:
        # Check if we need to advance to next phase
        if current_phase < len(phase_episodes) - 1:
            if trainer.episode_count >= phase_start_episode + phase_episodes[current_phase]:
                current_phase += 1
                phase_start_episode = trainer.episode_count
                env_params['num_opponents'] = phase_opponents[current_phase]
                
                # Recreate environment pool with new parameters
                trainer.env_pool = ParallelEnvPool(num_parallel_envs, env_params)
                
                print(f"\n{'='*60}")
                print(f"Phase {current_phase + 1}: Training with {phase_opponents[current_phase]} opponents")
                print(f"{'='*60}")
        
        # Update self-play opponents periodically
        if (trainer.episode_count % trainer.self_play_update_interval == 0 and 
            trainer.episode_count > 0):
            trainer.update_self_play_opponents()
            print(f"Updated self-play opponents at episode {trainer.episode_count}")
        
        # Collect experience from parallel environments
        trainer.collect_parallel_experience(num_steps=steps_per_collection)
        
        # Progress reporting
        if trainer.episode_count % 50 == 0 and trainer.episode_count > 0:
            elapsed = time.time() - start_time
            eps_per_hour = trainer.stats.get_episodes_per_hour(trainer.episode_count)
            
            # Get recent performance statistics
            recent_rewards = trainer.stats.episode_stats.get('total_reward', [])[-50:]
            recent_wins = trainer.stats.episode_stats.get('game_won', [])[-50:]
            recent_territory = trainer.stats.episode_stats.get('final_territory', [])[-50:]
            
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
                win_rate = np.mean(recent_wins) * 100 if recent_wins else 0
                avg_territory = np.mean(recent_territory) if recent_territory else 0
                
                print(f"Episode {trainer.episode_count}: "
                      f"Avg Reward={avg_reward:.2f}, "
                      f"Win Rate={win_rate:.1f}%, "
                      f"Avg Territory={avg_territory:.1f}, "
                      f"Eps/hour={eps_per_hour:.0f}, "
                      f"ε={trainer.epsilon:.3f}")
        
        # Periodic checkpointing and visualization
        if trainer.episode_count - last_checkpoint_episode >= checkpoint_interval:
            trainer.save_checkpoint(trainer.episode_count)
            trainer.stats.plot_training_curves()
            last_checkpoint_episode = trainer.episode_count
            
            # Estimate time remaining
            if eps_per_hour > 0:
                remaining_episodes = total_episodes - trainer.episode_count
                eta_hours = remaining_episodes / eps_per_hour
                print(f"ETA: {eta_hours:.1f} hours ({eta_hours/24:.1f} days)")
    
    # Final evaluation and statistics
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Save final checkpoint and summary
    trainer.save_checkpoint(trainer.episode_count)
    final_summary = trainer.stats.save_summary()
    
    # Print comprehensive training summary
    total_time = time.time() - start_time
    print(f"\nTraining Summary:")
    print(f"Total time: {total_time / 3600:.1f} hours")
    print(f"Total episodes: {trainer.episode_count}")
    print(f"Episodes per hour: {trainer.episode_count / (total_time / 3600):.0f}")
    
    if 'final_metrics' in final_summary:
        metrics = final_summary['final_metrics']
        print(f"\nFinal Performance (last 100 episodes):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.2f}")
    
    # Generate final plots
    trainer.stats.plot_training_curves()
    
    # Clean up resources
    trainer.writer.close()
    
    return trainer, final_summary

def main():
    """
    Main entry point for the training script.
    
    Sets up multiprocessing and runs the optimized training pipeline.
    """
    # Set up multiprocessing for parallel environments
    mp.set_start_method('spawn', force=True)
    
    # Run optimized training
    trainer, summary = train_optimized()
    
    print("\nTraining complete! Check the following directories for results:")
    print("- training_stats/: CSV files and plots")
    print("- checkpoints/: Saved models")
    print("- runs/: TensorBoard logs")
    
    # Save final summary
    print("\nFinal summary saved to: training_stats/training_summary.json")

if __name__ == "__main__":
    main()