#!/usr/bin/env python3
"""
REINFORCE Agent Training Script for Generals.io

This script trains a REINFORCE agent with a value function baseline against a frozen DQN opponent
in the Generals.io environment. The agent uses policy gradient methods with advantage estimation
and includes comprehensive reward shaping for strategic gameplay.

Most of this code is AI Generated.

"""

import os
import random
import sys
import argparse
from typing import Dict, List, Tuple, Optional, Set
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from generals import CellType
from policy_network import PolicyNetwork
from generals_rl_env_gpu import GeneralsEnv
from DQN_agent import QNetwork, select_action


# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPISODES = 3000
DEFAULT_LOG_EVERY = 10
DEFAULT_SAVE_EVERY = 100
DEFAULT_GAMMA = 0.99
DEFAULT_ENTROPY_BETA = 0.01
DEFAULT_GRADIENT_CLIP = 0.5

# Reward shaping constants
REWARD_CONSTANTS = {
    'win_bonus': 1000.0,
    'loss_penalty': -200.0,
    'city_capture_bonus': 50.0,
    'enemy_territory_bonus': 10.0,
    'territory_gain_multiplier': 5.0,
    'step_penalty': -0.002,
    'invalid_move_penalty': -0.1,
    'movement_bonus': 0.005,
    'frontier_bonus': 0.01,
    'frontier_growth_bonus': 0.02,
    'exploration_bonus': 0.005,
    'enemy_city_spotting_bonus': 1.0
}

# Game constants
MAX_EPISODE_LENGTH = 2750
EXPLORATION_RATE = 0.20


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output."""
    with open(os.devnull, 'w') as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout


def total_params(model: nn.Module) -> int:
    """
    Calculate the total number of parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def safe_load_state(model: nn.Module, path: str) -> nn.Module:
    """
    Safely load model state, handling missing keys gracefully.
    
    Args:
        model: PyTorch model to load state into
        path: Path to the checkpoint file
        
    Returns:
        Model with loaded state (or original model if loading fails)
    """
    if not os.path.exists(path):
        print(f"Warning: Model file {path} not found. Using untrained model.")
        return model
    
    try:
        state_dict = torch.load(path, map_location=DEVICE)
        model_dict = model.state_dict()
        
        # Filter out missing keys
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        
        # Update model dict with filtered state
        model_dict.update(filtered_state_dict)
        model.load_state_dict(model_dict)
        
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        print("Using untrained model instead.")
        return model


def get_random_valid_action(env: GeneralsEnv) -> Optional[int]:
    """
    Get a random valid action from the environment.
    
    Args:
        env: Generals environment
        
    Returns:
        Random valid action index or None if no valid actions
    """
    valid_actions = env.get_valid_actions()
    if len(valid_actions) > 0:
        return int(random.choice(valid_actions))
    return None


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    """
    Compute discounted returns for a sequence of rewards.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        Tensor of discounted returns
    """
    ret, rets = 0.0, []
    for r in reversed(rewards):
        ret = r + gamma * ret
        rets.insert(0, ret)
    return torch.tensor(rets, dtype=torch.float32, device=DEVICE)


def save_checkpoint(episode: int, model: nn.Module, optimizer: optim.Optimizer, 
                   outdir: str) -> str:
    """
    Save a training checkpoint.
    
    Args:
        episode: Current episode number
        model: Model to save
        optimizer: Optimizer to save
        outdir: Output directory
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f'policy_ep{episode:05d}.pth')
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)
    return ckpt_path


# =============================================================================
# GAME STATE ANALYSIS FUNCTIONS
# =============================================================================

def get_valid_actions_mask(env: GeneralsEnv) -> torch.Tensor:
    """
    Create a binary mask for valid actions.
    
    Args:
        env: Generals environment
        
    Returns:
        Binary tensor mask where 1 indicates valid actions
    """
    np_mask = env.get_valid_actions()
    valid_indices = np.nonzero(np_mask)[0]

    mask = torch.zeros(env.action_space.n, device=DEVICE)
    if valid_indices.size > 0:
        idx_tensor = torch.tensor(valid_indices, device=DEVICE)
        mask[idx_tensor] = 1.0
    return mask


def get_dqn_action(state: torch.Tensor, env: GeneralsEnv, dqn_net: QNetwork) -> int:
    """
    Get action from frozen DQN opponent.
    
    Args:
        state: Current game state
        env: Generals environment
        dqn_net: DQN network
        
    Returns:
        Action index selected by DQN
    """
    with torch.no_grad():
        # Move tensor to CPU before converting to numpy
        state_cpu = state.cpu()
        return select_action(dqn_net, state_cpu, 0.0, env.player_id)


def get_state_summary(state: torch.Tensor) -> Dict:
    """
    Extract summary statistics from game state.
    
    Args:
        state: Game state tensor
        
    Returns:
        Dictionary containing state summary statistics
    """
    s = state.squeeze(0).cpu().numpy()
    
    # Count armies and territories for player and enemies
    player_armies = np.sum(s[:, :, 1] * (s[:, :, 0] == 0)) * 100
    player_terr = np.sum(s[:, :, 0] == 0)
    enemy_armies = np.sum(s[:, :, 1] * (s[:, :, 0] > 0)) * 100
    enemy_terr = np.sum(s[:, :, 0] > 0)
    
    # Extract city and fort positions
    cities = []
    forts = []
    for y in range(s.shape[0]):
        for x in range(s.shape[1]):
            if s[y, x, 2] > 0:  # is_city channel
                cities.append((x, y))
            if s[y, x, 3] > 0:  # is_general channel
                forts.append((x, y))
    
    return {
        'player_armies': player_armies,
        'player_territories': player_terr,
        'enemy_armies': enemy_armies,
        'enemy_territories': enemy_terr,
        'cities': cities,
        'forts': forts
    }


def count_frontier_cells(env: GeneralsEnv, player_id: int) -> int:
    """
    Count cells that are adjacent to enemy or neutral territory.
    
    Args:
        env: Generals environment
        player_id: Player ID to check frontiers for
        
    Returns:
        Number of frontier cells
    """
    frontier_count = 0
    for y in range(env.grid_height):
        for x in range(env.grid_width):
            cell = env.game.grid[y][x]
            if cell.owner == player_id:
                # Check all 4 directions
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < env.grid_width and 0 <= ny < env.grid_height:
                        neighbor = env.game.grid[ny][nx]
                        if neighbor.owner != player_id and neighbor.type != CellType.MOUNTAIN:
                            frontier_count += 1
                            break  # Count each cell only once
    return frontier_count


# =============================================================================
# REWARD SHAPING FUNCTIONS
# =============================================================================

def calculate_territory_rewards(state: torch.Tensor, next_state: torch.Tensor, 
                               env: GeneralsEnv) -> float:
    """
    Calculate rewards based on territory changes.
    
    Args:
        state: Current state
        next_state: Next state
        env: Generals environment
        
    Returns:
        Territory-based reward
    """
    cur_tiles = int((next_state.squeeze(0)[:, :, 0] == env.player_id).sum().item())
    prev_tiles = int((state.squeeze(0)[:, :, 0] == env.player_id).sum().item())
    territory_change = cur_tiles - prev_tiles
    
    reward = REWARD_CONSTANTS['territory_gain_multiplier'] * territory_change
    
    # Enemy territory capture bonus
    if territory_change > 0:
        prev_enemy_tiles = sum(1 for y in range(env.grid_height) for x in range(env.grid_width) 
                             if state.squeeze(0)[y, x, 0].item() > 0)
        cur_enemy_tiles = sum(1 for y in range(env.grid_height) for x in range(env.grid_width) 
                            if next_state.squeeze(0)[y, x, 0].item() > 0)
        enemy_territory_captured = prev_enemy_tiles - cur_enemy_tiles
        if enemy_territory_captured > 0:
            reward += REWARD_CONSTANTS['enemy_territory_bonus'] * enemy_territory_captured
    
    return reward


def calculate_city_rewards(cur_sum: Dict, next_state: torch.Tensor, env: GeneralsEnv,
                          owned_cities: Set, enemy_cities: Set, quiet: bool) -> Tuple[float, Set, Set]:
    """
    Calculate rewards based on city captures and discoveries.
    
    Args:
        cur_sum: Current state summary
        next_state: Next state
        env: Generals environment
        owned_cities: Set of cities owned by player
        enemy_cities: Set of enemy cities discovered
        quiet: Whether to suppress output
        
    Returns:
        Tuple of (reward, updated owned_cities, updated enemy_cities)
    """
    reward = 0.0
    
    for city_pos in cur_sum['cities']:
        city_owner = next_state.squeeze(0)[city_pos[1], city_pos[0], 0].item()
        if city_owner == env.player_id and city_pos not in owned_cities:
            owned_cities.add(city_pos)
            reward += REWARD_CONSTANTS['city_capture_bonus']
            if not quiet:
                print(f"Milestone: Captured city! +{REWARD_CONSTANTS['city_capture_bonus']} bonus")
        elif city_owner != env.player_id and city_pos not in enemy_cities:
            enemy_cities.add(city_pos)
            reward += REWARD_CONSTANTS['enemy_city_spotting_bonus']
    
    return reward, owned_cities, enemy_cities


def calculate_frontier_rewards(env: GeneralsEnv, prev_frontiers: int) -> Tuple[float, int]:
    """
    Calculate rewards based on frontier expansion.
    
    Args:
        env: Generals environment
        prev_frontiers: Previous frontier count
        
    Returns:
        Tuple of (reward, current_frontiers)
    """
    current_frontiers = count_frontier_cells(env, env.player_id)
    frontier_delta = current_frontiers - prev_frontiers
    
    reward = REWARD_CONSTANTS['frontier_bonus'] * current_frontiers
    if frontier_delta > 0:
        reward += REWARD_CONSTANTS['frontier_growth_bonus'] * frontier_delta
    
    return reward, current_frontiers


def calculate_exploration_rewards(env: GeneralsEnv, explored_cells: Set) -> Tuple[float, Set]:
    """
    Calculate rewards based on exploration of new cells.
    
    Args:
        env: Generals environment
        explored_cells: Set of previously explored cells
        
    Returns:
        Tuple of (reward, updated explored_cells)
    """
    current_visible_cells = set()
    for y in range(env.grid_height):
        for x in range(env.grid_width):
            cell = env.game.grid[y][x]
            if env.player_id in cell.visible_to:
                current_visible_cells.add((x, y))
    
    new_explored = current_visible_cells - explored_cells
    reward = REWARD_CONSTANTS['exploration_bonus'] * len(new_explored)
    explored_cells.update(new_explored)
    
    return reward, explored_cells


# =============================================================================
# STATISTICS TRACKING
# =============================================================================

class GameStats:
    """Track and summarize training statistics."""
    
    def __init__(self):
        """Initialize statistics tracking."""
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = 0
        self.total_armies = []
        self.total_territories = []
        self.invalid_moves = 0
        self.episodes = 0
        self.rewards = 0
        self.entropies = []
        self.values = []
        self.losses = []
    
    def update(self, reward: float, length: int, won: bool, armies: int, 
               territories: int, invalid_moves: int, entropy: Optional[float] = None,
               value: Optional[float] = None, loss: Optional[float] = None):
        """
        Update statistics with episode results.
        
        Args:
            reward: Episode total reward
            length: Episode length
            won: Whether the episode was won
            armies: Final army count
            territories: Final territory count
            invalid_moves: Number of invalid moves
            entropy: Policy entropy
            value: Value function estimate
            loss: Training loss
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.wins += int(won)
        self.total_armies.append(armies)
        self.total_territories.append(territories)
        self.invalid_moves += invalid_moves
        self.episodes += 1
        self.rewards += reward
        
        if entropy is not None:
            self.entropies.append(entropy)
        if value is not None:
            self.values.append(value)
        if loss is not None:
            self.losses.append(loss)
    
    def get_summary(self, window: int = 10) -> str:
        """
        Get a formatted summary of recent statistics.
        
        Args:
            window: Number of recent episodes to average over
            
        Returns:
            Formatted summary string
        """
        if not self.episodes:
            return "No episodes completed yet"
        
        # Calculate win rate
        win_rate = 100 * self.wins / self.episodes
        
        # Calculate averages over last window episodes
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        recent_armies = self.total_armies[-window:]
        recent_territories = self.total_territories[-window:]
        recent_entropies = self.entropies[-window:] if self.entropies else []
        recent_values = self.values[-window:] if self.values else []
        recent_losses = self.losses[-window:] if self.losses else []
        
        # Calculate averages
        avg_reward = np.mean(recent_rewards)
        avg_length = np.mean(recent_lengths)
        avg_army = np.mean(recent_armies)
        avg_territory = np.mean(recent_territories)
        avg_entropy = np.mean(recent_entropies) if recent_entropies else 0
        avg_value = np.mean(recent_values) if recent_values else 0
        avg_loss = np.mean(recent_losses) if recent_losses else 0
        
        # Calculate invalid move rate
        invalid_rate = 100 * self.invalid_moves / sum(self.episode_lengths)
        
        return (f"Ep {self.episodes:4d} | "
                f"Win% {win_rate:5.1f} | "
                f"Rew {avg_reward:6.2f} | "
                f"Len {avg_length:5.1f} | "
                f"Army {avg_army:6.1f} | "
                f"Terr {avg_territory:5.1f} | "
                f"Ent {avg_entropy:5.3f} | "
                f"Val {avg_value:6.2f} | "
                f"Loss {avg_loss:6.3f} | "
                f"Inv% {invalid_rate:4.1f}")


# =============================================================================
# EPISODE EXECUTION
# =============================================================================

def run_episode(episode: int, stats: GameStats, policy_net: PolicyNetwork, 
                dqn_net: QNetwork, env: GeneralsEnv, args: argparse.Namespace,
                optimizer: optim.Optimizer) -> Tuple[float, List[float], List[torch.Tensor], 
                                                    List[torch.Tensor], int]:
    """
    Run a single training episode.
    
    Args:
        episode: Episode number
        stats: Statistics tracker
        policy_net: Policy network
        dqn_net: DQN opponent network
        env: Generals environment
        args: Command line arguments
        optimizer: Optimizer for policy network
        
    Returns:
        Tuple of (loss, episode_rewards, episode_logps, episode_values, invalid_moves)
    """
    # Initialize episode
    obs = env.reset()
    state = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    prev_sum = get_state_summary(state)
    
    # Tracking variables
    owned_cities = set()
    enemy_cities = set()
    prev_frontiers = 0
    explored_cells = set()
    
    episode_logps = []
    episode_rewards = []
    episode_values = []
    episode_logits = []
    invalid_moves = 0
    
    done = False
    turns = 0
    
    while not done and turns < MAX_EPISODE_LENGTH:
        turns += 1
        
        # Agent's move
        dist = policy_net.get_action_dist(state)
        valid_mask = get_valid_actions_mask(env)
        
        if valid_mask.any():
            # Mask out invalid actions
            logits_masked = dist.logits + (valid_mask - 1) * 1e9
            dist = Categorical(logits=logits_masked)
            
            # Get value estimate
            _, value = policy_net(state)
            episode_values.append(value.squeeze(-1))
            episode_logits.append(dist)
            
            # Action selection with exploration
            if random.random() < EXPLORATION_RATE:
                action = get_random_valid_action(env)
            else:
                valid_probs = dist.probs * valid_mask
                valid_probs = valid_probs / valid_probs.sum()
                action = torch.multinomial(valid_probs, 1).item()
        else:
            # No valid actions, use no-op
            action = 0
            episode_values.append(torch.tensor(0.0, device=DEVICE))
            episode_logits.append(None)
        
        logp = dist.log_prob(torch.tensor(action, device=DEVICE))
        
        # Environment step
        with suppress_stdout():
            next_obs, raw_r, done, info = env.step(action)
        
        next_state = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        cur_sum = get_state_summary(next_state)
        
        # Calculate reward
        reward = REWARD_CONSTANTS['step_penalty']  # Base step penalty
        
        # Territory rewards
        reward += calculate_territory_rewards(state, next_state, env)
        
        # City rewards
        city_reward, owned_cities, enemy_cities = calculate_city_rewards(
            cur_sum, next_state, env, owned_cities, enemy_cities, args.quiet)
        reward += city_reward
        
        # Frontier rewards
        frontier_reward, prev_frontiers = calculate_frontier_rewards(env, prev_frontiers)
        reward += frontier_reward
        
        # Exploration rewards
        exploration_reward, explored_cells = calculate_exploration_rewards(env, explored_cells)
        reward += exploration_reward
        
        # Movement reward
        if action != 0:
            reward += REWARD_CONSTANTS['movement_bonus']
        
        # Invalid move penalty
        if info.get("invalid_move", False):
            reward += REWARD_CONSTANTS['invalid_move_penalty']
            invalid_moves += 1
        
        # Game over rewards
        if done:
            if info.get("winner") == env.player_id:
                reward += REWARD_CONSTANTS['win_bonus']
                if not args.quiet:
                    print(f"Milestone: Victory! +{REWARD_CONSTANTS['win_bonus']} bonus")
            else:
                reward += REWARD_CONSTANTS['loss_penalty']
        
        episode_rewards.append(reward)
        episode_logps.append(logp)
        
        state = next_state
        prev_sum = cur_sum
        
        # Opponent move
        if not done:
            state6 = state.squeeze(0)
            state4_flat = state6[..., :4].flatten()
            opp_a = get_dqn_action(state4_flat, env, dqn_net)
            
            with suppress_stdout():
                opp_obs, _, done, _ = env.step_with_reduced_action(opp_a)
            
            state = torch.as_tensor(opp_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_sum = get_state_summary(state)
    
    # Update statistics
    stats.episodes += 1
    stats.rewards += sum(episode_rewards)
    stats.episode_lengths.append(len(episode_rewards))
    if info.get("winner") == env.player_id:
        stats.wins += 1
    
    # Calculate returns and advantages
    returns = compute_returns(episode_rewards, args.gamma)
    values = torch.cat(episode_values)
    
    # Calculate advantages using value function baseline
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Calculate losses
    policy_loss = -(torch.stack(episode_logps) * advantages).mean()
    value_loss = 0.5 * (returns - values).pow(2).mean()
    entropy = torch.stack([dist.entropy() for dist in episode_logits if dist is not None]).mean()
    
    # Total loss
    loss = policy_loss + value_loss - DEFAULT_ENTROPY_BETA * entropy
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), DEFAULT_GRADIENT_CLIP)
    optimizer.step()
    
    # Update statistics
    stats.update(
        reward=sum(episode_rewards),
        length=len(episode_rewards),
        won=info.get("winner") == env.player_id,
        armies=info.get("army", 0),
        territories=info.get("territory", 0),
        invalid_moves=invalid_moves,
        entropy=entropy.item(),
        value=values.mean().item(),
        loss=loss.item()
    )
    
    return loss.item(), episode_rewards, episode_logps, episode_values, invalid_moves


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description='Train REINFORCE agent against DQN opponent in Generals.io'
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to the DQN model to use for opponent'
    )
    
    # Optional arguments
    parser.add_argument(
        '--lr', type=float, default=DEFAULT_LEARNING_RATE,
        help='Learning rate for policy network'
    )
    parser.add_argument(
        '--episodes', type=int, default=DEFAULT_EPISODES,
        help='Number of episodes to train for'
    )
    parser.add_argument(
        '--log-every', type=int, default=DEFAULT_LOG_EVERY,
        help='Log statistics every N episodes'
    )
    parser.add_argument(
        '--save-every', type=int, default=DEFAULT_SAVE_EVERY,
        help='Save model checkpoint every N episodes'
    )
    parser.add_argument(
        '--gamma', type=float, default=DEFAULT_GAMMA,
        help='Discount factor for future rewards'
    )
    parser.add_argument(
        '--outdir', type=str, default='runs',
        help='Output directory for runs'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress messages'
    )
    
    return parser.parse_args()


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Initialize environment and models
    env = GeneralsEnv(player_id=0, num_opponents=3, device=DEVICE)
    state_dim = len(env.reset())
    action_dim = env.action_space.n
    
    # Initialize networks
    dqn_net = QNetwork(state_dim, action_dim).to(DEVICE)
    policy_net = PolicyNetwork(n_actions=action_dim).to(DEVICE)
    
    # Load DQN model
    print(f"[INFO] Loading DQN model from {args.model_path}")
    dqn_net = safe_load_state(dqn_net, args.model_path).eval()
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    
    # Print model information
    print(f"[INFO] Policy params: {total_params(policy_net):,} | "
          f"DQN params: {total_params(dqn_net):,}")
    
    # Training loop
    stats = GameStats()
    
    for ep in range(args.episodes):
        if not args.quiet:
            print(f"\nEpisode {ep+1}/{args.episodes}")
        
        loss, ep_rewards, ep_logps, ep_values, invalid_moves = run_episode(
            ep, stats, policy_net, dqn_net, env, args, optimizer
        )
        
        # Save checkpoint
        if (ep + 1) % args.save_every == 0:
            checkpoint_path = save_checkpoint(ep + 1, policy_net, optimizer, args.outdir)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Log statistics
        if (ep + 1) % args.log_every == 0:
            print(stats.get_summary())
    
    # Save final model
    final_path = os.path.join(args.outdir, 'final_model.pth')
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    main() 