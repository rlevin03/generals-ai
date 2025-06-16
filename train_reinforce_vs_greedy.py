#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN REINFORCE VS GREEDY
#  – Combined version –
#     * logs a compact summary every 10 episodes
#     * saves the policy checkpoint every 100 episodes
#     * keeps DQN opponent frozen
# ─────────────────────────────────────────────────────────────────────────────

import os, random, time, math, argparse, json, collections, sys
import numpy as np
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from   datetime import datetime
from generals import CellType
from policy_network import PolicyNetwork
from generals_rl_env_gpu import GeneralsEnv
from DQN_agent import QNetwork, select_action

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Train REINFORCE agent against greedy opponent')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the DQN model to use for opponent')
    
    # Optional arguments
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate for policy network')
    parser.add_argument('--episodes', type=int, default=3000,
                      help='Number of episodes to train for')
    parser.add_argument('--log-every', type=int, default=10,
                      help='Log statistics every N episodes')
    parser.add_argument('--save-every', type=int, default=100,
                      help='Save model checkpoint every N episodes')
    parser.add_argument('--gamma', type=float, default=0.99,
                      help='Discount factor for future rewards')
    parser.add_argument('--outdir', type=str, default='runs',
                      help='Output directory for runs')
    parser.add_argument('--quiet', action='store_true',
                      help='Suppress progress messages')
    
    return parser.parse_args()

# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------
def total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def safe_load_state(model, path):
    """Safely load model state, handling missing keys"""
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

def get_random_valid_action(env):
    """Get a random valid action"""
    valid_actions = env.get_valid_actions()
    if len(valid_actions) > 0:
        return int(random.choice(valid_actions))
    return None

def compute_returns(rewards, gamma):
    ret, rets = 0.0, []
    for r in reversed(rewards):
        ret = r + gamma * ret
        rets.insert(0, ret)
    return torch.tensor(rets, dtype=torch.float32, device=DEVICE)

def save_checkpoint(ep, net, opt, outdir):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f'policy_ep{ep:05d}.pth')
    torch.save({'episode': ep,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opt.state_dict()}, ckpt_path)
    return ckpt_path

# ---------------------------------------------------------------------------
#  Build env + models
# ---------------------------------------------------------------------------
ARGS = parse_args()  # Parse command line arguments

env            = GeneralsEnv(player_id=0, num_opponents=3, device=DEVICE)
state_dim      = len(env.reset())         # flattened board length
action_dim     = env.action_space.n

# Load DQN model once at the start
print(f"[INFO] Loading DQN model from {ARGS.model_path}")
dqn_model = QNetwork(state_dim, action_dim).to(DEVICE)
dqn_model = safe_load_state(dqn_model, ARGS.model_path).eval()

policy_net = PolicyNetwork(n_actions=action_dim).to(DEVICE)
optimizer = optim.Adam(policy_net.parameters(), lr=ARGS.lr)
entropy_beta = 0.01

print(f"[INFO]  Policy params: {total_params(policy_net):,}  |  "
      f"DQN params: {total_params(dqn_model):,}")

# ---------------------------------------------------------------------------
#  Training helpers
# ---------------------------------------------------------------------------
class GameStats:
    def __init__(self):
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
    
    def update(self, reward, length, won, armies, territories, invalid_moves, entropy=None, value=None, loss=None):
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
    
    def get_summary(self, window=10):
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

def get_valid_actions_mask(env):
    """Returns a 0/1 mask over the *full* action space of size N."""
    np_mask = env.get_valid_actions()          # np.array of bools, shape (N,)
    valid_indices = np.nonzero(np_mask)[0]     # integer indices where True

    mask = torch.zeros(env.action_space.n, device=DEVICE)
    if valid_indices.size > 0:
        idx_tensor = torch.tensor(valid_indices, device=DEVICE)
        mask[idx_tensor] = 1.0
    return mask

def get_dqn_action(state, env, dqn_net):
    """Get action from frozen DQN opponent"""
    with torch.no_grad():  # Ensure no gradients are computed
        return select_action(dqn_net, state, 0.0, env.action_space.n, env.player_id)

def get_state_summary(t):
    s = t.squeeze(0).cpu().numpy()
    # Count armies in all cells owned by player (0) - denormalize by multiplying by 100
    player_armies = np.sum(s[:, :, 1] * (s[:, :, 0] == 0)) * 100
    player_terr = np.sum(s[:, :, 0] == 0)
    # Count armies in all cells owned by enemies (any owner > 0) - denormalize by multiplying by 100
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
    
    return dict(
        player_armies=player_armies,
        player_territories=player_terr,
        enemy_armies=enemy_armies,
        enemy_territories=enemy_terr,
        cities=cities,
        forts=forts
    )

def count_frontier_cells(env, player_id):
    """Count cells that are adjacent to enemy or neutral territory"""
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

def scale_reward(raw, info, prev_sum=None, cur_sum=None):
    r = 0.0
    if info.get("game_over", False):
        r += 2.0 if raw > 0 else -2.0  # win/loss reward
    if info.get("captured_general", False):
        r += 1.0  # general capture reward
    if prev_sum and cur_sum:
        r += 0.02 * (cur_sum["player_territories"] - prev_sum["player_territories"])  # territory gain
        r += 0.01 * (cur_sum["player_armies"] - prev_sum["player_armies"])  # army growth
        rel_strength = cur_sum["player_armies"] / (cur_sum["enemy_armies"] + 1e-6)
        r += 0.1 * (rel_strength - 1.0)  # relative strength
    r -= 0.002  # step penalty
    if info.get("invalid_move", False):
        r -= 0.05  # invalid move penalty
    return r

def run_episode(ep, baseline, stats, policy_net, dqn_net, env, ARGS):
    obs = env.reset()
    state = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    prev_sum = get_state_summary(state)
    
    # Track initial tile ownership
    start_tiles = (state.squeeze(0)[:, :, 0] == env.player_id).sum()
    
    # Track milestones and strategic positions
    saw_10_tiles = False
    saw_20_tiles = False
    saw_enemy_fort = False
    owned_cities = set()  # Track cities we own
    enemy_cities = set()  # Track enemy cities we can see
    prev_frontiers = 0  # Track previous frontier count
    
    # Milestone rewards
    tiles_10_bonus = 1.0
    tiles_20_bonus = 1.0
    enemy_fort_bonus = 5.0
    win_bonus = 1000.0  # Massive win bonus
    city_capture_bonus = 50.0  # Big bonus for capturing cities
    enemy_territory_bonus = 10.0  # Bonus for capturing enemy territory
    frontier_bonus = 0.01  # Minimal frontier bonus
    frontier_growth_bonus = 0.02  # Minimal frontier growth bonus
    exploration_bonus = 0.005  # Minimal exploration bonus

    # Track previous distance for delta calculation
    prev_min_distance = float('inf')
    prev_frontier_cells = set()  # Track previous frontier cells
    explored_cells = set()  # Track explored cells

    episode_logps = []
    episode_rewards = []
    episode_values = []  # Track value estimates
    episode_logits = []  # Store distributions for entropy calculation
    invalid_moves = 0

    done = False
    turns = 0
    while not done and turns < 2750:
        turns += 1
        # — agent's move — -------------------------------------------------
        # Get action distribution and value estimate
        dist = policy_net.get_action_dist(state)
        valid_mask = get_valid_actions_mask(env)
        
        # If we have valid actions, force the agent to take one
        if valid_mask.any():
            # Mask out invalid actions with large negative value
            logits_masked = dist.logits + (valid_mask - 1) * 1e9
            dist = torch.distributions.Categorical(logits=logits_masked)
            
            # Get value estimate
            _, value = policy_net(state)
            episode_values.append(value.squeeze(-1))
            
            # Store distribution for entropy calculation
            episode_logits.append(dist)
            
            # Increase exploration rate to 20% to encourage more moves
            if random.random() < 0.20:  # 20% random actions
                action = get_random_valid_action(env)
            else:
                # Take the best valid action
                valid_probs = dist.probs * valid_mask
                valid_probs = valid_probs / valid_probs.sum()  # re-normalize to a proper distribution
                action = torch.multinomial(valid_probs, 1).item()
        else:
            # No valid actions, use a no-op
            action = 0
            episode_values.append(torch.tensor(0.0, device=DEVICE))
            episode_logits.append(None)  # No distribution for no-op

        logp = dist.log_prob(torch.tensor(action, device=DEVICE))
        # Suppress stdout during step
        with open(os.devnull, 'w') as f:
            old_stdout = sys.stdout
            sys.stdout = f
            next_obs, raw_r, done, info = env.step(action)
            sys.stdout = old_stdout
        next_state = torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        cur_sum = get_state_summary(next_state)

        # ---------- reward shaping ----------
        # Cast tile counts to int to avoid tensor/float mixing
        cur_tiles = int((next_state.squeeze(0)[:, :, 0] == env.player_id).sum().item())  # tiles you own now
        prev_tiles = int((state.squeeze(0)[:, :, 0] == env.player_id).sum().item())  # tiles owned in previous step
        territory_change = cur_tiles - prev_tiles  # actual change in territory this step
        
        # Base rewards - focus on territory capture
        reward = 5.0 * territory_change  # Big reward for any territory gain
        reward -= 0.002  # Minimal step penalty

        # City capture rewards
        for city_pos in cur_sum['cities']:
            city_owner = next_state.squeeze(0)[city_pos[1], city_pos[0], 0].item()
            if city_owner == env.player_id and city_pos not in owned_cities:
                owned_cities.add(city_pos)
                reward += city_capture_bonus  # Big bonus for capturing cities
                if not ARGS.quiet:
                    print(f"Milestone: Captured city! +{city_capture_bonus} bonus")
            elif city_owner != env.player_id and city_pos not in enemy_cities:
                enemy_cities.add(city_pos)
                reward += 1.0  # Small reward for spotting enemy city

        # Enemy territory capture bonus
        if territory_change > 0:
            # Check if we captured enemy territory (not neutral)
            prev_enemy_tiles = sum(1 for y in range(env.grid_height) for x in range(env.grid_width) 
                                 if state.squeeze(0)[y, x, 0].item() > 0)
            cur_enemy_tiles = sum(1 for y in range(env.grid_height) for x in range(env.grid_width) 
                                if next_state.squeeze(0)[y, x, 0].item() > 0)
            enemy_territory_captured = prev_enemy_tiles - cur_enemy_tiles
            if enemy_territory_captured > 0:
                reward += enemy_territory_bonus * enemy_territory_captured

        # Calculate frontier rewards (minimal)
        current_frontiers = count_frontier_cells(env, env.player_id)
        frontier_delta = current_frontiers - prev_frontiers
        reward += frontier_bonus * current_frontiers
        if frontier_delta > 0:
            reward += frontier_growth_bonus * frontier_delta
        prev_frontiers = current_frontiers

        # Exploration reward (minimal)
        current_visible_cells = set()
        for y in range(env.grid_height):
            for x in range(env.grid_width):
                cell = env.game.grid[y][x]
                if env.player_id in cell.visible_to:
                    current_visible_cells.add((x, y))
        new_explored = current_visible_cells - explored_cells
        reward += exploration_bonus * len(new_explored)
        explored_cells.update(new_explored)

        # Movement reward (minimal)
        if action != 0:  # If agent made a move (not no-op)
            reward += 0.005  # Small reward for making a move

        # Invalid move penalty
        if info.get("invalid_move", False):
            reward -= 0.1  # Increased invalid move penalty

        # Game over rewards
        if done:
            if info.get("winner") == env.player_id:
                reward += win_bonus
                if not ARGS.quiet:
                    print(f"Milestone: Victory! +{win_bonus} bonus")
            else:
                reward -= 200.0  # Significant penalty for losing
        episode_rewards.append(reward)

        if info.get('invalid_move', False):
            invalid_moves += 1

        episode_logps.append(logp)

        state = next_state
        prev_sum = cur_sum

        # — opponent move — ----------------------------------------------
        if not done:
            # Slice off extra channels before passing to DQN
            state6 = state.squeeze(0)                # shape (20,25,6)
            state4_flat = state6[..., :4].flatten()  # take only the first 4 channels → size 2000
            opp_a = get_dqn_action(state4_flat, env, dqn_net)
            # Suppress stdout during opponent step
            with open(os.devnull, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                opp_obs, _, done, _ = env.step_with_reduced_action(opp_a)
                sys.stdout = old_stdout
            state = torch.as_tensor(opp_obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            prev_sum = get_state_summary(state)

    # Update statistics
    stats.episodes += 1
    stats.rewards += sum(episode_rewards)
    stats.episode_lengths.append(len(episode_rewards))
    if info.get("winner") == env.player_id:
        stats.wins += 1

    # Calculate returns and advantages
    returns = []
    advantages = []
    R = 0
    for r in reversed(episode_rewards):
        R = r + ARGS.gamma * R
        returns.insert(0, R)
    
    returns = torch.tensor(returns, device=DEVICE)
    values = torch.cat(episode_values)
    
    # Calculate advantages using value function baseline
    advantages = returns - values.detach()
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Calculate policy loss with value function baseline
    policy_loss = -(torch.stack(episode_logps) * advantages).mean()
    
    # Calculate value function loss
    value_loss = 0.5 * (returns - values).pow(2).mean()
    
    # Calculate entropy bonus
    entropy = torch.stack([dist.entropy() for dist in episode_logits if dist is not None]).mean()
    
    # Total loss
    loss = policy_loss + value_loss - entropy_beta * entropy
    
    # Optimize
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)  # Gradient clipping
    optimizer.step()
    
    # Update statistics with all metrics
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

# ---------------------------------------------------------------------------
#  Main loop
# ---------------------------------------------------------------------------
def main():
    # Create output directory
    os.makedirs(ARGS.outdir, exist_ok=True)
    
    # Initialize environment and models
    env = GeneralsEnv(player_id=0, num_opponents=3, device=DEVICE)
    state_dim = len(env.reset())
    action_dim = env.action_space.n
    
    # Initialize networks
    dqn_net = QNetwork(state_dim, action_dim).to(DEVICE)
    policy_net = PolicyNetwork(n_actions=action_dim).to(DEVICE)
    
    # Load DQN model
    print(f"[INFO] Loading DQN model from {ARGS.model_path}")
    dqn_net = safe_load_state(dqn_net, ARGS.model_path).eval()
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=ARGS.lr)
    
    # Training loop
    baseline = 0
    stats = GameStats()
    
    for ep in range(ARGS.episodes):
        if not ARGS.quiet:
            print(f"\nEpisode {ep+1}/{ARGS.episodes}")
        loss, ep_rewards, ep_logps, ep_values, invalid_moves = run_episode(ep, baseline, stats, policy_net, dqn_net, env, ARGS)
        
        # Save checkpoint
        if (ep + 1) % ARGS.save_every == 0:
            checkpoint_path = os.path.join(ARGS.outdir, f'checkpoint_{ep+1}.pth')
            torch.save({
                'episode': ep + 1,
                'model_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Log statistics
        if (ep + 1) % ARGS.log_every == 0:
            print(stats.get_summary())
    
    # Save final model
    final_path = os.path.join(ARGS.outdir, 'final_model.pth')
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Saved final model to {final_path}")

if __name__ == "__main__":
    ARGS = parse_args()  # Parse arguments before main
    main() 