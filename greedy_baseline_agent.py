"""
Greedy Baseline Agent for Generals.io Reinforcement Learning Environment

This module implements a greedy heuristic agent that makes decisions based on
immediate rewards without memory between turns. The agent prioritizes capturing
neutral cities, expanding territory, and attacking weaker enemy positions.

Most of this code is AI generated.

"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
from typing import List, Tuple, Optional, Dict

from generals import CellType, Game
from generals_rl_env_gpu import GeneralsEnv


# =============================================================================
# Environment Patches and Utilities
# =============================================================================

def _patch_get_reduced_action_space(self: GeneralsEnv) -> Tuple[List[int], Dict[int, Tuple[int, int, int, int]]]:
    """
    Generate a reduced action space for the environment.
    
    Returns a tuple containing:
    - List of legal action indices
    - Dictionary mapping action indices to (from_x, from_y, to_x, to_y) coordinates
    
    This method uses the environment's own validators to ensure only legal moves
    are included in the action space.
    """
    legal_actions = []
    action_mapping = {}
    action_index = 0
    
    # Define movement directions: Up, Down, Left, Right
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    
    for y in range(self.grid_height):
        for x in range(self.grid_width):
            cell = self.game.grid[y][x]
            
            # Skip if cell doesn't belong to player or has no army
            if cell.owner != self.player_id or cell.army < 1:
                continue
                
            # Check all four directions
            for dx, dy in directions:
                new_x, new_y = x + dx, y + dy
                
                # Check bounds
                if not (0 <= new_x < self.grid_width and 0 <= new_y < self.grid_height):
                    continue
                    
                # Check if move is valid
                if not self._is_valid_move(x, y, new_x, new_y):
                    continue
                    
                action_mapping[action_index] = (x, y, new_x, new_y)
                legal_actions.append(action_index)
                action_index += 1
    
    # Store mapping on environment for step_with_reduced_action
    self.action_mapping = action_mapping
    return legal_actions, action_mapping


def _silent_step_with_reduced_action(self: GeneralsEnv, action_idx: int):
    """
    Execute a step using reduced action space without debug output.
    
    Args:
        action_idx: Index of the action to execute
        
    Returns:
        Tuple of (observation, reward, done, info)
    """
    if not getattr(self, "action_mapping", None):
        return self.step(0)  # No-op if nothing legal yet
        
    if action_idx not in self.action_mapping:
        return (self._get_observation(), self.invalid_move_penalty, 
                False, {"invalid_move": True})

    from_x, from_y, to_x, to_y = self.action_mapping[action_idx]
    
    # Convert to flat action index
    flat_action = from_y * self.grid_width * 4 + from_x * 4
    if to_y < from_y:      # Up
        flat_action += 0
    elif to_y > from_y:    # Down
        flat_action += 1
    elif to_x < from_x:    # Left
        flat_action += 2
    else:                  # Right
        flat_action += 3
        
    return self.step(flat_action)


def _quiet_execute_move(self, move):
    """Execute a move without printing debug information."""
    with contextlib.redirect_stdout(io.StringIO()):
        return Game._orig_execute_move(self, move)


def _quiet_get_valid_source_cells(self):
    """Get valid source cells without debug output."""
    return [(x, y) for y in range(self.grid_height) 
            for x in range(self.grid_width)
            if (cell := self.game.grid[y][x]).owner == self.player_id and cell.army >= 1]


def _setup_environment_patches():
    """Apply patches to the environment to reduce verbosity and add functionality."""
    # Add reduced action space method
    if not hasattr(GeneralsEnv, "get_reduced_action_space"):
        GeneralsEnv.get_reduced_action_space = _patch_get_reduced_action_space
    
    # Replace step_with_reduced_action with silent version
    GeneralsEnv.step_with_reduced_action = _silent_step_with_reduced_action
    
    # Suppress Game.execute_move debug output
    if not hasattr(Game, "_orig_execute_move"):
        Game._orig_execute_move = Game.execute_move
        Game.execute_move = _quiet_execute_move
    
    # Replace get_valid_source_cells with quiet version
    if hasattr(GeneralsEnv, "get_valid_source_cells"):
        GeneralsEnv.get_valid_source_cells = _quiet_get_valid_source_cells


# =============================================================================
# Greedy Agent Implementation
# =============================================================================

class GreedyAgent:
    """
    A greedy heuristic agent for the Generals.io environment.
    
    This agent makes decisions based on immediate rewards without memory between
    turns. It prioritizes:
    1. Capturing neutral cities (highest priority)
    2. Expanding to neutral territory
    3. Attacking weaker enemy positions
    4. Moving larger army stacks
    """
    
    def select(self, env: GeneralsEnv) -> Optional[int]:
        """
        Select the best action for the current game state.
        
        Args:
            env: The current game environment
            
        Returns:
            Action index to execute, or None if no legal actions available
        """
        legal_actions, action_mapping = env.get_reduced_action_space()
        
        if not legal_actions:
            return None  # Skip turn if nothing legal
        
        best_score = float('-inf')
        best_action = random.choice(legal_actions)  # Default to random choice
        
        for action_id in legal_actions:
            from_x, from_y, to_x, to_y = action_mapping[action_id]
            source_cell = env.game.grid[from_y][from_x]
            target_cell = env.game.grid[to_y][to_x]
            
            score = self._evaluate_move(source_cell, target_cell, env.player_id)
            
            if score > best_score:
                best_score = score
                best_action = action_id
                
        return best_action
    
    def _evaluate_move(self, source_cell, target_cell, player_id: int) -> float:
        """
        Evaluate the score of a potential move.
        
        Args:
            source_cell: The cell we're moving from
            target_cell: The cell we're moving to
            player_id: Our player ID
            
        Returns:
            Score for this move (higher is better)
        """
        score = 0.0
        
        # Priority 1: Capture neutral cities (highest reward)
        if (target_cell.type == CellType.CITY and 
            target_cell.owner == -1 and 
            source_cell.army > target_cell.army):
            score += 1000
            
        # Priority 2: Expand to neutral territory
        elif target_cell.owner == -1:
            score += 100
            
        # Priority 3: Attack weaker enemy positions
        elif (target_cell.owner not in (-1, player_id) and 
              source_cell.army > target_cell.army):
            score += 50 + (target_cell.army - source_cell.army)
            
        # Priority 4: Prefer moves that mobilize larger army stacks
        score += source_cell.army * 0.1
        
        return score


# =============================================================================
# Episode Management
# =============================================================================

def run_episode(agent: GreedyAgent, grid_size: int) -> bool:
    """
    Run a single episode with the given agent.
    
    Args:
        agent: The agent to use for decision making
        grid_size: Size of the game grid (assumed square)
        
    Returns:
        True if the agent won, False otherwise
    """
    env = GeneralsEnv(player_id=0, num_opponents=1, training_mode=True)
    
    # Resize grid if needed
    if grid_size != env.grid_width:
        env.grid_width = env.grid_height = grid_size
    
    env.reset()
    
    done = False
    while not done:
        action_id = agent.select(env)
        
        if action_id is None:
            # No legal actions, take a dummy step
            _, _, done, _ = env.step(0)
        else:
            _, _, done, _ = env.step_with_reduced_action(action_id)
    
    return env.game.winner == env.player_id


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run the greedy agent evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a greedy baseline agent for Generals.io"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=200,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--grid", 
        type=int, 
        default=25,
        help="Size of the game grid (assumed square)"
    )
    
    args = parser.parse_args()
    
    # Setup environment patches
    _setup_environment_patches()
    
    # Create agent and run evaluation
    agent = GreedyAgent()
    wins = 0
    
    print(f"Running {args.episodes} episodes on {args.grid}x{args.grid} grid...")
    
    for episode in range(1, args.episodes + 1):
        if run_episode(agent, args.grid):
            wins += 1
            
        # Print progress every 20 episodes or at the end
        if episode % 20 == 0 or episode == args.episodes:
            win_rate = wins / episode
            print(f"Episode {episode}/{args.episodes} – Win rate: {win_rate:.1%}")
    
    print(f"\nFinal Results:")
    print(f"Total episodes: {args.episodes}")
    print(f"Wins: {wins}")
    print(f"Win rate: {wins/args.episodes:.1%}")


if __name__ == "__main__":
    main()
