"""
Generals Game Reinforcement Learning Environment

Agent-agnostic environment: owns the Game, handles reset/step/done.
Provides the controller with raw state (basic tensor) and masked valid actions
(explicit moves). Receives explicit actions from the controller and returns
result (state, reward, done, info).

Most of this code is AI generated.
"""

from game import Game, GRID_WIDTH, GRID_HEIGHT, CellType
import gym
import numpy as np
import random
import torch
from typing import Tuple, Dict, Any, List, Optional


class GeneralsEnv(gym.Env):
    """
    A Gym-compatible environment for the Generals game.
    
    This environment wraps the Generals game with a reinforcement learning interface,
    providing observations as multi-channel tensors and actions as discrete moves.
    The environment supports GPU acceleration and includes opponent simulation.
    
    Attributes:
        player_id (int): The ID of the learning agent (default: 0)
        num_opponents (int): Number of AI opponents (default: 3)
        render_mode (str): Rendering mode for visualization
        training_mode (bool): Whether in training mode (affects delays)
        device (str): Device for tensor operations ('cpu' or 'cuda')
        grid_width (int): Width of the game grid
        grid_height (int): Height of the game grid
        action_space (gym.spaces.Discrete): Action space for all possible moves
        observation_space (gym.spaces.Box): Observation space with 6 channels
        invalid_move_penalty (float): Penalty for invalid moves
        max_steps (int): Maximum steps per episode
    """
    
    def __init__(self, 
                 player_id: int = 0, 
                 num_opponents: int = 3, 
                 render_mode: Optional[str] = None, 
                 grid_size: Optional[Tuple[int, int]] = None, 
                 training_mode: bool = True, 
                 device: str = 'cpu') -> None:
        """
        Initialize the Generals environment.
        
        Args:
            player_id: The ID of the learning agent
            num_opponents: Number of AI opponents to simulate
            render_mode: Rendering mode ('human', 'rgb_array', etc.)
            grid_size: Custom grid size (width, height), uses default if None
            training_mode: Whether in training mode (affects game delays)
            device: Device for tensor operations ('cpu' or 'cuda')
        """
        super().__init__()
        
        # Environment configuration
        self.player_id = player_id  # kept for backward compat; acting player is current_player_index
        self.num_players = 4
        self.current_player_index = 0  # whose turn to act (0-3); env maintains this, advances after each step
        self.num_opponents = num_opponents
        self.render_mode = render_mode
        self.training_mode = training_mode
        self.device = device
        
        # Grid dimensions (use default from original game)
        if grid_size is not None:
            self.grid_width, self.grid_height = grid_size
        else:
            self.grid_width = GRID_WIDTH
            self.grid_height = GRID_HEIGHT
        
        # Game state
        self.game = None
        
        # Action space: from_cell * 4 directions
        self.action_space = gym.spaces.Discrete(self.grid_width * self.grid_height * 4)
        
        # Observation space: 6-channel spatial representation
        # Channels: [owner, army, is_city, is_general, is_mountain, is_visible]
        self.observation_space = gym.spaces.Box(
            low=-1, high=1000, shape=(self.grid_height, self.grid_width, 6), dtype=np.float32
        )
        
        # Training parameters
        self.invalid_move_penalty = -0.1
        self.step_count = 0
        self.max_steps = 5000
        self.last_territory_count = 0
        self.last_army_count = 0
        self.episode_reward = 0
        
        # Action mapping for reduced action space
        self.action_mapping = {}
        
        # Direction vectors for movement
        self.directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

    # -------------------------------------------------------------------------
    # Agent-agnostic API (for controller)
    # -------------------------------------------------------------------------

    def get_state(self) -> np.ndarray:
        """Current raw state as a basic tensor (H, W, C). Agent-agnostic."""
        return self._get_state()

    def get_valid_actions_explicit(self) -> List[Tuple[int, int, int, int]]:
        """Valid actions as explicit moves: list of (from_x, from_y, to_x, to_y)."""
        valid = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner != self.current_player_index or cell.army < 1:
                    continue
                for dx, dy in self.directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                        to_cell = self.game.grid[ny][nx]
                        if self.current_player_index not in to_cell.visible_to or to_cell.type != CellType.MOUNTAIN:
                            valid.append((x, y, nx, ny))
        return valid

    def step_explicit(
        self, action: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Apply one explicit action from the controller.
        action: (from_x, from_y, to_x, to_y) or None for no-op.
        Returns: (state, reward, done, info).
        """
        self.step_count += 1
        if action is not None:
            from_x, from_y, to_x, to_y = action
            if not self._is_valid_move(from_x, from_y, to_x, to_y):
                self.episode_reward += self.invalid_move_penalty
                return self._get_state(), self.invalid_move_penalty, False, {
                    "invalid_move": True, "episode_reward": self.episode_reward,
                    "territory": self._count_territory(), "army": self._count_army(), "step": self.step_count,
                }
            cell = self.game.grid[from_y][from_x]
            army_to_move = self._calculate_army_to_move(cell.army)
            success = self.game.queue_move(from_x, from_y, to_x, to_y, army_to_move, self.current_player_index)
            if not success:
                self.episode_reward += self.invalid_move_penalty
                return self._get_state(), self.invalid_move_penalty, False, {
                    "invalid_move": True, "episode_reward": self.episode_reward,
                    "territory": self._count_territory(), "army": self._count_army(), "step": self.step_count,
                }
        # Only current player acts; no opponent simulation. Advance to next player after game update.
        self.game.update()
        self._advance_current_player()
        reward = self._calculate_reward()
        self.episode_reward += reward
        done = self._is_done()
        info = {
            "turn": self.game.get_turn_number(),
            "territory": self._count_territory(),
            "army": self._count_army(),
            "step": self.step_count,
            "episode_reward": self.episode_reward,
            "current_player_index": self.current_player_index,
            "winner": getattr(self.game, "winner", None),
        }
        return self._get_state(), reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            np.ndarray: Initial observation of shape (height, width, 6)
        """
        # Initialize new game
        self.game = Game()
        self.step_count = 0
        self.episode_reward = 0
        self.current_player_index = 0
        
        # Initialize tracking variables for reward calculation (per current player)
        self.last_territory_count = self._count_territory()
        self.last_army_count = self._count_army()
        
        return self._get_state()
    
    def _advance_current_player(self) -> None:
        """Advance current_player_index to next alive player (0-3 round-robin)."""
        start = self.current_player_index
        for _ in range(self.num_players):
            self.current_player_index = (self.current_player_index + 1) % self.num_players
            if self.game.players[self.current_player_index].is_alive:
                return
        self.current_player_index = start  # all dead or one left; keep index
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer representing the action to take
            
        Returns:
            Tuple containing:
                - observation: Current game state observation
                - reward: Reward for this step
                - done: Whether episode is finished
                - info: Additional information dictionary
        """
        self.step_count += 1
        
        # Decode action to movement coordinates
        from_x, from_y, to_x, to_y = self._decode_action(action)
        
        # Validate move before execution
        if not self._is_valid_move(from_x, from_y, to_x, to_y):
            reward = self.invalid_move_penalty
            self.episode_reward += reward
            return self._get_state(), reward, False, {
                "invalid_move": True,
                "reason": "Invalid cell selection or movement",
                "episode_reward": self.episode_reward
            }
        
        # Calculate army size to move
        cell = self.game.grid[from_y][from_x]
        army_to_move = self._calculate_army_to_move(cell.army)
        
        # Execute the move
        success = self.game.queue_move(from_x, from_y, to_x, to_y, army_to_move, self.player_id)
        
        if not success:
            reward = self.invalid_move_penalty
            self.episode_reward += reward
            return self._get_state(), reward, False, {
                "invalid_move": True,
                "reason": "Move queue failed",
                "episode_reward": self.episode_reward
            }
        
        # Simulate opponent actions
        self._simulate_opponents()
        
        # Update game state
        self.game.update()
        
        # Calculate reward and check termination
        reward = self._calculate_reward()
        self.episode_reward += reward
        done = self._is_done()
        
        # Prepare info dictionary
        info = {
            "turn": self.game.get_turn_number(),
            "territory": self._count_territory(),
            "army": self._count_army(),
            "step": self.step_count,
            "episode_reward": self.episode_reward
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current game state as a raw multi-channel array (agent-agnostic).
        Shape (height, width, 6): [owner, army, is_city, is_general, is_mountain, is_visible]
        """
        obs = np.zeros((self.grid_height, self.grid_width, 6), dtype=np.float32)
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                
                # Check if cell is visible to the current (acting) player
                if self.current_player_index in cell.visible_to:
                    # Visible cell - provide full information
                    obs[y, x, 0] = cell.owner  # -1 for neutral, 0-3 for players
                    obs[y, x, 1] = min(cell.army, 999) / 100.0  # Normalized army count
                    obs[y, x, 2] = float(cell.type == CellType.CITY)
                    obs[y, x, 3] = float(cell.type == CellType.GENERAL)
                    obs[y, x, 4] = float(cell.type == CellType.MOUNTAIN)
                    obs[y, x, 5] = 1.0  # Visibility flag
                else:
                    # Fog of war - minimal information
                    obs[y, x, 0] = -1  # Unknown owner
                    obs[y, x, 5] = 0.0  # Not visible
        
        return obs

    def _decode_action(self, action: int) -> Tuple[int, int, int, int]:
        """
        Decode action integer to movement coordinates.
        
        Args:
            action: Integer representing the action
            
        Returns:
            Tuple of (from_x, from_y, to_x, to_y) coordinates
        """
        from_idx = action // 4
        direction = action % 4
        
        from_x = from_idx % self.grid_width
        from_y = from_idx // self.grid_width
        
        # Apply direction vector
        dx, dy = self.directions[direction]
        to_x = from_x + dx
        to_y = from_y + dy
        
        return from_x, from_y, to_x, to_y
    
    def _is_valid_move(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """
        Check if a move is valid according to game rules.
        
        Args:
            from_x, from_y: Source cell coordinates
            to_x, to_y: Destination cell coordinates
            
        Returns:
            bool: True if move is valid, False otherwise
        """
        # Check boundary conditions
        if not (0 <= from_x < self.grid_width and 0 <= from_y < self.grid_height):
            return False
        if not (0 <= to_x < self.grid_width and 0 <= to_y < self.grid_height):
            return False
        
        # Check ownership
        from_cell = self.game.grid[from_y][from_x]
        if from_cell.owner != self.current_player_index:
            return False
        
        # Check army availability
        if from_cell.army < 1:
            return False
        
        # Check destination validity
        to_cell = self.game.grid[to_y][to_x]
        if self.current_player_index in to_cell.visible_to and to_cell.type == CellType.MOUNTAIN:
            return False
        
        # Allow moves to any non-mountain cell (encourages exploration)
        return True
    
    def _calculate_army_to_move(self, army_count: int) -> int:
        """
        Calculate how many armies to move from a cell.
        
        Args:
            army_count: Total armies in the source cell
            
        Returns:
            int: Number of armies to move
        """
        if army_count == 1:
            return 1  # Move the single army
        else:
            return max(1, army_count // 2)  # Move half for multiple armies
    
    def _simulate_opponents(self) -> None:
        """
        Simulate opponent actions using simple random AI.
        
        Each opponent makes 1-3 random valid moves per turn.
        """
        for player_id in range(len(self.game.players)):
            # Skip if this is the learning agent or player is eliminated
            if player_id == self.player_id or not self.game.players[player_id].is_alive:
                continue
            
            # Find all possible moves for this opponent
            possible_moves = self._get_possible_moves_for_player(player_id)
            
            # Execute 1-3 random moves
            if possible_moves:
                num_moves = min(len(possible_moves), random.randint(1, 3))
                for _ in range(num_moves):
                    if possible_moves:
                        move = random.choice(possible_moves)
                        from_x, from_y, to_x, to_y, army = move
                        self.game.queue_move(from_x, from_y, to_x, to_y, army, player_id)
                        possible_moves.remove(move)  # Avoid duplicate moves
    
    def _get_possible_moves_for_player(self, player_id: int) -> List[Tuple[int, int, int, int, int]]:
        """
        Get all possible moves for a given player.
        
        Args:
            player_id: ID of the player to find moves for
            
        Returns:
            List of tuples (from_x, from_y, to_x, to_y, army_to_move)
        """
        possible_moves = []
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner == player_id and cell.army >= 1:
                    # Check all 4 directions
                    for dx, dy in self.directions:
                        to_x, to_y = x + dx, y + dy
                        if 0 <= to_x < self.grid_width and 0 <= to_y < self.grid_height:
                            to_cell = self.game.grid[to_y][to_x]
                            if to_cell.type != CellType.MOUNTAIN:
                                army_to_move = self._calculate_army_to_move(cell.army)
                                possible_moves.append((x, y, to_x, to_y, army_to_move))
        
        return possible_moves
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward based on game state changes.
        
        Returns:
            float: Reward value for the current step
        """
        reward = 0.0
        
        # Territory expansion reward
        current_territory = self._count_territory()
        territory_gain = current_territory - self.last_territory_count
        reward += territory_gain * 1.0
        
        # Army growth reward
        current_army = self._count_army()
        army_gain = current_army - self.last_army_count
        reward += army_gain * 0.01
        
        # Strategic position rewards
        reward += self._calculate_strategic_rewards()
        
        # Game completion rewards
        if self.game.game_over:
            if self.game.winner == self.current_player_index:
                reward += 100.0  # Victory bonus
            else:
                reward -= 50.0   # Defeat penalty
        
        # Efficiency penalty (encourages faster play)
        reward -= 0.01
        
        # Update tracking variables
        self.last_territory_count = current_territory
        self.last_army_count = current_army
        
        return reward
    
    def _calculate_strategic_rewards(self) -> float:
        """
        Calculate rewards for strategic achievements.
        
        Returns:
            float: Strategic reward component
        """
        reward = 0.0
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner == self.current_player_index:
                    if cell.type == CellType.CITY:
                        reward += 0.1  # City ownership bonus
                    elif cell.type == CellType.GENERAL and (x, y) != self.game.players[self.current_player_index].general_pos:
                        reward += 50.0  # Enemy general capture bonus
        
        return reward
    
    def _count_territory(self) -> int:
        """
        Count territories owned by the current (acting) player.
        """
        count = 0
        for row in self.game.grid:
            for cell in row:
                if cell.owner == self.current_player_index:
                    count += 1
        return count
    
    def _count_army(self) -> int:
        """
        Count total army size for the current (acting) player.
        """
        count = 0
        for row in self.game.grid:
            for cell in row:
                if cell.owner == self.current_player_index:
                    count += cell.army
        return count
    
    def _is_done(self) -> bool:
        """
        Check if the episode should terminate.
        
        Returns:
            bool: True if episode should end, False otherwise
        """
        # Game completion
        if self.game.game_over:
            return True
        
        # Player elimination (current player dead)
        if not self.game.players[self.current_player_index].is_alive:
            return True
        
        # Step limit reached
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def is_done(self) -> bool:
        """True if the game has ended (game over or step limit)."""
        return self._is_done()
    
    def get_valid_actions(self) -> np.ndarray:
        """
        Get a boolean mask of valid actions for the current state.
        
        Returns:
            np.ndarray: Boolean array where True indicates valid actions
        """
        valid = np.zeros(self.action_space.n, dtype=bool)
        
        for action in range(self.action_space.n):
            from_x, from_y, to_x, to_y = self._decode_action(action)
            if self._is_valid_move(from_x, from_y, to_x, to_y):
                valid[action] = True
        
        return valid
    
    def get_valid_actions_tensor(self) -> torch.Tensor:
        """
        Get valid actions mask as a PyTorch tensor.
        
        Returns:
            torch.Tensor: Boolean tensor of valid actions on specified device
        """
        valid = self.get_valid_actions()
        return torch.from_numpy(valid).to(self.device)
    
    def get_valid_source_cells(self) -> List[Tuple[int, int]]:
        """
        Get list of cells that can make moves (have armies).
        
        Returns:
            List[Tuple[int, int]]: List of (x, y) coordinates of valid source cells
        """
        valid_sources = []
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner == self.current_player_index and cell.army >= 1:
                    valid_sources.append((x, y))
        
        return valid_sources
    
    def step_with_reduced_action(self, action_idx: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute action using a reduced action space based on valid moves.
        
        This method provides a more efficient action space by only considering
        currently valid moves, reducing the action space from the full grid
        to just the available moves.
        
        Args:
            action_idx: Index in the reduced action space
            
        Returns:
            Tuple containing (observation, reward, done, info)
        """
        # Update action mapping based on current valid moves
        valid_sources = self.get_valid_source_cells()
        self.action_mapping = {}
        
        # Build mapping from reduced action space to full actions
        for idx, (from_x, from_y) in enumerate(valid_sources):
            cell = self.game.grid[from_y][from_x]
            if cell.army >= 1:
                # Check all 4 directions
                for dx, dy in self.directions:
                    to_x, to_y = from_x + dx, from_y + dy
                    if 0 <= to_x < self.grid_width and 0 <= to_y < self.grid_height:
                        to_cell = self.game.grid[to_y][to_x]
                        if to_cell.type != CellType.MOUNTAIN:
                            self.action_mapping[idx] = (from_x, from_y, to_x, to_y)
        
        # Handle case with no valid actions
        if not self.action_mapping:
            # Execute a no-op step
            dummy_action = 0
            result = self.step(dummy_action)
            obs, reward, done, info = result
            
            # Override penalty if it was due to no valid moves
            if 'invalid_move' in info and info['invalid_move']:
                reward = -0.01  # Minimal time penalty
                info['no_valid_actions'] = True
                info['invalid_move'] = False
            
            return obs, reward, done, info
        
        # Execute the mapped action
        if action_idx not in self.action_mapping:
            return self._get_state(), self.invalid_move_penalty, False, {"invalid_move": True}
        
        from_x, from_y, to_x, to_y = self.action_mapping[action_idx]
        
        # Convert to original action space
        original_action = self._encode_action(from_x, from_y, to_x, to_y)
        
        return self.step(original_action)
    
    def _encode_action(self, from_x: int, from_y: int, to_x: int, to_y: int) -> int:
        """
        Encode movement coordinates to action integer.
        
        Args:
            from_x, from_y: Source coordinates
            to_x, to_y: Destination coordinates
            
        Returns:
            int: Encoded action
        """
        # Calculate base action for source cell
        original_action = from_y * self.grid_width * 4 + from_x * 4
        
        # Add direction offset
        if to_y < from_y:
            original_action += 0  # Up
        elif to_y > from_y:
            original_action += 1  # Down
        elif to_x < from_x:
            original_action += 2  # Left
        else:
            original_action += 3  # Right
        
        return original_action
