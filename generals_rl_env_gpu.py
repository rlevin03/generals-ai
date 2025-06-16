from generals import Game, GRID_WIDTH, GRID_HEIGHT, CellType
import gym
import numpy as np
import random
import torch


class GeneralsEnv(gym.Env):
    def __init__(self, player_id=0, num_opponents=3, render_mode=None, 
                 grid_size=None, training_mode=True, device='cpu'):
        self.player_id = player_id
        self.num_opponents = num_opponents
        self.render_mode = render_mode
        self.training_mode = training_mode  # Note: Can't actually disable delays with original Game
        self.device = device
        
        # Use default grid size from original game
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
            
        self.game = None
        
        # Action space: from_cell * 4 directions
        self.action_space = gym.spaces.Discrete(self.grid_width * self.grid_height * 4)
        
        # Observation space: keep spatial structure
        # Channels: [owner, army, is_city, is_general, is_mountain, is_visible]
        self.observation_space = gym.spaces.Box(
            low=-1, high=1000, shape=(self.grid_height, self.grid_width, 6), dtype=np.float32
        )
        
        # Track game statistics
        self.invalid_move_penalty = -0.1
        self.step_count = 0
        self.max_steps = 5000  # Increased from 1000 to 5000 to allow longer episodes
        self.last_territory_count = 0
        self.last_army_count = 0
        self.episode_reward = 0
        
        # Initialize action mapping
        self.action_mapping = {}
        
    def reset(self):
        """Reset the environment"""
        # Create game with original Game class
        self.game = Game()
        self.step_count = 0
        self.episode_reward = 0
        
        # Get initial counts for reward calculation
        self.last_territory_count = self._count_territory()
        self.last_army_count = self._count_army()
        
        return self._get_observation()
    
    def step(self, action):
        """Execute one step in the environment"""
        self.step_count += 1
        
        # Decode and validate action
        from_x, from_y, to_x, to_y = self._decode_action(action)
        
        # Check if move is valid before executing
        if not self._is_valid_move(from_x, from_y, to_x, to_y):
            # Invalid move penalty
            reward = self.invalid_move_penalty
            self.episode_reward += reward
            return self._get_observation(), reward, False, {
                "invalid_move": True,
                "reason": "Invalid cell selection or movement",
                "episode_reward": self.episode_reward
            }
        
        # Calculate army to move
        cell = self.game.grid[from_y][from_x]
        # Allow moving all armies including the last one
        if cell.army == 1:
            army_to_move = 1  # Move the single army
        else:
            army_to_move = max(1, cell.army // 2)  # Move half for multiple armies
        
        # Queue the move
        success = self.game.queue_move(from_x, from_y, to_x, to_y, army_to_move, self.player_id)
        
        if not success:
            reward = self.invalid_move_penalty
            self.episode_reward += reward
            return self._get_observation(), reward, False, {
                "invalid_move": True,
                "reason": "Move queue failed",
                "episode_reward": self.episode_reward
            }
        
        # Simulate opponent moves (simple random AI)
        self._simulate_opponents()
        
        # Update game state (no delays in training mode)
        self.game.update()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.episode_reward += reward
        
        # Check if game is over
        done = self._is_done()
        
        # Additional info
        info = {
            "turn": self.game.get_turn_number(),
            "territory": self._count_territory(),
            "army": self._count_army(),
            "step": self.step_count,
            "episode_reward": self.episode_reward
        }
        
        return self._get_observation(), reward, done, info
    
    def _get_observation(self):
        """Get the current observation - optimized for GPU processing"""
        obs = np.zeros((self.grid_height, self.grid_width, 6), dtype=np.float32)
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                
                # Check visibility
                if self.player_id in cell.visible_to:
                    # Visible cell
                    obs[y, x, 0] = cell.owner  # -1 for neutral, 0-3 for players
                    obs[y, x, 1] = min(cell.army, 999) / 100.0  # Normalize army count
                    obs[y, x, 2] = float(cell.type == CellType.CITY)
                    obs[y, x, 3] = float(cell.type == CellType.GENERAL)
                    obs[y, x, 4] = float(cell.type == CellType.MOUNTAIN)
                    obs[y, x, 5] = 1.0  # Is visible
                else:
                    # Fog of war
                    obs[y, x, 0] = -1
                    obs[y, x, 5] = 0.0  # Not visible
        
        return obs
    
    def _get_observation_tensor(self):
        """Get observation directly as a PyTorch tensor on the specified device"""
        obs = self._get_observation()
        return torch.from_numpy(obs).float().to(self.device)
    
    def _decode_action(self, action):
        """Decode action integer to movement coordinates"""
        from_idx = action // 4
        direction = action % 4
        
        from_x = from_idx % self.grid_width
        from_y = from_idx // self.grid_width
        
        # Direction: 0=up, 1=down, 2=left, 3=right
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        to_x = from_x + dx
        to_y = from_y + dy
        
        return from_x, from_y, to_x, to_y
    
    def _is_valid_move(self, from_x, from_y, to_x, to_y):
        """Check if a move is valid"""
        # Check bounds
        if not (0 <= from_x < self.grid_width and 0 <= from_y < self.grid_height):
            return False
        if not (0 <= to_x < self.grid_width and 0 <= to_y < self.grid_height):
            return False
        
        # Check ownership
        from_cell = self.game.grid[from_y][from_x]
        if from_cell.owner != self.player_id:
            return False
        
        # Check if we have army to move - allow moves with 1 army
        if from_cell.army < 1:
            return False
        
        # Check if destination is visible and is mountain
        to_cell = self.game.grid[to_y][to_x]
        if self.player_id in to_cell.visible_to and to_cell.type == CellType.MOUNTAIN:
            return False
            
        # Allow moves to any non-mountain cell, even if we can't see it
        # This encourages exploration and aggressive play
        return True
    
    def _simulate_opponents(self):
        """Simple AI for opponents - random valid moves"""
        for player_id in range(len(self.game.players)):
            if player_id == self.player_id or not self.game.players[player_id].is_alive:
                continue
            
            # Find all possible moves for this player
            possible_moves = []
            for y in range(self.grid_height):
                for x in range(self.grid_width):
                    cell = self.game.grid[y][x]
                    if cell.owner == player_id and cell.army >= 1:  # Allow moves with 1 army
                        # Check all 4 directions
                        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                            to_x, to_y = x + dx, y + dy
                            if 0 <= to_x < self.grid_width and 0 <= to_y < self.grid_height:
                                to_cell = self.game.grid[to_y][to_x]
                                if to_cell.type != CellType.MOUNTAIN:
                                    # Calculate army to move
                                    if cell.army == 1:
                                        army_to_move = 1  # Move the single army
                                    else:
                                        army_to_move = max(1, cell.army // 2)  # Move half
                                    possible_moves.append((x, y, to_x, to_y, army_to_move))
            
            # Make 1-3 random moves
            if possible_moves:
                num_moves = min(len(possible_moves), random.randint(1, 3))
                for _ in range(num_moves):
                    if possible_moves:
                        move = random.choice(possible_moves)
                        from_x, from_y, to_x, to_y, army = move
                        self.game.queue_move(from_x, from_y, to_x, to_y, army, player_id)
                        # Remove this move to avoid duplicates
                        possible_moves.remove(move)
    
    def _calculate_reward(self):
        """Calculate reward based on game state changes"""
        reward = 0.0
        
        # Territory expansion reward
        current_territory = self._count_territory()
        territory_gain = current_territory - self.last_territory_count
        reward += territory_gain * 1.0
        
        # Army growth reward
        current_army = self._count_army()
        army_gain = current_army - self.last_army_count
        reward += army_gain * 0.01
        
        # Capturing cities/generals
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner == self.player_id:
                    if cell.type == CellType.CITY:
                        reward += 0.1  # Small bonus for owning cities
                    elif cell.type == CellType.GENERAL and (x, y) != self.game.players[self.player_id].general_pos:
                        reward += 50.0  # Big bonus for capturing enemy general
        
        # Game over rewards
        if self.game.game_over:
            if self.game.winner == self.player_id:
                reward += 100.0  # Win bonus
            else:
                reward -= 50.0  # Loss penalty
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        # Update counters
        self.last_territory_count = current_territory
        self.last_army_count = current_army
        
        return reward
    
    def _count_territory(self):
        """Count territories owned by the player"""
        count = 0
        for row in self.game.grid:
            for cell in row:
                if cell.owner == self.player_id:
                    count += 1
        return count
    
    def _count_army(self):
        """Count total army owned by the player"""
        count = 0
        for row in self.game.grid:
            for cell in row:
                if cell.owner == self.player_id:
                    count += cell.army
        return count
    
    def _is_done(self):
        """Check if episode is done"""
        # Game over conditions
        if self.game.game_over:
            return True
        
        # Player eliminated
        if not self.game.players[self.player_id].is_alive:
            return True
        
        # Max steps reached
        if self.step_count >= self.max_steps:
            return True
        
        return False
    
    def render(self, mode='human'):
        """Render the game state (optional)"""
        if self.render_mode == 'human' and not self.training_mode:
            # You could integrate with pygame rendering here
            pass
    
    def get_valid_actions(self):
        """Get mask of valid actions for the current state"""
        valid = np.zeros(self.action_space.n, dtype=bool)
        
        for action in range(self.action_space.n):
            from_x, from_y, to_x, to_y = self._decode_action(action)
            if self._is_valid_move(from_x, from_y, to_x, to_y):
                valid[action] = True
        
        return valid
    
    def get_valid_actions_tensor(self):
        """Get valid actions mask as a tensor on the specified device"""
        valid = self.get_valid_actions()
        return torch.from_numpy(valid).to(self.device)
    
    def get_valid_source_cells(self):
        """Get cells that can make moves"""
        valid_sources = []
        owned_cells = 0
        cells_with_armies = 0
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.game.grid[y][x]
                if cell.owner == self.player_id:
                    owned_cells += 1
                    if cell.army >= 1:  # Changed from > 1 to >= 1 to allow moves with 1 army
                        cells_with_armies += 1
                        # Add cell as valid source if it has at least 1 army
                        valid_sources.append((x, y))
        
        # Debug output
        if len(valid_sources) == 0 and owned_cells > 0:
            print(f"Debug: Player {self.player_id} owns {owned_cells} cells, {cells_with_armies} have >=1 army, but no valid moves!")
            if cells_with_armies > 0:
                # Check why no valid moves
                for y in range(self.grid_height):
                    for x in range(self.grid_width):
                        cell = self.game.grid[y][x]
                        if cell.owner == self.player_id and cell.army >= 1:
                            print(f"  Cell ({x},{y}) has {cell.army} armies")
                            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                nx, ny = x + dx, y + dy
                                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                                    to_cell = self.game.grid[ny][nx]
                                    print(f"    -> ({nx},{ny}): type={to_cell.type}, visible={self.player_id in to_cell.visible_to}")
        
        return valid_sources
    
    def step_with_reduced_action(self, action_idx):
        """Execute action using reduced action space"""
        # Update action mapping based on current valid moves
        valid_sources = self.get_valid_source_cells()
        self.action_mapping = {}
        
        for idx, (from_x, from_y) in enumerate(valid_sources):
            cell = self.game.grid[from_y][from_x]
            if cell.army >= 1:  # Only consider cells with armies
                # Check all 4 directions
                for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    to_x, to_y = from_x + dx, from_y + dy
                    if 0 <= to_x < self.grid_width and 0 <= to_y < self.grid_height:
                        to_cell = self.game.grid[to_y][to_x]
                        if to_cell.type != CellType.MOUNTAIN:
                            self.action_mapping[idx] = (from_x, from_y, to_x, to_y)
        
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
        
        if action_idx not in self.action_mapping:
            # Invalid action
            return self._get_observation(), self.invalid_move_penalty, False, {"invalid_move": True}
        
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