import pygame
import numpy as np
from generals import CellType, PLAYER_COLORS
import time

class TrainingVisualizer:
    """Visualize game state during training"""
    
    def __init__(self, grid_width=25, grid_height=20, cell_size=20):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size
        self.window_width = grid_width * cell_size + 200
        self.window_height = grid_height * cell_size + 100
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (64, 64, 64)
        self.FOG = (40, 40, 40)
        
        self.screen = None
        self.font = None
        self.small_font = None
        self.initialized = False
        
    def init_pygame(self):
        """Initialize pygame (only when needed)"""
        if not self.initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Training Visualization")
            self.font = pygame.font.Font(None, 16)
            self.small_font = pygame.font.Font(None, 12)
            self.initialized = True
    
    def close(self):
        """Close pygame window"""
        if self.initialized:
            pygame.quit()
            self.initialized = False
    
    def visualize_state(self, env, episode, step, reward, action_info=None):
        """Visualize current game state"""
        if not self.initialized:
            self.init_pygame()
        
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Get game state
        game = env.game
        player_id = env.player_id
        
        # Draw grid
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = game.grid[y][x]
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                
                # Determine cell color
                if player_id in cell.visible_to:
                    if cell.type == CellType.MOUNTAIN:
                        color = self.BLACK
                    elif cell.owner >= 0:
                        base_color = PLAYER_COLORS[cell.owner]
                        color = tuple(max(0, c - 30) for c in base_color)
                    else:
                        color = self.GRAY
                else:
                    color = self.FOG
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.DARK_GRAY, rect, 1)
                
                # Draw cell contents if visible
                if player_id in cell.visible_to:
                    # Draw army count
                    if cell.army > 0:
                        text = self.small_font.render(str(cell.army), True, self.WHITE)
                        text_rect = text.get_rect(center=rect.center)
                        self.screen.blit(text, text_rect)
                    
                    # Mark special cells
                    if cell.type == CellType.GENERAL:
                        pygame.draw.circle(self.screen, self.WHITE, rect.center, 5, 2)
                    elif cell.type == CellType.CITY:
                        pygame.draw.rect(self.screen, self.WHITE, 
                                       (rect.centerx - 3, rect.centery - 3, 6, 6), 2)
        
        # Draw action if provided
        if action_info:
            from_x, from_y, to_x, to_y = action_info
            if 0 <= from_x < self.grid_width and 0 <= from_y < self.grid_height:
                from_rect = pygame.Rect(from_x * self.cell_size, from_y * self.cell_size,
                                      self.cell_size, self.cell_size)
                to_rect = pygame.Rect(to_x * self.cell_size, to_y * self.cell_size,
                                    self.cell_size, self.cell_size)
                
                # Highlight source and destination
                pygame.draw.rect(self.screen, (255, 255, 0), from_rect, 3)
                pygame.draw.rect(self.screen, (0, 255, 0), to_rect, 3)
                
                # Draw arrow
                pygame.draw.line(self.screen, (255, 255, 0), 
                               from_rect.center, to_rect.center, 2)
        
        # Draw stats
        y_offset = 10
        stats = [
            f"Episode: {episode}",
            f"Step: {step}",
            f"Reward: {reward:.2f}",
            f"Territory: {env._count_territory()}",
            f"Army: {env._count_army()}",
            ""
        ]
        
        # Add player stats
        for player in game.players:
            if player.is_alive:
                color = PLAYER_COLORS[player.id]
                text = f"P{player.id}: Army {player.total_army}, Land {player.total_land}"
                if player.id == player_id:
                    text = ">> " + text + " <<"
                rendered = self.font.render(text, True, color)
                self.screen.blit(rendered, (self.grid_width * self.cell_size + 10, y_offset))
                y_offset += 20
        
        # Update display
        pygame.display.flip()
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        
        return True

# Add this to your training loop
def add_visualization_to_training(trainer):
    """Add visualization capability to existing trainer"""
    
    # Create visualizer
    visualizer = TrainingVisualizer()
    
    # Store original collect_experience method
    original_collect = trainer.collect_parallel_experience
    
    def collect_with_viz(num_steps=100):
        """Modified collection with visualization"""
        # Only visualize one environment (first one)
        viz_env_idx = 0
        viz_interval = 100  # Visualize every N steps
        
        # Call original method but intercept for visualization
        states = trainer.states
        
        for step in range(num_steps):
            # Get valid actions
            valid_actions_list, _ = trainer.env_pool.get_reduced_actions()
            
            # Select actions
            actions = trainer.act_batch(states, valid_actions_list)
            
            # Visualize before step (occasionally)
            if trainer.step_count % viz_interval == 0:
                env = trainer.env_pool.envs[viz_env_idx]
                
                # Get action info for visualization
                if viz_env_idx < len(actions) and actions[viz_env_idx] in env.action_mapping:
                    action_info = env.action_mapping[actions[viz_env_idx]]
                else:
                    action_info = None
                
                # Visualize
                visualizer.visualize_state(
                    env, 
                    trainer.episode_count,
                    trainer.step_count,
                    trainer.episode_rewards[viz_env_idx] if hasattr(trainer, 'episode_rewards') else 0,
                    action_info
                )
                
                # Small delay to see the move
                time.sleep(0.1)
            
            # Continue with normal training...
            # (rest of the collect_parallel_experience logic)
            next_states, rewards, dones, infos = trainer.env_pool.step(actions)
            
            # ... (continue with original implementation)
            
    # Replace method
    trainer.collect_parallel_experience = collect_with_viz
    
    return visualizer
