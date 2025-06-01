import pygame
import random
import math
import time
from enum import Enum
from typing import List, Tuple, Optional, Dict, Deque
from collections import deque
import sys

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
GRID_WIDTH = 25
GRID_HEIGHT = 20
CELL_SIZE = 30
GRID_OFFSET_X = 50
GRID_OFFSET_Y = 50

# Game timing (in seconds)
TURN_DURATION = 1  # Each "turn" is 0.5 seconds like the original
ARMY_GENERATION_INTERVAL = 25 * TURN_DURATION  # Every 25 "turns" (12.5 seconds)

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (100, 255, 100)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (192, 192, 192)
FOG_COLOR = (40, 40, 40)

# Player colors
PLAYER_COLORS = [
    (255, 100, 100),  # Red
    (100, 100, 255),  # Blue
    (100, 255, 100),  # Green
    (255, 255, 100),  # Yellow
    (255, 100, 255),  # Magenta
    (100, 255, 255),  # Cyan
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
]

class CellType(Enum):
    EMPTY = 0
    MOUNTAIN = 1
    CITY = 2
    GENERAL = 3

class MoveCommand:
    def __init__(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int], army_count: int, player_id: int):
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.army_count = army_count
        self.player_id = player_id
        self.timestamp = time.time()

class Cell:
    def __init__(self, x: int, y: int, cell_type: CellType = CellType.EMPTY):
        self.x = x
        self.y = y
        self.type = cell_type
        self.owner = -1  # -1 for neutral, 0-7 for players
        self.army = 0
        self.visible_to = set()  # Which players can see this cell
        
        # Initialize army count based on cell type
        if cell_type == CellType.CITY:
            self.army = random.randint(40, 50)
        elif cell_type == CellType.GENERAL:
            self.army = 1

class Player:
    def __init__(self, player_id: int, color: Tuple[int, int, int]):
        self.id = player_id
        self.color = color
        self.general_pos = None
        self.is_alive = True
        self.total_army = 0
        self.total_land = 0
        self.move_queue = deque()  # Queue for move commands

class Game:
    def __init__(self):
        self.grid = [[Cell(x, y) for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
        self.players = [Player(i, PLAYER_COLORS[i]) for i in range(4)]  # Start with 4 players
        self.game_start_time = time.time()
        self.last_army_generation = time.time()
        self.last_general_tick = time.time()
        self.selected_cell = None
        self.current_player = 0  # For input handling
        self.game_over = False
        self.winner = -1
        
        # Generate map
        self.generate_map()
        self.place_generals()
        self.update_visibility()
        
    def generate_map(self):
        """Generate mountains and cities on the map"""
        # Add some mountains (impassable terrain)
        num_mountains = random.randint(15, 25)
        for _ in range(num_mountains):
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if self.grid[y][x].type == CellType.EMPTY:
                self.grid[y][x].type = CellType.MOUNTAIN
        
        # Add cities
        num_cities = random.randint(8, 12)
        for _ in range(num_cities):
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            if self.grid[y][x].type == CellType.EMPTY:
                self.grid[y][x].type = CellType.CITY
                self.grid[y][x].army = random.randint(40, 50)
    
    def place_generals(self):
        """Place generals for each player ensuring minimum distance"""
        positions = []
        min_distance = 15  # Minimum Manhattan distance between generals
        
        for player in self.players:
            attempts = 0
            while attempts < 100:  # Prevent infinite loop
                x = random.randint(1, GRID_WIDTH - 2)
                y = random.randint(1, GRID_HEIGHT - 2)
                
                # Check if position is valid (empty cell)
                if self.grid[y][x].type != CellType.EMPTY:
                    attempts += 1
                    continue
                
                # Check minimum distance from other generals
                valid = True
                for px, py in positions:
                    distance = abs(x - px) + abs(y - py)  # Manhattan distance
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    self.grid[y][x].type = CellType.GENERAL
                    self.grid[y][x].owner = player.id
                    self.grid[y][x].army = 1
                    player.general_pos = (x, y)
                    positions.append((x, y))
                    break
                
                attempts += 1
    
    def get_neighbors(self, x: int, y: int, include_diagonals: bool = False) -> List[Tuple[int, int]]:
        """Get valid neighboring coordinates"""
        neighbors = []
        # Cardinal directions (up, down, left, right)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                neighbors.append((nx, ny))
        
        # Diagonal directions for visibility
        if include_diagonals:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def update_visibility(self):
        """Update fog of war based on owned territories"""
        # Clear all visibility
        for row in self.grid:
            for cell in row:
                cell.visible_to.clear()
        
        # Add visibility for each player's territories and adjacent cells (including diagonals)
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = self.grid[y][x]
                if cell.owner >= 0:  # Player-owned cell
                    # The cell itself is visible to the owner
                    cell.visible_to.add(cell.owner)
                    
                    # Adjacent cells (including diagonals) are also visible
                    for nx, ny in self.get_neighbors(x, y, include_diagonals=True):
                        self.grid[ny][nx].visible_to.add(cell.owner)
    
    def queue_move(self, from_x: int, from_y: int, to_x: int, to_y: int, army_count: int, player_id: int):
        """Queue a move command - now allows moves into fog of war"""
        if self.game_over:
            return False
            
        from_cell = self.grid[from_y][from_x]
        
        # Validate the move
        if from_cell.owner != player_id:
            print(f"Move failed: Cell not owned by player {player_id}")
            return False
        if army_count >= from_cell.army:
            print(f"Move failed: Not enough army ({army_count} >= {from_cell.army})")
            return False
        if army_count <= 0:
            print(f"Move failed: Army count too low ({army_count})")
            return False
        if abs(to_x - from_x) + abs(to_y - from_y) != 1:  # Must be adjacent
            print(f"Move failed: Not adjacent ({from_x},{from_y}) to ({to_x},{to_y})")
            return False
        
        # Check if destination is valid (only check what we can see)
        to_cell = self.grid[to_y][to_x]
        
        # If we can see the destination and it's a mountain, reject the move
        if player_id in to_cell.visible_to and to_cell.type == CellType.MOUNTAIN:
            print(f"Move failed: Target is mountain")
            return False
        
        # Add to player's move queue
        move = MoveCommand((from_x, from_y), (to_x, to_y), army_count, player_id)
        self.players[player_id].move_queue.append(move)
        
        # Provide feedback based on what we can see
        if player_id in to_cell.visible_to:
            # We can see the destination
            if to_cell.owner == player_id:
                print(f"Move queued: {army_count} armies from ({from_x},{from_y}) to own territory ({to_x},{to_y})")
            elif to_cell.owner == -1 or army_count > to_cell.army:
                print(f"Attack queued: {army_count} armies attacking ({to_x},{to_y}) with {to_cell.army} defenders")
            else:
                print(f"Risky attack queued: {army_count} armies vs {to_cell.army} defenders at ({to_x},{to_y})")
        else:
            # Moving into fog of war
            print(f"Fog move queued: {army_count} armies from ({from_x},{from_y}) into unexplored territory ({to_x},{to_y})")
        
        return True
    
    def execute_move(self, move: MoveCommand):
        """Execute a single move command - handles fog of war discoveries"""
        from_x, from_y = move.from_pos
        to_x, to_y = move.to_pos
        
        from_cell = self.grid[from_y][from_x]
        to_cell = self.grid[to_y][to_x]
        
        # Revalidate the move (army might have changed)
        if from_cell.owner != move.player_id:
            print(f"Execute failed: Cell ownership changed")
            return False
        if move.army_count >= from_cell.army:
            # Adjust army count if it's too high now
            move.army_count = max(1, from_cell.army - 1)
            if move.army_count <= 0:
                print(f"Execute failed: No army to move")
                return False
        
        # Check if we're moving into a mountain (discovered through fog)
        if to_cell.type == CellType.MOUNTAIN:
            print(f"Execute failed: Discovered mountain at ({to_x},{to_y}) through fog of war")
            return False
        
        print(f"Executing move: {move.army_count} from ({from_x},{from_y}) to ({to_x},{to_y})")
        
        # Handle the move
        if to_cell.owner == from_cell.owner:
            # Moving to own territory - just transfer armies
            to_cell.army += move.army_count
            from_cell.army -= move.army_count
            print(f"Moved to own territory: {to_cell.army} armies now at destination")
        else:
            # Battle!
            attacking_army = move.army_count
            defending_army = to_cell.army
            
            print(f"Battle: {attacking_army} vs {defending_army}")
            
            if attacking_army > defending_army:
                # Attack succeeds
                remaining_army = attacking_army - defending_army
                old_owner = to_cell.owner
                to_cell.owner = from_cell.owner
                to_cell.army = remaining_army
                from_cell.army -= move.army_count
                
                if old_owner == -1:
                    print(f"Captured neutral territory with {remaining_army} armies")
                else:
                    print(f"Attack succeeded: Captured enemy territory with {remaining_army} armies")
                
                # Check if we captured a general
                if to_cell.type == CellType.GENERAL:
                    print(f"GENERAL CAPTURED! Player {old_owner} defeated by Player {from_cell.owner}")
                    self.capture_general(old_owner, from_cell.owner)
            else:
                # Attack fails
                to_cell.army -= attacking_army
                from_cell.army -= move.army_count
                print(f"Attack failed: Defender has {to_cell.army} armies remaining")
        
        return True
    
    def capture_general(self, captured_player: int, capturing_player: int):
        """Handle general capture - transfer all armies and territory"""
        if captured_player < 0 or captured_player >= len(self.players):
            return
        
        # Mark player as dead
        self.players[captured_player].is_alive = False
        
        # Transfer all territory and armies to the capturing player
        for row in self.grid:
            for cell in row:
                if cell.owner == captured_player:
                    cell.owner = capturing_player
        
        # Check for game over
        alive_players = [p for p in self.players if p.is_alive]
        if len(alive_players) == 1:
            self.game_over = True
            self.winner = alive_players[0].id
    
    def update(self):
        """Real-time game update"""
        current_time = time.time()
        
        # Process move queues for all players every TURN_DURATION
        if current_time - self.last_general_tick >= TURN_DURATION:
            self.last_general_tick = current_time
            
            # Execute one move from each player's queue
            for player in self.players:
                if not player.is_alive:
                    continue
                if player.move_queue:
                    move = player.move_queue.popleft()
                    self.execute_move(move)
            
            # Generals and cities generate armies every turn (0.5 seconds)
            for row in self.grid:
                for cell in row:
                    if cell.owner >= 0:
                        if cell.type == CellType.GENERAL:
                            cell.army += 1
                        elif cell.type == CellType.CITY:
                            cell.army += 1
            
            self.update_visibility()
        
        # Generate armies for all territories every 25 turns (12.5 seconds)
        if current_time - self.last_army_generation >= ARMY_GENERATION_INTERVAL:
            self.last_army_generation = current_time
            self.generate_armies()
    
    def generate_armies(self):
        """Generate 1 army for each owned territory every 25 turns"""
        for row in self.grid:
            for cell in row:
                if cell.owner >= 0 and cell.type not in [CellType.MOUNTAIN]:
                    cell.army += 1
    
    def update_player_stats(self):
        """Update player statistics"""
        for player in self.players:
            player.total_army = 0
            player.total_land = 0
            
            for row in self.grid:
                for cell in row:
                    if cell.owner == player.id:
                        player.total_army += cell.army
                        player.total_land += 1
    
    def get_game_time(self):
        """Get elapsed game time in seconds"""
        return time.time() - self.game_start_time
    
    def get_turn_number(self):
        """Get current turn number (for display)"""
        return int(self.get_game_time() / TURN_DURATION)

class GameRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(None, 20)
        self.small_font = pygame.font.Font(None, 16)
    
    def render(self, game: Game):
        self.screen.fill(BLACK)
        
        # Render grid
        current_player = game.current_player
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                cell = game.grid[y][x]
                rect = pygame.Rect(
                    GRID_OFFSET_X + x * CELL_SIZE,
                    GRID_OFFSET_Y + y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                
                # Determine if cell is visible to current player
                visible = current_player in cell.visible_to
                
                # Draw cell background
                if not visible:
                    # Fog of war
                    color = FOG_COLOR
                    if cell.type == CellType.CITY:
                        color = BLACK
                elif cell.owner >= 0:
                    # Player-owned territory
                    base_color = PLAYER_COLORS[cell.owner]
                    # Darken the color slightly for better contrast
                    color = tuple(max(0, c - 30) for c in base_color)
                else:
                    # Neutral territory
                    color = LIGHT_GRAY
                if cell.type == CellType.MOUNTAIN:
                    color = BLACK
                
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, WHITE, rect, 1)
                
                # Draw cell contents
                if visible:
                    center_x = rect.centerx
                    center_y = rect.centery
                    
                    # Draw special markers
                    if cell.type == CellType.GENERAL:
                        # Draw crown symbol for general
                        pygame.draw.polygon(self.screen, WHITE, [
                            (center_x - 8, center_y - 8),
                            (center_x - 4, center_y - 12),
                            (center_x, center_y - 8),
                            (center_x + 4, center_y - 12),
                            (center_x + 8, center_y - 8),
                            (center_x + 6, center_y + 8),
                            (center_x - 6, center_y + 8)
                        ])
                    elif cell.type == CellType.CITY:
                        # Draw building symbol for city
                        pygame.draw.rect(self.screen, WHITE, 
                                       (center_x - 6, center_y - 8, 4, 8))
                        pygame.draw.rect(self.screen, WHITE, 
                                       (center_x - 1, center_y - 10, 4, 10))
                        pygame.draw.rect(self.screen, WHITE, 
                                       (center_x + 4, center_y - 6, 4, 6))
                    
                    # Draw army count
                    if cell.army > 0:
                        army_text = str(cell.army)
                        text_surface = self.font.render(army_text, True, BLACK)
                        text_rect = text_surface.get_rect(center=(center_x, center_y))
                        self.screen.blit(text_surface, text_rect)
                
                # Highlight selected cell
                if game.selected_cell and game.selected_cell == (x, y):
                    pygame.draw.rect(self.screen, WHITE, rect, 3)
        
        # Draw UI
        self.draw_ui(game)
    
    def draw_ui(self, game: Game):
        """Draw game UI"""
        # Update player stats
        game.update_player_stats()
        
        # Draw player info
        y_offset = 10
        for player in game.players:
            if not player.is_alive:
                continue
                
            color = player.color
            queue_size = len(player.move_queue)
            text = f"Player {player.id + 1}: Army {player.total_army}, Land {player.total_land}"
            if queue_size > 0:
                text += f" (Queued: {queue_size})"
            
            # Highlight current player for input
            if player.id == game.current_player:
                text = f">>> {text} <<<"
            
            text_surface = self.font.render(text, True, color)
            self.screen.blit(text_surface, (GRID_OFFSET_X + GRID_WIDTH * CELL_SIZE + 20, y_offset))
            y_offset += 25
        
        # Draw game time and turn
        game_time = game.get_game_time()
        turn_num = game.get_turn_number()
        time_text = f"Time: {game_time:.1f}s, Turn: {turn_num}"
        text_surface = self.font.render(time_text, True, WHITE)
        self.screen.blit(text_surface, (GRID_OFFSET_X + GRID_WIDTH * CELL_SIZE + 20, y_offset + 20))
        
        # Draw next army generation countdown
        time_since_last_gen = game_time - (game.last_army_generation - game.game_start_time)
        time_until_next_gen = ARMY_GENERATION_INTERVAL - time_since_last_gen
        if time_until_next_gen <= 0:
            time_until_next_gen = ARMY_GENERATION_INTERVAL
        gen_text = f"Next army generation: {time_until_next_gen:.1f}s"
        text_surface = self.small_font.render(gen_text, True, WHITE)
        self.screen.blit(text_surface, (GRID_OFFSET_X + GRID_WIDTH * CELL_SIZE + 20, y_offset + 45))
        
        # Draw instructions
        instructions = [
            "REAL-TIME GAMEPLAY",
            "Click to select your territory",
            "WASD/Arrow Keys to chain moves",
            "Tab to switch player view",
            "Hold Shift for half army move",
            "ESC to clear selection"
        ]
        
        for i, instruction in enumerate(instructions):
            color = WHITE if i > 0 else (255, 200, 100)
            text_surface = self.small_font.render(instruction, True, color)
            self.screen.blit(text_surface, (10, WINDOW_HEIGHT - 110 + i * 15))
        
        # Game over screen
        if game.game_over:
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(128)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            winner_text = f"Player {game.winner + 1} Wins!"
            text_surface = pygame.font.Font(None, 48).render(winner_text, True, PLAYER_COLORS[game.winner])
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))
            self.screen.blit(text_surface, text_rect)
            
            restart_text = "Press R to restart"
            text_surface = self.font.render(restart_text, True, WHITE)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(text_surface, text_rect)

def main():
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Generals.io - Real Time Strategy")
    clock = pygame.time.Clock()
    
    game = Game()
    renderer = GameRenderer(screen)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and game.game_over:
                    game = Game()
                elif event.key == pygame.K_TAB:
                    # Switch player view (for local multiplayer)
                    alive_players = [p.id for p in game.players if p.is_alive]
                    if len(alive_players) > 1:
                        current_idx = alive_players.index(game.current_player)
                        game.current_player = alive_players[(current_idx + 1) % len(alive_players)]
                
                elif event.key == pygame.K_ESCAPE:
                    # Clear selection
                    game.selected_cell = None
                    print("Selection cleared")
                
                # WASD movement
                elif not game.game_over and game.selected_cell:
                    from_x, from_y = game.selected_cell
                    to_x, to_y = from_x, from_y
                    
                    if event.key == pygame.K_w or event.key == pygame.K_UP:
                        to_y = from_y - 1
                    elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                        to_y = from_y + 1
                    elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        to_x = from_x - 1
                    elif event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        to_x = from_x + 1
                    
                    # Execute the move if valid
                    if (to_x, to_y) != (from_x, from_y) and 0 <= to_x < GRID_WIDTH and 0 <= to_y < GRID_HEIGHT:
                        from_cell = game.grid[from_y][from_x]
                        current_player = game.current_player
                        
                        print(f"Attempting WASD move from ({from_x},{from_y}) to ({to_x},{to_y})")
                        
                        # Determine army count to move
                        keys = pygame.key.get_pressed()
                        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                            army_count = from_cell.army // 2
                        else:
                            army_count = from_cell.army - 1
                        
                        print(f"Army count to move: {army_count} (from {from_cell.army} total)")
                        
                        if army_count > 0:
                            success = game.queue_move(from_x, from_y, to_x, to_y, 
                                                    army_count, current_player)
                            if success:
                                # Always auto-select the destination cell for chaining moves
                                # This allows continuous movement without reselecting
                                game.selected_cell = (to_x, to_y)
                                print(f"Auto-selected destination ({to_x},{to_y}) for chaining")
                        else:
                            print(f"Cannot move: army count too low")
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if game.game_over:
                    continue
                    
                # Convert mouse position to grid coordinates
                mouse_x, mouse_y = event.pos
                grid_x = (mouse_x - GRID_OFFSET_X) // CELL_SIZE
                grid_y = (mouse_y - GRID_OFFSET_Y) // CELL_SIZE
                
                if 0 <= grid_x < GRID_WIDTH and 0 <= grid_y < GRID_HEIGHT:
                    current_player = game.current_player
                    
                    if event.button == 1:  # Left click - select
                        cell = game.grid[grid_y][grid_x]
                        if cell.owner == current_player and cell.army > 1:
                            game.selected_cell = (grid_x, grid_y)
                            print(f"Selected cell ({grid_x},{grid_y}) with {cell.army} armies")
                        else:
                            print(f"Cannot select: Owner={cell.owner}, Army={cell.army}, Player={current_player}")
                    
                    elif event.button == 3:  # Right click - queue move (fallback)
                        if game.selected_cell:
                            from_x, from_y = game.selected_cell
                            from_cell = game.grid[from_y][from_x]
                            
                            # Check if it's a valid adjacent move
                            if abs(grid_x - from_x) + abs(grid_y - from_y) == 1:
                                # Determine army count to move
                                keys = pygame.key.get_pressed()
                                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                                    army_count = from_cell.army // 2
                                else:
                                    army_count = from_cell.army - 1
                                
                                if army_count > 0:
                                    success = game.queue_move(from_x, from_y, grid_x, grid_y, 
                                                            army_count, current_player)
                                    if success:
                                        game.selected_cell = None
        
        # Real-time game update
        game.update()
        
        renderer.render(game)
        pygame.display.flip()
        clock.tick(60)  # 60 FPS for smooth gameplay
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()