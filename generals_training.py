# Add this modified Game class to your existing generals.py file
# Or create a new file generals_training.py

class Game:
    def __init__(self, grid_width=None, grid_height=None, num_players=4, training_mode=False):
        # Allow custom grid sizes
        self.grid_width = grid_width or GRID_WIDTH
        self.grid_height = grid_height or GRID_HEIGHT
        self.training_mode = training_mode
        
        # Initialize grid with custom size
        self.grid = [[Cell(x, y) for x in range(self.grid_width)] for y in range(self.grid_height)]
        
        # Initialize players
        self.players = [Player(i, PLAYER_COLORS[i % len(PLAYER_COLORS)]) for i in range(num_players)]
        
        # Timing variables
        self.game_start_time = time.time()
        self.last_army_generation = time.time()
        self.last_general_tick = time.time()
        self.selected_cell = None
        self.current_player = 0
        self.game_over = False
        self.winner = -1
        
        # Turn duration based on mode
        self.turn_duration = 0.001 if training_mode else TURN_DURATION  # 1ms vs 1s
        self.army_generation_interval = 25 * self.turn_duration
        
        # Turn counter for training mode
        self.turn_counter = 0
        
        # Generate map
        self.generate_map()
        self.place_generals()
        self.update_visibility()
    
    def generate_map(self):
        """Generate mountains and cities on the map - scaled to grid size"""
        # Scale counts based on map size
        map_scale = (self.grid_width * self.grid_height) / (GRID_WIDTH * GRID_HEIGHT)
        
        # Add mountains
        num_mountains = int(random.randint(15, 25) * map_scale)
        for _ in range(num_mountains):
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if self.grid[y][x].type == CellType.EMPTY:
                self.grid[y][x].type = CellType.MOUNTAIN
        
        # Add cities
        num_cities = int(random.randint(8, 12) * map_scale)
        for _ in range(num_cities):
            x = random.randint(0, self.grid_width - 1)
            y = random.randint(0, self.grid_height - 1)
            if self.grid[y][x].type == CellType.EMPTY:
                self.grid[y][x].type = CellType.CITY
                self.grid[y][x].army = random.randint(40, 50)
    
    def place_generals(self):
        """Place generals for each player ensuring minimum distance"""
        positions = []
        # Scale minimum distance based on map size
        min_distance = max(5, int(15 * (self.grid_width * self.grid_height) / (GRID_WIDTH * GRID_HEIGHT)))
        
        for player in self.players:
            attempts = 0
            while attempts < 100:
                x = random.randint(1, self.grid_width - 2)
                y = random.randint(1, self.grid_height - 2)
                
                if self.grid[y][x].type != CellType.EMPTY:
                    attempts += 1
                    continue
                
                valid = True
                for px, py in positions:
                    distance = abs(x - px) + abs(y - py)
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
    
    def get_neighbors(self, x: int, y: int, include_diagonals: bool = False):
        """Get valid neighboring coordinates"""
        neighbors = []
        # Cardinal directions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                neighbors.append((nx, ny))
        
        # Diagonal directions for visibility
        if include_diagonals:
            for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def update_visibility(self):
        """Update fog of war based on owned territories"""
        # Clear all visibility
        for row in self.grid:
            for cell in row:
                cell.visible_to.clear()
        
        # Add visibility for each player's territories and adjacent cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell = self.grid[y][x]
                if cell.owner >= 0:
                    cell.visible_to.add(cell.owner)
                    
                    # Adjacent cells are also visible
                    for nx, ny in self.get_neighbors(x, y, include_diagonals=True):
                        self.grid[ny][nx].visible_to.add(cell.owner)
    
    def queue_move(self, from_x: int, from_y: int, to_x: int, to_y: int, army_count: int, player_id: int):
        """Queue a move command"""
        if self.game_over:
            return False
            
        from_cell = self.grid[from_y][from_x]
        
        # Validate the move
        if from_cell.owner != player_id:
            if not self.training_mode:
                print(f"Move failed: Cell not owned by player {player_id}")
            return False
        if army_count > from_cell.army:  # Changed from >= to > to allow moving all armies
            if not self.training_mode:
                print(f"Move failed: Not enough army ({army_count} > {from_cell.army})")
            return False
        if army_count <= 0:
            if not self.training_mode:
                print(f"Move failed: Army count too low ({army_count})")
            return False
        if abs(to_x - from_x) + abs(to_y - from_y) != 1:
            if not self.training_mode:
                print(f"Move failed: Not adjacent ({from_x},{from_y}) to ({to_x},{to_y})")
            return False
        
        # Check if destination is valid
        to_cell = self.grid[to_y][to_x]
        
        # If we can see the destination and it's a mountain, reject
        if player_id in to_cell.visible_to and to_cell.type == CellType.MOUNTAIN:
            if not self.training_mode:
                print(f"Move failed: Target is mountain")
            return False
        
        # Add to player's move queue
        move = MoveCommand((from_x, from_y), (to_x, to_y), army_count, player_id)
        self.players[player_id].move_queue.append(move)
        
        # Only print feedback in non-training mode
        if not self.training_mode:
            if player_id in to_cell.visible_to:
                if to_cell.owner == player_id:
                    print(f"Move queued: {army_count} armies from ({from_x},{from_y}) to own territory ({to_x},{to_y})")
                elif to_cell.owner == -1 or army_count > to_cell.army:
                    print(f"Attack queued: {army_count} armies attacking ({to_x},{to_y}) with {to_cell.army} defenders")
                else:
                    print(f"Risky attack queued: {army_count} armies vs {to_cell.army} defenders at ({to_x},{to_y})")
            else:
                print(f"Fog move queued: {army_count} armies from ({from_x},{from_y}) into unexplored territory ({to_x},{to_y})")
        
        return True
    
    def execute_move(self, move: MoveCommand):
        """Execute a single move command"""
        from_x, from_y = move.from_pos
        to_x, to_y = move.to_pos
        
        from_cell = self.grid[from_y][from_x]
        to_cell = self.grid[to_y][to_x]
        
        # Revalidate the move
        if from_cell.owner != move.player_id:
            if not self.training_mode:
                print(f"Execute failed: Cell ownership changed")
            return False
        if move.army_count > from_cell.army:  # Changed from >= to > to allow moving all armies
            move.army_count = from_cell.army  # Changed to allow moving all armies
            if move.army_count <= 0:
                if not self.training_mode:
                    print(f"Execute failed: No army to move")
                return False
        
        # Check if we're moving into a mountain
        if to_cell.type == CellType.MOUNTAIN:
            if not self.training_mode:
                print(f"Execute failed: Discovered mountain at ({to_x},{to_y})")
            return False
        
        # Handle the move
        if to_cell.owner == from_cell.owner:
            # Moving to own territory
            to_cell.army += move.army_count
            from_cell.army -= move.army_count
            if not self.training_mode:
                print(f"Moved to own territory: {to_cell.army} armies now at destination")
        else:
            # Battle!
            attacking_army = move.army_count
            defending_army = to_cell.army
            
            if not self.training_mode:
                print(f"Battle: {attacking_army} vs {defending_army}")
            
            if attacking_army > defending_army:
                # Attack succeeds
                remaining_army = attacking_army - defending_army
                old_owner = to_cell.owner
                to_cell.owner = from_cell.owner
                to_cell.army = remaining_army
                from_cell.army -= move.army_count
                
                if not self.training_mode:
                    if old_owner == -1:
                        print(f"Captured neutral territory with {remaining_army} armies")
                    else:
                        print(f"Attack succeeded: Captured enemy territory with {remaining_army} armies")
                
                # Check if we captured a general
                if to_cell.type == CellType.GENERAL:
                    if not self.training_mode:
                        print(f"GENERAL CAPTURED! Player {old_owner} defeated by Player {from_cell.owner}")
                    self.capture_general(old_owner, from_cell.owner)
            else:
                # Attack fails
                to_cell.army -= attacking_army
                from_cell.army -= move.army_count
                if not self.training_mode:
                    print(f"Attack failed: Defender has {to_cell.army} armies remaining")
        
        return True
    
    def capture_general(self, captured_player: int, capturing_player: int):
        """Handle general capture"""
        if captured_player < 0 or captured_player >= len(self.players):
            return
        
        # Mark player as dead
        self.players[captured_player].is_alive = False
        
        # Transfer all territory and armies
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
        """Real-time game update - fast in training mode"""
        if self.training_mode:
            # In training mode, increment turn counter and generate armies every turn
            self.turn_counter += 1
            
            # Execute moves
            for player in self.players:
                if not player.is_alive:
                    continue
                if player.move_queue:
                    move = player.move_queue.popleft()
                    self.execute_move(move)
            
            # Generate armies for generals and cities every turn
            for row in self.grid:
                for cell in row:
                    if cell.owner >= 0:
                        if cell.type == CellType.GENERAL:
                            cell.army += 1
                        elif cell.type == CellType.CITY:
                            cell.army += 1
            
            # Generate armies for all territories every 25 turns
            if self.turn_counter % 25 == 0:
                for row in self.grid:
                    for cell in row:
                        if cell.owner >= 0 and cell.type not in [CellType.MOUNTAIN]:
                            cell.army += 1
            
            # Update visibility after all changes
            self.update_visibility()
            
            # Update player stats
            self.update_player_stats()
        else:
            # Original real-time update logic
            current_time = time.time()
            
            # Process moves
            if current_time - self.last_general_tick >= self.turn_duration:
                self.last_general_tick = current_time
                
                # Execute moves
                for player in self.players:
                    if not player.is_alive:
                        continue
                    if player.move_queue:
                        move = player.move_queue.popleft()
                        self.execute_move(move)
                
                # Generate armies for generals and cities
                for row in self.grid:
                    for cell in row:
                        if cell.owner >= 0:
                            if cell.type == CellType.GENERAL:
                                cell.army += 1
                            elif cell.type == CellType.CITY:
                                cell.army += 1
                
                self.update_visibility()
            
            # Generate armies for all territories
            if current_time - self.last_army_generation >= self.army_generation_interval:
                self.last_army_generation = current_time
                self.generate_armies()
    
    def generate_armies(self):
        """Generate armies for owned territories"""
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
        """Get elapsed game time"""
        return time.time() - self.game_start_time
    
    def get_turn_number(self):
        """Get current turn number"""
        return int(self.get_game_time() / self.turn_duration)