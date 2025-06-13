from generals import Game, GRID_WIDTH, GRID_HEIGHT, CellType

def evaluate_move(from_cell, to_cell):
    if to_cell.type == CellType.MOUNTAIN:
        return -999  # avoid
    if to_cell.owner == -1:
        score = 3  # neutral tile
        if to_cell.type == CellType.CITY:
            score += 7  # prioritize cities
        score += max(0, from_cell.army - to_cell.army)
        return score
    elif to_cell.owner != from_cell.owner:
        if from_cell.army > to_cell.army:
            return 5 + (from_cell.army - to_cell.army)  # good attack
        else:
            return -10  # suicidal
    else:
        return 0  # skip own land

def choose_greedy_move(game, player_id):
    best_score = -float("inf")
    best_move = None

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            cell = game.grid[y][x]
            if cell.owner == player_id and cell.army > 1:
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT:
                        neighbor = game.grid[ny][nx]
                        if player_id in neighbor.visible_to:
                            score = evaluate_move(cell, neighbor)
                            if score > best_score:
                                best_score = score
                                best_move = (x, y, nx, ny)
    return best_move

class GreedyBot:
    def __init__(self, player_id):
        self.player_id = player_id

    def act(self, game: Game):
        move = choose_greedy_move(game, self.player_id)
        if move:
            from_x, from_y, to_x, to_y = move
            from_cell = game.grid[from_y][from_x]
            army = max(1, from_cell.army // 2)
            game.queue_move(from_x, from_y, to_x, to_y, army, self.player_id)
