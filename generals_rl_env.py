from generals import Game, GRID_WIDTH, GRID_HEIGHT, CellType
import gym
import numpy as np


class GeneralsEnv(gym.Env):
    def reset(self):
        self.game = Game()
        return self.extract_state()
    
    def step(self, action):
        from_x, from_y, to_x, to_y = self.decode_action(action)
        army = self.get_move_army(from_x, from_y)
        success = self.game.queue_move(from_x, from_y, to_x, to_y, army, self.player_id)
        self.game.update()
        next_state = self.extract_state()
        reward = self.compute_reward()
        done = self.game.game_over
        return next_state, reward, done, {}
    
    def extract_state(self):
        state = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
        for i in range(GRID_HEIGHT):
            for j in range(GRID_WIDTH):
                cell = self.game.grid[i][j]
                if cell.player_id in cell.visible_to:
                    state[i][j] = [
                        cell.owner if cell.owner != -1 else -1,
                        cell.army,
                        int(cell.type == CellType.CITY),
                        int(cell.type == CellType.GENERAL)
                    ]
                else:
                    state[i][j] = [-2, 0, 0, 0]
        return state
    


    def decode_action(self, action):
        from_index = action // 4
        direction = action % 4
        from_x = from_index % GRID_WIDTH
        from_y = from_index // GRID_WIDTH
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        to_x = from_x + dx
        to_y = from_y + dy
        return from_x, from_y, to_x, to_y
    
    def get_move_army(self, x, y):
        cell = self.game.grid[y][x]
        return max(1, cell.army // 2) if cell.army > 1 else 0
    

    def compute_reward(self):
        player = self.game.players[self.player_id]
        reward = player.total_land * 0.1 + player.total_army * 0.01
        if self.game.game_over:
            reward += 10 if self.game.winner == self.player_id else -10
        return reward
