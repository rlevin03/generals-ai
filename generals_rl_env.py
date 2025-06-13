from generals import Game, GRID_WIDTH, GRID_HEIGHT, CellType
import gym
import numpy as np


class GeneralsEnv(gym.Env):
    def __init__(self, player_id):
        self.player_id = player_id
        self.game = Game()
        self.action_space = gym.spaces.Discrete(GRID_WIDTH * GRID_HEIGHT * 4)
        self.observation_space = gym.spaces.Box(
            low=-2, high=1000, shape=(GRID_HEIGHT, GRID_WIDTH, 4), dtype=np.float32
        )

    
    # at the beginning, reset the environment
    def reset(self):
        self.game = Game()
        return self.extract_state()
    
    # the step method record the new state after one action
    def step(self, action):
        # use decode_action to get the from x, from y and other data with a single int
        from_x, from_y, to_x, to_y = self.decode_action(action)
        # use the move_army method to move the armys
        army = self.move_army(from_x, from_y)
        # use queue_move method to try to attack or relocate the army
        success = self.game.queue_move(from_x, from_y, to_x, to_y, army, self.player_id)

        if not success:
            return self.extract_state(), -1.0, False, {"incorrect move"}
        
        # use update method to deal with all the state change above
        self.game.update()
        # extract the new state of the game
        next_state = self.extract_state()
        # compute the reward of current state
        reward = self.compute_reward()
        # check if the game is over
        done = self.game.game_over
        return next_state, reward, done, {}
    


    # get the current state of the grid
    def extract_state(self):
        # create a 3 dimension numpy array, the 4 represent the id of the cell belong to, number of armies, city, general
        state = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4))
        # the y axis
        for i in range(GRID_HEIGHT):
            # the x axis
            for j in range(GRID_WIDTH):
                # get the information of the current cell
                cell = self.game.grid[i][j]
                # check if the cell can be visible to players
                if self.player_id in cell.visible_to:
                    state[i][j] = [
                        # get the four element of the cell
                        cell.owner if cell.owner != -1 else -1,
                        cell.army,
                        int(cell.type == CellType.CITY),
                        int(cell.type == CellType.GENERAL)
                    ]
                else:
                    # if is not visible, use fog to represent it
                    state[i][j] = [-2, 0, 0, 0]
        return state
    

    # use an int action to represent the location and direction of one movement
    def decode_action(self, action):
        # get the from index of the grid based on action
        from_index = action // 4
        # get the direction of the cell based on action
        direction = action % 4
        # get the from cell location
        from_x = from_index % GRID_WIDTH
        from_y = from_index // GRID_WIDTH
        # get the movement count based on direction
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][direction]
        # apply movement to from location
        to_x = from_x + dx
        to_y = from_y + dy
        return from_x, from_y, to_x, to_y
    

    # move the army
    def move_army(self, x, y):
        cell = self.game.grid[y][x]
        # if the army count is less than 1, not move, else move half of the army or 1
        return max(1, cell.army // 2) if cell.army > 1 else 0
    

    # get the reward of the current state
    def compute_reward(self):
        player = self.game.players[self.player_id]
        reward = player.total_land * 1 + player.total_army * 0.1
        if self.game.game_over:
            reward += 1000 if self.game.winner == self.player_id else -500
        return reward
