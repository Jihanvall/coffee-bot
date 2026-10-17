import numpy as np

# Cell types 
EMPTY, WALL, WATER, COFFEE, OUTLET = 0, 1, 2, 3, 4

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

LAYOUT = [
    "S . . # . . .",
    ". # . # . W .",
    ". # . . . . .",
    ". . . # # . .",
    "W . . . . . .",
    ". # # . # . .",
    ". . . . . . C",
]

class CoffeeEnv:
    """A 7x7 grid. The robot starts at S and must reach the coffee C.

    Walls (#) block movement, water (W) ends the episode with a penalty.
    """

    REWARD_COFFEE = 10.0
    REWARD_WATER = -10.0
    REWARD_STEP = -0.1
    REWARD_WALL = -0.5

    def __init__(self, layout=None, max_steps=100):
        rows = [line.split() for line in (layout or LAYOUT)]
        self.rows = len(rows)
        self.cols = max(len(row) for row in rows)
        self.size = self.rows
        self.grid = np.zeros((self.rows, self.cols), dtype=int)
        self.start = (0, 0)
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.grid[r, c] = WALL
                elif ch == "W":
                    self.grid[r, c] = WATER
                elif ch == "C":
                    self.grid[r, c] = COFFEE
                elif ch == "S":
                    self.start = (r, c)
        self.max_steps = max_steps
        self.n_actions = 4
        self.n_states = self.rows * self.cols
        self.reset()
    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self._state()

    def _state(self):
        return self.pos[0] * self.cols + self.pos[1]

    def step(self, action):
        if action not in ACTION_DELTAS:
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1
        dr, dc = ACTION_DELTAS[action]
        r, c = self.pos[0] + dr, self.pos[1] + dc

        reward = self.REWARD_STEP
        done = False

        out_of_bounds = not (0 <= r < self.rows and 0 <= c < self.cols)
        if out_of_bounds or self.grid[r, c] == WALL:
            reward += self.REWARD_WALL
        else:
            self.pos = (r, c)
            cell = self.grid[r, c]
            if cell == COFFEE:
                reward = self.REWARD_COFFEE
                done = True
            elif cell == WATER:
                reward = self.REWARD_WATER
                done = True
        if self.steps >= self.max_steps:
            done = True

        return self._state(), reward, done