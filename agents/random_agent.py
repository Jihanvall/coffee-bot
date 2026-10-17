import random

class RandomAgent:
    def __init__(self, n_actions, seed=None):
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.randrange(self.n_actions)