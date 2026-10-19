import numpy as np


class QAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, seed=None):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions))

    def act(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.best_action(state)

    def best_action(self, state):
        row = self.q[state]
        best = np.flatnonzero(row == row.max())
        return int(self.rng.choice(best))

    def learn(self, state, action, reward, next_state, done):
        target = reward if done else reward + self.gamma * self.q[next_state].max()
        self.q[state, action] += self.alpha * (target - self.q[state, action])

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy(self):
        return GreedyPolicy(self)


class GreedyPolicy:
    def __init__(self, agent):
        self.agent = agent

    def act(self, state):
        return self.agent.best_action(state)