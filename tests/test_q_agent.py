import numpy as np

from agents.evaluate import evaluate
from agents.q_agent import QAgent
from agents.random_agent import RandomAgent
from agents.train import train
from env.coffee_env import CoffeeEnv

def test_q_table_has_the_right_shape():
    env = CoffeeEnv()
    agent = QAgent(env.n_states, env.n_actions)
    assert agent.q.shape == (49, 4)

def test_learn_moves_value_toward_the_target():
    agent = QAgent(3, 2, alpha=0.5, gamma=0.9)
    agent.learn(0, 1, 10.0, 2, True)
    assert agent.q[0, 1] == 5.0

def test_epsilon_decays_but_not_below_the_minimum():
    agent = QAgent(3, 2, epsilon=1.0, epsilon_min=0.5, epsilon_decay=0.1)
    agent.end_episode()
    assert agent.epsilon == 0.5

def test_best_action_picks_the_highest_value():
    agent = QAgent(2, 3, seed=0)
    agent.q[0] = [0.0, 5.0, 1.0]
    assert agent.best_action(0) == 1

def test_training_beats_the_random_baseline():
    env = CoffeeEnv()
    agent = QAgent(env.n_states, env.n_actions, seed=0)
    train(env, agent, episodes=2000)
    trained = evaluate(CoffeeEnv(), agent.greedy(), episodes=200)
    baseline = evaluate(CoffeeEnv(), RandomAgent(4, seed=0), episodes=200)
    assert trained["success_rate"] > baseline["success_rate"]
    assert trained["avg_reward"] > baseline["avg_reward"] 