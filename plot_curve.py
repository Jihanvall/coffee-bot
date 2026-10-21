from agents.plot import save_learning_curve
from agents.q_agent import QAgent
from agents.train import train
from env.coffee_env import CoffeeEnv

env = CoffeeEnv()
agent = QAgent(env.n_states, env.n_actions, seed=0)
rewards = train(env, agent, episodes=2000)
save_learning_curve(rewards, "learning_curve.png")
print("Saved learning_curve.png")