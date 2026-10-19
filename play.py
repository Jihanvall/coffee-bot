from agents.q_agent import QAgent
from agents.train import train
from env.coffee_env import CoffeeEnv
from render.pygame_view import PygameView

STAGES = [0, 30, 300, 2000]
SECONDS_PER_STAGE = 1
MAX_WATCH_STEPS = 40

env = CoffeeEnv()
agent = QAgent(env.n_states, env.n_actions, seed=0)
view = PygameView(env)


def watch(label):
    state = env.reset()
    for _ in range(MAX_WATCH_STEPS):
        if view.should_quit():
            return False
        view.draw(label)
        view.tick()
        state, _, done = env.step(agent.act(state))
        if done:
            view.draw(label)
            view.tick(1)
            break
    return True


trained = 0
running = True
for stage in STAGES:
    if stage > trained:
        train(env, agent, episodes=stage - trained)
        trained = stage
    saved_epsilon = agent.epsilon
    label = f"after {stage} episodes"
    for _ in range(3):
        running = watch(label)
        if not running:
            break
    agent.epsilon = saved_epsilon
    if not running:
        break

view.close()