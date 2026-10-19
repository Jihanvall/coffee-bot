def train(env, agent, episodes=2000):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total += reward
        agent.end_episode()
        rewards.append(total)
    return rewards