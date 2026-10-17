def evaluate(env, agent, episodes=500):
    successes = 0
    water = 0
    total_reward = 0.0

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        last_reward = 0.0

        while not done:
            action = agent.act(state)
            state, reward, done = env.step(action)
            episode_reward  += reward
            last_reward = reward

        total_reward += episode_reward
        if last_reward == env.REWARD_COFFEE:
            successes += 1
        elif last_reward == env.REWARD_WATER:
            water += 1

    return {
        "success_rate": successes / episodes,
        "water_rate": water /episodes,
        "avg_reward": total_reward / episodes,
    }
