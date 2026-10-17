from agents.evaluate import evaluate
from agents.random_agent import RandomAgent
from env.coffee_env import CoffeeEnv


def test_random_agent_returns_valid_actions():
    agent = RandomAgent(4, seed=0)
    actions = {agent.act(0) for _ in range(200)}
    assert actions <= {0, 1, 2, 3}
    assert len(actions) == 4


def test_same_seed_gives_same_actions():
    a = RandomAgent(4, seed=1)
    b = RandomAgent(4, seed=1)
    assert [a.act(0) for _ in range(20)] == [b.act(0) for _ in range(20)]


def test_evaluate_returns_the_three_metrics():
    result = evaluate(CoffeeEnv(), RandomAgent(4, seed=0), episodes=50)
    assert set(result) == {"success_rate", "water_rate", "avg_reward"}


def test_rates_are_between_zero_and_one():
    result = evaluate(CoffeeEnv(), RandomAgent(4, seed=0), episodes=50)
    assert 0 <= result["success_rate"] <= 1
    assert 0 <= result["water_rate"] <= 1
    assert result["success_rate"] + result["water_rate"] <= 1


def test_evaluate_is_repeatable_with_the_same_seed():
    first = evaluate(CoffeeEnv(), RandomAgent(4, seed=3), episodes=50)
    second = evaluate(CoffeeEnv(), RandomAgent(4, seed=3), episodes=50)
    assert first == second