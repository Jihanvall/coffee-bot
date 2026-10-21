import pytest

from env.coffee_env import CoffeeEnv, UP, DOWN, LEFT, RIGHT

SMALL = [
    "S W",
    "# C",
]


def test_reset_returns_start_state():
    env = CoffeeEnv()
    assert env.reset() == 0


def test_move_right_changes_state():
    env = CoffeeEnv(SMALL.copy())
    env.reset()
    state, reward, done = env.step(RIGHT)
    assert state == 1
    assert done is True  # (0,1) is water


def test_wall_blocks_movement_and_penalizes():
    env = CoffeeEnv(SMALL.copy())
    env.reset()
    state, reward, done = env.step(DOWN)  # (1,0) is a wall
    assert state == 0
    assert reward == pytest.approx(env.REWARD_STEP + env.REWARD_WALL)
    assert done is False


def test_leaving_the_grid_is_treated_like_a_wall():
    env = CoffeeEnv(SMALL.copy())
    env.reset()
    state, reward, done = env.step(UP)
    assert state == 0
    assert reward < env.REWARD_STEP


def test_reaching_coffee_gives_reward_and_ends_episode():
    layout = ["S C"]
    env = CoffeeEnv(layout)
    env.reset()
    state, reward, done = env.step(RIGHT)
    assert reward == env.REWARD_COFFEE
    assert done is True


def test_water_ends_episode_with_penalty():
    layout = ["S W C"]
    env = CoffeeEnv(layout)
    env.reset()
    _, reward, done = env.step(RIGHT)
    assert reward == env.REWARD_WATER
    assert done is True


def test_episode_ends_after_max_steps():
    env = CoffeeEnv(["S . . C"], max_steps=3)
    env.reset()
    done = False
    for _ in range(3):
        _, _, done = env.step(LEFT)  # keeps bumping into the border
    assert done is True


def test_invalid_action_raises():
    env = CoffeeEnv()
    with pytest.raises(ValueError):
        env.step(9)


def test_default_map_has_a_path_to_the_coffee():
    env = CoffeeEnv()
    seen, stack = {env.start}, [env.start]
    reached = False
    while stack:
        r, c = stack.pop()
        if env.grid[r, c] == 3:
            reached = True
            break
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < env.size
                and 0 <= nc < env.size
                and (nr, nc) not in seen
                and env.grid[nr, nc] not in (1, 2)
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    assert reached
def test_no_slip_by_default():
    env = CoffeeEnv(["S . C"])
    env.reset()
    state, _, _ = env.step(RIGHT)
    assert state == 1


def test_always_slipping_still_gives_valid_states():
    env = CoffeeEnv(slip_prob=1.0, seed=0)
    env.reset()
    for _ in range(50):
        state, _, done = env.step(RIGHT)
        assert 0 <= state < env.n_states
        if done:
            env.reset()


def test_slip_is_repeatable_with_the_same_seed():
    def run():
        env = CoffeeEnv(slip_prob=0.5, seed=3)
        env.reset()
        return [env.step(RIGHT)[0] for _ in range(10)]

    assert run() == run()