import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

from env.coffee_env import CoffeeEnv
from render.pygame_view import BAR_HEIGHT, CELL, PygameView


def test_view_draws_without_a_screen():
    env = CoffeeEnv()
    env.reset()
    view = PygameView(env)
    view.draw("after 0 episodes")
    expected = (env.cols * CELL, env.rows * CELL + BAR_HEIGHT)
    assert view.screen.get_size() == expected
    view.close()