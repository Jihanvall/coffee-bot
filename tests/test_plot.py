import os 

from agents.plot import moving_average, save_learning_curve

def test_moving_average_smooths_values():
    result = moving_average([0, 10, 0, 10], window=2)
    assert list(result) == [5.0, 5.0, 5.0]

def test_moving_average_handles_short_input():
    assert len(moving_average([1, 2, 3], window=50)) == 1

def test_curve_is_saved_to_a_file(tmp_path):
    path = tmp_path / "curve.png"
    save_learning_curve([1, 2, 3, 4, 5], str(path), window=2)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0