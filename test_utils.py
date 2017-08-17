import pytest

from utils import find_value_intervals, get_first_and_last_from_iterator


def test_first_and_last_from_iter():
    assert get_first_and_last_from_iterator(iter([1, 2, 3, 4])) == (1, 4)


___ = 999


@pytest.mark.parametrize('data, intervals', [
    [[], []],
    [[0, 0, 0], []],
    [[0, 1, 2, ___, ___, 5, 6, 7, 8, ___, ___, ___], [(3, 5), (9, 12)]],
])
def test_find_value_intervals(data, intervals):
    assert find_value_intervals(data, ___) == intervals
