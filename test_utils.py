import numpy as np
import pytest

from utils import get_first_and_last_from_iterator, intervals_where, timerepr


def test_first_and_last_from_iter():
    assert get_first_and_last_from_iterator(iter([1, 2, 3, 4])) == (1, 4)


BIG = 999


@pytest.mark.parametrize('data, intervals', [
    [[], []],
    [[0, 0, 0], []],
    [[0, 1, 2, BIG, BIG, 5, 6, 7, 8, BIG, BIG, BIG], [(3, 5), (9, 12)]],
])
def test_intervals_where(data, intervals):
    data = np.array(data)
    assert list(intervals_where(data == BIG)) == intervals
    assert list(intervals_where(data > BIG - 1)) == intervals


def to_milliseconds(hour, minute, second, millis):
    minute += hour * 60
    second += minute * 60
    millis += second * 1000
    return millis


@pytest.mark.parametrize('millis, output', [
    (to_milliseconds(1000, 3, 42, 23), '1000:03:42.023'),
    (to_milliseconds(12, 3, 42, 23), '12:03:42.023'),
    (to_milliseconds(0, 3, 0, 999), '03:00.999'),
    (to_milliseconds(0, 0, 0, 0), '00:00.000'),
])
def test_timerepr(millis, output):
    assert timerepr(millis) == output
