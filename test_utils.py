import pytest

from utils import flatten, timerepr


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


def test_flatten():
    assert [1, 2, 3, 4, 5, 6] == flatten([[1, 2, 3], [4, 5, 6]])
