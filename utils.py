
from itertools import groupby


def get_first_and_last_from_iterator(iterator):
    "Returns the first and last items from an iterator"
    first = last = next(iterator)
    for last in iterator:
        pass
    return first, last


def find_value_intervals(data, value):
    """Returns the list of intervals in data
    that contain continuous occurrences of value"""

    def interval(group):
        first, last = get_first_and_last_from_iterator(i for i, v in group)
        return first, last + 1

    value_groups = groupby(enumerate(data), lambda (i, v): v == value)
    return [interval(g) for k, g in value_groups if k]


def timerepr(millis):
    '''based on
    https://www.darklaunch.com/2009/10/06/python-time-duration-human-friendly-timestamp'''

    seconds, millis = divmod(millis, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return '{:02d}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, seconds, millis)
    else:
        return '{:02d}:{:02d}.{:03d}'.format(minutes, seconds, millis)
