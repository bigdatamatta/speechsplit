
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
