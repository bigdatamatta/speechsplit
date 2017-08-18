
from itertools import groupby

import numpy as np
import pydub


def get_first_and_last_from_iterator(iterator):
    "Returns the first and last items from an iterator"
    first = last = next(iterator)
    for last in iterator:
        pass
    return first, last


def intervals_where(mask):
    where = np.where(mask)[0]
    for __, group in groupby(enumerate(where), lambda (i, x): x - i):
        first, last = get_first_and_last_from_iterator(x for _, x in group)
        yield first, last + 1


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
