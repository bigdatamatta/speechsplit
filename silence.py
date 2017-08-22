
from itertools import product

from pydub.silence import detect_silence


def split_on_silence_keep_before(audio_segment, min_silence_len=500,
                                 silence_thresh=-42):
    '''Splits audios segments in chuncks separated by silence.
    Keep the silence in the begginning of each chunk, as possible,
    and attach silence at the end to the last chunk.'''

    silent_ranges = detect_silence(
        audio_segment, min_silence_len, silence_thresh)
    len_seg = len(audio_segment)

    # if there is no silence or the whole segment is silent, return it all
    if not silent_ranges or silent_ranges == [[0, len_seg]]:
        return [(0, len_seg)]

    # collect all silence starts as chunk starts
    # don't split on last silence if it ends the segment
    starts = [ini for ini, end in silent_ranges if not end == len_seg]
    # add chunk before first silence if the segment does not start on silence
    if not starts or starts[0] != 0:
        starts.insert(0, 0)

    return zip(starts, starts[1:] + [len_seg])


SILENCE_LEVELS = list(product(range(-42, -33), range(500, 200, -100)))


def seek_split(audio):
    for min_silence_len, silence_thresh in SILENCE_LEVELS:
        split = split_on_silence_keep_before(audio,
                                             min_silence_len=min_silence_len,
                                             silence_thresh=silence_thresh)
        if len(split) > 1:
            return split, min_silence_len, silence_thresh
