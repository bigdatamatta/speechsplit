
from itertools import product

from pydub.silence import detect_silence


def detect_silence_and_audible(audio_segment, min_silence_len=500,
                               silence_thresh=-42):
    '''Splits audios segments in chunks separated by silence.
    Keep the silence in the beginning of each chunk, as possible,
    and ignore silence after the last chunk.'''

    silent_ranges = detect_silence(audio_segment,
                                   min_silence_len, silence_thresh)
    len_seg = len(audio_segment)

    # make sure there is a silence at the beginning (even an empty one)
    if not silent_ranges or silent_ranges[0][0] is not 0:
        silent_ranges.insert(0, (0, 0))
    # make sure there is a silence at the end (even an empty one)
    if silent_ranges[-1][1] is not len_seg:
        silent_ranges.append((len_seg, len_seg))

    return [(start, end, start_next)
            for (start, end), (start_next, __) in zip(silent_ranges,
                                                      silent_ranges[1:])]


SILENCE_LEVELS = list(product(range(-42, -33), range(500, 200, -100)))


def seek_split(audio):
    for min_silence_len, silence_thresh in SILENCE_LEVELS:
        split = detect_silence_and_audible(audio,
                                           min_silence_lenf=min_silence_len,
                                           silence_thresh=silence_thresh)
        if len(split) > 1:
            return split, min_silence_len, silence_thresh
