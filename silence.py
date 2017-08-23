
import itertools
import os
from hashlib import sha1

import yaml
from functools32 import lru_cache
from pydub.silence import detect_silence

SILENCE_LEVELS = [{'silence_thresh': t, 'min_silence_len': l}
                  for t in range(-42, -33)
                  for l in range(500, 100, -100)]
NO_LABEL = '?'


def detect_silence_and_audible(audio_segment, level=0):
    '''Splits audios segments in chunks separated by silence.
    Keep the silence in the beginning of each chunk, as possible,
    and ignore silence after the last chunk.'''

    silent_ranges = detect_silence(audio_segment, seek_step=10,
                                   **SILENCE_LEVELS[level])
    len_seg = len(audio_segment)

    # make sure there is a silence at the beginning (even an empty one)
    if not silent_ranges or silent_ranges[0][0] is not 0:
        silent_ranges.insert(0, (0, 0))
    # make sure there is a silence at the end (even an empty one)
    if silent_ranges[-1][1] is not len_seg:
        silent_ranges.append((len_seg, len_seg))

    return [[start, end, start_next, level]
            for (start, end), (start_next, __) in zip(silent_ranges,
                                                      silent_ranges[1:])]


def seek_split(audio, level=0):
    for level in range(level, len(SILENCE_LEVELS)):
        chunks = detect_silence_and_audible(audio, level)
        if len(chunks) > 1:
            return chunks
    else:
        return chunks


def get_audio_id(audio):
    return sha1(audio[:1000].get_array_of_samples()).hexdigest()[:10]


def get_audio_fragments_filename(audio, iteration='final'):
    audio_id = get_audio_id(audio)
    return '{}.{}.fragments.yaml'.format(audio_id, iteration)


def save_fragments(audio, chunks, iteration='final'):
    filename = get_audio_fragments_filename(audio, iteration)
    with open(filename, 'w') as fragments_file:
        yaml.dump(chunks, fragments_file)
    return filename


@lru_cache()
def do_fragment(audio, max_audible_allowed_size=5000):
    chunks = [d + [NO_LABEL] for d in detect_silence_and_audible(audio)]
    for iteration in itertools.count(1):
        for pos, (silence_start, start, end, level, label
                  ) in enumerate(chunks):
            if (end - start > max_audible_allowed_size and
                    level + 1 < len(SILENCE_LEVELS)):
                subsplit = seek_split(audio[start:end], level + 1)
                if len(subsplit) > 1:
                    # shift all chunks by "start" and add label placeholder
                    # notice we erase the previous label after splitting
                    subsplit = [[s + start, i + start, e + start, l, NO_LABEL]
                                for s, i, e, l in subsplit]
                    # attach previous silence to first chunk of subsplit
                    subsplit[0][0] = silence_start
                    # last end must be the silence start of next global chunk
                    if pos + 1 < len(chunks):
                        chunks[pos + 1][0] = subsplit[-1][2]
                    chunks[pos:pos + 1] = subsplit
                    last_saved = save_fragments(audio, chunks, iteration)
                    break
        else:
            # there's nothing more to split
            if iteration > 1:
                os.remove(last_saved)
            save_fragments(audio, chunks)
            break
    return chunks
