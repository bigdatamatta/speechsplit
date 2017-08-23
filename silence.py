
import itertools
import os
from hashlib import sha1

import yaml
from pydub.silence import detect_silence

SILENCE_LEVELS = [{'silence_thresh': t, 'min_silence_len': l}
                  for t in range(-42, -33)
                  for l in range(500, 100, -100)]


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


def save_fragments(audio, iteration, chunks):
    filename = get_audio_fragments_filename(audio, iteration)
    with open(filename, 'w') as fragments_file:
        yaml.dump(chunks, fragments_file)
    return filename


def fragment(audio, max_audible_allowed_size=5000):
    chunks = [[0, 0, len(audio), -1]]
    for iteration in itertools.count():
        for pos, (silence_start, start, end, level) in enumerate(chunks):
            if (end - start > max_audible_allowed_size and
                    level + 1 < len(SILENCE_LEVELS)):
                subsplit = seek_split(audio[start:end], level + 1)
                # shift all chunks by "start"
                subsplit = [[s + start, i + start, e + start, l]
                            for s, i, e, l in subsplit]
                # attach previous silence to first chunk of subsplit
                subsplit[0][0] = silence_start
                # last end must be the silence start of next global chunk
                if pos + 1 < len(chunks):
                    chunks[pos + 1][0] = subsplit[-1][2]
                chunks[pos:pos + 1] = subsplit
                break
        else:
            # there's nothing more to split
            break
        last_filename = save_fragments(audio, iteration, chunks)
    # save with final name
    os.rename(last_filename, get_audio_fragments_filename(audio))
    return chunks
