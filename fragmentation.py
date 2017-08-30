
import itertools
import os
from hashlib import sha1

import yaml
from bunch import Bunch
from functools32 import lru_cache
from pydub.silence import detect_silence


class Chunk(Bunch):

    """Audio fragment metadata for classification"""

    def __init__(self, silence_start, start, end, level=-1,
                 truth=None, label=None):
        self.silence_start = silence_start  # start of silence at the beginning
        self.start = start  # start of audible part
        self.end = end      # end of audible part
        self.level = level  # level of split that created this chunk
        self.truth = truth  # ground truth label
        self.label = label  # label obtained by the classifier

    @property
    def len(self):
        return self.end - self.silence_start

    @property
    def audible_len(self):
        return self.end - self.start

    def cut(self, audio):
        return audio[self.start:self.end]

    def __hash__(self):
        return hash((self.silence_start, self.start, self.end, self.level,
                     self.truth, self.label))


def get_audio_hash(audio):
    return sha1(audio.get_array_of_samples()).hexdigest()[:10]


DATA_DIR = 'data'


def get_audio_chunks_filename(audio):
    chunks_extension = '.chunks.yaml'
    if hasattr(audio, 'filename'):
        basename = os.path.splitext(audio.filename)[0]
        return basename + chunks_extension
    else:
        audio_id = os.path.join(DATA_DIR, get_audio_hash(audio))
    return '{}/{}.chunks.yaml'.format(DATA_DIR, audio_id)


def save_chunks(audio, chunks):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    filename = get_audio_chunks_filename(audio)
    with open(filename, 'w') as chunks_file:
        yaml.safe_dump(chunks, chunks_file)


def load_chunks(audio):
    filename = get_audio_chunks_filename(audio)
    if os.path.exists(filename):
        with open(filename, 'r') as chunks_file:
            return [Chunk(**kwargs)
                    for kwargs in yaml.safe_load(chunks_file)]


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

    return [Chunk(silence_start, start, end, level)
            for (silence_start, start), (end, __) in zip(silent_ranges,
                                                         silent_ranges[1:])]


def seek_split(audio, level=0):
    for level in range(level, len(SILENCE_LEVELS)):
        chunks = detect_silence_and_audible(audio, level)
        if len(chunks) > 1:
            return chunks
    else:
        return chunks


def _gen_join_almost_silent(chunks, min_audible_len):
    'join each almost silent chunk as a silence beginning the following one'

    almost_silence_start = None
    for chunk in chunks:
        if chunk.audible_len < min_audible_len:
            # remember for following silence start
            # note that more than one "almost silence" can accumulate
            almost_silence_start = almost_silence_start or chunk.silence_start
        else:
            if almost_silence_start:
                chunk.silence_start = almost_silence_start
                almost_silence_start = None  # reset
            yield chunk


@lru_cache()
def get_chunks(audio, min_audible_len=300, target_audible_len=2000,
               load_if_available=True):

    # try to load from disk
    if load_if_available:
        loaded = load_chunks(audio)
        if loaded:
            return loaded

    chunks = detect_silence_and_audible(audio)
    for iteration in itertools.count(1):
        for pos, chunk in enumerate(chunks):
            if (chunk.audible_len > target_audible_len and
                    chunk.level + 1 < len(SILENCE_LEVELS)):
                subsplit = seek_split(chunk.cut(audio),
                                      chunk.level + 1)
                if len(subsplit) > 1:
                    # shift all sub chunks by "start"
                    # notice the previous label is discarded after splitting
                    for sub in subsplit:
                        sub.silence_start += chunk.start
                        sub.start += chunk.start
                        sub.end += chunk.start
                        # keep ground truth after split
                        sub.truth = chunk.truth
                    # attach previous silence to first chunk of subsplit
                    subsplit[0].silence_start = chunk.silence_start
                    # the end of the last sub chunk must
                    # be the silence start of next global chunk
                    if pos + 1 < len(chunks):
                        chunks[pos + 1].silence_start = subsplit[-1].end
                    # replace chunk with it's subsplit
                    chunks[pos:pos + 1] = subsplit
                    break
        else:
            # there's nothing more to split
            break

    chunks = list(_gen_join_almost_silent(chunks, min_audible_len))
    save_chunks(audio, chunks)
    return chunks
