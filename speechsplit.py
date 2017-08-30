import timeit
from collections import defaultdict
from math import ceil

import numpy as np
import python_speech_features
from choice import Menu
from functools32 import lru_cache
from pydub import AudioSegment
from sklearn.svm import SVC

from fragmentation import Chunk, get_chunks
from utils import flatten, play, save_yaml, timerepr

# FEATURES ###################################################################


def get_numpy_array_of_samples(audio):
    return np.fromstring(audio._data, dtype=np.dtype(audio.array_type))


WINDOW_STEP = 10    # the step between successive windows in milliseconds
WINDOW_LENGTH = 25  # the length of the analysis window in milliseconds


@lru_cache()
def get_features(audio, max_windows_per_segment=10 * 6000):  # 6000 -> 1 min
    """Compute audio features of pydub audio:
        Mel-filterbank energy features and loudness (in dBFS).

    :param audio:
    :param max_windows_per_segment: max lenght of segment for mfcc computation,
                                    in number of windows

    :returns: 2 values:
        first value is a numpy array
        with the MFCC coefficients (from 2nd to 13th) for the windowed audio

        second value is a numpy array with the same length
        with the loudness measurements (in dBFS)
        of each corresponding coefficients window
    """
    def build_mfcc(segment):
        signal = get_numpy_array_of_samples(segment)
        return python_speech_features.mfcc(signal, segment.frame_rate,
                                           winlen=WINDOW_LENGTH / 1000.0,
                                           winstep=WINDOW_STEP / 1000.0,
                                           appendEnergy=False)[:, 1:]

    # Build mfcc in slices to avoid out of memory issues
    # We need some margin before and after the slices,
    # otherwise windowing would distort the values at the edges

    mfcc_margin = 10  # in number of windows
    msg = 'Segmentation size has to be greater than {}'.format(mfcc_margin)
    assert max_windows_per_segment > mfcc_margin, msg

    audio_margin = mfcc_margin * WINDOW_STEP
    audio_segmentation_size = max_windows_per_segment * WINDOW_STEP

    ranges = [(0 if start == 0 else start - audio_margin,  # slice start
               start + audio_segmentation_size + audio_margin,   # slice end
               0 if start == 0 else mfcc_margin  # result offset
               )
              for start in range(0, len(audio), audio_segmentation_size)]

    mfcc_segments = [
        build_mfcc(audio[start:end])[
            result_offset:result_offset + max_windows_per_segment]
        for start, end, result_offset in ranges
    ]

    # the mfcc conputation discards one window at the end
    # to keep things tidy we repeat the last window (of the last segment)
    mfcc_segments.append(mfcc_segments[-1][-1:])

    mfcc = np.concatenate(mfcc_segments)

    def gen_dBFS():
        "Generate the sequence of loudness measures (dBFS) of the windows"
        for index in range(len(mfcc)):
            start = index * WINDOW_STEP
            end = start + WINDOW_LENGTH
            yield audio[start:end].dBFS

    loudness = np.fromiter(gen_dBFS(), dtype=np.dtype(float), count=len(mfcc))

    return mfcc, loudness


# CLASSIFICATION #############################################################


SPEAKER, TRANSLATOR, BOTH = 'speaker', 'translator', 'both'
VOICES = [SPEAKER, TRANSLATOR]
TRUTH_OPTIONS = [BOTH] + VOICES
CLASSES = {SPEAKER: 1, TRANSLATOR: 2}


def get_some_chunks_with_set_truth(chunks, operation_on_chunk=None,
                                   # 10 seconds
                                   min_duration=10000):
    '''Aggregate some chunks by ground truth.
    Each group must have at least min_duration total audible duration'''

    samples = defaultdict(list)
    for chunk in chunks:

        if operation_on_chunk:
            operation_on_chunk(chunk)

        # accumulate segment on proper sample
        if chunk.truth in VOICES:
            samples[chunk.truth].append(chunk)

        if all(sum(c.audible_len for c in chunk_list) >= min_duration
               for chunk_list in samples.values()):
            break
    return samples


def loudness_between(min=-float('inf'), max=float('inf')):
    return lambda mfcc, loudness: mfcc[(loudness >= min) & (loudness < max)]


def louder_than(dbfs):
    return lambda mfcc, loudness: mfcc[loudness >= dbfs]


DEFAULT_FILTER = louder_than(-33)


def get_mfcc_from_chunk(features, chunk, filter=DEFAULT_FILTER):
    # cut features to this chunk
    start, end = chunk.start / WINDOW_STEP, chunk.end / WINDOW_STEP
    mfcc, loudness = features
    mfcc, loudness = mfcc[start:end], loudness[start:end]
    return filter(mfcc, loudness)


def build_training_data(features, training_chunks, filter=DEFAULT_FILTER):

    label_to_mfcc = [
        (label, np.concatenate([get_mfcc_from_chunk(features, chunk, filter)
                                for chunk in training_chunks[label]]))
        for label in VOICES]

    X_all = np.concatenate([mfcc for label, mfcc in label_to_mfcc])
    y_all = np.concatenate([np.repeat(CLASSES[label], len(mfcc))
                            for label, mfcc in label_to_mfcc])
    return X_all, y_all


def predict_one_chunk(clf, features, chunk, filter=DEFAULT_FILTER):
    mfcc = get_mfcc_from_chunk(features, chunk, filter)
    prediction = clf.predict(mfcc)
    count_voice, voice = max(
        (np.count_nonzero(prediction == CLASSES[voice]), voice)
        for voice in VOICES)
    return (voice, count_voice / float(len(prediction)))


def predict_chunks(clf, features, chunks, filter=DEFAULT_FILTER):
    for chunk in chunks:
        chunk.label = predict_one_chunk(clf, features, chunk, filter)
    return chunks


def get_best_labeled(chunks, limit=None, min_audible_len=1000):
    # enforce min audible length and  give preference to larger chunks
    return sorted([c for c in chunks if c.audible_len > min_audible_len],
                  key=lambda c: (round(c.label[1], 2), c.audible_len),
                  reverse=True)[:limit]


def get_percentile_best_labeled(chunks, percentile, min_audible_len=1000):
    limit = int(ceil(len(chunks) * percentile / 100.0))
    return get_best_labeled(chunks, limit, min_audible_len)


def refit(clf, features, training_chunks):
    clf.fit(*build_training_data(features, training_chunks))


def refit_and_predict_chunks(clf, features, training_chunks, chunks):
    start_time = timeit.default_timer()
    print('.......... refit and re-predict started ..........')

    refit(clf, features, training_chunks)
    predict_chunks(clf, features, chunks)

    elapsed = timeit.default_timer() - start_time
    print('---------- refit and re-predict DONE (in {}) ----------'.format(
        timerepr(int(elapsed * 1000))
    ))


def start_classification(audio):
    '''Bootstrap classification returning a classifier fit to some small sample
    of chunks with set ground truth from each class'''

    def ask_operation(chunk):
        'ask the operator to classify this chunk'

        question = Menu(TRUTH_OPTIONS,
                        title="Who's speaking in the audio you just heard?")
        if chunk.truth not in TRUTH_OPTIONS:
            play(chunk.cut(audio))
            chunk.truth = question.ask()

    clf = SVC(C=1, gamma=0.001, kernel='rbf', random_state=0)
    features = get_features(audio)
    chunks = get_chunks(audio)
    # pre label if nececessary
    get_some_chunks_with_set_truth(chunks, ask_operation)

    refit_and_predict_chunks(
        clf, features, get_some_chunks_with_set_truth(chunks), chunks)
    return clf


def copy_chunks(chunks):
    return [Chunk(**c.copy()) for c in chunks]


def error_in_chunks(chunks):
    wrong = [c for c in chunks if c.truth != c.label[0]]
    return len(wrong) / float(len(chunks))


def refit_from_best(clf, audio, percentile=5):
    remaining = chunks = get_chunks(audio)
    features = get_features(audio)
    labeled = []
    evolution = []

    try:
        while(remaining):
            if not labeled:
                # begin with just pre labeled data
                labeled = flatten(
                    get_some_chunks_with_set_truth(chunks).values())
                for c in labeled:
                    c.label = (c.truth, 1)
            else:
                best = [get_percentile_best_labeled(
                    [c for c in remaining if c.label[0] == voice],
                    percentile, 1000)
                    for voice in VOICES]
                if not all(best):
                    break
                best = flatten(best)
                labeled += best
            remaining = [c for c in remaining if c not in labeled]
            training_chunks = {
                voice: [c for c in labeled if c.label[0] == voice]
                for voice in VOICES}

            refit_and_predict_chunks(clf, features, training_chunks, chunks)
            # important to make a copy of chunks because it changes over time
            evolution.append(map(copy_chunks, (labeled, remaining, chunks)))
            print(':::::: ', map(len, (labeled, remaining, chunks)))
            # print(map(lambda x: len(x) / float(len(chunks)),
            #           (labeled, remaining, chunks)))
            print(error_in_chunks(labeled),
                  error_in_chunks(remaining), error_in_chunks(chunks))

    except Exception as e:
        print("XXXX ERROR XXXX. "
              "Couldn't proceed after percentile: ", percentile)
        print e

    print('\nThese were the error percentages in chunk classification'
          ' at each percentile:')
    print('iteration | in labeled | in remaining | in all chunks')
    for i, (labeled, remaining, chunks) in enumerate(evolution):
        print(i, error_in_chunks(labeled),
              error_in_chunks(remaining), error_in_chunks(chunks))
    return evolution


def load_run_experiment_and_save(filename):
    clf = SVC(C=1, gamma=0.001, kernel='rbf', random_state=0)
    audio = AudioSegment.from_wav(filename)

    evolution = refit_from_best(clf, audio)

    exp_filename = 'data/experiments/' + filename.split('/')[-1]
    exp_filename = exp_filename.replace('.wav', '.yaml')
    save_yaml(exp_filename, evolution)

    return evolution
