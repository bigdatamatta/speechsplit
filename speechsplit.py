
import threading
import timeit
from collections import defaultdict
from math import ceil

import numpy as np
import python_speech_features
from choice import Menu
from functools32 import lru_cache
from pydub import AudioSegment
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC

from silence import Chunk, get_chunks
from utils import flatten, intervals_where, play, timerepr

# DATA TREATMENT  #######################################################


def smooth_bumps(data, value, width, margin):

    for start, end in intervals_where(data == value):

        # if the interval is a narrow bump ...
        if end - start <= width:
            before = data[start - margin:start]
            after = data[end: end + margin]
            surrounding_set = set(before) | set(after)

            # ... and is surrounded by a unique value
            if len(surrounding_set) == 1:
                surrounding_value = surrounding_set.pop()
                # replace all the bump with the surrounding value
                data[start:end] = surrounding_value


# FEATURE EXTRACTION  #######################################################


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
        with the MFCC coefficients (from 2nd to 12th) for the windowed audio

        second value is a numpy array with the same length
        with the loudness measurements (in dBFS)
        of each corresponding coefficients window
    """
    def build_mfcc(segment):
        signal = get_numpy_array_of_samples(segment)
        return python_speech_features.mfcc(signal, segment.frame_rate,
                                           winlen=WINDOW_LENGTH / 1000.0,
                                           winstep=WINDOW_STEP / 1000.0,
                                           appendEnergy=False)[:, 1:13]

    # Build mfcc in slices to avoid out of memory issues
    # We need some margin before and after the slices,
    # otherwise windowing would distort the values at the edges

    mfcc_margin = 10  # in number of windows
    msg = 'Segmentation size has to be greater than {}'.format(mfcc_margin)
    assert max_windows_per_segment > mfcc_margin, msg

    audio_margin = mfcc_margin * WINDOW_STEP
    audio_segmentation_size = max_windows_per_segment * WINDOW_STEP

    ranges = [(0 if start == 0 else start - audio_margin,  # splice start
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


def get_mfcc(audio):
    return get_features(audio)[0]


def get_loudness(audio):
    return get_features(audio)[1]


# CLASSIFICATION ############################################################

SPEAKER, TRANSLATOR, BOTH = 'speaker', 'translator', 'both'
VOICES = [SPEAKER, TRANSLATOR]
TRUTH_OPTIONS = [BOTH] + VOICES
CLASSES = {SPEAKER: 1, TRANSLATOR: 2}


def get_some_chunks_with_set_truth(chunks, min_duration=5000):
    # TODO: use this in pre_label !!!!
    samples = defaultdict(list)
    for chunk in chunks:
        # accumulate segment on proper sample
        if chunk.truth in VOICES:
            samples[chunk.truth].append(chunk)

        if all(sum(c.audible_len for c in chunk_list) >= min_duration
               for chunk_list in samples.values()):
            break
    return samples


def pre_label(audio, min_duration=5000):
    question = Menu(TRUTH_OPTIONS,
                    title="Who's speaking in the audio you just heard?")
    accumulated_samples = defaultdict(lambda: AudioSegment.silent(0))

    for chunk in get_chunks(audio):
        if chunk.truth not in TRUTH_OPTIONS:
            play(chunk.cut(audio))
            chunk.truth = question.ask()

        # label from ground truth and accumulate segment on proper sample
        if chunk.truth in VOICES:
            chunk.label = (chunk.truth, 1)
            accumulated_samples[chunk.truth] += chunk.cut(audio)

        # terminate if we have enough labeled data
        if all(len(a) >= min_duration for a in accumulated_samples.values()):
            break

    return accumulated_samples


def loudness_between(min=-float('inf'), max=float('inf')):
    return lambda mfcc, loudness: mfcc[(loudness >= min) & (loudness < max)]


def louder_than(dbfs):
    return lambda mfcc, loudness: mfcc[loudness >= dbfs]


DEFAULT_FILTER = louder_than(-33)


def build_training_data(features, training_chunks, filter=DEFAULT_FILTER):

    label_to_mfcc = [
        (label, np.concatenate([get_mfcc_from_chunk(features, chunk, filter)
                                for chunk in training_chunks[label]]))
        for label in VOICES]

    X_all = np.concatenate([mfcc for label, mfcc in label_to_mfcc])
    y_all = np.concatenate([np.repeat(CLASSES[label], len(mfcc))
                            for label, mfcc in label_to_mfcc])
    return X_all, y_all


def score_prediction(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target, y_pred)


def train_and_score(clf, X_all, y_all):
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=.20, random_state=0)
    clf.fit(X_train, y_train)
    f1_score_train = score_prediction(clf, X_train, y_train)
    f1_score_test = score_prediction(clf, X_test, y_test)
    return f1_score_train, f1_score_test


GRID_SEARCH_PARAMETERS = [
    # {'C': [1, 10, 100], 'kernel': ['linear']},
    {'C': [1, 10, 100], 'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001]},
]


def grid_search(X_all, y_all, parameters=GRID_SEARCH_PARAMETERS):
    clf = SVC(random_state=0)
    f1_scorer = make_scorer(f1_score)
    grid = GridSearchCV(clf, parameters, scoring=f1_scorer, n_jobs=4)
    f1_score_train, f1_score_test = train_and_score(grid, X_all, y_all)
    print('F1 score on the train an test data:')
    print(f1_score_train, f1_score_test)
    return grid.best_estimator_


def get_mfcc_from_chunk(features, chunk, filter=DEFAULT_FILTER):
    # cut features to this chunk
    start, end = chunk.start / WINDOW_STEP, chunk.end / WINDOW_STEP
    mfcc, loudness = features
    mfcc, loudness = mfcc[start:end], loudness[start:end]
    return filter(mfcc, loudness)


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


def collect_most_certain(chunks, min_proportion):
    collected = defaultdict(list)
    for chunk in chunks:
        if chunk.label:
            voice, proportion = chunk.label
            if proportion >= min_proportion:
                collected[voice].append(chunk)
    return collected


def to_segments(audio, chunks):
    return [chunk.cut(audio) for chunk in chunks]


def get_best_labeled(chunks, limit=None):
    # give preference to larger chunks
    return sorted(chunks, key=lambda c: (c.label[1], c.audible_len),
                  reverse=True)[:limit]


def get_percentile_best_labeled(chunks, percentile):
    limit = int(ceil(len(chunks) * percentile / 100.0))
    return get_best_labeled(chunks, limit)


def get_chunks_by_truth(chunks):
    return {voice: [c for c in chunks if c.truth == voice]
            for voice in VOICES}


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


def spawn_refit_and_predict(clf, features, training_chunks, chunks):
    # spawn a refit e re-predict thread if not already running
    refit_is_running = [t for t in threading.enumerate()
                        if t.name == 'refit_and_predict']
    if not refit_is_running:
        refit_thread = threading.Thread(
            name='refit_and_predict',
            target=refit_and_predict_chunks,
            args=(clf, features, training_chunks, chunks))
        refit_thread.setDaemon(True)
        refit_thread.start()


def confirm_truth(clf, audio, chunk_group_or_voice,
                  group=10, limit=10, speed=1):
    chunks = get_chunks(audio)
    print('Confirm label classifications.')
    print('Type:')
    print('      * just ENTER to confirm'
          '      * "s" to set SPEAKER as ground truth\n'
          '      * "t" to set TRANSLATOR as ground truth\n'
          '      * "b" to set BOTH as ground truth\n'
          '      * "a" to hear it again\n'
          '      * "/" to inspect one by one\n'
          '      * and anything else to stop.')

    while(limit):
        limit = limit - 1  # we need to explicitly decrement to enable repeat
        if chunk_group_or_voice in VOICES:
            unknown = [c for c in chunks
                       if not c.truth and c.label[0] == chunk_group_or_voice]
        else:
            unknown = [c for c in chunk_group_or_voice if not c.truth]

        best_first = get_best_labeled(unknown, group)

        if not best_first:
            break

        for best in best_first:
            print('#' * 30, best.label, chunks.index(best))
        play(sum(best.cut(audio) for best in best_first), speed)

        typed = raw_input().strip().lower()
        truth_option = {'s': SPEAKER, 't': TRANSLATOR, 'b': BOTH}.get(typed)

        if not typed:
            # default to label as ground truth
            for best in best_first:
                best.truth = best.label[0]
            spawn_refit_and_predict(clf, audio, get_chunks_by_truth(chunks))
        elif truth_option:
            # set explicitly
            for best in best_first:
                best.truth = truth_option
            spawn_refit_and_predict(clf, audio, get_chunks_by_truth(chunks))
        elif typed == 'a':
            # play again
            limit = limit + 1  # restore limit
            continue
        elif typed == '/':
            # start inspecting one by one
            group = 1
            continue
        else:
            break


def copy_chunks(chunks):
    return [Chunk(**c.copy()) for c in chunks]


def error_in_chunks(chunks):
    wrong = [c for c in chunks if c.truth != c.label[0]]
    return len(wrong) / float(len(chunks))


def get_best_percentile(chunks, percentile):
    # have at least pre labeled for the refit
    pre_labeled = flatten(get_some_chunks_with_set_truth(chunks).values())
    best_labeled = get_percentile_best_labeled(chunks, percentile)
    chunks = set(best_labeled) | set(pre_labeled)

    return {voice: [c for c in chunks if c.label[0] == voice]
            for voice in VOICES}


def run_experiment_refit_by_increasing_percentiles(clf, audio, step=5):
    chunks = get_chunks(audio)
    features = get_features(audio)

    # begin with just pre labeled data
    clf.fit(*build_training_data(features,
                                 get_some_chunks_with_set_truth(chunks)))
    predict_chunks(clf, features, chunks)
    evolution = [(0, copy_chunks(chunks))]

    try:
        for percentile in range(step, 101, step):
            training_chunks = get_best_percentile(chunks, percentile)
            refit_and_predict_chunks(clf, features, training_chunks, chunks)
            evolution.append((percentile, copy_chunks(chunks)))
    except Exception as e:
        print("XXXX ERROR XXXX. "
              "Couldn't proceed after percentile: ", percentile)
        print e

    print('\nThese were the error percentages in chunk classification'
          ' at each percentile:')
    for p, cc in evolution:
        print p, error_in_chunks(cc)
    return evolution


def run_experiment_refit_separating_best(clf, audio, percentile=5):
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
                labeled += get_percentile_best_labeled(remaining, percentile)
            remaining = [c for c in remaining if c not in labeled]
            training_chunks = {
                voice: [c for c in labeled if c.label[0] == voice]
                for voice in VOICES}

            refit_and_predict_chunks(clf, features, training_chunks, chunks)
            # important to make a copy of chunks because it changes over time
            evolution.append((labeled, remaining, copy_chunks(chunks)))
            print(map(len, (labeled, remaining, chunks)))
            print(map(lambda x: len(x) / float(len(chunks)),
                      (labeled, remaining, chunks)))
            print(error_in_chunks(labeled),
                  error_in_chunks(remaining), error_in_chunks(chunks))

    except Exception as e:
        print("XXXX ERROR XXXX. "
              "Couldn't proceed after percentile: ", percentile)
        print e

    print('\nThese were the error percentages in chunk classification'
          ' at each percentile:')
    print('iteration | in laaeled | in remaining | in all chunks')
    for i, cc in enumerate(evolution):
        print(i, error_in_chunks(labeled),
              error_in_chunks(remaining), error_in_chunks(chunks))
    return evolution
