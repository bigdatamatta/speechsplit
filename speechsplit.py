
from collections import defaultdict

import numpy as np
import python_speech_features
from choice import Menu
from functools32 import lru_cache
from pydub import AudioSegment
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC

from silence import get_chunks
from utils import intervals_where, play

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
def get_features(audio, max_windows_per_segment=100000):
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


def pre_label(audio, min_duration=5000):
    question = Menu(TRUTH_OPTIONS,
                    title="Who's speaking in the audio you just heard?")
    accumulated_samples = defaultdict(lambda: AudioSegment.silent(0))

    for chunk in get_chunks(audio):
        if chunk.truth not in TRUTH_OPTIONS:
            play(chunk.cut(audio))
            chunk.truth = question.ask()

        # accumulate segment on proper sample
        if chunk.truth in VOICES:
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


def build_training_data(labeled_audios, filter=DEFAULT_FILTER):

    label_to_mfcc = [
        (label, filter(*get_features(labeled_audios[label])))
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


def predict(clf, audio, filter=DEFAULT_FILTER):
    mfcc = filter(*get_features(audio))
    prediction = clf.predict(mfcc)
    count_voice, voice = max(
        (np.count_nonzero(prediction == CLASSES[voice]), voice)
        for voice in VOICES)
    return (voice, count_voice / float(len(prediction)))


def predict_chunks(clf, audio, filter=DEFAULT_FILTER):
    chunks = get_chunks(audio)
    for chunk in chunks:
        if chunk.truth not in TRUTH_OPTIONS:
            chunk.label = predict(clf, chunk.cut(audio), filter)
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
