
import numpy as np
import pydub
import python_speech_features
from functools32 import lru_cache
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC

from utils import intervals_where, timerepr

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


def extract_audio_features(audio, window_length=25, window_step=10,
                           max_windows_per_segment=100000):
    """Compute audio features of pydub audio:
        Mel-filterbank energy features and loudness (in dBFS).

    :param audio:
    :param window_length: the length of the analysis window in milliseconds.
    :param window_step: the step between successive windows in milliseconds.
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
        return python_speech_features.mfcc(
            signal, segment.frame_rate,
            winlen=window_length / 1000.0, winstep=window_step / 1000.0,
            appendEnergy=False)[:, 1:13]

    # Build mfcc in slices to avoid out of memory issues
    # We need some margin before and after the slices,
    # otherwise windowing would distort the values at the edges

    mfcc_margin = 10  # in number of windows
    msg = 'Segmentation size has to be greater than {}'.format(mfcc_margin)
    assert max_windows_per_segment > mfcc_margin, msg

    audio_margin = mfcc_margin * window_step
    audio_segmentation_size = max_windows_per_segment * window_step

    ranges = [(0 if start == 0 else start - audio_margin,  # splice start
               start + audio_segmentation_size + audio_margin,   # slice end
               0 if start == 0 else mfcc_margin  # result offset
               )
              for start in range(0, len(audio), audio_segmentation_size)]

    mfcc = np.concatenate([
        build_mfcc(audio[start:end])[
            result_offset:result_offset + max_windows_per_segment]
        for start, end, result_offset in ranges
    ])

    def gen_dBFS():
        "Generate the sequence of loudness measures (dBFS) of the windows"
        for index in range(len(mfcc)):
            start = index * window_step
            end = start + window_length
            yield audio[start:end].dBFS

    loudness = np.fromiter(gen_dBFS(), dtype=np.dtype(float), count=len(mfcc))

    return mfcc, loudness


def loudness_filter(min=-float('inf'), max=float('inf')):
    return lambda mfcc, loudness: mfcc[(loudness >= min) & (loudness < max)]


def louder_than(dbfs):
    return lambda mfcc, loudness: mfcc[loudness >= dbfs]


class AudioSegment(pydub.AudioSegment):

    def __init__(self, *args, **kwargs):
        super(AudioSegment, self).__init__(*args, **kwargs)

    @property
    @lru_cache()
    def features(self):
        return extract_audio_features(self)

    @property
    def mfcc(self):
        return self.features[0]

    @property
    def loudness(self):
        return self.features[1]

    def __repr__(self):
        return 'Audio (length: {})'.format(timerepr(len(self)))


def split_by_silence(audio, silence_thresh, min_silence_len=500):
    pass

# CLASSIFICATION ############################################################


SPEAKER_CLASS, TRANSLATOR_CLASS = 1, 2


def build_training_data(speaker_features, translator_features, filter):

    speaker_mfcc, translator_mfcc = [filter(*speaker_features),
                                     filter(*translator_features)]
    X_all = np.concatenate((speaker_mfcc, translator_mfcc))
    y_all = np.concatenate([np.repeat(y, len(mfcc)) for y, mfcc in [
        (SPEAKER_CLASS, speaker_mfcc),
        (TRANSLATOR_CLASS, translator_mfcc)]])

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
    {'C': [1, 10, 100], 'kernel': ['linear']},
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


def predict(clf, audio, filter):
    mfcc = filter(*extract_audio_features(audio))
    pred = clf.predict(mfcc)
    return [100 * np.count_nonzero(pred == k) / float(len(pred)) for k in (0, 1)]
