import numpy as np
import pytest
from mock import patch
from pydub.generators import Sine

from speechsplit import (CLASSES, SPEAKER, TRANSLATOR, build_training_data,
                         get_features)


def example_lines(example):
    '''Test util that converts a spec string to a list of lines

    Ignore:
        * empty lines
        * lines starting with "___"
        * lines starting with "#"

    and discard "|" at line borders'''
    return [line.strip('|') for line in example.strip().splitlines()
            if (line and
                not line.startswith('___') and
                not line.startswith('|___') and
                not line.startswith('#'))]


AUDIO_STUB = Sine(440).to_audio_segment(10100)


# remove lru caching for testing
get_features = get_features.__wrapped__


@pytest.mark.parametrize('size', [100, 333])
def test_segmentation_does_not_change_get_features(size):
    '''The segmentation made to avoid out of memory
    does not affect the the final result'''

    assert len(AUDIO_STUB) == 10100
    mfcc1, loud1 = get_features(AUDIO_STUB)  # no segmentation
    mfcc2, loud2 = get_features(
        AUDIO_STUB, max_windows_per_segment=size)
    assert np.all(np.isclose(mfcc1, mfcc2))
    assert np.all(np.isclose(loud1, loud2))


@pytest.mark.xfail(raises=AssertionError)
def test_segmentation_too_small_in_get_features():
    get_features(AUDIO_STUB, max_windows_per_segment=10)


SPE, TRA = [CLASSES[v] for v in SPEAKER, TRANSLATOR]
___, BIG = False, True


@pytest.mark.skip('TODO... fix this')
@pytest.mark.parametrize(
    'speaker_features, translator_features, X_all, y_all', [[
        (
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [___, BIG, ___, ___, ___, BIG, BIG, BIG, BIG, ___],
        ),
        (
            [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            [___, BIG, ___, ___, ___, BIG, BIG],
        ),
        [1.0, 5.0, 6.0, 7.0, 8.0, 1.1, 5.1, 6.1],
        [SPE, SPE, SPE, SPE, SPE, TRA, TRA, TRA]
    ], ])
def test_build_training_data(
        speaker_features, translator_features, X_all, y_all):

    speaker_features, translator_features, X_all, y_all = map(
        np.array, (speaker_features, translator_features, X_all, y_all))

    # mock get_features as the identity function...
    with patch('speechsplit.get_features', side_effect=lambda x: x):

        # ... and simply make the audio stubs directly equal to their features
        labeled_audios = {SPEAKER: speaker_features,
                          TRANSLATOR: translator_features}

        X, y = build_training_data(
            labeled_audios,
            lambda mfcc, loudness: mfcc[loudness.astype(bool)])
        assert all(X_all == X)
        assert all(y_all == y)
