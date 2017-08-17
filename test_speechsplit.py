
import numpy as np
import pytest

from speechsplit import (SPEAKER_CLASS, TRANSLATOR_CLASS, build_training_data,
                         smooth_bumps)

smooth_bumps_examples = '''
_______________|
    ...    ....|
           ....|
_______________|
..     ......  |
       ......  |
_______________|
'''
lines = (l.strip('|') for l in smooth_bumps_examples.strip().splitlines()
         if not l.startswith('___'))
smooth_bumps_examples = zip(lines, lines)


@pytest.mark.parametrize('data, output', smooth_bumps_examples)
def test_smooth_bumps(data, output):
    data = np.array(list(data))
    smooth_bumps(data, '.', margin=4, width=3)
    assert ''.join(data) == output


SPE, TRA = SPEAKER_CLASS, TRANSLATOR_CLASS
___, BIG = False, True


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

    X, y = build_training_data(
        speaker_features, translator_features,
        lambda mfcc, loudness: mfcc[loudness.astype(bool)])
    assert all(X_all == X)
    assert all(y_all == y)
