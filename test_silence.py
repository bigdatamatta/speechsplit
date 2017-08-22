import pytest
from mock import MagicMock, patch

from silence import SILENCE_LEVELS, detect_silence_and_audible


@pytest.mark.parametrize('silence_ranges, split_ranges', [

    ([(0, 100)], []),     # whole segment is silence
    ([], [[0, 0, 100]]),  # whole segment is audible

    # start: silence, end: silence (last silence ignored)
    ([(0, 10), (20, 100)], [[0, 10, 20]]),
    ([(0, 10), (20, 30), (40, 100)], [[0, 10, 20], [20, 30, 40]]),
    # start: silence, end: audible
    ([(0, 10)], [[0, 10, 100]]),
    ([(0, 10), (20, 30)], [[0, 10, 20], [20, 30, 100]]),
    # start: audible, end: silence (last silence ignored)
    ([(20, 100)], [[0, 0, 20]]),
    # start: audible, end: audible
    ([(10, 20)], [[0, 0, 10], [10, 20, 100]]),
])
def test_detect_silence_and_audible(silence_ranges, split_ranges):

    audio_stub = MagicMock()
    audio_stub.__len__.return_value = 100

    with patch('silence.detect_silence',
               return_value=silence_ranges) as mock_detect_silence:

        level = 0
        silence_thresh, min_silence_len = SILENCE_LEVELS[level]

        assert split_ranges == [
            d[:3] for d in detect_silence_and_audible(audio_stub, level)]
        mock_detect_silence.assert_called_once_with(
            audio_stub, min_silence_len, silence_thresh)
