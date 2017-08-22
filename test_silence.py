import pytest
from mock import MagicMock, patch

from silence import split_on_silence_keep_before


@pytest.mark.parametrize('silence_ranges, audio_length, split_ranges', [

    ([(0, 100)], 100, [(0, 100)]),  # whole segment is silence
    ([], 100, [(0, 100)]),          # whole segment is audible

    # start: silence, end: silence
    ([(0, 10), (20, 100)], 100, [(0, 100)]),
    ([(0, 10), (20, 30), (40, 100)], 100, [(0, 20), (20, 100)]),
    # start: silence, end: audible
    ([(0, 10)], 100, [(0, 100)]),
    ([(0, 10), (20, 30)], 100, [(0, 20), (20, 100)]),
    # start: audible, end: silence
    ([(20, 100)], 100, [(0, 100)]),
    # start: audible, end: audible
    ([(10, 20)], 100, [(0, 10), (10, 100)]),
])
def test_split_on_silence(silence_ranges, audio_length, split_ranges):

    audio_stub = MagicMock()
    audio_stub.__len__.return_value = audio_length

    with patch('silence.detect_silence',
               return_value=silence_ranges) as mock_detect_silence:

        assert split_on_silence_keep_before(audio_stub, 999, -99) == split_ranges
        mock_detect_silence.assert_called_once_with(audio_stub, 999, -99)
