import pytest
from mock import MagicMock, patch

from silence import detect_silence_and_audible


@pytest.mark.parametrize('silence_ranges, split_ranges', [

    ([(0, 100)], []),     # whole segment is silence
    ([], [(0, 0, 100)]),  # whole segment is audible

    # start: silence, end: silence (last silence ignored)
    ([(0, 10), (20, 100)], [(0, 10, 20)]),
    ([(0, 10), (20, 30), (40, 100)], [(0, 10, 20), (20, 30, 40)]),
    # start: silence, end: audible
    ([(0, 10)], [(0, 10, 100)]),
    ([(0, 10), (20, 30)], [(0, 10, 20), (20, 30, 100)]),
    # start: audible, end: silence (last silence ignored)
    ([(20, 100)], [(0, 0, 20)]),
    # start: audible, end: audible
    ([(10, 20)], [(0, 0, 10), (10, 20, 100)]),
])
def test_split_on_silence(silence_ranges, split_ranges):

    audio_stub = MagicMock()
    audio_stub.__len__.return_value = 100

    with patch('silence.detect_silence',
               return_value=silence_ranges) as mock_detect_silence:

        assert split_ranges == detect_silence_and_audible(
            audio_stub, 999, -99)
        mock_detect_silence.assert_called_once_with(audio_stub, 999, -99)
