import pytest
from mock import MagicMock, patch
from pydub.generators import Sine

from silence import detect_silence_and_audible, do_fragment


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

        assert split_ranges == [
            d[:3] for d in detect_silence_and_audible(audio_stub, 0)]
        mock_detect_silence.assert_called_once()
        assert (audio_stub,) == mock_detect_silence.call_args[0]


HI = Sine(440).to_audio_segment(1000)
LO = HI.apply_gain(-50)

do_fragment = do_fragment.__wrapped__  # remove lru cache for testing


def test_fragment():
    audio = LO + HI * 2 + LO[:400] + HI * 2 + LO + HI * 10
    # simply turn off yaml saving and os.remove
    with patch('silence.save_fragments'):
        with patch('os.remove'):
            assert [[0, 1000, 5400, 0, '?'],
                    [5400, 6400, 16400, 0, '?']] == do_fragment(audio, 5000)
            assert [[0, 1000, 3000, 1, '?'],
                    [3000, 3400, 5400, 1, '?'],
                    [5400, 6400, 16400, 0, '?']] == do_fragment(audio, 3000)
