import pytest
from mock import MagicMock, patch
from pydub.generators import Sine

from silence import Chunk, detect_silence_and_audible, get_chunks


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
            [c.silence_start, c.start, c.end]
            for c in detect_silence_and_audible(audio_stub, 0)]
        mock_detect_silence.assert_called_once()
        assert (audio_stub,) == mock_detect_silence.call_args[0]


HI = Sine(440).to_audio_segment(1000)
LO = HI.apply_gain(-50)

get_chunks = get_chunks.__wrapped__  # remove lru cache for testing


def test_get_chunks():
    audio = LO + HI * 2 + LO[:400] + HI * 2 + LO + HI * 10

    with patch('silence.save_chunks'):  # simply turn off saving
        assert [Chunk(0, 1000, 5400, 0),
                Chunk(5400, 6400, 16400, 0)] == get_chunks(
                    audio, target_audible_len=5000)
        assert [Chunk(0, 1000, 3000, 1),
                Chunk(3000, 3400, 5400, 1),
                Chunk(5400, 6400, 16400, 0)] == get_chunks(
                    audio, target_audible_len=3000)
