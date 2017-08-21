import pytest

import silence
from silence import split_on_silence_keep_before


class AudioSegmentStub(object):

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


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
def test_split_on_silence(monkeypatch, silence_ranges, audio_length, split_ranges):

    audio_stub = AudioSegmentStub(audio_length)

    def mock_detect_silence(audio_segment, min_silence_len, silence_thresh):
        assert audio_segment == audio_stub
        return silence_ranges

    monkeypatch.setattr(silence, 'detect_silence', mock_detect_silence)
    assert split_on_silence_keep_before(audio_stub) == split_ranges
