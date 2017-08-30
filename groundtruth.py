
import threading

from fragmentation import get_chunks
from speechsplit import (BOTH, SPEAKER, TRANSLATOR, VOICES, get_best_labeled,
                         get_features, refit_and_predict_chunks)
from utils import play


def spawn_refit_and_predict(clf, features, training_chunks, chunks):
    # spawn a refit e re-predict thread if not already running
    refit_is_running = [t for t in threading.enumerate()
                        if t.name == 'refit_and_predict']
    if not refit_is_running:
        refit_thread = threading.Thread(
            name='refit_and_predict',
            target=refit_and_predict_chunks,
            args=(clf, features, training_chunks, chunks))
        refit_thread.setDaemon(True)
        refit_thread.start()


def confirm_truth(clf, audio, chunk_group_or_voice,
                  group=10, limit=10, speed=1):
    '''Successively confirm truth suggestions of groups of chunks.
    Spawns refit and repredict on new ground truth
    whenever possible, on a separate thread.'''

    features = get_features(audio)
    chunks = get_chunks(audio)
    print('Confirm label classifications.')
    print('Type:')
    print('      * just ENTER to confirm'
          '      * "s" to set SPEAKER as ground truth\n'
          '      * "t" to set TRANSLATOR as ground truth\n'
          '      * "b" to set BOTH as ground truth\n'
          '      * "a" to hear it again\n'
          '      * "/" to inspect one by one\n'
          '      * and anything else to stop.')

    def _refit_and_predict():
        training_chunks = {voice: [c for c in chunks if c.truth == voice]
                           for voice in VOICES}
        spawn_refit_and_predict(clf, features, training_chunks, chunks)

    while(limit):
        limit = limit - 1  # we need to explicitly decrement to enable repeat
        if chunk_group_or_voice in VOICES:
            unknown = [c for c in chunks
                       if not c.truth and c.label[0] == chunk_group_or_voice]
        else:
            unknown = [c for c in chunk_group_or_voice if not c.truth]

        best_first = get_best_labeled(unknown, group, 1000)

        if not best_first:
            # give up min audible length
            best_first = get_best_labeled(unknown, group, 0)
            if not best_first:
                # really done
                break

        for best in best_first:
            print('#' * 30, best.label, chunks.index(best))
        play(sum(best.cut(audio) for best in best_first), speed)

        typed = raw_input().strip().lower()
        truth_option = {'s': SPEAKER, 't': TRANSLATOR, 'b': BOTH}.get(typed)

        if not typed:
            # default to label as ground truth
            for best in best_first:
                best.truth = best.label[0]
            _refit_and_predict()
        elif truth_option:
            # truth value set explicitly
            for best in best_first:
                best.truth = truth_option
            _refit_and_predict()
        elif typed == 'a':
            # play again
            limit = limit + 1  # restore limit
            continue
        elif typed == '/':
            # start inspecting one by one
            # increase limit to inspect at least all group
            limit = limit + group
            group = 1
            speed = 1  # slow down
            continue
        else:
            break


def alternate_confirm_truth(clf, audio, group=10, limit=10, speed=1):
    for i in range(limit):
        for voice in VOICES:
            confirm_truth(clf, audio, voice, group, 1, speed)
