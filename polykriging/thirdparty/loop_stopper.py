from threading import Timer

"""
This module provides a class that can be used to stop a loop after a given
amount of time.

References
----------
https://www.quora.com/In-Python-how-can-I-skip-an-iteration-in-a-for-loop-if-it-takes-longer-than-5-secs
"""

class LoopStopper:

    def __init__(self, seconds):
        self._loop_stop = False
        self._seconds = seconds

    def _stop_loop(self):
        self._loop_stop = True

    def run(self, generator_expression, task):
        """ Execute a task a number of times based on the generator_expression"""
        t = Timer(self._seconds, self._stop_loop)
        t.start()
        for i in generator_expression:
            task(i)
            if self._loop_stop:
                break
        t.cancel()  # Cancel the timer if the loop ends ok.


if __name__ == '__main__':
    ls = LoopStopper(15)
    ls.run(range(10000), print)
