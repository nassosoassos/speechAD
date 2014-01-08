###########################################################################
#
# File:         acctrack.py
# Date:         Thu 8 Jan 2009 16:04
# Author:       Ken Basye
# Description:  Classification accuracy tracking and printing
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
###########################################################################

"""
    Processor-based tracking and printing of classification accuracy
"""

from __future__ import division, with_statement
from onyx.util.streamprocess import ProcessorBase
from onyx.util.debugprint import dcheck, DebugPrint

class AccTracker(ProcessorBase):
    """
    A processor to track and print classification accuracy

    Incoming events should be in the form output by classifying processors, see
    AdaptingGmmClassProcessor.  Events are sent through without modification.
    The debug flag 'acc_track' can be turned on to cause interesting information
    to be output in chartable form.

    >>> at = AccTracker(sending=False)

    >>> right = ('A', ((1, 'A'), (0, 'B')))
    >>> wrong = ('B', ((1, 'A'), (0, 'B')))
    >>> input = tuple(([right]* 25 + [wrong] * 10) * 40)
    >>> len(input)
    1400
    >>> with DebugPrint('acc_track'):
    ...     for item in input:
    ...         at.process(item)
    |                   |                   |                   |              *    |                   |
    |                   |                   |                   |              *    |                   |
    |                   |                   |                   |         x  +      |                   |
    |                   |                   |                   |         x +       |                   |
    |                   |                   |                   |         x +       |                   |
    |                   |                   |                   |         x+        |                   |
    |                   |                   |                   |         x+        |                   |
    |                   |                   |                   |          +x       |                   |
    |                   |                   |                   |           +  x    |                   |
    |                   |                   |                   |           *       |                   |
    |                   |                   |                   |         x+        |                   |
    |                   |                   |                   |         x+        |                   |
    |                   |                   |                   |         x+        |                   |
    |                   |                   |                   |         x+        |                   |
    """

    def __init__(self, sendee=None, sending=True, short_term_window_size = 100, reporting_interval=100):
        super(AccTracker, self).__init__(sendee, sending=sending)
        self._window_size = short_term_window_size
        self._interval = reporting_interval
        self._long_term_count = 0
        self._long_term_correct = 0
        self._short_term_count = 0
        self._short_term_correct = 0
        self._line_template = '|' + (' ' * 19 + '|') * 5


    def process(self, event):
        label, scores = event
        self.send(event)
        # We try to do as little as possible unless we're going to do printing, so
        # there are many early outs here.
        if label is None: 
            return
        self._long_term_count += 1
        self._short_term_count += 1
        if label == scores[0][1]:
            self._long_term_correct += 1
            self._short_term_correct += 1

        if self._long_term_count % self._interval != 0:
            return
        
        dc = dcheck("acc_track")
        if not dc:
            return

        to_print = list(self._line_template)
        lt_acc = int(100 * self._long_term_correct / self._long_term_count)
        assert 0 <= lt_acc <= 100
        to_print[lt_acc] = '+'
        st_acc = int(100 * self._short_term_correct / self._short_term_count)
        assert 0 <= st_acc <= 100
        to_print[st_acc] = 'x'            
        if lt_acc == st_acc:
            to_print[st_acc] = '*'            
        if self._short_term_count > self._window_size:
            self._short_term_count = self._short_term_correct = 0

        dc(DebugPrint.NO_PREFIX, ''.join(to_print))
        

def fifo_test():
    from time import sleep
    at = AccTracker(sending=False)

    right = ('A', ((1, 'A'), (0, 'B')))
    wrong = ('B', ((1, 'A'), (0, 'B')))
    input = tuple(([right]* 25 + [wrong] * 10) * 100)
    with open('/tmp/fifo.acc', 'w') as fifo:
        with DebugPrint(fifo, 'acc_track'):
            while 1:
                sleep(0.1)
                for item in input:
                    at.process(item)


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    args = argv[1:]
    if '--fifotest' in args:
        fifo_test()
    


