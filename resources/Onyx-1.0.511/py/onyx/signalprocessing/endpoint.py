###########################################################################
#
# File:         endpoint.py
# Date:         8-Jan-2009
# Author:       Hugh Secker-Walker
# Description:  Utterance tagging
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
    Support for tagging utterance frames as speech/background.  Currently
    includes tracker values along with tags in order to support demoing
    instrumentation.  But really raises intersting issues regarding integrated
    instrumentation.
"""

from __future__ import with_statement
from __future__ import division
from collections import deque
from itertools import chain
import numpy as N
from onyx.util.debugprint import DebugPrint, dcheck, dprint

class RangeTracker(object):
    """
    Simple, one-dimensional range tracking.

    >>> tracker = RangeTracker(min_hunt=1, max_hunt=-2, dtype=N.int)
    >>> for sample in range(5) + range(5, 0, -1):
    ...   sample *= 4
    ...   print sample, tracker(sample)
    0 (0, 0)
    4 (1, -2)
    8 (2, 8)
    12 (3, 12)
    16 (4, 16)
    20 (5, 20)
    16 (6, 18)
    12 (7, 16)
    8 (8, 14)
    4 (4, 12)

    >>> for i in xrange(8):
    ...   print 3, tracker(3)
    3 (3, 10)
    3 (3, 8)
    3 (3, 6)
    3 (3, 4)
    3 (3, 2)
    3 (3, 3)
    3 (3, 3)
    3 (3, 3)
    """
    def __init__(self, min_hunt, max_hunt, dtype=N.float32):
        assert min_hunt > 0 and max_hunt < 0, str(min_hunt) + ' ' + str(max_hunt)
        self.hunt = N.array((min_hunt, max_hunt), dtype=dtype)
        self.state = None
        self.prev_gt_max = False
    @property
    def range(self):
        return tuple(self.state)
    def __call__(self, value):
        min_hunt, max_hunt = self.hunt
        state = self.state
        if state is None:
            state = self.state = N.array((value, value), dtype=self.hunt.dtype)
        else:
            state += self.hunt
            # XXX not obvious how to use Numpy for this decision-based work....
            # maybe vectorize on less-than via (value < state_min) (-value <
            # -state_max); for now we're falling back on Python
            state_min, state_max = state
            if value < state_min:
                state[0] = value                
            this_gt_max = (value > state_max)
            if this_gt_max:
                # not so aggressive on the max, need to see two in a row
                if self.prev_gt_max:
                    state[1] = value
            self.prev_gt_max = this_gt_max
        return tuple(state)

UTT = 'utt'
BAC = 'bac'
class UtteranceTagger(object):
    """
    Utterance tagging based on simple tracking.

    >>> tagger = UtteranceTagger(1, -2, 10, 200, 600, 20, 10, 25, 15)
    >>> tags = tuple(tagger(value) for value in chain([20, 25], [60] * 50, [20] * 40, [60] * 50, [20] * 40))
    >>> tagger.range
    (20.0, 20.0)

    >>> tagger(30)
    ((False, (20.0, 20.0)),)
    >>> tagger(30)
    ((False, (20.0, 20.0)),)

    Subtle: these False tags, and the range are the categorization of earlier
    samples from the ending set of 20's we pumped in above, not the 30's we just
    pumped.  Note that the current range has adapted to the 30's.

    >>> tagger.range
    (22.0, 30.0)

    >>> tuple(tuple(tagx[0] for tagx in tag) if tag else () for tag in tags)
    ((), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True,), (True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,), (False,))

    >>> tags
    ((), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ((True, (20.0, 20.0)), (True, (21.0, 18.0)), (True, (22.0, 60.0)), (True, (23.0, 60.0)), (True, (24.0, 60.0)), (True, (25.0, 60.0)), (True, (26.0, 60.0)), (True, (27.0, 60.0)), (True, (28.0, 60.0)), (True, (29.0, 60.0)), (True, (30.0, 60.0)), (True, (31.0, 60.0)), (True, (32.0, 60.0)), (True, (33.0, 60.0)), (True, (34.0, 60.0)), (True, (35.0, 60.0)), (True, (36.0, 60.0)), (True, (37.0, 60.0)), (True, (38.0, 60.0)), (True, (39.0, 60.0))), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ((True, (40.0, 60.0)),), ((True, (41.0, 60.0)),), ((True, (42.0, 60.0)),), ((True, (43.0, 60.0)),), ((True, (44.0, 60.0)),), ((True, (45.0, 60.0)),), ((True, (46.0, 60.0)),), ((True, (47.0, 60.0)),), ((True, (48.0, 60.0)),), ((True, (49.0, 60.0)),), ((True, (50.0, 60.0)), (True, (51.0, 60.0)), (True, (52.0, 60.0)), (True, (53.0, 60.0)), (True, (54.0, 60.0)), (True, (55.0, 60.0)), (True, (56.0, 60.0)), (True, (57.0, 60.0)), (True, (58.0, 60.0)), (True, (59.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (20.0, 58.0)), (True, (20.0, 56.0)), (True, (20.0, 54.0))), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ((False, (20.0, 52.0)),), ((False, (20.0, 50.0)),), ((False, (20.0, 48.0)),), ((False, (20.0, 46.0)),), ((False, (20.0, 44.0)),), ((False, (20.0, 42.0)),), ((False, (20.0, 40.0)),), ((False, (20.0, 38.0)),), ((False, (20.0, 36.0)),), ((False, (20.0, 34.0)),), ((False, (20.0, 32.0)),), ((False, (20.0, 30.0)),), ((False, (20.0, 28.0)),), ((False, (20.0, 26.0)),), ((False, (20.0, 24.0)),), ((False, (20.0, 22.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 18.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 20.0)),), ((True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (20.0, 20.0)), (True, (21.0, 60.0)), (True, (22.0, 60.0)), (True, (23.0, 60.0)), (True, (24.0, 60.0)), (True, (25.0, 60.0)), (True, (26.0, 60.0)), (True, (27.0, 60.0)), (True, (28.0, 60.0)), (True, (29.0, 60.0)), (True, (30.0, 60.0))), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ((True, (31.0, 60.0)),), ((True, (32.0, 60.0)),), ((True, (33.0, 60.0)),), ((True, (34.0, 60.0)),), ((True, (35.0, 60.0)),), ((True, (36.0, 60.0)),), ((True, (37.0, 60.0)),), ((True, (38.0, 60.0)),), ((True, (39.0, 60.0)),), ((True, (40.0, 60.0)),), ((True, (41.0, 60.0)),), ((True, (42.0, 60.0)),), ((True, (43.0, 60.0)),), ((True, (44.0, 60.0)),), ((True, (45.0, 60.0)),), ((True, (46.0, 60.0)),), ((True, (47.0, 60.0)),), ((True, (48.0, 60.0)),), ((True, (49.0, 60.0)),), ((True, (50.0, 60.0)), (True, (51.0, 60.0)), (True, (52.0, 60.0)), (True, (53.0, 60.0)), (True, (54.0, 60.0)), (True, (55.0, 60.0)), (True, (56.0, 60.0)), (True, (57.0, 60.0)), (True, (58.0, 60.0)), (True, (59.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (60.0, 60.0)), (True, (20.0, 58.0)), (True, (20.0, 56.0)), (True, (20.0, 54.0)), (True, (20.0, 52.0))), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), (), ((False, (20.0, 50.0)),), ((False, (20.0, 48.0)),), ((False, (20.0, 46.0)),), ((False, (20.0, 44.0)),), ((False, (20.0, 42.0)),), ((False, (20.0, 40.0)),), ((False, (20.0, 38.0)),), ((False, (20.0, 36.0)),), ((False, (20.0, 34.0)),), ((False, (20.0, 32.0)),), ((False, (20.0, 30.0)),), ((False, (20.0, 28.0)),), ((False, (20.0, 26.0)),), ((False, (20.0, 24.0)),), ((False, (20.0, 22.0)),), ((False, (20.0, 20.0)),), ((False, (20.0, 18.0)),))

    >>> with DebugPrint('UtteranceTagger'):
    ...   tagger = UtteranceTagger(1, -2, 10, 200, 600, 20, 10, 25, 15)
    ...   tagger(0)
    ...   tagger(30)
    ...   tagger(25)
    UtteranceTagger: __init__: 1 -2 0.2 0.6 10 20 10 25 15
    UtteranceTagger: state bac  count 0  queue 0  value 0  range 0.0 0.0
    ()
    UtteranceTagger: state bac  count 0  queue 1  value 30  range 1.0 -2.0
    UtteranceTagger: new_count 1
    ()
    UtteranceTagger: state bac  count 1  queue 2  value 25  range 2.0 25.0
    UtteranceTagger: new_count 2
    ()
    """
    def __init__(self, min_hunt, max_hunt, min_range, low_per_mil, high_per_mil, start_window, start_count, stop_window, stop_count):
        assert high_per_mil > low_per_mil
        assert min_range > 0
        assert start_window > start_count
        assert stop_window > stop_count
        self.tracker = RangeTracker(min_hunt, max_hunt)
        self.data = low_per_mil / 1000, high_per_mil / 1000, min_range, start_window, start_count, stop_window, stop_count
        dprint('UtteranceTagger', '__init__:', min_hunt, max_hunt, *self.data)
        count = 0
        self.state = BAC, count, deque()        
    @property
    def range(self):
        return self.tracker.range
    # this returns the flag and the tracker values, supporting a demo: but this
    # really raises questions about how much debugging info to carry along in
    # order to support integrated visual display of information....
    def __call__(self, value):
        dc = dcheck('UtteranceTagger')
        low_factor, high_factor, min_range, start_window, start_count, stop_window, stop_count = self.data
        state, count, queue = self.state        
        # update the range trackers
        range_min, range_max = tracker_range = self.tracker(value)
        dc and dc('state', state, ' count', count, ' queue', len(queue), ' value', value, ' range', range_min, range_max)
        if state is BAC:
            # in background state, so we're looking for enough speechy stuff;
            # we require a range so as not to trigger in noise
            range_max = max(range_max, range_min + min_range)
            high_t = range_min + (range_max - range_min) * high_factor
            if value > high_t:
                queue.append((1, tracker_range))
                count += 1
                dc and dc('new_count', count)
            else:
                queue.append((0, tracker_range))
            if len(queue) < start_window:
                self.state = state, count, queue
                return ()
            assert len(queue) == start_window
            if count >= start_count:
                ret = tuple((True, outrange) for inc, outrange in queue)
                assert len(ret) == start_window
                state = UTT
                dc and dc('new_state', state)
                count = 0
                queue.clear()
                self.state = state, count, queue
                return ret
            tag = False
        else:
            assert state is UTT
            # in utterance state, so we're looking for enough backgroundy stuff;
            # we don't limit the range here, thus we don't get trapped when the
            # background level undergoes a sudden and persistent rise in level
            low_t = range_min + (range_max - range_min) * low_factor
            if value <= low_t:
                queue.append((1, tracker_range))
                count += 1
                dc and dc('new_count', count)
            else:
                queue.append((0, tracker_range))
            if len(queue) < stop_window:
                self.state = state, count, queue
                return ()
            assert len(queue) == stop_window
            if count >= stop_count:
                ret = tuple((True, outrange) for inc, outrange in queue)
                assert len(ret) == stop_window
                state = BAC
                dc and dc('new_state', state)
                count = 0
                queue.clear()
                self.state = state, count, queue
                return ret
            tag = True
        # queue is full, but no state change
        dc and dc('tag', tag)
        inc, outrange = queue.popleft()
        count -= inc
        assert 0 <= count <= len(queue)
        self.state = state, count, queue
        return ((tag, outrange),)

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
