###########################################################################
#
# File:         pointprocess.py (directory: ./py/onyx/util)
# Date:         17-Jun-2009
# Author:       Hugh Secker-Walker
# Description:  Utilities for working with point process data
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
    >>> True
    True
"""
from __future__ import with_statement
from __future__ import division
import os
import collections
import onyx.builtin
from onyx.graph import dynamicgraph
from onyx.util import iterutils

class PointProcessSamplingWindow(object):
    """
    Sampling of user-maintained statistics using a sliding window of point
    process events.

    This object handles the timestamp bookkeeping and making of callbacks to
    support getting regular samples of statistics from point-process events
    using a sliding window.

    At construction the client specifies the numerical sampling interval and
    sliding window length, and provides three callbacks: add, remove, sample.
    The sampling interval specifies the period between callbacks to the provided
    sample() callable.

    The client then calls the event() method each time it has a point process
    event.  The PointProcessSamplingWindow object maintains recent events in the
    client by making callbacks to add() and to remove() in the context of an
    event() call.  Also, as appropriate, in the context of an event() call, the
    sample() callback will be made to tell the client to sample whatever
    statistics it is maintaining for the events in the window.

    This contrived example just demonstrates the mechanics of what happens.  It
    shows that sample() is called prior to the first call to add().  It also
    shows what happens when the timestamps are far enough apart that sample()
    will be called while there are no events in the window.

    >>> def add(timestamp, obj): print 'add', timestamp, obj
    >>> def remove(timestamp, obj): print 'remove', timestamp, obj
    >>> def sample(timestamp): print 'sample', timestamp
    >>> ppw = PointProcessSamplingWindow(10, 20, add, remove, sample)
    >>> for i in xrange(0, 10, 2): ppw.event(i*i, (i, 'a')); ppw.event(i*i, (i, 'b')); ppw.event(i*i+1, (i, 'c')); 
    sample 0
    add 0 (0, 'a')
    add 0 (0, 'b')
    add 1 (0, 'c')
    add 4 (2, 'a')
    add 4 (2, 'b')
    add 5 (2, 'c')
    sample 10
    add 16 (4, 'a')
    add 16 (4, 'b')
    add 17 (4, 'c')
    sample 20
    remove 0 (0, 'a')
    remove 0 (0, 'b')
    remove 1 (0, 'c')
    remove 4 (2, 'a')
    remove 4 (2, 'b')
    remove 5 (2, 'c')
    sample 30
    add 36 (6, 'a')
    add 36 (6, 'b')
    add 37 (6, 'c')
    remove 16 (4, 'a')
    remove 16 (4, 'b')
    remove 17 (4, 'c')
    sample 40
    sample 50
    remove 36 (6, 'a')
    remove 36 (6, 'b')
    remove 37 (6, 'c')
    sample 60
    add 64 (8, 'a')
    add 64 (8, 'b')
    add 65 (8, 'c')


    This real example uses an online graph to maintain statistics about social network behavior.

    Constants
    
    >>> secs_per_minute = 60
    >>> secs_per_hour = secs_per_minute * 60
    >>> secs_per_day = secs_per_hour * 24
    >>> secs_per_week = secs_per_day * 7
    >>> secs_per_year = secs_per_day * 365.2425

    An object that maintains statistics 

    >>> g = dynamicgraph.UndirectedOnlineInvariantsGraph()

    Set up callbacks to gather stats about the maximum of each feature

    >>> maxen = collections.defaultdict(lambda:collections.defaultdict(int))
    >>> def add(timestamp, event):
    ...   g.add_edge(event)
    >>> def remove(timestamp, event):
    ...   g.remove_edge(event)
    >>> import numpy
    >>> def sample(timestamp):
    ...     features = g.invariants
    ...     for key, value in features.iteritems():
    ...       if value > maxen[key][key]:
    ...          assert features['scan1'] == g._scan1_brute
    ...          features2 = dict(features)
    ...          features2['timestamp'] = timestamp
    ...          maxen[key] = features2
    ...     incidence, nodes = g.incidence_matrix
    ...     if False and incidence.any():
    ...       mad_upper, mad_lower = int(numpy.linalg.eigvalsh(incidence).max()*1000), features['mad_lower_k']
    ...       print timestamp, mad_upper, mad_lower, mad_upper / mad_lower

    >>> w = PointProcessSamplingWindow(secs_per_day, secs_per_week, add, remove, sample)

    Get the invariants at which each of the features is first maximal

    >>> filename = os.path.join(_module_dir, '..', 'graph', 'enron_subset.csv')
    >>> fields = 'employeeIndexDict[From]', 'employeeIndexDict[To]', 'Epoch', 'Topic'
    >>> with open(filename, 'rb') as infile:
    ...   for msg_index, (msg_from, msg_to, msg_epoch, msg_topic) in enumerate(iterutils.csv_itemgetter(infile, fields)):
    ...      w.event(int(msg_epoch), (msg_from, msg_to))

    Look at the maximal features
    
    >>> for key in sorted(maxen): print key, maxen[key][key], ' ', maxen[key]
    mad_lower_k 6690   {'num_triangles': 177, 'timestamp': 1002909837, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'max_degree': 52}
    max_degree 52   {'num_triangles': 118, 'timestamp': 1002737037, 'scan1': 81, 'mad_lower_k': 5571, 'size': 224, 'order': 116, 'max_degree': 52}
    num_triangles 177   {'num_triangles': 177, 'timestamp': 1002909837, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'max_degree': 52}
    order 123   {'num_triangles': 177, 'timestamp': 1002909837, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'max_degree': 52}
    scan1 91   {'num_triangles': 177, 'timestamp': 1002909837, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'max_degree': 52}
    size 283   {'num_triangles': 177, 'timestamp': 1002909837, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'max_degree': 52}
    """
    def __init__(self, sample_interval, window_interval, add, remove, sample):
        """
        The sample_interval and window_interval arguments are numerical
        specifiers for the sampling period and the length of the sliding
        event-window.  Each of the add, remove, and sample callbacks must be
        callable.  Both add() and remove() take two arguments, the timestamp for
        the event and the object that constitutes the event.  The sample()
        callback takes a single timestamp argument, the time at which the sample
        is being taken.

        It is an error if sample_interval is greater than window_interval.  It
        is an error if any of add, remove, or sample is not callable.
        """
        if sample_interval > window_interval:
            raise ValueError("expected the sample_interval, %r, to be less than or equal to the window_interval, %r" % (sample_interval, window_interval))
        if set(callable(x) for x in (add, remove, sample)) != set((True,)):
            raise TypeError("expected all of add, remove, and sample to be callable")
        self.data = sample_interval, window_interval, collections.deque(), add, remove, sample
        prev_time, next_sample_time = self.times = None, None

    def event(self, timestamp, obj):
        """
        Add a new point-process event to the system.  The timestamp argument is
        the time at which the event occured, and the obj argument is the object
        representing the event.

        At least one callback to add() will occur in the context of the event()
        call.  One or more callbacks to remove() and sample() may occur in the
        context of the event() call.  The particular order of any of the
        callbacks that occur is not specified.

        The add() callback will be called exactly once, with the timestamp and
        obj arguments, in the context of the event() call.

        The remove() and sample() callbacks may be called one or more times in
        the context of the event() call in order to maintain the client's view
        of the set of events in the window.  It is guaranteed that the
        timestamps in the successive calls to remove() and sample() will not
        decrease.

        It is an error if the timestamp argument is smaller than the timestamp
        provided to the previous call to event().
        """
        # absorb a new event, maintaining the fifo of objects in the window and making the callbacks
        sample_interval, window_interval, fifo, add, remove, sample = self.data
        prev_time, next_sample_time = self.times
        if next_sample_time is None:
            # bootstrap
            assert prev_time is None
            next_sample_time = timestamp
            sample(next_sample_time)
            next_sample_time += sample_interval
        else:
            if timestamp < prev_time:
                raise ValueError("expected non-decreasing timestamps, but previous is %r and got %r" % (prev_time, timestamp))
            while timestamp > next_sample_time:
                # remove old objects and/or emit samples
                cutoff = next_sample_time - window_interval, None
                # window support is [cutoff, next_sample_time)
                while fifo and fifo[0] < cutoff:
                    ts, ob = fifo.popleft()
                    remove(ts, ob)
                sample(next_sample_time)
                next_sample_time += sample_interval
        # put the new object into the window
        fifo.append((timestamp, obj))
        add(timestamp, obj)
        self.times = timestamp, next_sample_time

if __name__ == '__main__':
    import os
    import onyx
    _module_dir, _module_name = os.path.split(__file__)
    onyx.onyx_mainstartup()
