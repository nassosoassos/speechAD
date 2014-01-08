###########################################################################
#
# File:         join.py (directory: ./py/onyx/dataflow)
# Date:         Thu 11 Dec 2008 16:42
# Author:       Ken Basye
# Description:  Join processing elements, that is, elements which can be sent data from multiple streams simultaneously.
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008, 2009 The Johns Hopkins University
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
    Join processing elements, that is, elements which can be sent data from
    multiple streams simultaneously.

    Join elements handle more than one incoming stream simultaneously.  That is,
    their process functions may be called by multiple upstream callers, possibly
    running in different threads.  Several types are available, including the
    SerializingJoin, which just sends whatever comes in to the sendee (but which
    handles the locking necessary to be called from multiple threads), the
    RoundRobinJoin, which enforces a round-robin protocol on callers of process
    functions, and two flavors of slot-filling join, which produce sorted tuples
    of processed events.
"""

from __future__ import division
from threading import Lock
from time import sleep
from functools import partial
from itertools import izip, count
from collections import deque
from onyx.util.streamprocess import ProcessorBase
from onyx.dataflow.source import IteratorSource

class SerializingJoin(ProcessorBase):
    """
    A very simple join processor which will take things sent into it by multiple
    callers of the process function and send them along to the sendee.

    Suitable for use when more than one thread might each be calling the process
    function, and all you need is to get the events into a single stream.  Note
    that this element does not use its own thread, so each event pushed through
    will be using the thread which called process() to send the data on.
    Neither is there any buffering done here, so threads calling process() will
    block until any currently on-going process()/send() calls complete.

    >>> result = list()
    >>> join0 = SerializingJoin(result.append)
    >>> for i in xrange(10):
    ...    join0.process(i)
    >>> result
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> del result[:]

    >>> range_size = 50
    >>> small_range = xrange(100, 100+range_size)
    >>> large_range = xrange(1000, 1000+range_size)
    >>> small = set(small_range)
    >>> large = set(large_range)
    >>> correct = small | large
    >>> is0 = IteratorSource(small_range, sendee=join0.process)
    >>> is1 = IteratorSource(large_range, sendee=join0.process)

    Use real-time simulation to get nearly deterministic interleaving of data

    >>> real_time_base = 1 / 32
    >>> is0.simulate_realtime(real_time_base)
    >>> is1.simulate_realtime(real_time_base)
    >>> def go():
    ...   # use a function to avoid per-statement threading delays from doctest
    ...   is0.start()
    ...   is1.start()
    ...   # 1.125 times the time necessary to get everything
    ...   sleep(1.125 * range_size * real_time_base)
    >>> go()
    >>> len(result)
    100
    >>> set(result) == correct
    True

    >>> small_in_first_half = set(result[:len(result)//2]) & small
    >>> len(small_in_first_half) == range_size // 2
    True
    """

    def __init__(self, sendee=None, sending=True):
        super(SerializingJoin, self).__init__(sendee, sending=sending)
        self._lock = Lock()

    def process(self, item):
        self._lock.acquire()
        self.send(item)
        self._lock.release()

        

class RoundRobinJoin(ProcessorBase):
    """
    A join processor that enforces a round-robin protocol on callers of its
    process functions.

    This element sends all events processed to the sendee, taking one event at a
    time from each prearranged incoming stream.  Note that this element doesn't
    have the usual process() function.  Instead, clients should call
    get_process_function() which will dynamically allocate a process function to
    be called in the usual way.  Each client who calls get_process_function() is
    implicitly given a position in the round.  When all positions have been
    allocated, call the start() function, which will allow processing to begin.
    The first allocated function goes first, and so on.  Calls to the allocated
    process functions will block until it's that caller's turn, at which point
    their event will be sent to the sendee.  Note that this element does not use
    its own thread, so each event pushed through will be using the thread which
    called the allocated process function.

    >>> result = list()
    >>> rrj0 = RoundRobinJoin(result.append)

    Here's a test with a single thread.  This means we'd better make the calls
    to the process functions in the right order or we'll hang our process!  This
    way of setting things up is not suggested for real use; you probably want to
    use a real source of some kind to feed this join.

    >>> proc0 = rrj0.create_process_function()
    >>> proc1 = rrj0.create_process_function()
    >>> proc2 = rrj0.create_process_function()
    >>> rrj0.num_registered
    3

    >>> rrj0.start()
    >>> for i in xrange(5):
    ...     proc0(i)
    ...     proc1(i+10)
    ...     proc2(i+100)
    >>> result
    [0, 10, 100, 1, 11, 101, 2, 12, 102, 3, 13, 103, 4, 14, 104]
    >>> del result[:]

    Here's a multithreaded test.  Note that we're reusing the processor
    constructed above, and that it's currently proc0's turn.  These
    IteratorSources each use their own thread and will be calling their sendees
    asynchronously.  The RoundRobinJoin enforces turn-taking between the
    threads.
    
    >>> is0 = IteratorSource(xrange(0,5), sendee=proc0)
    >>> is1 = IteratorSource(xrange(10,15), sendee=proc1)
    >>> is2 = IteratorSource(xrange(100,105), sendee=proc2)

    We start by turning on is1, but it's not his turn, so there's no output
    
    >>> is1.start()
    >>> sleep(1/16)
    >>> result
    []

    Now turn on is0.  It's his turn, so he gets one, then is1 gets one turn,
    then things lock up again since is2 still isn't on.
    
    >>> is0.start()
    >>> sleep(1/16)
    >>> result
    [0, 10]

    Finally turn is2 on, whereon things run to completion.

    >>> is2.start()
    >>> sleep(1/16)
    >>> result
    [0, 10, 100, 1, 11, 101, 2, 12, 102, 3, 13, 103, 4, 14, 104]
    
    """

    def __init__(self, sendee=None, sending=True):
        super(RoundRobinJoin, self).__init__(sendee, sending=sending)
        self._process_locks = dict()
        self._num_registered = 0
        self._position_queue = deque()
        self._processing = False

    @property
    def num_registered(self):
        return self._num_registered
    
    # For now, work is divided into two non-overlapping phases, setup and
    # processing.  During setup we hand out processing functions and positions,
    # but do no processing.  Similarly, during processing we don't give out any
    # new positions.  In the future, it might be useful to be able to add new
    # positions during processing, and to give up positions.
    def create_process_function(self):
        new_position = self._num_registered
        if self._processing:
            raise ValueError("Can't add any more callers once start() has been called")
        ret = partial(self._process, position=new_position)
        self._process_locks[new_position] = Lock()
        self._process_locks[new_position].acquire()
        self._position_queue.append(new_position)
        self._num_registered += 1
        return ret
    
    # FUTURE: make it possible to give up your position
    def release_process_function(self, fn):
        pass


    def start(self):
        if self._processing:
            raise ValueError("start() has already been called")
        elif self._num_registered == 0:
            raise ValueError("no positions allocated yet")
            
        self._processing = True
        self._process_locks[self._position_queue[0]].release()

    def _process(self, item, position):
        if not self._processing:
            raise ValueError("start() has not been called yet")
        self._process_locks[position].acquire()  # this may block

        # If we get here, it must be our turn
        assert position == self._position_queue[0]
        
        self.send(item)
        # Rotate our position to the back of the queue
        self._position_queue.rotate(-1)

        # Release the lock on the next position in the queue
        next_position = self._position_queue[0]
        self._process_locks[next_position].release()


class BlockingSlotFillingJoin(ProcessorBase):
    """
    A join processor that fills the slots of a resulting event with single
    events from callers of its process functions.

    This element sends bundles of events processed to the sendee, taking one
    event from each prearranged incoming stream to make a complete bundle.  Note
    that this element doesn't have the usual process() function.  Instead,
    clients should call create_process_function() which will dynamically
    allocate a process function to be called in the usual way.  Each call to
    create_process_function() is passed an immutable tag, which must be
    different from all previous tags.  When all process functions have been
    allocated, call the start() function, which will allow processing to begin.
    Calls to the allocated process functions will be used to fill in a tuple of
    events whose order is that same as the sorted order of the tags.  Calls made
    for tags which are already filled will block until a complete tuple is
    emitted and an empty slot is again available.  Note that this element does
    not use its own thread, so each event pushed through will be using the
    thread which called the allocated process function which completed the
    tuple.

    >>> result = list()
    >>> sfj0 = BlockingSlotFillingJoin(result.append)

    >>> proc0 = sfj0.create_process_function(2)
    >>> proc1 = sfj0.create_process_function(1)
    >>> proc2 = sfj0.create_process_function(3)
    >>> sfj0.sorted_tag_set
    [1, 2, 3]

    >>> sfj0.start()

    get_process_function returns the process function associated with a
    particular tag, or None if no function has that tag.
    
    >>> proc1 == sfj0.get_process_function(1)
    True
    >>> sfj0.get_process_function('a') is None
    True

    Here's a test with a single thread.  This means we must make the calls to
    the process functions in complete groups since if one slot is filled twice
    before the others are filled once, our process will hang!  This way of
    setting things up is not suggested for real use; you probably want to use a
    real source of some kind to feed this join.

    >>> for i in xrange(5):
    ...     proc0(i)
    ...     proc1(i+10)
    ...     proc2(i+100)
    >>> result
    [(10, 0, 100), (11, 1, 101), (12, 2, 102), (13, 3, 103), (14, 4, 104)]

    >>> del result[:]

    Here's a multithreaded test.  Note that we're reusing the processor
    constructed above, and that there are no slots currently filled.  These
    IteratorSources each use their own thread and will be calling their sendees
    asynchronously.
    
    >>> is0 = IteratorSource(xrange(0,5), sendee=proc0)
    >>> is1 = IteratorSource(xrange(10,15), sendee=proc1)
    >>> is2 = IteratorSource(xrange(100,110), sendee=proc2)

    We start by turning on is1, but this only fills one slot, so there's no
    output
    
    >>> is1.start()
    >>> sleep(1/16)
    >>> result
    []

    Now turn on is0.  There's still no output, since we can't fill
    an entire tuple.
    
    >>> is0.start()
    >>> sleep(1/16)
    >>> result
    []

    Finally turn is2 on, whereon things run to completion.

    >>> is2.start()
    >>> sleep(1/16)

    >>> result
    [(10, 0, 100), (11, 1, 101), (12, 2, 102), (13, 3, 103), (14, 4, 104)]

    Note that even though is2 has another 5 events, there are no corresponding is0 and is1
    events, so no tuples can be made.  XXX So we might want some sort of flush operation.
    """

    def __init__(self, sendee=None, sending=True):
        super(BlockingSlotFillingJoin, self).__init__(sendee, sending=sending)
        self._function_lock_pairs = dict()
        self._tag_set = set()
        self._processing = False
        self._tag_to_index_map = None

    @property
    def sorted_tag_set(self):
        return sorted(self._tag_set)
    
    # Work is divided into two non-overlapping phases, setup and processing.
    # During setup we hand out processing functions, but do no processing.
    # Similarly, during processing we don't give out any new functions.
    def create_process_function(self, new_tag):
        if self._processing:
            raise ValueError("can't add any more inputs once start() has been called")
        elif new_tag in self._tag_set:
            raise ValueError("can't add input with tag %s; it's already been used" % (new_tag,))
            
        ret = partial(self._process, tag=new_tag)
        self._function_lock_pairs[new_tag] = (ret, Lock())
        self._function_lock_pairs[new_tag][1].acquire()
        self._tag_set.add(new_tag)
        return ret
    

    def get_process_function(self, tag):
        return self._function_lock_pairs[tag][0] if self._function_lock_pairs.has_key(tag) else None


    def start(self):
        if self._processing:
            raise ValueError("start() has already been called")
        elif len(self._tag_set) == 0:
            raise ValueError("no slots allocated yet")

        self._tag_set = frozenset(self._tag_set)
        self._tag_to_index_map = dict(izip(sorted(self._tag_set), count()))
        self._processing = True
        self._prepare_for_new_bundle()

    def _prepare_for_new_bundle(self):
        self._current_bundle = [None for i in xrange(len(self._tag_set))]
        self._current_open_slots = set(self._tag_set)
        for tag in self._tag_set:
            assert self._function_lock_pairs[tag][1].locked()
            self._function_lock_pairs[tag][1].release()


    def _process(self, item, tag):
        if not self._processing:
            raise ValueError("start() has not been called yet")
        self._function_lock_pairs[tag][1].acquire()  # this may block

        # If we get here, our slot must be open
        index = self._tag_to_index_map[tag]
        assert self._current_bundle[index] is None

        # Fill our slot
        self._current_bundle[index] = item
        self._current_open_slots.remove(tag)

        # Maybe send entire bundle
        if len(self._current_open_slots) == 0:
            self.send(tuple(self._current_bundle))
            self._prepare_for_new_bundle()




class AccumulatingSlotFillingJoin(ProcessorBase):
    """
    A join processor that fills the slots of a resulting event with accumulated
    events from callers of its process functions.

    This element sends bundles of events processed to the sendee, taking
    multiple events from each prearranged incoming stream to make a complete
    bundle.  Note that this element doesn't have the usual process() function.
    Instead, clients should call create_process_function() which will
    dynamically allocate a process function to be called in the usual way.  Each
    call to create_process_function() is passed an immutable tag, which must be
    different from all previous tags.  When all process functions have been
    allocated, call the start() function, which will allow processing to begin.
    Calls to the allocated process functions will be used to fill in an (outer)
    tuple of event (inner) tuples.  The outer tuple's order is that of the
    sorted order of the tags.  Calls made for tags which are already filled will
    accumulate in an inner tuple until a complete outer tuple is emitted.  This
    means that there will always be one innter tuple of length 1.  Note that
    this element does not use its own thread, so each event pushed through will
    be using the thread which called the allocated process function which
    completed the tuple.

    >>> result = list()
    >>> sfj0 = AccumulatingSlotFillingJoin(result.append)

    >>> proc0 = sfj0.create_process_function(2)
    >>> proc1 = sfj0.create_process_function(1)
    >>> proc2 = sfj0.create_process_function(3)
    >>> sfj0.sorted_tag_set
    [1, 2, 3]

    >>> sfj0.start()

    get_process_function returns the process function associated with a
    particular tag, or None if no function has that tag.
    
    >>> proc1 == sfj0.get_process_function(1)
    True
    >>> sfj0.get_process_function('a') is None
    True

    >>> for i in xrange(5):
    ...     proc0(i)
    ...     proc1(i+10)
    ...     proc1(i+20)
    ...     proc2(i+100)
    >>> result
    [((10, 20), (0,), (100,)), ((11, 21), (1,), (101,)), ((12, 22), (2,), (102,)), ((13, 23), (3,), (103,)), ((14, 24), (4,), (104,))]

    >>> del result[:]

    Here's a multithreaded test.  Note that we're reusing the processor
    constructed above, and that all slots are currently empty.  These
    IteratorSources each use their own thread and will be calling their sendees
    asynchronously.
    
    >>> is0 = IteratorSource(xrange(0,5), sendee=proc0)
    >>> is1 = IteratorSource(xrange(10,15), sendee=proc1)
    >>> is2 = IteratorSource(xrange(100,105), sendee=proc2)

    We start by turning on is1, but this only fills one slot, so there's no
    output
    
    >>> is1.start()
    >>> sleep(1/16)
    >>> result
    []

    Now turn on is0.  There's still no output, since we can't fill
    an entire tuple.
    
    >>> is0.start()
    >>> sleep(1/16)
    >>> result
    []

    Finally turn is2 on, whereon we get our first output

    >>> is2.start()
    >>> sleep(1/16)
    >>> result
    [((10, 11, 12, 13, 14), (0, 1, 2, 3, 4), (100,))]

    Note that this is all we're going to get, since is0 and is1 are now empty
    and no triple can be filled.  XXX So we might want some kind of 'flush'
    operation.
    """

    def __init__(self, sendee=None, sending=True):
        super(AccumulatingSlotFillingJoin, self).__init__(sendee, sending=sending)
        self._functions = dict()
        self._tag_set = set()
        self._processing = False
        self._tag_to_index_map = None
        self._process_lock = Lock()
        self._process_lock.acquire()

    @property
    def sorted_tag_set(self):
        return sorted(self._tag_set)
    
    # Work is divided into two non-overlapping phases, setup and processing.
    # During setup we hand out processing functions, but do no processing.
    # Similarly, during processing we don't give out any new functions.
    def create_process_function(self, new_tag):
        if self._processing:
            raise ValueError("can't add any more inputs once start() has been called")
        elif new_tag in self._tag_set:
            raise ValueError("can't add input with tag %s; it's already been used" % (new_tag,))
            
        ret = partial(self._process, tag=new_tag)
        self._functions[new_tag] = ret
        self._tag_set.add(new_tag)
        return ret
    

    def get_process_function(self, tag):
        return self._functions[tag] if self._functions.has_key(tag) else None


    def start(self):
        if self._processing:
            raise ValueError("start() has already been called")
        elif len(self._tag_set) == 0:
            raise ValueError("no slots allocated yet")

        self._tag_set = frozenset(self._tag_set)
        self._tag_to_index_map = dict(izip(sorted(self._tag_set), count()))
        self._processing = True
        self._prepare_for_new_bundle()
        self._process_lock.release()

    def _prepare_for_new_bundle(self):
        self._current_bundle = [None for i in xrange(len(self._tag_set))]
        self._current_open_slots = set(self._tag_set)


    def _process(self, item, tag):
        if not self._processing:
            raise ValueError("start() has not been called yet")
        self._process_lock.acquire()  # this may block

        # Accumulate in our slot
        index = self._tag_to_index_map[tag]
        if self._current_bundle[index] is None:
            self._current_bundle[index] = list((item,))
            self._current_open_slots.remove(tag)
        else:
            self._current_bundle[index].append(item)
            
        # Maybe send entire bundle
        if len(self._current_open_slots) == 0:
            self.send(tuple((tuple(_list) for _list in self._current_bundle)))
            self._prepare_for_new_bundle()
        self._process_lock.release()

class SynchronizingSequenceJoin(ProcessorBase):
    """
    A synchronizing join processor.  Process functions are obtained by calls to
    get_process_function().  The order of calls to get_process_function
    establishes the order of the slots in the event sequences that get sent on.
    Events arriving at each slot are queued up.  An event is built and sent on
    when every slot has at least one item in its queue (and the last slot that
    got an item has only that single item in its queue).  The sent event is a
    tuple containing one item popped off each queue.  Thus, events are
    synchronized and are emitted at the rate (and latency) of the slowest
    source.  Both get_process_function() and the returned functions are
    thread-safe and single-thread safe.

    >>> ssj = SynchronizingSequenceJoin()
    >>> funcs = list(ssj.get_process_function() for i in xrange(5))

    Don't need no sendee yet since we don't fill the first slot

    >>> for i, f in enumerate(funcs[1:]): f(i)
    >>> for i, f in enumerate(funcs[1:]): f(10+i)

    Now create a sendee, and it gets the synchronized result

    >>> result = list()
    >>> ssj.set_sendee(result.append)
    >>> funcs[0](-1)
    >>> result
    [(-1, 0, 1, 2, 3)]

    No new result

    >>> for i, f in enumerate(funcs[1:]): f(20+i)
    >>> result
    [(-1, 0, 1, 2, 3)]

    Now get out the next two synchronized results

    >>> funcs[0](-10)
    >>> result
    [(-1, 0, 1, 2, 3), (-10, 10, 11, 12, 13)]
    >>> funcs[0](-20)
    >>> result
    [(-1, 0, 1, 2, 3), (-10, 10, 11, 12, 13), (-20, 20, 21, 22, 23)]

    Now the other slots are empty, so no new result
    >>> funcs[0](-30)
    >>> result
    [(-1, 0, 1, 2, 3), (-10, 10, 11, 12, 13), (-20, 20, 21, 22, 23)]

    Add a new inputter (this is thread safe too)

    >>> funcs.append(ssj.get_process_function())
    >>> for f in funcs: f(None)
    >>> result
    [(-1, 0, 1, 2, 3), (-10, 10, 11, 12, 13), (-20, 20, 21, 22, 23), (-30, None, None, None, None, None)]
    """
    def __init__(self, sendee=None, sending=True):
        super(SynchronizingSequenceJoin, self).__init__(sendee=sendee, sending=sending)
        self.lock = Lock()
        self.queues = list()
    def get_process_function(self):
        lock = self.lock
        queues = self.queues
        queue = deque()
        lock.acquire()
        queues.append(queue)
        lock.release()
        def process(value):
            lock.acquire()
            assert not all(queues)
            queue.append(value)
            if all(queues):
                self.send(tuple(q.popleft() for q in queues))
            assert not all(queues)
            lock.release()
        return process

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



