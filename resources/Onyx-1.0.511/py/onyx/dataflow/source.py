###########################################################################
#
# File:         source.py (directory: ./py/onyx/dataflow)
# Date:         11-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Support for sources of data
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
    Source elements that push data into a dataflow network

    The classes here support the common needs of clients for getting data into
    dataflow graphs.  The key feature of each such source is that it internally
    manages a thread that pushes data through the graph.  These sources differ in
    terms of how they handle obtaining the data that gets pushed into the graph.
    These classes provide non-blocking supervisor functions, e.g. Start(),
    Stop(), Pause(), which provide the basic dataflow controls associated with
    streaming data.

    Two types of sources are available:

    IteratorSource
      for cases where the client has a pull model for generating data that will
      go into the network, e.g. obtaining data by iterating from a file.

    NonBlockingSource
      for cases where the client is already configured with a push model for
      generating data that will go into the network, e.g. from a real-time
      streaming source.

    The IteratorSource is an "impedance-matching" processor that is inserted
    between a pull-based source of data and the push-based dataflow graph.  It
    has a thread that pulls on a client-supplied iterator that generates data,
    e.g. a file object.  It has three modes of operation selected via its
    throttle interface.  In the as-needed mode, it pulls on the iterator on an
    as-needed basis whenever the queue of unsent data is below some low-water
    mark.  In the simulate-real-time mode it pulls on the iterator at a
    client-supplied interval of wall-clock time.  The third mode is a limit of
    the simulate-real-time mode when the interval goes to zero and the source
    pulls in the data as quickly as it can.

    The NonBlockingSource is simply a buffering processor for the push model of
    dataflow.  It has a sendee, and it has a process() function.  However,
    unlike simple processors, its process() function is non-blocking; this
    function simply queues up the data and returns quickly.  The
    NonBlockingSource handles the buffering and it has a separate thread so as
    to call the sendee asynchronously whenever there is data available that has
    been put in the queue via the process() interface.  The idea here is that
    the client is handling some push-based data source, e.g. a live input, that
    must be serviced rapidly, so the client cannot wait for process() to do a
    lot of work.  The NonBlockingSource provides the buffering that simplifies
    the interface between such a time-sensitive source and the dataflow graph.

    The IteratorSource and NonBlockingSource classes are subclasses of
    DataflowSourceBase.  This base class handles most of the semantics needed
    for the asynchronous sending of data into the network.  It can be subclassed
    by clients with very specialized needs that are not met by either
    NonBlockingSource or IteratorSource.

    >>> len((DataflowSourceBase(), NonBlockingSource(), IteratorSource(())))
    3
"""

from __future__ import with_statement
from __future__ import division
import time
from collections import deque
from itertools import count, repeat
from threading import Thread, Lock
from onyx.util.streamprocess import ProcessorBase

class DataflowSourceBase(ProcessorBase):
    """
    Baseclass for objects which serve as sources of dataflow packets.

    This class has a thread that pushes data into the sendee when there's data
    in the queue.  It has supervisor functions, e.g. start(), stop(), pause(),
    for managing the collecting of data and the emptying of the queue.

    This baseclass object takes care of emptying the queue asynchronously via
    calls to send().

    Subclasses need to implement a way to append items to the queue (eventually
    calling _enqueue() while the critical section is locked) when in the Running
    or Paused states.
    """

    # states
    Stopped = 'stopped'
    Running = 'running'
    Paused = 'paused'
    Done = 'done'

    # semantic sets: we're in none of these if we're Done; we're in exactly two
    # of these otherwise
    #
    # we can receive if we're Running or Paused, but not if we're Stopped
    Receiving = Running, Paused
    # we will send if we're Running or Stopped, but not if we're Paused
    Sending = Running, Stopped
    # we are waiting if we're Paused or Stopped, but not if we're Running
    Waiting = Paused, Stopped

    def __init__(self, sendee=None, sending=True):
        super(DataflowSourceBase, self).__init__(sendee, sending=sending)

        # lock protecting our own changes of state; it must not be held for
        # long, specifically, not while doing any real work; exposed functions
        # use _enter() and _exit() to use this lock and to have the semantics of
        # the other named locks implemented.
        self.state = self.Stopped
        self.state_lock = Lock()
        locked = self._state_lock_acquire()
        assert locked

        self.index = 0
        self.deque = deque()

        self.thread_locks = list()

        # lock to prevent spinning in _worker function; only released when
        # there's (a change of state that may mean there's) more work to be done
        self.work_lock = Lock()
        self.work_lock.acquire()
        self.thread_locks.append(self.work_lock)

        # thread that handles the asynchronous emptying of the queue, making
        # calls to send() which can do (potentially blocking) work in the
        # dataflow network
        self.work_thread = Thread(target=self._worker)
        self.work_thread.setDaemon(True)
        self.work_thread.start()

        # lock for subclasses that have a getter thread; _worker() releases this
        # whenever it pops an item off the queue to notify getter thread of
        # queue size reductions
        self.get_lock = Lock()
        self.get_lock.acquire()
        self.thread_locks.append(self.get_lock)

        #self.state_lock.release()
        self._exit()

    # critical section support
    def _state_lock_acquire(self):
        # acquire the state lock; return True if we're not Done; else, release
        # the lock and return False
        self.state_lock.acquire()
        if self.state is not self.Done:
            return True
        else:
            self.state_lock.release()
            return False

    def _enter(self, check_name=None):
        # hand in a check_name string if calling from a supervisory method and
        # you want an error throw on violation of the API semantics
        is_acquired = self._state_lock_acquire()
        if not is_acquired and check_name is not None:
            raise ValueError("can't call %s() after calling done()" % (check_name,))
        return is_acquired

    def _exit(self):
        assert self.state_lock.locked()
        # all acquired thread locks get released any time someone's been through
        # the critical section
        for lock in self.thread_locks:
            if lock.locked():
                lock.release()
        self.state_lock.release()

    def _work_to_do(self):
        assert self.state_lock.locked()
        #return self.state in self.Sending and len(self.deque) > 0
        return self.state in self.Sending and len(self.deque) > 0


    # for pushing onto the queue, used by subclasses
    def _enqueue(self, value):
        assert self.state_lock.locked()
        # really need two counters: number of enqueue requests, and the index of
        # the guys that are actually enqueued; for now we count requests as this
        # is needed for real-time work....
        self.index += 1
        if self.state in self.Receiving:
            self.deque.append(value)
        else:
            # XXX gap management
            pass


    # supervisor properties and methods
    @property
    def backlog(self):
        """
        This property is the number of unprocessed samples in the queue.
        """
        return len(self.deque)

    def start(self):
        """
        Start accepting samples into the queue and start sending queued samples
        to the sendee.
        """
        self._enter('start')
        self.state = self.Running
        self._exit()

    def stop(self, flush=False):
        """
        Stop accepting or sending samples.  If flush is True, the existing queue
        of samples will be sent to the sendee, otherwise the queue will be
        cleared (the default).
        """
        self._enter('stop')
        self.state = self.Stopped
        # flush=True means to leave the deque alone so as to let worker thread
        # process the existing queue of values; flush=False means to drop the
        # enqueued values (clear())
        if not flush:
            self.deque.clear()
        self._exit()

    def pause(self):
        """
        Stop sending samples.  Samples will still be accepted and queued up.
        """
        self._enter('pause')
        self.state = self.Paused
        self._exit()

    def wait(self):
        # XXX semantics aren't clear -- what's it mean to wait while you're paused
        """
        If stopped or paused, wait until any pending processing of queued
        samples is done or the state changes to running, then return.  This is a
        blocking method for the caller.  Don't call this unless you really need
        to wait for the queue to empty, e.g. for (more) reproducible testing of
        the simulated real-time processing.
        """
        # sleep, yuk, 1/64 of a second
        sleep_time = 1 / 64
        self._enter('wait')
        while self.state in self.Waiting and self._work_to_do():
            self._exit()
            time.sleep(sleep_time)
            self._enter('wait')
        self._exit()

    def done(self):
        """
        This call permanently shuts down the processor and finishes the worker
        threads.  It *must* be called in order for the processor and its thread
        resources to get freed up.  It can only successfully be called once.

        The queue of pending samples is cleared.  This function blocks until the
        worker thread has shutdown after it finishes processing (sending)
        whatever sample it was working on.

        If you need to process the existing queue before shutting down, call
        stop(True), then wait() (which blocks until the queue has been
        processed), and, then call done().
        """
        if self.state == self.Done:
            return
        self._enter('done')
        if self.state != self.Done:
            self.state = self.Done
            self.deque.clear()
            work_thread = self.work_thread
            # this is essential so that when the thread object calls join() it has
            # no references and so goes away and releases its reference to our
            # self._worker method, so we can go away....
            self.work_thread = None
        self._exit()
        # note: unlike most other supervisory methods, this method blocks until
        # the worker thread finishes
        work_thread.join()

    def __del__(self):
        self.done()

    # the function that the worker thread runs; asynchronously pops items off
    # the queue and send()s them
    def _worker(self):
        while True:
            # note: wait here; blocking call, outside the critical section; we
            # acquire the lock whenever anyone goes through _exit()
            self.work_lock.acquire()

            # begin critical section
            if not self._state_lock_acquire():
                return

            work_to_do = self._work_to_do()
            if work_to_do:
                # note: don't remove the value from the queue yet, so that
                # _work_to_do() remains True
                value = self.deque[0]                
                # this release allows the getter thread, if any, to run again
                # without waiting for some high-level state change and its call
                # to _exit()
                if self.get_lock.locked():
                    self.get_lock.release()

            # end critical section
            self.state_lock.release()

            if work_to_do:
                # do real work outside the critical section because the send()
                # call could take an indefinite amount of time doing work in the
                # dataflow graph
                self.send(value)

                if not self._state_lock_acquire():
                    return

                # queue could have been cleared by Stop/no-flush
                if len(self.deque) > 0:
                    # here's where we actually remove the guy
                    v = self.deque.popleft()
                    assert v is value
                if self._work_to_do() and self.work_lock.locked():
                    # this allows us (_worker) to loop again without waiting for
                    # some high-level state change and its call to _exit()
                    self.work_lock.release()
                self.state_lock.release()

class NonBlockingSource(DataflowSourceBase):
    """
    A stream processor object with a non-blocking process() function, and
    non-blocking control functions.  This implements a thread barrier.  The
    process() function will not block.  This means that calls made to send() by
    this processor are asynchronous.  It is intended that this processor be used
    at the periphery of a graph in order to buffer data from an asynchronous
    source that requires responsive handling of its calls to process(),
    e.g. some form of live input.  Also, except for wait() and done(), the
    supervisor methods do not block.

    Set up processor that dumps into a list.
    
    >>> result = list()
    >>> nbb = NonBlockingSource(sendee=result.append)

    Start the processor, fiddle its state and process stuff.  In a
    real-world application, the process calls would come from a
    different thread and we would not block that thread.

    >>> nbb.start()
    >>> for i in xrange(10): nbb.pause(); nbb.start(); nbb.process(i); nbb.start(); nbb.pause(); nbb.process(i); nbb.start(); nbb.pause(); nbb.start()
    >>> for i in xrange(10, 20): nbb.process(i); nbb.process(i)

    Stop it, letting the queue get processed.

    >>> nbb.stop(True)

    Polling wait for the queue to empty.

    >>> while nbb.backlog > 0: time.sleep(1/32)
    >>> size = len(result)
    >>> size == 40
    True

    Because the processor is stopped, these items do not go into the
    queue; they are dropped.

    >>> for i in xrange(20, 30): nbb.process(i)
    >>> assert nbb.backlog == 0
    >>> assert len(result) == 40

    When paused, things get into the queue, but aren't processed.

    >>> nbb.pause()
    >>> for i in xrange(30, 40): nbb.process(i); nbb.process(i)
    >>> assert nbb.backlog == 20

    Now, stop accepting new samples, but allow the existing queue to
    keep being processed.

    >>> nbb.stop(True)

    We're stopped, so these new guys get dropped.  But, we can't yet make claims
    about the state of the queue.

    >>> for i in xrange(50, 60): nbb.process(i); nbb.process(i)

    Now, wait() blocks until the queue is empty

    >>> nbb.wait()
    >>> nbb.backlog
    0
    >>> assert len(result) == 60

    Back in pause mode, dump more stuff in there

    >>> nbb.pause()
    >>> for i in xrange(70, 80): nbb.process(i); nbb.process(i)
    >>> assert nbb.backlog == 20

    Stop and allow the queue to empty; wait again for the queue (that got filled
    while we were paused) to get empty.

    >>> nbb.stop(True)
    >>> nbb.wait()
    >>> result
    [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79]

    When stop() is used, the queue is cleared without any pending items being processed.

    >>> nbb.start()
    >>> for i in xrange(90, 100): nbb.process(i); nbb.process(i)
    >>> nbb.stop()
    >>> assert nbb.backlog == 0

    Given the asynchronous calls to send, we can't make any strong
    assertions about how many of the (90, 100) guys got sent and
    put into result.

    >>> assert len(result) >= 60

    The blocking call to done() shuts down the processor and its
    associated sending thread.  After this call returns, the processor
    will never send an item and cannot be used.

    >>> nbb.done()

    However, a done processor silently bit-buckets items send to process().

    >>> for i in xrange(100, 110): nbb.process(i); nbb.process(i)

    Once done() is called, it's an error to try to use the processor.

    >>> nbb.start()
    Traceback (most recent call last):
      ...
    ValueError: can't call start() after calling done()

    >>> nbb.stop()
    Traceback (most recent call last):
      ...
    ValueError: can't call stop() after calling done()

    >>> nbb.pause()
    Traceback (most recent call last):
      ...
    ValueError: can't call pause() after calling done()

    >>> nbb.wait()
    Traceback (most recent call last):
      ...
    ValueError: can't call wait() after calling done()

    It's reasonable to allow done() to be called again, e.g. from __del__

    >>> nbb.done()

    And, as we said, a done processor silently bit-buckets items.

    >>> for i in xrange(110, 120): nbb.process(i); nbb.process(i)
    """

    # an external (client) thread should call this with values to be processed
    def process(self, value):
        # no error even if we're done, so no check_name argument to _enter()
        if self._enter():
            # note: _enqueue() handles dropping samples if we're stopped
            self._enqueue(value)
            self._exit()


class IteratorSource(DataflowSourceBase):
    """
    A source that pulls on a user-supplied iterable for its samples.  The
    default is to pull on the iterator as-needed based on the sendee's
    consumption of samples.  E.g. this can be used to pull on a file-based
    iterator.  Running in simulated real-time is also possible, in which case
    the iterator is pulled on at the simulated real-time interval, regardless of
    whether the sendee can keep up.

    Set up some stuff, including fractions of a second.

    >>> base_interval = 1 / 8
    >>> base_by_4_plus_epsilon = base_interval * 4 + base_interval / 4

    Set up processor that dumps into a list and start it.  Wait until the
    iterator stops.  It empties the xrange source iterator with on-demand
    pulling.

    >>> result = list()
    >>> nbb = IteratorSource(xrange(10), sendee=result.append)
    >>> nbb.is_iter_stopped
    False
    >>> nbb.start()
    >>> nbb.wait_iter_stopped()
    True
    >>> nbb.is_iter_stopped
    True
    >>> nbb.stop(True)
    >>> nbb.wait()
    >>> result
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> del result[:]

    Show simulated real-time with an interval of base_interval.

    >>> nbb.simulate_realtime(base_interval)
    >>> nbb.set_iter() and nbb.is_iter_stopped
    False
    >>> nbb.start()

    Sleep so that the empty iteration from the prior set_iter() call gets
    exhausted; then check that this is so.

    >>> time.sleep(base_interval)
    >>> nbb.is_iter_stopped
    True

    In the following we use a function to run the steps in the dance here in
    order to avoid test-disrupting delays that would occur if the doctest
    machinery had to process each line separately.

    Run with a non-empty iterator.  Wait a total of just over four of the
    base_interval intervals, giving five interval-bounding items.

    >>> def go():
    ...   results = list()
    ...   nbb.set_iter(count())
    ...   results.append(nbb.wait_iter_stopped(base_interval))
    ...   time.sleep(base_by_4_plus_epsilon - base_interval)
    ...   results.append(nbb.is_iter_stopped)
    ...   nbb.stop(True)
    ...   nbb.wait()
    ...   results.append(nbb.is_iter_stopped)
    ...   return tuple(results)
    >>> go()
    (False, False, False)
    >>> result
    [0, 1, 2, 3, 4]
    >>> del result[:]

    Show switching iterators during simulated real-time processing.  Each call
    to set_iter() resets the real-time clock.  Note that we capture and return
    the values of the previous count() and repeat() iterators.

    >>> def go2():
    ...   results = list()
    ...   results.append(nbb.set_iter())
    ...   nbb.start()
    ...   results.append(nbb.set_iter(repeat(1)))
    ...   time.sleep(base_by_4_plus_epsilon)
    ...   results.append(nbb.set_iter(repeat(2)))
    ...   time.sleep(base_by_4_plus_epsilon)
    ...   nbb.stop(True)
    ...   nbb.wait()
    ...   return tuple(results)
    >>> go2() #doctest: +ELLIPSIS
    (count(5), <tupleiterator object at 0x...>, repeat(1))

    Since set_iter resets the real-time clock, each iterator gives five
    interval-bounding items.

    >>> result
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    >>> del result[:]

    Show that pausing does not interfere with pulling on the iterator.  

    >>> def go3():
    ...   nbb.set_iter()
    ...   nbb.start()
    ...   nbb.set_iter(count())
    ...   time.sleep(base_by_4_plus_epsilon)
    ...   nbb.pause()
    ...   nbb.wait()
    ...   # result1 gets the samples bounding 4 intervals
    ...   result1 = list(result)
    ...   time.sleep(base_by_4_plus_epsilon)
    ...   # no change to result since we're wait()-ing (but, the internal queue has changed)
    ...   result2 = list(result)
    ...   # call to start() is not strictly necessary since stop(True)
    ...   # will pull everything even if we're paused when it's called
    ...   nbb.start()
    ...   nbb.stop(True)
    ...   nbb.wait()
    ...   # has all 8 sample-bounding intervals
    ...   result3 = list(result)
    ...   return result1, result2, result3
    >>> result1, result2, result3 = go3()
    >>> result1
    [0, 1, 2, 3, 4]
    >>> result2
    [0, 1, 2, 3, 4]
    >>> result3
    [0, 1, 2, 3, 4, 5, 6, 7, 8]

    >>> nbb.done()
    """

    # minimum number of items to (try to) keep in the queue in non-throttled
    # (consumption-based) pulling mode; could become a per-instance variable
    queue_low_water = 1

    def __init__(self, iterable=(), sendee=None, sending=True):
        super(IteratorSource, self).__init__(sendee, sending=sending)

        self.iter = None
        self.set_iter(iterable)
        self.simulate_realtime(None)

        # thread uses DataflowSourceBase.get_lock in _getter to
        # throttle the work; note that we do not keep a reference to
        # the thread because we don't call join and we need the thread
        # to go away when it exits (it's arguably a bug in Python
        # 2.5.2 that the Thread object holds onto its target function
        # even after that function returns....)
        get_thread = Thread(target=self._getter)
        get_thread.setDaemon(True)
        get_thread.start()

    def simulate_realtime(self, interval):
        """
        Set the non-negative simulated real-time interval, in seconds, for
        pulling samples from the iterator.  An interval of None will cause
        need-based pulling of items.

        An interval of zero can be specified in order to drain the iterator very
        responsively.  This is useful for pulling on an iterator that has its
        own rate limiting, e.g. a stream-iterator from a live audio source.
        However, using a zero or very small interval on an unbounded,
        non-rate-limited iterator can exhaust memory in our queue if the sendee
        cannot keep up with the data we push into it.

        Calling this function resets the real-time clock.
        """
        if interval is not None:
            if not isinstance(interval, (float, int)):
                raise ValueError("expected interval to be None or a non-negative number, got a %s" % (type(interval).__name__,))
            if interval < 0:
                raise ValueError("expected interval to be None or non-negative, got %s" % (interval,))
        self._enter('simulate_realtime')
        self.interval = interval
        self.index = 0
        self.start_time = time.time()
        self._exit()

    @property
    def is_iter_stopped(self):
        """
        This property is True if the current iterator from which this source is
        pulling has raised StopIteration, otherwise the property is False.  See
        also set_iter().
        """
        self._enter('is_iter_stopped')
        is_iter_stopped = self._is_iter_stopped
        self._exit()
        return is_iter_stopped
    
    def wait_iter_stopped(self, timeout=None):
        """
        If timeout is None (default), block unconditionally until the
        .is_iter_stopped property is True and return True.  Otherwise, timeout
        is a non-negative number and the call will block for at most timeout
        seconds, returning True if .is_iter_stopped becomes True or returning
        False otherwise.
        """
        if timeout is not None:
            assert timeout >= 0
            timeout += time.time()
        self._enter('wait_iter_stopped')
        is_iter_stopped = self._is_iter_stopped
        self._exit()
        wait = 1 / 64
        while is_iter_stopped == False and (timeout is None or time.time() < timeout):
            time.sleep(wait)
            self._enter('wait_iter_stopped')
            is_iter_stopped = self._is_iter_stopped
            self._exit()
        return is_iter_stopped
    
    def set_iter(self, iterable=tuple()):
        """
        Using iterable, set up the iterator from which this source will pull.
        Calling this function with no argument is equivalent to calling it with
        the empty tuple and will suspend the processing that fills the queue.
        Calling this function resets the real-time clock.

        Sets the .is_iter_stopped property to False.  When the iterator is
        exhausted and raises StopIteration, the .is_iter_stopped property will
        become True.

        Returns the iterator that was set up during the previous call to
        set_iter().
        """
        self._enter('set_iter')
        old_iter = self.iter
        self.iter = iter(iterable)
        self._is_iter_stopped = False
        self.index = 0
        self.start_time = time.time()
        self._exit()
        return old_iter

    def _throttle(self):
        # support for both consumption-based pulling and simulated real-time
        # pulling on iterator: return None if there's no getter work to be done;
        # return a positive number if the getter should sleep; return a negative
        # number if the getter should pull for a sample
        assert self.state_lock.locked()
        assert self.state is not self.Done
        # XXX semantics of stopping and waiting aren't clear!
        if self.state is self.Stopped:
            return None
        assert self.state in self.Receiving
        if self.interval is None:
            # consumption-based throttling: return non-positive to pull or None to wait for state change
            return -1 if len(self.deque) <= self.queue_low_water else None
            #return -1 if self.state in self.Receiving and len(self.deque) <= self.queue_low_water else None
        else:
            # simulated real-time throttling: return non-positive for pulling (catch up) or positive amount to wait
            return -1 if self.interval == 0 else (self.start_time + self.index * self.interval) - time.time()

    def _getter(self):
        # thread function responsible for pulling on iterator and enqueueing the
        # values; implements the throttle_interval semantics to handle both
        # consumption-based pulling and simulated real-time pulling, including
        # switching between the modes; see _throttle() for description of
        # semantics of throttle_interval
        throttle_interval = None
        while True:
            if throttle_interval is None:
                # wait for a state change, e.g. _worker drains the queue, or
                # start(), a new iterator, new mode, etc
                self.get_lock.acquire()

            if not self._state_lock_acquire():
                return

            # throttle_interval controls what we do on this pass through the loop
            throttle_interval = self._throttle()
            self.state_lock.release()

            if throttle_interval is None:
                # nothing to do, so loop and wait for the get_lock
                pass
            elif throttle_interval > 0:
                # we're ahead of "real time", so wait...
                time.sleep(throttle_interval)
            else:
                assert throttle_interval <= 0
                # pull on iterator so as to catch up to "real time" in real-time
                # mode or to enqueue up to queue_low_water items in on-demand
                # mode
                while throttle_interval is not None and throttle_interval <= 0:
                    try:
                        # pull data from the client's iterator; could take a
                        # long time; this is where we will be waiting if the
                        # iterator we're pulling on is itself rate-limited, or
                        # doing work if the iterator does work
                        value = self.iter.next()
                    except StopIteration:
                        if not self._state_lock_acquire():
                            print 'source.py: StopIteration and not _state_lock_acquire()'
                            import sys
                            print >> sys.stderr, 'source.py: StopIteration and not _state_lock_acquire()'
                            return
                        self._is_iter_stopped = True                        
                        throttle_interval = None
                        self.state_lock.release()
                    else:
                        if not self._state_lock_acquire():
                            return
                        self._enqueue(value)
                        # this release allows the _worker thread to loop again
                        # without waiting for some high-level state change and
                        # its call to _exit()
                        if self.work_lock.locked():
                            self.work_lock.release()
                        throttle_interval = self._throttle()
                        self.state_lock.release()


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
