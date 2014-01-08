###########################################################################
#
# File:         streamprocess.py
# Date:         Mon 2008-01-28 17:48:32 -0500
# Author:       Hugh Secker-Walker (with mods by Ken Basye)
# Description:  Simple stream-based processing chain
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
Baseclass and some simple subclasses for dataflow processing of streaming data.
"""

from itertools import izip
from numpy import array, append, delete, s_, arange
from onyx.graph.graphtools import FrozenGraph, GraphBuilder
import numpy as N

# a function that's useful for the doctests
def _printer(value):
    print repr(value)

class ProcessorBase(object):
    """
    Baseclass for processor elements in a chain.  Subclasses must implement process(value) which
    does work on value to do work based on value, and to call self.send(result) when the work
    produces a result to push on down the chain.

    >>> a = ProcessorBase(sendee=_printer)
    >>> a.process(None)
    Traceback (most recent call last):
       ...
    NotImplementedError: ProcessorBase must implement self.process(value) as a method or attribute
    """

    def __init__(self, sendee=None, sending=True, label=None):
        self._sendee = None
        self._sending = sending

        if sendee is not None:
            self.set_sendee(sendee)
        
        self._label = label if label is not None else type(self).__name__
        gb = GraphBuilder()
        node = gb.new_node(self._label)
        self._graph = FrozenGraph(gb)

    # XXX revisit sendee baseclass semantics given the multiple sendees
    # implemented by splits, e.g. SplitProcessor

    def set_sendee(self, sendee):
        """Clients call this to set up the callable where the
        processor will send its results."""
        if not callable(sendee):
            raise ValueError("expected sendee to be callable, got a %s" % (type(sendee).__name__,))
        self._sendee = sendee

    def set_sending(self, sending):
        """Clients call this to turn sending from a processor on or off."""
        if sending is not False and sending is not True:
            raise ValueError("expected either True or False, got %s" % (sending,))
        self._sending = sending

    @property
    def sendee(self):
        return self._sendee

    @property
    def sending(self):
        return self._sending

    def send(self, result):
        """
        Internal function that pushes result into the sendee.  Implementations of process(value)
        must call this to push results.  To set up the sendee, (the target of the push), clients of
        the processor must either initialize the object with a sendee, or call set_sendee().
        Processors created with a sendee of False will never send, but will not error if send is
        called.
        """
        if not self.sending:
            return
        if not hasattr(self, '_sendee'):
            name = type(self).__name__
            raise AttributeError("no '_sendee' attribute on instance of %s: perhaps this subclass of"
                                 "ProcessorBase did not call the baseclass's __init__ method from "
                                 "its constructor, e.g. via super(%s, self).__init__(sendee)"
                                 % (name, name,))
        sendee = self._sendee
        if sendee is None:
            raise ValueError("nowhere to send result: client of %s processor needs "
                             "to construct instance with a sendee argument or "
                             "call set_sendee(sendee) prior to using the "
                             "processor"
                             % (type(self).__name__,))
        assert callable(sendee)
        sendee(result)

    def process(self, value):
        """
        Subclasses must override the self.process(value) function.

        The function accepts a single value.  If, after processing the single
        value, it has a new result or results, the function must call
        self.send(result) for each result that is calculated.  For a give call
        to process(), the function can call send() zero, one, or multiple times
        depending on the semantics that the processor type implements.
        """
        raise NotImplementedError("%s must implement self.process(value) as a method or attribute"
                                  % (type(self).__name__,))

    @property
    def label(self):
        """
        Return a label for this processor.  By default this is just the name of
        the class; derived classes may wish to override this property by
        providing a different label to __init__.
        """
        return _label
        
    @property
    def graph(self):
        """
        Return a graph for this processor.  By default this is just a single
        node whose label is the label of processor; derived classes may wish to
        override this property.
        """
        return self._graph
        
class EpsilonProcessor(ProcessorBase):
    """
    An epsilon processor.  Just sends the value passed to process().

    An example where a user-defined function is the sendee; and in
    this case that function just prints out the value.
    >>> e = EpsilonProcessor()
    >>> e.set_sendee(_printer)
    >>> e.process('abc')
    'abc'
    >>> e.process('def')
    'def'
    """
    def __init__(self, sendee=None, sending=True):
        super(EpsilonProcessor, self).__init__(sendee, sending=sending)
    def process(self, value):
        self.send(value)
        

class ChainProcessor(ProcessorBase):
    """
    Processor that chains together an iterable sequence of processors.

    >>> e = ChainProcessor(tuple(EpsilonProcessor() for i in xrange(5)), sendee=_printer)
    >>> for x in 'abc': e.process(x)
    'a'
    'b'
    'c'

    >>> e.graph
    FrozenGraph(GraphTables((('EpsilonProcessor', 'EpsilonProcessor', 'EpsilonProcessor', 'EpsilonProcessor', 'EpsilonProcessor'), (0, 1, 2, 3), (1, 2, 3, 4), (None, None, None, None))))
    """
    def __init__(self, processors, sendee=None, sending=True):
        super(ChainProcessor, self).__init__(sendee, sending=sending)
        processors = tuple(processors)
        if not processors:
            raise ValueError("expected at least one element in processors chain, got zero")

        # front of the chain, where we push stuff
        self.head = processors[0]


        # create chain of processors, linking each processor's send function to
        # each successor's process() function....
        gb = GraphBuilder()
        nodes, starts, ends = gb.add_graph(processors[0].graph)
        assert len(starts) == 1 and len(ends) == 1
        if len(processors) > 1:
            succ_iter = iter(processors)
            succ_iter.next()
            for pred, succ in izip(processors, succ_iter):
                pred.set_sendee(succ.process)
                nodes, new_starts, new_ends = gb.add_graph(succ.graph)
                gb.new_arc(ends[0], new_starts[0])
                starts, ends = new_starts, new_ends
                assert len(starts) == 1 and len(ends) == 1
            assert succ == processors[-1]

        # set up the list that will collect what the final element pushes
        self.collector = list()
        processors[-1].set_sendee(self.collector.append)

        self._graph = FrozenGraph(gb)

                                   
##         # it is possible to implement the process attribute as a generator...
##         def process():
##             head_process = self.head.process
##             send = self.send
##             collector = self.collector
##             while True:
##                 assert not collector
##                 item = yield
##                 # this pushes stuff through the chain
##                 head_process(item)
##                 # see if anything got pushed out the far end
##                 if collector:
##                     # if so, send it on
##                     for result in collector:
##                         send(result)
##                     del collector[:]
##         self.process = process().send
##         self.process(None)
##         # ... but we don't
##         del self.process

    @property
    def graph(self):
        return self._graph

    def process(self, item):
        collector = self.collector
        assert not collector
        # this pushes stuff through the chain
        self.head.process(item)
        # see if anything got pushed out the far end
        if collector:
            # if so, send it on
            send = self.send
            for result in collector:
                send(result)
            del collector[:]
        
class WatchProcessor(ProcessorBase):
    """
    A debugging processor.  Prints a label and the value, then sends
    the value on down the stream.

    >>> recipient = list()
    >>> e = WatchProcessor('hoo hoo:', sendee=recipient.append)
    >>> for x in xrange(4): e.process(x)
    hoo hoo: 0
    hoo hoo: 1
    hoo hoo: 2
    hoo hoo: 3
    >>> recipient
    [0, 1, 2, 3]
    """
    def __init__(self, watch_label, sendee=None, sending=True):
        super(WatchProcessor, self).__init__(sendee, sending=sending)
        self._watch_label = watch_label
    def process(self, value):
        print self._watch_label, value
        self.send(value)
        
        
class CollectProcessor(ProcessorBase, list):
    """
    A collecting processor, a subclass of list.  Collects the values
    as well as passing them on.  Useful for collecting elements in the
    middle of a chain.

    This example doesn't really do the collect processor justice
    because it puts the collector at the end of the chain.  See usage
    in main() for the more useful case of collecting values from
    somewhere in the middle of a chain.
    >>> recipient = list()
    >>> e = CollectProcessor(sendee=recipient.append)
    >>> for x in xrange(4): e.process(x)
    >>> recipient
    [0, 1, 2, 3]
    >>> e
    [0, 1, 2, 3]
    """
    def __init__(self, sendee=None, sending=True):
        super(CollectProcessor, self).__init__(sendee, sending=sending)
    def process(self, value):
        self.append(value)
        self.send(value)
        
class FunctionProcessor(ProcessorBase):
    """
    Apply a callable to the argument to process(), send the result.

    >>> c = FunctionProcessor(int, sendee=_printer)
    >>> for x in xrange(7): c.process(-0.825 * x)
    0
    0
    -1
    -2
    -3
    -4
    -4
    """
    def __init__(self, function, sendee=None, label=None, sending=True):
        # XXX Doing all this before the base class initialization makes me nervous :-<
        if label is not None:
            lab = label
        elif hasattr(function, "__name__"):
            lab = function.__name__ + "()"
        # Maybe function is an object with __call__ defined
        elif hasattr(function, "__class__"):
            lab = function.__class__.__name__ + "()"
        else:
            lab = None # give up and let the base class do what it will

        super(FunctionProcessor, self).__init__(sendee, label=lab, sending=sending)
        if not callable(function):
            raise ValueError("expected argument function to be a callable, got an instance of %s" % (type(function).__name__,))
        self.function = function

    def process(self, value):
        self.send(self.function(value))

        
class SequenceFunctionProcessor(ProcessorBase):
    """
    Apply a callable to the argument to process(); the callable returns a
    sequence; send each item in the returned sequence.

    >>> c = SequenceFunctionProcessor(lambda x: [x] * abs(x), sendee=_printer)
    >>> for x in xrange(3, -4, -1): c.process(x)
    3
    3
    3
    2
    2
    1
    -1
    -2
    -2
    -3
    -3
    -3
    """
    def __init__(self, function, sendee=None, label=None, sending=True):
        # XXX Doing all this before the base class initialization makes me nervous :-<
        if label is not None:
            lab = label
        elif hasattr(function, "__name__"):
            lab = function.__name__
        # Maybe function is an object with __call__ defined
        elif hasattr(function, "__class__"):
            lab = function.__class__.__name__
        else:
            lab = None # give up and let the base class do what it will
        super(SequenceFunctionProcessor, self).__init__(sendee, sending=sending, label=lab)
        if not callable(function):
            raise ValueError("expected argument function to be a callable, got an instance of %s" % (type(function).__name__,))
        self.function = function
    def process(self, value):
        for item in self.function(value):
            self.send(item)
    
class DecimatingProcessor(ProcessorBase):
    """
    A decimating processor.  Pushes value of every nth call to
    process(), starting with the value of the first call to process().

    >>> recipient = list()
    >>> d = DecimatingProcessor(2, sendee=recipient.append)
    >>> for x in xrange(10): d.process(x)
    >>> recipient
    [0, 2, 4, 6, 8]
    >>> d = DecimatingProcessor(3)
    >>> d.set_sendee(recipient.append)
    >>> for x in xrange(10): d.process(x)
    >>> recipient
    [0, 2, 4, 6, 8, 0, 3, 6, 9]
    """
    def __init__(self, nth, sendee=None, sending=True):
        super(DecimatingProcessor, self).__init__(sendee, sending=sending)
        if not isinstance(nth, int):
            raise TypeError("expected nth to be %s, got %s" % (int.__name__, type(nth).__name__))
        if nth < 1:
            raise ValueError("expected nth to be one or greater, got %d" % (nth,))
        self.nth = nth
        self.countdown = 1

    def process(self, value):
        self.countdown -= 1
        if self.countdown == 0:
            self.countdown = self.nth
            self.send(value)


class SlidingWindowProcessor(ProcessorBase):
    """
    >>> s = SlidingWindowProcessor(5, 2, sendee=_printer)
    >>> for i in xrange(2):
    ...   s.process(arange(1, dtype=int))
    ...   s.process(arange(10,12, dtype=int))
    ...   s.process(arange(20,23, dtype=int))
    ...   s.process(arange(30,40, dtype=int))
    array([ 0, 10, 11, 20, 21])
    array([11, 20, 21, 22, 30])
    array([21, 22, 30, 31, 32])
    array([30, 31, 32, 33, 34])
    array([32, 33, 34, 35, 36])
    array([34, 35, 36, 37, 38])
    array([36, 37, 38, 39,  0])
    array([38, 39,  0, 10, 11])
    array([ 0, 10, 11, 20, 21])
    array([11, 20, 21, 22, 30])
    array([21, 22, 30, 31, 32])
    array([30, 31, 32, 33, 34])
    array([32, 33, 34, 35, 36])
    array([34, 35, 36, 37, 38])
    """
    def __init__(self, window_len, hop_len, sendee=None, sending=True):
        super(SlidingWindowProcessor, self).__init__(sendee, sending=sending)
        self.window_len = window_len
        self.window_slice = s_[0:window_len]
        self.hop_slice = s_[0:hop_len]
        self.window = array((), dtype=int)
        self.data = [self.window, self.window_len, self.window_slice, self.hop_slice, self.send]

    def process(self, value):
        window, window_len, window_slice, hop_slice, send = data = self.data
        window = append(window, value)
        # XXX this could be made more efficient for the case of a long
        # window by keeping track of both the start and end of the
        # slice to send and then doing one delete at the end
        while len(window) >= window_len:
            send(window[window_slice])
            window = delete(window, hop_slice)
        data[0] = window

    
class SplitProcessor(ProcessorBase):
    """
    A processor for fanning out data, i.e. a split.  Sendees must be added with
    add_sendee().  Pushes value of each call to process() to every sendee.
    Sendees are a set, so each sendee is called only once per process() even if
    it was added to SplitProcessor via multiple calls to add_sendee().

    >>> recipient1 = list()
    >>> recipient2 = list()
    >>> d = SplitProcessor()
    >>> d.add_sendee(recipient1.append)
    >>> r2a = recipient2.append  # note: see commentary in add_sendee()
    >>> d.add_sendee(r2a)
    >>> d.add_sendee(r2a)
    >>> for x in xrange(10): d.process(x)
    >>> recipient1
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> recipient2
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    Show that non-ndarrays don't get copied, but ndarrays do get copied

    >>> del recipient1[:], recipient2[:]
    >>> d.process(range(5))
    >>> d.process(N.arange(5))
    >>> recipient1[0] is recipient2[0]
    True
    >>> recipient1[1] is recipient2[1]
    False

    >>> d.add_sendee(0)
    Traceback (most recent call last):
       ...
    ValueError: expected a callable sendee, got an instance of int
    """
    def __init__(self, sending=True):
        # XXX what about label?
        super(SplitProcessor, self).__init__(sending=sending)
        self.sendees = dict()

    def set_sendee(self, sendee):
        """Not implemented for this processor; use add_sendee()"""
        raise NotImplementedError("use add_sendee() to add to the set of targets")

    def add_sendee(self, sendee):
        """
        Clients call this to add a callable to the set to which this processor
        will send its results.  Has set semantics based on the id() of sendee;
        so, as with set objects, adding a sendee more than once does not change
        the runtime behavior.  Sends copies of ndarrays.
        """
        if not callable(sendee):
            raise ValueError("expected a callable sendee, got an instance of %s" % (type(sendee).__name__,))

        # note: use of id() is necessary to get a hashable object; but we're
        # straying into a grey area of the interpreter, e.g. for a list object
        # x, repeated evaluations of x.append do not necessarily return an
        # object with the same id.
        id_ = id(sendee)
        if id_ not in self.sendees:
            self.sendees[id_] = sendee
        
    def process(self, value):
        if not self.sending:
            return
        if not self.sendees:
            raise ValueError("nowhere to send result: client of %s processor needs "
                             "to add one or more targets using add_sendee(sendee) "
                             "prior to using the processor"
                             % (type(self).__name__,))
        # XXX we're sidestepping ._sendee and .send()
        for target in self.sendees.itervalues():
            assert callable(target)
            # XXX we need to get the copy semantics less implicit....
            # XXX no need to copy the final guy....
            target(value.copy() if isinstance(value, N.ndarray) else value)

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
