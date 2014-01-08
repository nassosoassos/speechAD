###########################################################################
#
# File:         eventchain.py
# Date:         7-Dec-2007
# Author:       Hugh Secker-Walker
# Description:  Example of objects and coding styles involved in an event-processing chain
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 - 2009 The Johns Hopkins University
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
Demonstration of event-processing chain.  A "simple" event detector on
the smoothed average-energy of a decimated two-tone signal.  Also
demonstrates composition of two chains.

Chain that decimates, windows, and computes energy.
>>> stage1 = ChainProcessor((DecimatingProcessor(2), window(10, 10), energy()))

Chain that does the signal conditioning we want and the peak/valley detection
>>> stage2 = ChainProcessor((smooth(0.825),
...                          FunctionProcessor(int),
...                          events({'peak':True, 'valley':True})))

Composed processor chains the two stages, with an epsilon processor in the middle,
just because
>>> event_detector = ChainProcessor((stage1, EpsilonProcessor(), stage2))
>>> event_detector.set_sendee(printer)
>>> from math import sin, pi
>>> for i in xrange(1000):
...     event_detector.process(10 * sin(2*pi*i/200) + 15 * sin(2*pi*i/(pi*80)))
('valley', 5, 182)
('peak', 8, 265)
('valley', 11, 178)
('peak', 13, 192)
('valley', 26, 32)
('peak', 30, 52)
('valley', 32, 41)
('peak', 36, 127)
('valley', 38, 102)
('peak', 41, 234)
('valley', 44, 182)
('peak', 47, 315)
"""


from __future__ import division
from operator import add
from itertools import izip
from onyx.util.streamprocess import ProcessorBase, ChainProcessor, EpsilonProcessor
from onyx.util.streamprocess import DecimatingProcessor, FunctionProcessor, CollectProcessor

# a function that's useful for the doctests
def printer(value):
    print repr(value)

    
class window(ProcessorBase):
    """
    A simple sliding window.

    >>> recipient = list()
    >>> w = window(3, 2, recipient.append)
    >>> for x in xrange(10): w.process(x)
    >>> recipient
    [(0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, 8)]
    """
    def __init__(self, length, hop, sendee=None, sending=True):
        super(window, self).__init__(sendee, sending=sending)
        if not length >= hop:
            raise ValueError("expected length to be greater than or equal to hop, got length %s, hop %s" % (length, hop))
        self.length = length
        self.hop = hop
        self.window = list()

    def process(self, value):
        window = self.window
        window.append(value)
        assert len(window) <= self.length
        if len(window) == self.length:
            self.send(tuple(window))
            window[:] = window[self.hop:]
    

class energy(ProcessorBase):
    """
    Compute the average value of the energy in the iterable that's
    sent to process().

    An example where a container-growing function is the sendee; and
    then the filled container can be used....
    >>> recipient = list()
    >>> e = energy(recipient.append)
    >>> e.process(xrange(10))
    >>> map(int, recipient)
    [28]
    """
    def __init__(self, sendee=None, sending=True):
        super(energy, self).__init__(sendee, sending=sending)
    def process(self, iterable):
        samples = tuple(iterable)
        if len(samples) > 0:
            self.send(reduce(add, (x * x for x in samples)) / len(samples))
    
class smooth(ProcessorBase):
    """
    One-pole smoothing of sample sent to process().

    Example where we fill a container and then generate a plot from
    the data in the container.  Also, due to the behavior of smooth
    after it's been reset(), demonstrates a case where not every input
    leads to a send into the recipient.

    >>> e = smooth(0.75)
    >>> recipient = list()
    >>> e.set_sendee(recipient.append)

    Send a ramp through the one-pole
    >>> for x in xrange(10): e.process(x)
    >>> len(recipient)
    9

    Reset and then send part of a downward ramp
    >>> e.reset()
    >>> for x in reversed(xrange(5, 10)): e.process(x)
    >>> len(recipient)
    13

    And now look at some homogeneous response
    >>> for x in (0,) * 15: e.process(x)
    >>> for x in recipient: print ' ' * int(5 * x) + '*'
     *
       *
          *
             *
                 *
                     *
                          *
                              *
                                   *
                                               *
                                             *
                                          *
                                       *
                              *
                       *
                  *
               *
            *
          *
        *
       *
      *
     *
     *
     *
    *
    *
    *
    """
    
    def __init__(self, alpha, sendee=None, sending=True):
        """
        Create a one-pole IIR filter with the pole at alpha.
        """
        super(smooth, self).__init__(sendee, sending=sending)
        self.alpha = alpha
        self.scale = 1 - alpha
        self.reset()
    def reset(self):
        self.prev = None
    def process(self, sample):
        if self.prev is None:
            self.prev = sample
        else:
            self.prev = self.alpha * self.prev + self.scale * sample
            self.send(self.prev)
    
class events(ProcessorBase):
    """
    Find events in the stream of values passed to process().  Each
    result is a tuple: (event type, index in stream, value at index)
    of the event in the stream.  Note that a single cal to process()
    can push multiple events into the sendee.

    First we collect results from a smoother, and then send them to
    the event detector, and just use a printer for the end recipient.

    >>> samples = list()
    >>> s = smooth(0.75)
    >>> s.set_sendee(samples.append)

    Send in an up ramp
    >>> for x in xrange(10): s.process(x)

    Reset and send in a down ramp and then catenate the samples we've collected
    >>> s.reset()
    >>> for x in reversed(xrange(5, 10)): s.process(x)
    >>> stream = samples + samples

    Plot the stream values with their index in the stream
    >>> for index, value in enumerate(stream): print "%2d %s" % (index, ' ' * int(5 * value) + '*')
     0  *
     1    *
     2       *
     3          *
     4              *
     5                  *
     6                       *
     7                           *
     8                                *
     9                                            *
    10                                          *
    11                                       *
    12                                    *
    13  *
    14    *
    15       *
    16          *
    17              *
    18                  *
    19                       *
    20                           *
    21                                *
    22                                            *
    23                                          *
    24                                       *
    25                                    *

    Run an event detector, just printing out the event tuples.
    >>> e = events({'peak': True, 'valley':True, 'up_discontinuity':1.6, 'down_discontinuity':1.5}, printer)
    >>> for x in stream: e.process(x)
    ('up_discontinuity', 5, 3.533935546875)
    ('up_discontinuity', 6, 4.40045166015625)
    ('up_discontinuity', 7, 5.3003387451171875)
    ('up_discontinuity', 8, 6.2252540588378906)
    ('up_discontinuity', 9, 8.75)
    ('peak', 9, 8.75)
    ('down_discontinuity', 13, 0.25)
    ('valley', 13, 0.25)
    ('down_discontinuity', 14, 0.6875)
    ('down_discontinuity', 15, 1.265625)
    ('up_discontinuity', 18, 3.533935546875)
    ('up_discontinuity', 19, 4.40045166015625)
    ('up_discontinuity', 20, 5.3003387451171875)
    ('up_discontinuity', 21, 6.2252540588378906)
    ('up_discontinuity', 22, 8.75)
    ('peak', 22, 8.75)
    """

    # event types are class members
    peak = 'peak'
    valley = 'valley'
    up_discontinuity = 'up_discontinuity'
    down_discontinuity = 'down_discontinuity'

    def __init__(self, config, sendee=None, sending=True):
        super(events, self).__init__(sendee, sending=sending)
        self.config = config
        self.index = -1
        self.buffer = list()
    def process(self, sample):
        config = self.config
        buffer = self.buffer
        buffer.append(sample)
        self.index += 1

        # note: this event detector has a very weak model of acoustic discontinuities

        # this should be configurable
        bufferlen = 6

        # behavior based on values in self.config
        if len(buffer) >= bufferlen:
            left, center, right = buffer[-3:]
            
            # just using the presense of self.peak or self.valley in
            # config; these events use of the value at index - 1 (center)
            if self.peak in config and center > left and center > right:
                self.send((self.peak, self.index - 1, center))
            if self.valley in config and center < left and center < right:
                self.send((self.valley, self.index - 1, center))

            # using the presense of self.up_discontinuity or
            # self.down_discontinuity in config and also using the
            # value in config; these events use the value at index (right)
            if self.up_discontinuity in config:
                # quantile-based threshold
                prev = sorted(buffer[:-1])[-3]
                if right > config[self.up_discontinuity] * prev:
                    self.send((self.up_discontinuity, self.index, right))

            if self.down_discontinuity in config:
                # quantile-based threshold
                prev = sorted(buffer[:-1])[2]
                if right * config[self.down_discontinuity] < prev:
                    self.send((self.down_discontinuity, self.index, right))
            
            # prevent the buffer from growing
            self.buffer = buffer[-bufferlen:]

def main(args):
    """
    Args is an iterable sequence of filenames.

    Exercise the audiofile module by doing some event detection on the files.
    """
    from onyx.audio.audiodata import get_file_audio
    import os.path

    for filename in args:
        file_info, audio_info, wave_data = get_file_audio(filename, 'int16')

        # unpack the numpy arrays into Python lists
        # XXX revisit the processors in this chain and make them use numpy arrays
        num_channels, num_samples = wave_data.shape

        channels = tuple(channel.tolist() for channel in wave_data)

        print
        print 'filename:', os.path.basename(filename)
        for key in sorted(file_info.keys()):
            if key == 'file_name':
                print '  os.path.basename(' + key + ')', os.path.basename(file_info[key])
            elif key != 'file_name_full':
                print ' ', key, file_info[key]

        # assert that all channels have the same number of samples
        buflens = set(len(channel) for channel in channels)
        assert len(buflens) == 1
        print ' ', 'SamplesPerChannel', buflens.pop()

        # work as something like 11025
        base_rate = 11025
        decimation = audio_info.audio_sample_rate // base_rate  # integer division
        if decimation < 1:
            decimation = 1

        # we tap two places in the chain
        smootheds = CollectProcessor()
        discontinuties = CollectProcessor()
        eventdetector = ChainProcessor((DecimatingProcessor(decimation),
                                        window(440, 110),
                                        energy(),
                                        smooth(0.75),
                                        FunctionProcessor(int),
                                        smootheds,
                                        events({'up_discontinuity':2.5,
                                                'down_discontinuity':2.0}),
                                        discontinuties), sending=False)

        # just process the first channel
        for sample in channels[0]:
            eventdetector.process(sample)

        print 'smoothed_energy:'
        for index, smoothed in enumerate(smootheds):
            print ' ', index, smoothed
        print 'events:'
        # note: the index inside the discontinuity corresponds to the index of smoothed
        for index, discontinuity in enumerate(discontinuties):
            print ' ', index, repr(discontinuity)


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    args = argv[1:]
    if args:
        main(args)
