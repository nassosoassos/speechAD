###########################################################################
#
# File:         spectrum.py
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Spectral calculations like FFT, Mel filters, Discrete Cosine
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 The Johns Hopkins University
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
Module for spectrum generation.

This module has tools for generating spectral data from various sources.


Examples of FFT work using fft() and ifft().


Null sequence is OK

>>> fft([])
()

Singleton is trivial

>>> fft([1+1j])
((1+1j),)

Even/odd pair gives pure real/imaginary results

>>> fft([1-1j, 1+1j])
((2+0j), -2j)

Some other simple examples:

>>> fft([1, -1] * 4)
(0, 0j, 0j, 0j, 8, 0j, 0j, 0j)

>>> fft([1] * 2 + [0] * 2)
(2, (1-1j), 0, (1+1j))

Tranform of impulse is uniform in frequency

>>> fft([1] * 1 + [0] * (8 - 1))
(1, (1+0j), (1+0j), (1+0j), 1, (1+0j), (1+0j), (1+0j))

Now, helper rounding and scaling functions to make the following
results readable, and to hide small floating-point differences.

>>> def cint(c): return complex(int(round(c.real)), int(round(c.imag))) if isinstance(c, complex) else int(round(c))
>>> def scale_it(array): return tuple(cint(item * 1000) for item in array)

Delayed impulses get phase shift, but have same magnitudes

>>> scale_it(fft([0] * 1 + [1] * 1 + [0] * (8 - 2)))
(1000, (707-707j), -1000j, (-707-707j), -1000, (-707+707j), 1000j, (707+707j))

>>> scale_it(fft([0] * 2 + [1j] * 1 + [0] * (8 - 3)))
(1000j, (1000+0j), -1000j, (-1000+0j), 1000j, (1000+0j), -1000j, (-1000+0j))

>>> scale_it(fft([0] * 3 + [(1+1j)*sqrt(1/2)] * 1 + [0] * (8 - 4)))
((707+707j), -1000j, (-707+707j), (1000+0j), (-707-707j), 1000j, (707-707j), (-1000+0j))

Demonstrate superposition

>>> x = [1] * 2 + [0j] * 2 + [0] * (16 - 4)
>>> y = [0] * 2 + [1j] * 2 + [0] * (16 - 4)
>>> z = list(a+b for a,b in izip(x,y))
>>> scale_it(a+b for a,b in izip(fft(x), fft(y))) == scale_it(fft(z))
True

Create a 1/4 duty-cycle pulse

>>> pulse = [1] * 4 + [0] * (16 - 4)
>>> scale_it(pulse)
(1000, 1000, 1000, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

Look at sinx/x pattern with zeroes every 4th frequency

>>> scale_it(fft(pulse))
(4000, (3014-2014j), (1000-2414j), (-248-1248j), 0j, (834+166j), (1000-414j), (401-599j), 0, (401+599j), (1000+414j), (834-166j), 0j, (-248+1248j), (1000+2414j), (3014+2014j))

Show recovery of original signal with inverse DFT and vice versa.
Third example shows that equality in the first example is relying
on the rounding/truncating in scale_it.

>>> scale_it(ifft(fft(pulse))) == scale_it(pulse)
True

>>> scale_it(fft(ifft(pulse))) == scale_it(pulse)
True

>>> ifft(fft(pulse)) == pulse
False

Odd real sinusoid gives pure imaginary impulse:

>>> scale_it(fft(sin(3 * (2 * PI * n / 32)) for n in xrange(32)))
(0, 0j, 0j, -16000j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 16000j, 0j, 0j)

More superposition.  Note that the odd imaginary sine at 5 gives the pure real part of the result:

>>> scale_it(fft(complex(sin(3 * (2 * PI * n / 32)), sin(5 * (2 * PI * n / 32))) for n in xrange(32)))
(0j, 0j, 0j, -16000j, 0j, (16000+0j), 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, 0j, (-16000+0j), 0j, 16000j, 0j, 0j)

Regression test, showing that the twiddle cache works when you drop back from 32 to 16 samples....

>>> scale_it(fft([-1] * 2 + [-1j] * 2 + [0] * (16 - 4)))
((-2000-2000j), (-3555-707j), (-3414+1414j), (-1707+2555j), 2000j, (472+707j), 0j, (-293+58j), 0j, (141-707j), (-586-1414j), (-1707-1141j), (-2000+0j), (-1058+707j), 0j, (-293-1472j))

Show that there are no significant differences this implementation and
the reference
>>> tuple((i, int(abs(100000 * (ref-can)))) for i in xrange(11) for ref, can in izip(reference_fft(range(1 << i)), fft(range(1 << i))) if int(abs(100000 * (ref-can))))
()
"""


from __future__ import division
from math import pi as PI, sin, cos, sqrt
from cmath import exp as cexp
from itertools import izip, islice, chain, count, repeat
from functools import partial

from onyx.signalprocessing import sigprocbase
from onyx.signalprocessing.filter import type2stage
from onyx.builtin import attrdict

class PreEmphasis_draft(sigprocbase):
    """
    Pre-emphasis of wave data.  Boost higher frequencies with a
    one-zero filter.

    >>> pe = PreEmphasis_draft('3dB=2500*Hz*Hz/sec/usec*usec')
    >>> pe.configure(sample_rate=8000)
    * Hz
    * Hz
    / sec
    / usec
    * usec
    defaultdict(<type 'int'>, {'Hz': 2, 'usec': 0, 'sec': -1})
    """

    SERIAL_VERSION = 0
    def __init__(self, *args):
        super(PreEmphasis_draft, self).__init__(*args)
        

    def configure(self, **kwargs):
        db = self.init_options.get('3dB')
        return units(db)
        

class PreEmphasis(type2stage):
    """
    Pre-emphasis of wave data.  Boost higher frequencies.

    >>> pe = PreEmphasis()
    >>> pe
    PreEmphasis((0, 0), (1, -0.98046875), 1)
    
    >>> res = list()
    >>> pe.set_recipient(res.append)
    >>> pe.send(1)
    >>> res
    [1.0]
    >>> pe.send(0)
    >>> pe.send(0)
    >>> pe.send(0)
    >>> res
    [1.0, 1.0, -0.98046875, 0.0]
    >>> pe.send_many(xrange(5))
    >>> res
    [1.0, 1.0, -0.98046875, 0.0, 0.0, 1.0, 3.0, 4.01953125, 5.0390625]
    """

    def __init__(self, *_):
        # eight bits of precision
        alpha = 0xfb / (1 << 8);
        super(PreEmphasis, self).__init__((), (1, -alpha))
        
    def config(self, samplerate):
        pass

def units(string):
    import re
    from collections import defaultdict
    splitter = re.compile(r'([*/])').split
    parse = splitter(string)
    key, units = parse[0], parse[1:]
    d = defaultdict(int)
    ui = iter(units)
    for numden, name in izip(ui, ui):
        assert numden == '*' or numden == '/'
        print numden, name
        d[name] += (1 if numden == '*' else -1)
    print d
        

class Fft(sigprocbase):
    SERIAL_VERSION = 0

class MelFilter(sigprocbase):
    SERIAL_VERSION = 0

class Dct(sigprocbase):
    SERIAL_VERSION = 0
    


class SimpleFftMgr(object):
    """
    A class with a very simple recursive implementation of the
    decimate in time FFT algorithm for the DFT.  Each instance caches
    twiddle factors, so a single instance can be used for all
    power-of-two DFT work.  Instance is callable with a sequence or
    iterable of (possibly complex) numbers.

    The module functions fft() and ifft() use a shared instance of
    SimpleFftMgr.
    """

    def __init__(self):
        # twiddle cache (from zero to minus PI), and lowest frequency
        # in the cache; we prime this with two exact values; this
        # helps give integer results in some doctest examples
        self.twiddles = (1, -1j)
        self.freq = -2j * PI / 4
        
    def oddtwiddles(self):
        # return a generator for odd-indexed twiddles
        freq = self.freq
        return (cexp(freq * n) for n in xrange(1, len(self.twiddles) * 2, 2))
    
    def __call__(self, iterable):
        # the fft function

        array = iterable if hasattr(iterable, '__len__') else tuple(iterable)
        len_array = len(array)

        # early outs for lengths <= two
        if len_array <= 2:
            if len_array < 2:
                return tuple(array)
            x0, x1 = array
            return x0 + x1, x0 - x1
        
        # update the twiddle cache by doublings of its length, whereby
        # the old twiddles become the even twiddles in the new cache
        #
        # note: for thread safety of this object, this while loop and
        # the final assignment would need a mutex; that's it
        while 2 * len(self.twiddles) < len_array:
            self.freq /= 2
            self.twiddles = tuple(item
                                  for items in izip(self.twiddles, self.oddtwiddles())
                                  for item in items)
        twiddles = self.twiddles

        # simple recursive ftt implementation
        def fft_recursive(xx, twiddle_stride):
            if len(xx) > 4:
                twiddle_stride2 = twiddle_stride * 2
                evens = fft_recursive(xx[::2], twiddle_stride2)
                odds = fft_recursive(xx[1::2], twiddle_stride2)
                # assert len(evens) == len(odds) == len(xx) // 2
                twodds = tuple(twiddle * odd for twiddle, odd in izip(twiddles[::twiddle_stride], odds))                
                return tuple(chain((even + twodd for even, twodd in izip(evens, twodds)),
                                   (even - twodd for even, twodd in izip(evens, twodds))))

            # optimization: 4-tap DFT has exact coefficients
            x0, x1, x2, x3 = xx
            x0px2 = x0 + x2
            x0mx2 = x0 - x2
            x1px3 = x1 + x3
            jx1mjx3 = 1j * (x1 - x3)
            return x0px2 + x1px3, x0mx2 - jx1mjx3, x0px2 - x1px3, x0mx2 + jx1mjx3

        len_twiddles2 = len(twiddles) * 2
        if len_twiddles2 % len_array != 0:
            raise ValueError("expected a sequence with power of two length, got length %d" % (len_array,))
        return fft_recursive(array, len_twiddles2 // len_array)


# the shared instance
_fft = SimpleFftMgr()

def fft(iterable):
    """
    Calculates the Discrete Fourier Transform of the sequence of
    (possibly complex) values in the iterable argument.  Returns a
    tuple of the resulting (likely complex) values.
    """
    return _fft(iterable)

def ifft(iterable):
    """
    Calculates the inverse Discrete Fourier Transform of the sequence
    of (possibly complex) values in the iterable argument.  Returns a
    tuple of the resulting (likely complex) values.
    """

    # start with the dft
    result = _fft(iterable)

    len_result = len(result)
    if len_result <= 1:
        return result

    # Rearrange and scale to get the inverse.  This ain't pretty, but
    # this one-liner seems simpler than making the SimpleFftMgr's
    # twiddler run backwards.
    return tuple(item / len_result for item in chain(result[:1], result[-1:0:-1]))    



class power_of_two_padding_dict(attrdict):
    """an auto dict of zero-paddings needed for power-of-two lengths"""
    def __missing__(self, padding):
        power_of_two = 1
        while power_of_two < padding:
            power_of_two <<= 1
        return power_of_two - padding

# shared instance
_padding = power_of_two_padding_dict()

class FftMag(object):
    """
    A callable object that returns the magnitudes of the DFT of a
    sequence of values.  It pads the sequence to a power of two,
    calculates the DFT, and returns the magnitudes.  If the optional
    constructor argument truncate is 'half' the returned sequence will
    be truncated to one half the length of the padded sequence.  If
    truncate is 'half+' the returned sequence will be truncated to one
    half the length of the padded sequence plus one sample.

    Examples:

    Set up the three flavors of FftMag and a helper function

    >>> mag = FftMag()
    >>> maghalf = FftMag('half')
    >>> maghalfp = FftMag('half+')
    >>> def intify(iterable): return tuple(int(x) for x in iterable)

    Some simple examples

    >>> mag([])
    (0,)
    >>> mag([1])
    (1,)
    >>> mag([1] * 16)
    (16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    >>> cycle3 = tuple(sin(3 * (2*PI*n/32)) for n in xrange(32))
    >>> intify(mag(cycle3))
    (0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0)
    >>> intify(maghalf(cycle3))
    (0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    Showing why 'half+' is important

    >>> nyquist = tuple(item for item in [1, -1] * 16)
    >>> intify(maghalf(nyquist))
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    >>> intify(maghalfp(nyquist))
    (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32)
    >>> dd = [1, -1, 1, -1] + [0] * 16
    >>> intify(maghalf(dd))
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3)
    >>> intify(maghalfp(dd))
    (0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 2, 3, 3, 3, 4)

    >>> def energy(iterable): return sum(abs(x)*abs(x) for x in iterable)
    >>> energy(dd)
    4
    >>> magdd = mag(dd)
    >>> round(energy(x/sqrt(len(magdd)) for x in magdd), 5)
    4.0

    >>> intify(maghalfp(sin(3 * (2*PI*n/33)) for n in xrange(32)))
    (0, 0, 1, 15, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    """

    def __init__(self, truncate=None):
        if not (truncate is None or truncate == 'half' or truncate == 'half+'):
            raise ValueError("expected truncate to be one of None, 'half', or 'half+', got %s" % (repr(truncate),))
        self.truncate = truncate

    def __call__(self, iterable):
        """
        Calculates the Discrete Fourier Transform of the sequence of
        (possibly complex) values in the iterable argument.  Pads the
        sequence with zeroes to get a power-of-two length.

        Returns a tuple of the magnitudes of the DFT values.
        """

        array = iterable if hasattr(iterable, '__len__') else tuple(iterable)
        padding = _padding[len(array)]
        if padding > 0:
            array = tuple(chain(array, islice(repeat(0), padding)))
        dft = _fft(array)

        truncate = self.truncate
        if truncate is None:
            return tuple(abs(x) for x in dft)
        count = len(dft) // 2
        if truncate == 'half+':
            count += 1
        return tuple(abs(x) for x in islice(dft, count))



class sliceview(object):
    """
    Slice-based views of a sequence.

    >>> def printit(x):
    ...   print len(x), ':',
    ...   for i in x:
    ...     print i,
    ...   print

    >>> r = xrange(32)
    >>> s0 = sliceview(r, 0, 2)
    >>> printit(s0)
    16 : 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30

    >>> s1 = sliceview(r, 1, 2)
    >>> printit(s1)
    16 : 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31

    >>> sx = sliceview(s1, 1, 2)
    >>> printit(sx)
    8 : 3 7 11 15 19 23 27 31

    >>> sy = sliceview(sx, 0, 2)
    >>> printit(sy)
    4 : 3 11 19 27

    >>> sz = sliceview(sx, 1, 2)
    >>> printit(sz)
    4 : 7 15 23 31

    """

    def __init__(self, iterable, offset, inc):
        if isinstance(iterable, sliceview):
            seq, ooffset, len_seq, oinc = iterable.slice
            offset = ooffset + offset * oinc
            inc *= oinc
        else:
            seq = iterable if hasattr(iterable, '__len__') and hasattr(iterable, '__getitem__') else tuple(iterable)
            len_seq = len(seq)
        self.slice = seq, offset, len_seq, inc
        self.len_self = (len_seq - offset + inc - 1) // inc
        self.iterview = seq, xrange(offset, len_seq, inc)

    def __len__(self):
        return self.len_self

    def __iter__(self):
        seq, iterview = self.iterview
        return (seq[i] for i in iterview)


def reference_fft(x):
    """
    A very simple refernce implementation of the decimate in time FFT
    algorithm for the DFT.

    The length of the (possibly complex) array must be a power of two.

    >>> def scale_it(array): return tuple(int(round(abs(item) * 1000)) for item in array)

    >>> reference_fft([])
    ()

    >>> reference_fft([1])
    (1,)

    >>> scale_it(reference_fft([1] * 2))
    (2000, 0)

    >>> scale_it(reference_fft([1] * 2 + [0] * 2))
    (2000, 1414, 0, 1414)

    >>> scale_it(reference_fft([1] * 2 + [complex(0,1)] * 2 + [0] * (32 - 4)))
    (2828, 3310, 3625, 3754, 3696, 3460, 3073, 2571, 2000, 1410, 850, 368, 0, 227, 299, 218, 0, 326, 721, 1139, 1531, 1849, 2053, 2110, 2000, 1718, 1273, 688, 0, 747, 1501, 2212)

    >>> scale_it(reference_fft([1] * 1 + [0] * (16 - 1)))
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)

    >>> scale_it(reference_fft([1] * 8 + [0] * (32 - 8)))
    (8000, 7214, 5126, 2436, 0, 1500, 1800, 1115, 0, 915, 1203, 802, 0, 739, 1020, 711, 0, 711, 1020, 739, 0, 802, 1203, 915, 0, 1115, 1800, 1500, 0, 2436, 5126, 7214)

    >>> scale_it(reference_fft(sin(2 * PI * n * 3 / 32) for n in xrange(32)))
    (0, 0, 0, 16000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16000, 0, 0)

    >>> scale_it(reference_fft([1] * 2 + [complex(0,1)] * 2 + [0] * (16 - 4)))
    (2828, 3625, 3696, 3073, 2000, 850, 0, 299, 0, 721, 1531, 2053, 2000, 1273, 0, 1501)

    >>> tuple(complex(int(round(x.real)), int(round(x.imag))) for x in fft([-1000] * 2 + [-1000j] * 2 + [0] * (16 - 4)))
    ((-2000-2000j), (-3555-707j), (-3414+1414j), (-1707+2555j), 2000j, (472+707j), 0j, (-293+58j), 0j, (141-707j), (-586-1414j), (-1707-1141j), (-2000+0j), (-1058+707j), 0j, (-293-1472j))


    >>> for i in xrange(8):
    ...   scale_it(reference_fft(range(1 << i)))
    (0,)
    (1000, 1000)
    (6000, 2828, 2000, 2828)
    (28000, 10453, 5657, 4330, 4000, 4330, 5657, 10453)
    (120000, 41007, 20905, 14400, 11314, 9622, 8659, 8157, 8000, 8157, 8659, 9622, 11314, 14400, 20905, 41007)
    (496000, 163237, 82013, 55118, 41810, 33942, 28799, 25221, 22627, 20698, 19243, 18142, 17318, 16720, 16313, 16077, 16000, 16077, 16313, 16720, 17318, 18142, 19243, 20698, 22627, 25221, 28799, 33942, 41810, 55118, 82013, 163237)
    (2016000, 652161, 326474, 218087, 164027, 131698, 110237, 94987, 83620, 74844, 67883, 62244, 57598, 53718, 50442, 47650, 45255, 43188, 41397, 39840, 38486, 37308, 36284, 35399, 34637, 33987, 33440, 32989, 32627, 32350, 32155, 32039, 32000, 32039, 32155, 32350, 32627, 32989, 33440, 33987, 34637, 35399, 36284, 37308, 38486, 39840, 41397, 43188, 45255, 47650, 50442, 53718, 57598, 62244, 67883, 74844, 83620, 94987, 110237, 131698, 164027, 218087, 326474, 652161)
    (8128000, 2607856, 1304321, 869984, 652947, 522830, 436174, 374352, 328053, 292102, 263396, 239959, 220473, 204028, 189973, 177830, 167240, 157931, 149688, 142345, 135767, 129844, 124489, 119627, 115197, 111148, 107437, 104026, 100884, 97983, 95301, 92815, 90510, 88368, 86375, 84521, 82793, 81183, 79681, 78279, 76972, 75753, 74616, 73556, 72569, 71651, 70797, 70006, 69273, 68596, 67973, 67402, 66880, 66405, 65977, 65594, 65254, 64956, 64700, 64485, 64310, 64174, 64077, 64019, 64000, 64019, 64077, 64174, 64310, 64485, 64700, 64956, 65254, 65594, 65977, 66405, 66880, 67402, 67973, 68596, 69273, 70006, 70797, 71651, 72569, 73556, 74616, 75753, 76972, 78279, 79681, 81183, 82793, 84521, 86375, 88368, 90510, 92815, 95301, 97983, 100884, 104026, 107437, 111148, 115197, 119627, 124489, 129844, 135767, 142345, 149688, 157931, 167240, 177830, 189973, 204028, 220473, 239959, 263396, 292102, 328053, 374352, 436174, 522830, 652947, 869984, 1304321, 2607856)

    """

    x = tuple(x)
    if len(x) < 2:
        return x

    jnegtwopireciplength = -2j * PI / len(x)
    twiddles = tuple(cexp(jnegtwopireciplength * n) for n in xrange(len(x) // 2))

    def fft_recursive(xx, twiddle_stride):
        if len(xx) == 1:
            return xx
        evens = fft_recursive(xx[::2], twiddle_stride * 2)
        odds = fft_recursive(xx[1::2], twiddle_stride * 2)
        if len(evens) != len(odds):
            raise ValueError("expected a power of two, got len(x) %d" % (len(x),))
        twodds = tuple(twiddle * odd for twiddle, odd in izip(twiddles[::twiddle_stride], odds))        
        return tuple(chain((even + twodd for even, twodd in izip(evens, twodds)),
                           (even - twodd for even, twodd in izip(evens, twodds))))

    return fft_recursive(x, 1)

        # note: the sliceview approach which avoids copying is not as
        # fast in Python as the tuple-slicing approach, but it almost
        # certainly is faster in a compiled language; so the above
        # assignments to evens and odds could be replaced with these
        # versions:
        #evens = fft_recursive(sliceview(xx, 0, 2), twiddle_stride * 2)
        #odds = fft_recursive(sliceview(xx, 1, 2), twiddle_stride * 2)



def reference_splitradix_fft(x):
    """
    A very simple refernce implementation of the decimate in time FFT
    algorithm for the DFT.

    The length of the (possibly complex) array must be a power of two.

    >>> def scale_it(array): return tuple(int(round(abs(item) * 1000)) for item in array)

    >>> reference_splitradix_fft([])
    ()

    >>> reference_splitradix_fft([1])
    (1,)

    >>> scale_it(reference_splitradix_fft([1] * 2))
    (2000, 0)

    >>> scale_it(reference_splitradix_fft([1] * 2 + [0] * 2))
    (2000, 1414, 0, 1414)

    >>> scale_it(reference_splitradix_fft([1] * 2 + [complex(0,1)] * 2 + [0] * (32 - 4)))
    (2828, 3310, 3625, 3754, 3696, 3460, 3073, 2571, 2000, 1410, 850, 368, 0, 227, 299, 218, 0, 326, 721, 1139, 1531, 1849, 2053, 2110, 2000, 1718, 1273, 688, 0, 747, 1501, 2212)

    >>> scale_it(reference_splitradix_fft([1] * 1 + [0] * (16 - 1)))
    (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000)

    >>> scale_it(reference_splitradix_fft([1] * 8 + [0] * (32 - 8)))
    (8000, 7214, 5126, 2436, 0, 1500, 1800, 1115, 0, 915, 1203, 802, 0, 739, 1020, 711, 0, 711, 1020, 739, 0, 802, 1203, 915, 0, 1115, 1800, 1500, 0, 2436, 5126, 7214)

    >>> scale_it(reference_splitradix_fft(sin(2 * PI * n * 3 / 32) for n in xrange(32)))
    (0, 0, 0, 16000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16000, 0, 0)

    >>> scale_it(reference_splitradix_fft([1] * 2 + [complex(0,1)] * 2 + [0] * (16 - 4)))
    (2828, 3625, 3696, 3073, 2000, 850, 0, 299, 0, 721, 1531, 2053, 2000, 1273, 0, 1501)

    >>> tuple(complex(int(round(x.real)), int(round(x.imag))) for x in fft([-1000] * 2 + [-1000j] * 2 + [0] * (16 - 4)))
    ((-2000-2000j), (-3555-707j), (-3414+1414j), (-1707+2555j), 2000j, (472+707j), 0j, (-293+58j), 0j, (141-707j), (-586-1414j), (-1707-1141j), (-2000+0j), (-1058+707j), 0j, (-293-1472j))


    >>> for i in xrange(8):
    ...   scale_it(reference_splitradix_fft(range(1 << i)))
    (0,)
    (1000, 1000)
    (6000, 2828, 2000, 2828)
    (28000, 10453, 5657, 4330, 4000, 4330, 5657, 10453)
    (120000, 41007, 20905, 14400, 11314, 9622, 8659, 8157, 8000, 8157, 8659, 9622, 11314, 14400, 20905, 41007)
    (496000, 163237, 82013, 55118, 41810, 33942, 28799, 25221, 22627, 20698, 19243, 18142, 17318, 16720, 16313, 16077, 16000, 16077, 16313, 16720, 17318, 18142, 19243, 20698, 22627, 25221, 28799, 33942, 41810, 55118, 82013, 163237)
    (2016000, 652161, 326474, 218087, 164027, 131698, 110237, 94987, 83620, 74844, 67883, 62244, 57598, 53718, 50442, 47650, 45255, 43188, 41397, 39840, 38486, 37308, 36284, 35399, 34637, 33987, 33440, 32989, 32627, 32350, 32155, 32039, 32000, 32039, 32155, 32350, 32627, 32989, 33440, 33987, 34637, 35399, 36284, 37308, 38486, 39840, 41397, 43188, 45255, 47650, 50442, 53718, 57598, 62244, 67883, 74844, 83620, 94987, 110237, 131698, 164027, 218087, 326474, 652161)
    (8128000, 2607856, 1304321, 869984, 652947, 522830, 436174, 374352, 328053, 292102, 263396, 239959, 220473, 204028, 189973, 177830, 167240, 157931, 149688, 142345, 135767, 129844, 124489, 119627, 115197, 111148, 107437, 104026, 100884, 97983, 95301, 92815, 90510, 88368, 86375, 84521, 82793, 81183, 79681, 78279, 76972, 75753, 74616, 73556, 72569, 71651, 70797, 70006, 69273, 68596, 67973, 67402, 66880, 66405, 65977, 65594, 65254, 64956, 64700, 64485, 64310, 64174, 64077, 64019, 64000, 64019, 64077, 64174, 64310, 64485, 64700, 64956, 65254, 65594, 65977, 66405, 66880, 67402, 67973, 68596, 69273, 70006, 70797, 71651, 72569, 73556, 74616, 75753, 76972, 78279, 79681, 81183, 82793, 84521, 86375, 88368, 90510, 92815, 95301, 97983, 100884, 104026, 107437, 111148, 115197, 119627, 124489, 129844, 135767, 142345, 149688, 157931, 167240, 177830, 189973, 204028, 220473, 239959, 263396, 292102, 328053, 374352, 436174, 522830, 652947, 869984, 1304321, 2607856)

    """

    x = tuple(x)
    if len(x) < 2:
        return x

    negtwopireciplength = -2j * PI / len(x)
    twiddles = tuple(cexp(n * negtwopireciplength) for n in xrange(3 * len(x) // 4))
    # twiddles = tuple(cexp(n * negtwopireciplength) for n in xrange(len(x)))    
    twiddler = partial(islice, twiddles, 0, None)

    def fft_recursive(xx, twiddle_stride):
        if len(xx) == 1:
            return xx
        evens = fft_recursive(xx[::2], twiddle_stride * 2)
        if len(evens) >= 4:
            # we can do the split radix work
            odds1 = fft_recursive(xx[1::4], 4 * twiddle_stride)
            odds3 = fft_recursive(xx[3::4], 4 * twiddle_stride)
            if len(odds1) != len(odds3):
                raise ValueError("expected a power of two, got len(x) %d" % (len(x),))

            twodds1 = tuple(twiddle * odd1 for twiddle, odd1 in izip(twiddler(twiddle_stride), odds1))
            twodds3 = tuple(twiddle * odd3 for twiddle, odd3 in izip(twiddler(3 * twiddle_stride), odds3))
            assert len(twodds1) == len(twodds3)

            twoddsums = tuple(twodd1 + twodd3 for twodd1, twodd3 in izip(twodds1, twodds3))
            jtwodddiffs = tuple(1j * (twodd1 - twodd3) for twodd1, twodd3 in izip(twodds1, twodds3))

            return tuple(chain((even + twoddsum for even, twoddsum in izip(evens, twoddsums)),
                               (even - jtwodddiff for even, jtwodddiff in izip(islice(evens, len(evens)//2, None), jtwodddiffs)),
                               (even - twoddsum for even, twoddsum in izip(evens, twoddsums)),
                               (even + jtwodddiff for even, jtwodddiff in izip(islice(evens, len(evens)//2, None), jtwodddiffs))))

        odds = fft_recursive(xx[1::2], twiddle_stride * 2)
        if len(evens) != len(odds):
            raise ValueError("expected a power of two, got len(x) %d" % (len(x),))

        twodds = tuple(twiddle * odd for twiddle, odd in izip(twiddler(twiddle_stride), odds))

        return tuple(chain((even + twodd for even, twodd in izip(evens, twodds)),
                           (even - twodd for even, twodd in izip(evens, twodds))))

    return fft_recursive(x, 1)


def _timetest():

    count = 2000000

    # function-based work
    def mag(x):
        return abs(x)

    from time import time
    start = time()
    for x in xrange(count):
        y = mag(x)
        # assert y == x
    stop = time()
    print
    print "duration mag function:", round(stop-start, 6), '-->', (stop-start) / count, 'per call'


    # generator-based work
    def maggen(abs=abs):
        x = yield
        while 1:
            x = yield abs(x)

    mag = maggen().send
    mag(None)

    start = time()
    for x in xrange(count):
        y = mag(x)
        # assert y == x
    stop = time()
    print
    print "duration mag generator:", round(stop-start, 6), '-->', (stop-start) / count, 'per call'
    

    count2 = 100
    buf = tuple(complex(x, x//2) for x in range(512))

    start = time()
    for x in xrange(count2):
        _fft(buf)
    stop = time()
    print
    print "duration 100 512 pt-fft:", round(stop-start, 6), '-->', (stop-start) / count2, 'per call'
    

    bigshift = 16

    # prime the cache
    print
    start = time()
    _fft(0 for x in xrange(1<<bigshift))
    stop = time()
    print  "duration priming:", round(stop-start, 6)

    print
    print "sizes"
    for shift in xrange(bigshift+1):
        size = 1 << shift
        count3 = 1
        buf = tuple(complex(x, x//2) for x in range(size))
        start = time()
        for x in xrange(count3):
            _fft(buf)
        stop = time()
        print "duration", count3, size, "pt-fft:", round(stop-start, 6), '-->', (stop-start) / count3, 'per call'


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
    
    from sys import argv
    if '--timing' in argv:
        _timetest()
