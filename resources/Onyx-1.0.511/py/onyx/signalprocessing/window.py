###########################################################################
#
# File:         window.py
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Windowing functions
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
Windowing functionality.

"""

from __future__ import division
from onyx.dataflow import processor
from onyx.builtin import frozentuple, attrdict

from onyx.signalprocessing import sigprocbase

class Sliding(sigprocbase, processor):
    SERIAL_VERSION = 0
    def __init__(self, *_):
        super(Sliding, self).__init__(*_)

        self.signal = attrdict()

        # usec
        self.window = 25000
        self.hop = self.window // 3

    # XXX formalize configuration from signal sending
##     def send_signals(self, samplerate):
##         self.config(samplerate)

    def config(self, samplerate):
        sru = samplerate / 1000000
        win = int(self.window * sru)
        hop = int(self.hop * sru)
        print "win", win, "hop", hop
        self.send = self.senderator(win, hop, self.signal).send
        self.send(None)

        
    def send_signals(self, **kwargs):
        signal = self.signal
        signal.clear()
        signal.update(kwargs)
        return self.send(signal)

    def get_recipient(self):
        return self.send_signals(get_recipient=True)
    def set_recipient(self, recipient):
        self.send_signals(set_recipient=recipient)
        
    def reset(self):
        self.send_signals(reset=True)
        

    @staticmethod
    def senderator(length, hop, signal, recipient=None):
        assert length >= hop, "length %r  hop %r" % (length, hop)
        window = list()

        hasattr = signal.hasattr
        result = None
        while True:
            sample = yield result
            result = None

            if sample is signal:
                if hasattr.set_recipient:
                    recipient = signal.set_recipient
                if hasattr.get_recipient:
                    result = recipient
                if hasattr.reset:
                    window.clear()
                continue

            window.append(sample)
            if len(window) >= length:
                assert len(window) == length
                payload = window
                window = window[hop:]
                if recipient is not None:
                    recipient(payload)
                

class Hamming(sigprocbase):
    SERIAL_VERSION = 0

class Padding(sigprocbase):
    SERIAL_VERSION = 0

class Truncate(sigprocbase):
    SERIAL_VERSION = 0


from math import pi as PI, cos, sqrt

class FixedPointHammingMgr(object):
    """
    Fixedpoint Hamming window manager.  Caches Hamming windows.
    
    Hamming window is one of a parametric family of windowing
    functions that are a raised cosine.  The parameter for this
    family is the bottom of the function's range:

      range is (b, 1), where 0 < b < 1

    and the window function reaches b at the edges, and 1 in the
    center (if the length is odd).  Given b, set:

      a = (b + 1) / 2, where 1/2 < a < 1

    the Hamming window, with support (0, length - 1), is given by:

      w[index] = a - (1 - a) * cos(2 * pi * index / (length - 1))

    Most references to Hamming window use b = 0.08 (giving a = 0.54 and 1 - a =
    0.46), but b = 0.07677... is also used.  In order to reduce floating-point
    issues, we choose a value for b that is slightly less than 0.08, but which
    has only 16 significant bits.

    """
   
    BOTTOM = 0xa3d7 / float(1 << 19)
    # note: a = COS_OFFSET
    COS_OFFSET = (BOTTOM + 1) / 2
    COS_SCALE = 1 - COS_OFFSET

    __slots__ = ('windows',)
    def __init__(self):
        self.windows = dict()

    def __call__(self, length, shift, norm=None):
        key = length, shift, norm
        windows = self.windows
        hamming = windows.get(key)

        if hamming is None:

            # check preconditions etc
            if int(length) != length or length < 0:
                raise ValueError("expected non-negative integral length, got %s" % (length,))
            length = int(length)
            if int(shift) != shift or shift < 0:
                raise ValueError("expected non-negative integral shift, got %s" % (shift,))
            shift = int(shift)
            if not (norm is None or norm == 'sum' or norm == 'sumsq' or norm == 'mode'):
                raise ValueError("expected norm of None or 'sum' or 'mode', got %s" % (repr(norm),))
            key = frozentuple(key)

            if length == 0:
                hamming = frozentuple()
            elif length == 1:
                hamming = frozentuple([float(1 << shift)])
            else:
                # create unscaled Hamming window values

                bottom = self.BOTTOM
                cos_offset = self.COS_OFFSET
                cos_scale = self.COS_SCALE

                length1 = length - 1
                twopireciplength1 = 2 * PI / length1
                hamming = [None] * length

                left, right = 0, length1
                numbottom = 0
                while left <= right:
                    val = cos_offset - cos_scale * cos(twopireciplength1 * left)
                    assert bottom <= val <= 1
                    if val == bottom:
                        numbottom += 1
                    hamming[left] = hamming[right] = val
                    left += 1
                    right -= 1
                assert numbottom > 0
                assert (hamming[length1 // 2] == 1
                        or (length % 2 == 0 and hamming[(length1-1) // 2] == hamming[length1 // 2]))

                # scale and truncate

                # note: we truncate rather than round; for the
                # positive data in the window this errs on the side of
                # slightly less energy than if we rounded, but it avoids
                # the problem of rounding making the sum be more than
                # 1 for clients using the 'sum'-to-one normalization

                if norm is None:
                    norm_sum = 1
                elif norm == 'mode':
                    norm_sum = hamming[length1 // 2]
                    assert norm_sum == hamming[length // 2]
                elif norm == 'sum':
                    norm_sum = sum(hamming)
                elif norm == 'sumsq':
                    norm_sum = sqrt(sum(h*h for h in hamming))
                else:
                    assert False, "unreachable"

                assert norm_sum > 0
                norm_scale = float(1 << shift) / norm_sum
                hamming = windows[key] = frozentuple(int(h * norm_scale) for h in hamming)

        assert type(hamming) is frozentuple
        return hamming            

_FixedPointHammingMgr = FixedPointHammingMgr()
def iHamming(length, shift_scale, norm=None):
    """
    Function returns a frozentuple with a fixed-point hamming window of 'length'
    with values scaled by 1 << 'scale'.

    Optional 'norm' selects the normalization to use prior to applying the
    scale.  If not given, or None, no normalization is applied.  Otherwise the
    values are scaled so that a computed value is one.  The legal values of
    'norm' are 'sum', 'sumsq', or 'mode' which cause the computed value to be
    the sum of the samples, the sum of the squares of the samples, or the mode
    of the samples in the window respectively.

    The returned values are cached, so it is not expensive to use this function
    repeatedly with the same arguments.

    >>> length_range = 7
    >>> scale_shift = 10
    >>> for norm in None, 'sum', 'sumsq', 'mode':
    ...   print
    ...   print "normalization:", repr(norm)
    ...   for length in xrange(length_range):
    ...     print length, iHamming(length, scale_shift, norm)
    <BLANKLINE>
    normalization: None
    0 frozentuple(())
    1 frozentuple((1024.0,))
    2 frozentuple((81, 81))
    3 frozentuple((81, 1024, 81))
    4 frozentuple((81, 788, 788, 81))
    5 frozentuple((81, 552, 1024, 552, 81))
    6 frozentuple((81, 407, 934, 934, 407, 81))
    <BLANKLINE>
    normalization: 'sum'
    0 frozentuple(())
    1 frozentuple((1024.0,))
    2 frozentuple((512, 512))
    3 frozentuple((70, 882, 70))
    4 frozentuple((48, 463, 463, 48))
    5 frozentuple((36, 246, 457, 246, 36))
    6 frozentuple((29, 146, 335, 335, 146, 29))
    <BLANKLINE>
    normalization: 'sumsq'
    0 frozentuple(())
    1 frozentuple((1024.0,))
    2 frozentuple((724, 724))
    3 frozentuple((81, 1017, 81))
    4 frozentuple((74, 720, 720, 74))
    5 frozentuple((64, 437, 810, 437, 64))
    6 frozentuple((58, 288, 661, 661, 288, 58))
    <BLANKLINE>
    normalization: 'mode'
    0 frozentuple(())
    1 frozentuple((1024.0,))
    2 frozentuple((1024, 1024))
    3 frozentuple((81, 1024, 81))
    4 frozentuple((106, 1024, 1024, 106))
    5 frozentuple((81, 552, 1024, 552, 81))
    6 frozentuple((89, 446, 1024, 1024, 446, 89))
    """

    return _FixedPointHammingMgr(length, shift_scale, norm)


# junk
def xHamming(index, length, shift):
    """
    Integer Hamming calculation.  Returns the value for zero-based 'index' in a
    Hamming window of 'length' scaled by 1 << 'shift'.

    >>> xHamming(1, 3, 10)
    81
    """

    # for a value range is (b, 1) where 0 < b < 1
    # a =  (b + 1) / 2
    # typically, b = 0.08, giving a = 0.54
    # but binary-based floating point doesn't represent these values exactly...

    b = 0.08
    a = (b + 1) / 2
    val = a + (1 - a) * cos(2 * PI * index / (length - 1))

    return int(val * (1 << shift))


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
