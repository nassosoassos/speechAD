###########################################################################
#
# File:         filter.py
# Date:         11-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Support for FIR and IIR filters
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
    Traditional FIR and IIR filtering

    An unstable second-order filter
    >>> f1 = type2stage((5, 6), (1, 2), 2)
    >>> f1
    type2stage((5, 6), (1, 2), 2)
    >>> f1.process_one(100)
    [200]
    >>> f1.process_one(0)
    [-800]
    >>> f1.process_one(0)
    [3200]

    An FIR filter, using default coeff_k=1
    >>> f2 = type2stage((), (2, 3))
    >>> f2.process_some((1, 0, 0, 0, -1, -2, -3, 0, 0, 0))
    [1, 2, 3, 0, -1, -4, -10, -12, -9, 0]

    A first-order filter
    >>> for y in type2stage((-0.75,), (), 30).process_some((1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)):
    ...   print '-' * int(y), int(y)
    ------------------------------ 30
    ---------------------- 22
    ---------------- 16
    ------------ 12
    --------- 9
    ------- 7
    ----- 5
    ---- 4
    --- 3
    -- 2
    - 1
    - 1

    A stable second-order filter
    >>> for y in type2stage((-1.5, 0.875), (), 10).process_some((1,) * 20):
    ...   print '-' * int(y), int(y)
    ---------- 10
    ------------------------- 25
    -------------------------------------- 38
    ---------------------------------------------- 46
    --------------------------------------------- 45
    ------------------------------------- 37
    -------------------------- 26
    ----------------- 17
    ------------ 12
    ------------- 13
    ------------------- 19
    --------------------------- 27
    --------------------------------- 33
    ------------------------------------- 37
    ----------------------------------- 35
    ------------------------------- 31
    ------------------------- 25
    --------------------- 21
    ------------------- 19
    -------------------- 20


    A first-order, bi-linear 'pre-emphasis' filter run at a low and a
    high frequency
    >>> map(int, type2stage((0.5,), (-0.05,), 10).process_some((1, 0, -1, 0) * 4 + (0, 0, 0, 0) + (1, -1) * 8))
    [10, -5, -7, 4, 7, -4, -7, 4, 7, -4, -7, 4, 7, -4, -7, 4, -2, 1, 0, 0, 9, -15, 18, -19, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20]

    >>> f4 = type2stage((-1.5, 0.875), (), 10)
    >>> def yow(x): print 'yow', int(x)
    >>> f4.get_recipient()
    >>> f4.set_recipient(yow)
    >>> f4.send_many([1] * 20)
    yow 10
    yow 25
    yow 38
    yow 46
    yow 45
    yow 37
    yow 26
    yow 17
    yow 12
    yow 13
    yow 19
    yow 27
    yow 33
    yow 37
    yow 35
    yow 31
    yow 25
    yow 21
    yow 19
    yow 20
    >>> f4.get_recipient()  #doctest: +ELLIPSIS
    <function yow at 0x...>
    >>> f4.set_recipient(None)
    >>> f4.send_many([0] * 20)
    >>> f4.set_recipient(yow)
    >>> f4.send_many([0] * 20)
    yow 2
    yow 0
    yow -2
    yow -4
    yow -3
    yow -1
    yow 0
    yow 2
    yow 2
    yow 2
    yow 1
    yow 0
    yow -1
    yow -2
    yow -1
    yow 0
    yow 0
    yow 1
    yow 1
    yow 1
    >>> f4.send_many([1] * 5)
    yow 10
    yow 24
    yow 37
    yow 45
    yow 44
    >>> f4.send(0)
    yow 27

    Get the state, and then run it for a few
    >>> s = f4.get_state()
    >>> f4.send_many([1] * 5)
    yow 12
    yow 4
    yow 5
    yow 14
    yow 27

    Reset it and see that it's dead
    >>> f4.reset()
    >>> f4.send_many([0] * 3)
    yow 0
    yow 0
    yow 0

    Restore the state and get the same output as we got prior to the reset
    >>> s = f4.set_state(s)
    >>> f4.send_many([1] * 5)
    yow 12
    yow 4
    yow 5
    yow 14
    yow 27
    
    >>> f4.reset()
    >>> f4.send(0)
    yow 0

    >>> f4.send_many([1] * 5)
    yow 10
    yow 25
    yow 38
    yow 46
    yow 45
    >>> f4.send_signals(recipient=yow, reset=True)
    >>> f4.send(0)
    yow 0

    The trivial case
    >>> f3 = type2stage((), ())
    >>> f3
    type2stage((0, 0), (0, 0), 1)
    >>> f3.process_some(xrange(20))
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    A degenerate case
    >>> f4 = type2stage((), (), 0)
    >>> f4
    type2stage((0, 0), (0, 0), 0)
    >>> f4.process_some(xrange(20))
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


"""

from __future__ import division
from onyx.builtin import attrdict
from onyx.dataflow import processor
from collections import deque
from itertools import islice, chain, repeat, izip
import math, cmath

class type2stage(processor):
    """
    Implements the second-order difference equation

    y[n] + a1 * y[n-1] + a2 * y[n-2] = k * (x[n] + b1 * x[n-1] + b2 * x[n-2])

    """

    def __init__(self, coeffs_a, coeffs_b, coeff_k=1):
        """
        Construct a canonic Type II filter that implements the
        (inhomogeneous) second-order difference equation:

          y[n] + a1 * y[n-1] + a2 * y[n-2]  =  k * (x[n] + b1 * x[n-1] + b2 * x[n-2])
        
        generating the solution y[n] given input sequence x[n]:
        
          y[n]  =  -1 * (a1 * y[n-1] + a2 * y[n-2])  +  k * (x[n] + b1 * x[n-1] + b2 * x[n-2])  

        The 'coeffs_a' is a sequence of zero to two IIR coefficients,
        'a' in the above.  The 'coeffs_b' is a sequence of zero to two
        FIR coefficients, 'b' in the above.  In these two sequences,
        missing trailing coefficients are treated as having value 0.
        The optional 'coeff_k' argument is the overall scaling factor
        that defaults to one, 'k' in the above.
        """

        super(type2stage, self).__init__()

        if len(coeffs_b) > 2:
            raise ValueError("expected coeffs_b of length 2 or less, got %d" % (len(coeffs_b),))
        if len(coeffs_a) > 2:
            raise ValueError("expected coeffs_a of length 2 or less, got %d" % (len(coeffs_a),))

        coeffs_b = tuple(islice(chain(coeffs_b, repeat(0)), 2))
        coeffs_a = tuple(islice(chain(coeffs_a, repeat(0)), 2))
        self.coeff_k = coeff_k
        self.ba = tuple(izip(coeffs_b, coeffs_a))
        assert len(self.ba) == 2
        self.deque = deque((0, 0))
        
        self.data = self.coeff_k, self.ba, self.deque

        def senderator(coeffs_a, coeffs_b, coeff_k, signal, recipient=None):
            a1, a2 = coeffs_a
            b1, b2 = coeffs_b
            k = coeff_k

            # so we can use 'is' comparisons
            zero = 0
            a1 = a1 or zero
            a2 = a2 or zero
            b1 = b1 or zero
            b2 = b2 or zero
            w1, w2 = reset = zero, zero
            hasattr = signal.hasattr
            result = None

            while True:
                x = yield result
                result = None

                if x is signal:
                    if hasattr.get_recipient:
                        result = recipient
                    if hasattr.set_recipient:
                        recipient = signal.set_recipient
                    if hasattr.get_state:
                        result = w1, w2
                    if hasattr.set_state:
                        w1, w2 = signal.set_state
                    if hasattr.reset:
                        w1, w2 = reset
                    continue

                w = k * x
                if a1 is not zero: w -= a1 * w1
                if a2 is not zero: w -= a2 * w2

                y = w
                if b1 is not zero: y += b1 * w1
                if b2 is not zero: y += b2 * w2

                w2 = w1
                w1 = w

                if recipient is not None:
                    recipient(y)

        self.signal = attrdict()
        self.send = senderator(coeffs_a, coeffs_b, coeff_k, self.signal).send
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
        
    def get_state(self):
        return self.send_signals(get_state=True)
    def set_state(self, state):
        self.send_signals(set_state=state)
        
    def reset(self):
        self.send_signals(reset=True)

    def __repr__(self):
        coeff_k, ba, _ = self.data
        coeffs_b = tuple(b for b, a in ba)
        coeffs_a = tuple(a for b, a in ba)

        return "%s(%r, %r, %r)" % (type(self).__name__, coeffs_a, coeffs_b, coeff_k)

    def process_one(self, input):
        if input is self.null:
            return []
        
        coeff_k, ba, deque = self.data
        wn = input
        yn = 0
        for (b, a), w in izip(ba, deque):
            yn += b * w
            wn -= a * w
        yn += wn
        yn *= coeff_k
        deque.rotate(1)
        deque[0] = wn
        return [yn]



class nonlinear2pole(processor):
    """
    Implements a nonlinear two-pole system.
    """

    def __init__(self, r, theta):
        """
        A simple two-pole system, with zeros at 1 and -1.  Argument r
        is the radius to each pole, theta is the angle in radians.  In
        the running system, you can vary r via h.set_r(new_r_value)
        for nonlinear behavior.

        >>> for y in nonlinear2pole(0.97, 0.4).process_some((.1,) * 30):
        ...   y+=15;print '-' * int(y), int(y)
        ----------------- 17
        ---------------------- 22
        ------------------------- 25
        -------------------------- 26
        -------------------------- 26
        ------------------------ 24
        -------------------- 20
        ---------------- 16
        ------------ 12
        -------- 8
        ------ 6
        ----- 5
        ------ 6
        -------- 8
        ---------- 10
        -------------- 14
        ----------------- 17
        -------------------- 20
        --------------------- 21
        ---------------------- 22
        --------------------- 21
        -------------------- 20
        ----------------- 17
        --------------- 15
        ------------ 12
        ---------- 10
        --------- 9
        --------- 9
        --------- 9
        ----------- 11
        """

        super(nonlinear2pole, self).__init__()

        def H(z):
            # print
            # print 'r', r, ' theta', theta
            zsq = z ** 2
            num = zsq - 1
            num2 = (z - 1) * (z + 1)
            # print 'num', num, num - num2
            # print 'abs(num)', abs(num), 2 * math.sin(theta)
            #print 'z', z, abs(z), ' zsq', zsq, abs(zsq)
            den = zsq - 2 * r * math.cos(theta) * z + r ** 2
            den2 = (z - r * pow(cmath.e, 1j*theta)) *  (z - r * pow(cmath.e, -1j*theta))
            # print 'den', den, den - den2

            cos2theta = math.cos(2 * theta)
            asq = 1 - cos2theta ** 2
            bsq = asq + (r - cos2theta) ** 2
            assert bsq >= 0
            den3 = abs(1 - r) * math.sqrt(bsq)
            # print 'den3', den3, abs(den) - den3
            #print 'd', d
            #return 1
            
            return num / den

        coeffs_b = 0, -1

        # XXX write set_r, and use it here
        coeffs_a = -2 * r * math.cos(theta), r * r
        coeff_k = (2 * math.sin(theta)) / (1 - r)

        # print 'coeff_k', coeff_k, ' |H(theta)|', abs(H(pow(cmath.e, 1j*theta)))

        if len(coeffs_b) > 2:
            raise ValueError("expected coeffs_b of length 2 or less, got %d" % (len(coeffs_b),))
        if len(coeffs_a) > 2:
            raise ValueError("expected coeffs_a of length 2 or less, got %d" % (len(coeffs_a),))

        coeffs_b = tuple(islice(chain(coeffs_b, repeat(0)), 2))
        coeffs_a = tuple(islice(chain(coeffs_a, repeat(0)), 2))
        self.coeff_k = coeff_k
        self.ba = tuple(izip(coeffs_b, coeffs_a))
        assert len(self.ba) == 2
        self.deque = deque((0, 0))
        
        self.data = self.coeff_k, self.ba, self.deque

        def senderator(coeffs_a, coeffs_b, coeff_k, signal, recipient=None):
            a1, a2 = coeffs_a
            b1, b2 = coeffs_b
            k = coeff_k

            # so we can use 'is' comparisons
            zero = 0
            a1 = a1 or zero
            a2 = a2 or zero
            b1 = b1 or zero
            b2 = b2 or zero
            w1, w2 = reset = zero, zero
            hasattr = signal.hasattr
            result = None

            while True:
                x = yield result
                result = None

                if x is signal:
                    if hasattr.get_recipient:
                        result = recipient
                    if hasattr.set_recipient:
                        recipient = signal.set_recipient
                    if hasattr.get_state:
                        result = w1, w2
                    if hasattr.set_state:
                        w1, w2 = signal.set_state
                    if hasattr.reset:
                        w1, w2 = reset
                    continue

                w = k * x
                if a1 is not zero: w -= a1 * w1
                if a2 is not zero: w -= a2 * w2

                y = w
                if b1 is not zero: y += b1 * w1
                if b2 is not zero: y += b2 * w2

                w2 = w1
                w1 = w

                if recipient is not None:
                    recipient(y)

        self.signal = attrdict()
        self.send = senderator(coeffs_a, coeffs_b, coeff_k, self.signal).send
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
        
    def get_state(self):
        return self.send_signals(get_state=True)
    def set_state(self, state):
        self.send_signals(set_state=state)
        
    def reset(self):
        self.send_signals(reset=True)

    def __repr__(self):
        coeff_k, ba, _ = self.data
        coeffs_b = tuple(b for b, a in ba)
        coeffs_a = tuple(a for b, a in ba)

        return "%s(%r, %r, %r)" % (type(self).__name__, coeffs_a, coeffs_b, coeff_k)

    def process_one(self, input):
        if input is self.null:
            return []
        
        coeff_k, ba, deque = self.data
        wn = input
        yn = 0
        for (b, a), w in izip(ba, deque):
            yn += b * w
            wn -= a * w
        yn += wn
        yn *= coeff_k
        deque.rotate(1)
        deque[0] = wn
        return [yn]


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
