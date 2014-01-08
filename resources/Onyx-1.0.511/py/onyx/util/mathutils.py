###########################################################################
#
# File:         mathutils.py (directory: ./py/onyx/util)
# Date:         9-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Math utilities
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
    Numerical utilitites
    
    Module attribute 'primes' can be used as an (unbounded) generator
    of prime numbers, or as an (unbounded) sequence of prime numbers.

    Look at some primes:
    >>> tuple(islice(primes, 10))
    (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    >>> tuple(islice(primes, 20))
    (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71)

    The hundredth prime via iteration
    >>> tuple(islice(primes, 100))[-1]
    541

    The thousandth prime via indexing
    >>> primes[999]
    7919


    This module provides some routines for doing log-domain computations on
    Numpy arrays.

    >>> x = numpy.zeros((3,2), dtype=float) - 3.2
    >>> x
    array([[-3.2, -3.2],
           [-3.2, -3.2],
           [-3.2, -3.2]])
    
    By default, logsumexp_array does the summation over the entire array, just
    like Numpy's sum().

    >>> logsumexp_array(x) #doctest: +ELLIPSIS
    -1.40824053077...

    Here's summation along axis 1, which is the horizontal axis in a 2-d array.
    
    >>> logsumexp_array(x, axis=1)
    array([-2.50685282, -2.50685282, -2.50685282])

    To do the summation vertically, specify axis 0.

    >>> logsumexp_array(x, axis=0)
    array([-2.10138771, -2.10138771])


    -Inf is the additive identity for this operation, here's how to get them
     without triggering a warning message to stderr.

    >>> x[:,1] = quiet_log(0.0)
    >>> x
    array([[-3.2, -Inf],
           [-3.2, -Inf],
           [-3.2, -Inf]])
    
    >>> logsumexp_array(x, axis=1)
    array([-3.2, -3.2, -3.2])

    >>> w = numpy.zeros((3,3), dtype=float) - 3.2
    >>> logsumexp(w, w)
    array([[-2.50685282, -2.50685282, -2.50685282],
           [-2.50685282, -2.50685282, -2.50685282],
           [-2.50685282, -2.50685282, -2.50685282]])

    >>> z = log_zeros((3,3), dtype=float)
    >>> (logsumexp(w, z) == w).all()
    True

    >>> y = numpy.zeros((2,3), dtype=float) - 20.0
    >>> y[:,1] = quiet_log(0.0)
    >>> y
    array([[-20., -Inf, -20.],
           [-20., -Inf, -20.]])

    >>> log_dot(x,y)
    array([[-23.2,  -Inf, -23.2],
           [-23.2,  -Inf, -23.2],
           [-23.2,  -Inf, -23.2]])

    >>> log_dot(y,x)
    array([[-22.50685282,         -Inf],
           [-22.50685282,         -Inf]])

    >>> x_exp = numpy.exp(x)
    >>> y_exp = numpy.exp(y)
    >>> dot_exp = numpy.dot(y_exp, x_exp)
    >>> (log_dot(y,x) == quiet_log(dot_exp)).all()
    True

    Note that using a single maximum value can run into underflow problems
    pretty easily if the values in the array are very different from each other
    (about 705 for ordinary doubles).  Using an array of bias values makes this
    work more correctly.
    
    >>> test_arr = numpy.array((-1.0, -7006.1, -7260.1, -7469.1, -7568.2, -7675.8, -7814.9, -9617.2), dtype = float)
    >>> test_arr[0] = quiet_log(0.0)
    >>> print test_arr
    [   -Inf -7006.1 -7260.1 -7469.1 -7568.2 -7675.8 -7814.9 -9617.2]
    >>> z = log_zeros((len(test_arr),), dtype=float)
    >>> y = logsumexp(test_arr, z)
    >>> print y
    [   -Inf -7006.1 -7260.1 -7469.1 -7568.2 -7675.8 -7814.9 -9617.2]
    >>> (y == test_arr).all()
    True

    >>> test_arr2 = test_arr - 10000.0
    >>> test_arr2
    array([    -Inf, -17006.1, -17260.1, -17469.1, -17568.2, -17675.8,
           -17814.9, -19617.2])

    As might be expected, 'adding' numbers which are so much closer to '0' makes
    no difference.  In fact, the values added were actually 0 in this example.
    
    >>> logsumexp(test_arr, test_arr2)
    array([   -Inf, -7006.1, -7260.1, -7469.1, -7568.2, -7675.8, -7814.9,
           -9617.2])

"""

from __future__ import division, with_statement
from collections import defaultdict
from functools import partial
from itertools import islice, izip
from operator import mul
import numpy
from numpy import array, sqrt
from onyx.util import checkutils, discrete
from onyx.util.debugprint import dcheck
from onyx.util import floatutils


class Primes(object):
    """
    Generates primes based on the Eratosthenes sieve, but not
    requiring an end limit.  Instance is intended to be used (and
    reused) as an iterator or an indexed container.

    Module attribute 'primes' is an instance of Primes() intended for
    general use.

    >>> primes[20]
    73

    >>> tuple(islice(primes, 100, 101))
    (547,)
    >>> primes[42:44]
    [191, 193]

    >>> primes[:]
    Traceback (most recent call last):
       ...
    ValueError: expected non-negative value for indexing or slice.stop, got None

    """
    
    def __init__(self):
        # errr, prime the list of primes
        self.sieve = [2, 3]

    def extend(self):
        # grow the list, asymptotically proportionally
        sieve = self.sieve
        sieve_append = sieve.append
        candidate = sieve[-1]
            
        for _ in xrange((len(sieve) // 16) + 4):
            while True:
                candidate += 2
                primes = iter(sieve)
                # skip the first entry, 2
                primes.next()
                for prime in primes:
                    if candidate % prime == 0 or prime * prime > candidate:
                        break
                if candidate % prime != 0:
                    sieve_append(candidate)
                    break

    def __getitem__(self, itemspec):
        limit = itemspec.stop if isinstance(itemspec, slice) else itemspec
        if limit is None or limit < 0:
            raise ValueError("expected non-negative value for indexing or slice.stop, got %r" % (limit,))
        sieve = self.sieve
        extend = self.extend
        while len(sieve) <= limit:
            extend()
        return sieve[itemspec]

    def __iter__(self):
        extend = self.extend
        sieve = self.sieve
        maxprime = sieve[-1]
        next = iter(sieve).next
        while True:
            prime = next()
            yield prime
            if prime == maxprime:
                extend()
                maxprime = sieve[-1]

# the attribute
primes = Primes()

def prime_factors(value):
    """
    Return the ordered sequence of the prime factors of non-negative
    integer 'value'.

    See also factors().

    Look at some factorizations
    >>> for i in xrange(20):
    ...   i, prime_factors(i)
    (0, [0])
    (1, [1])
    (2, [2])
    (3, [3])
    (4, [2, 2])
    (5, [5])
    (6, [2, 3])
    (7, [7])
    (8, [2, 2, 2])
    (9, [3, 3])
    (10, [2, 5])
    (11, [11])
    (12, [2, 2, 3])
    (13, [13])
    (14, [2, 7])
    (15, [3, 5])
    (16, [2, 2, 2, 2])
    (17, [17])
    (18, [2, 3, 3])
    (19, [19])

    Verify, for a few factorizations, that the product of the returned
    items equals the argument
    >>> tuple((i, prime_factors(i)) for i in xrange(0, 1000) if reduce(mul, prime_factors(i)) != i)
    ()

    Errors
    >>> prime_factors(-1)
    Traceback (most recent call last):
    ...
    ValueError: expected a non-negative number, got -1

    >>> prime_factors('a')
    Traceback (most recent call last):
    ...
    TypeError: expected positive 'int', got 'str'
    """

    checkutils.check_instance(int, value, 'positive ')
    checkutils.check_nonnegative(value)

    if value < 4:
        return [value]

    ret = list()
    for prime in primes:
        if value == 1:
            assert len(ret) > 0
            return ret
        while value % prime == 0:
            ret.append(prime)
            value //= prime

def factors(value):
    """
    Return the ordered sequence of the factors of positive integer 'value'.

    See also prime_factors().

    >>> factors(720)
    (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 144, 180, 240, 360, 720)

    >>> for i in xrange(1, 13):
    ...   i, factors(i)
    (1, (1,))
    (2, (1, 2))
    (3, (1, 3))
    (4, (1, 2, 4))
    (5, (1, 5))
    (6, (1, 2, 3, 6))
    (7, (1, 7))
    (8, (1, 2, 4, 8))
    (9, (1, 3, 9))
    (10, (1, 2, 5, 10))
    (11, (1, 11))
    (12, (1, 2, 3, 4, 6, 12))

    >>> big = 2 * 2 * 3 * 3 * 5 * 5 * 7 * 7 * 11 * 11 * 13 * 13
    >>> big
    901800900
    >>> f = factors(big)
    >>> len(f)
    729
    
    One invariant of the factors
    >>> tuple((f1, f2, f1*f2) for f1, f2 in izip(f, reversed(f)) if f1 * f2 != big)
    ()
    """

    # get the prime factors, also does argument checking
    primes = prime_factors(value)
    checkutils.check_positive(value)

    # make a histogram of the prime counts
    prime_counts = defaultdict(int)
    for prime in primes:
        prime_counts[prime] += 1

    # make a discrete space of the factors that each prime can contribute
    factor_sets = list()
    for prime, count in prime_counts.iteritems():
        factors = [1]
        for _ in xrange(count):
            factors.append(factors[-1] * prime)
        factor_sets.append(tuple(factors))
    
    # do the combinatorics on the factors
    multiplyem = partial(reduce, mul)
    factors = set(multiplyem(factorpoint) for factorpoint in discrete.iterspace(factor_sets))

    ret = tuple(sorted(factors))
    assert ret[0] == 1 and ret[-1] == value
    return ret


def quiet_log(item):
    """
    Return a copy of item that's been transformed to the log-domain, but do the
    operation quietly even if there are zeros in the item.
    """
    with numpy.errstate(divide='ignore'):
        ret = numpy.log(item)
    return ret

LOG_ZERO = quiet_log(0.0)

def log_zeros(shape, dtype=float):
    """
    Produce an array with the given shape filled with log-domain 0 values,
    i.e., -Inf.
    """
    ret = numpy.zeros(shape, dtype) + LOG_ZERO
    return ret


def logsumexp_array(x, axis=None):
    """
    Compute log(sum(exp(x))) along a particular axis for Numpy arrays.  
    """
    # This implementation hasn't been tested on arrays with dimension > 2!
    if axis is None:
        max_ent = x.max()
        bias = max_ent
    else:
        max_ent = x.max(axis)
        bias = max_ent if axis == 0 else max_ent[:, numpy.newaxis]

    # In the no-axis case, if -Inf is the max value, it means *all* the entries
    # were -Inf, so we already know what's going to happen and we can skip doing
    # real work.
    if axis is None and max_ent == LOG_ZERO:
        return LOG_ZERO
        
    # Otherwise, there's a bit of trickiness here - subtracting -Inf results in
    # Nan, so we never want to use that as a bias.
    mask = (max_ent == LOG_ZERO)
    if mask.any():
        # If some rows or columns have only -Inf values, use a bias of 0 in just
        # those rows or cols.
        numpy.place(max_ent, mask, 0.0)
        numpy.place(bias, bias == LOG_ZERO, 0.0)
    return max_ent + quiet_log(numpy.sum(numpy.exp(x - bias), axis=axis))


def logsumexp(x, y):
    """
    Compute log(exp(x) + exp(y))) where x and y are either scalars, Numpy
    arrays of the same shape, or Numpy arrays that can be projected into the
    same shape.
    """
    max_ent = numpy.maximum(x, y)
    mask = (max_ent == LOG_ZERO)
    # If entries have -Inf values for both operands, use a bias of 0 in those
    # entries
    if mask.any():
        numpy.place(max_ent, mask, 0.0)
    return max_ent + quiet_log(numpy.exp(x - max_ent) + numpy.exp(y - max_ent))


def log_dot(a, b):
    """
    Compute the dot product of two 2-dimensional Numpy arrays in the log domain.
    """
    i,x = a.shape
    y,j = b.shape
    if x != y:
        raise ValueError("matrices are not aligned")

    # OK to use real zeros here since we're going to overwrite each entry
    ret = numpy.zeros((i,j), dtype=float)
    for row in xrange(i):
        for col in xrange(j):
            prod = (a[row,:] + b[:,col])
            ret[row,col] = logsumexp_array(prod)
    return ret


def safe_log_divide(x, y):
    """
    Compute x - y where x and y are either scalars, Numpy arrays of the same
    shape, or Numpy arrays that can be projected into the same shape, and make
    sure that (-Inf) - (-Inf) is handled correctly and quietly.  Complain if the
    second operand is -Inf but the first is not.

    >>> zero = quiet_log(0.0)
    >>> result = safe_log_divide(zero, 1.0)
    >>> result == zero
    True
    
    >>> safe_log_divide(1.0, zero)
    Traceback (most recent call last):
    ValueError: log division by zero

    >>> result = safe_log_divide(zero, zero)
    >>> result == zero
    True

    >>> x = numpy.zeros((3,2), dtype=float) - 3.2
    >>> x[:,1] = quiet_log(0.0)
    >>> x
    array([[-3.2, -Inf],
           [-3.2, -Inf],
           [-3.2, -Inf]])

    >>> safe_log_divide(x, x)
    array([[  0., -Inf],
           [  0., -Inf],
           [  0., -Inf]])

    >>> y = numpy.zeros((3,2), dtype=float) - 3.2
    >>> safe_log_divide(x, y)
    array([[  0., -Inf],
           [  0., -Inf],
           [  0., -Inf]])

    >>> safe_log_divide(y, x)
    Traceback (most recent call last):
    ValueError: log division by zero
    
    """
    with numpy.errstate(invalid='ignore'):
        POS_INF = 1.0 - LOG_ZERO
        ret = numpy.subtract(x, y)
    if isinstance(ret, numpy.ndarray):
        if (ret == POS_INF).any():
            dc = dcheck("math_raise")
            dc and dc("Arguments: numerator = \n%s\ndenominator = \n%s" % (x, y))
            raise ValueError("log division by zero")
        numpy.place(ret, numpy.isnan(ret), LOG_ZERO)
    elif numpy.isnan(ret):
        ret = LOG_ZERO
    elif ret == POS_INF:
        dc = dcheck("math_raise")
        dc and dc("Arguments: numerator = \n%s\ndenominator = \n%s" % (x, y))
        raise ValueError("log division by zero")
    return ret


def distance(v1, v2):
    """
    Euclidean distance between two vectors

    >>> orig = numpy.zeros(4)
    >>> one = numpy.ones(4)
    >>> d = distance(orig, one)
    >>> floatutils.float_to_readable_string(d)
    '+(+0001)0x0000000000000'
    >>> floatutils.float_to_readable_string(distance(orig, 2 * one))
    '+(+0002)0x0000000000000'
    """
    assert v1.shape == v2.shape
    assert v1.ndim == v2.ndim == 1
    diff = v1 - v2
    return sqrt(numpy.inner(diff, diff))


def find_cumulative_index(arr, limit):
    """
    Find the index of the first element in an iterable of values whose
    cumulative value is at or above a given limit

    Returns the index, or None if the total of the entire vector is greater than
    the limit.
    
    >>> arr = array((1, 2, 3, 4))
    >>> find_cumulative_index(arr, 0)
    0
    >>> find_cumulative_index(arr, 3)
    1
    >>> find_cumulative_index(arr, 7)
    3
    >>> find_cumulative_index(arr, 11) is None
    True
    
    """
    result = None
    cum = 0
    for i, value in enumerate(arr):
        cum += value
        if cum >= limit:
            result = i
            break
    return result

            


usage = '''
usage:  mathutils.py max

print prime numbers up to max
'''

def main(args):

    arg0 = args[0]
    if arg0 in ('--help', '-h', '/?'):
        print usage
        return

    max = int(arg0)
    next = iter(Primes()).next
    while True:
        p = next()
        if p > max:
            break
        print p
    
if __name__ == '__main__':

    import sys
    args = sys.argv[1:]

    if not args:
        from onyx import onyx_mainstartup
        onyx_mainstartup()
    else:
        main(args)



