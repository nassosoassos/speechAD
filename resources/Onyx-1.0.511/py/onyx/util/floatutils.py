###########################################################################
#
# File:         floatutils.py (directory: ./py/onyx/util)
# Date:         Wed 1 Oct 2008 13:15
# Author:       Ken Basye
# Description:  Python side of a wrapper for the floatutils library
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
    Simple routines to exactly serialize floats as strings in a readable
    fashion.  This wrapper also exposes the C99 fpclassify macro and the
    manifest constants it returns.

    Examples:
    
    >>> pi = 3.14159
    >>> s1 = float_to_readable_string(-100*pi)
    >>> s1
    '-(+0008)0x3a28b43958106'
    >>> pi0 = readable_string_to_float(s1)
    >>> pi0 == -100*pi
    True

    >>> fpclassify(pi0) == FP_NORMAL
    True
    >>> fpclassify(0.0) == FP_NORMAL
    False
    >>> fpclassify(0.0) == FP_ZERO
    True
    >>> fpclassify(0) == FP_ZERO
    True
    >>> # trying to use mathutils.quiet_log() introduces an import circularity,
    >>> # so we do the work directly
    >>> with numpy.errstate(divide='ignore'):
    ...   inf = numpy.log(0.0)
    >>> fpclassify(inf) == FP_INFINITE
    True
    >>> with numpy.errstate(invalid='ignore'):
    ...    nan = numpy.divide(inf, inf)
    >>> fpclassify(nan) == FP_NAN
    True
    >>> subnorm_readable = '+(-1023)0x0000000000010'
    >>> subnorm = readable_string_to_float(subnorm_readable)

    >>> fpclassify(subnorm) == FP_SUBNORMAL
    True

    >>> fpclassify("bad_arg")
    Traceback (most recent call last):
    ...
    ValueError: unable to classify object of type <type 'str'>

    >>> float_to_readable_string(subnorm)
    '+(-1023)0x0000000000010'
    >>> float_to_readable_string(inf)    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: unable to generate readable string for infinite value ...
    >>> float_to_readable_string(nan)    # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: unable to generate readable string for NaN value ...
    
    
    >>> p0 = readable_string_to_float(s1[1:])
    Traceback (most recent call last):
    ...
    ValueError: unable to decode string (+0008)0x3a28b43958106  --  bad string length, expected 23 characters but got 22


    >>> zero0 = 0.0
    >>> s = float_to_hex_string(pi)
    >>> s
    '6e861bf0f9210940'
    >>> s2 = float_to_hex_string(zero0)
    >>> s2
    '0000000000000000'
    >>> pi1 = hex_string_to_float(s)
    >>> pi1
    3.1415899999999999
    >>> pi == pi1
    True
    >>> zero = hex_string_to_float(s2)
    >>> zero
    0.0
    >>> zero == zero0
    True

"""

from __future__ import with_statement
import sys
import struct
import binascii
import numpy

# import onyx so that sys.path gets tweaked to find the built shared objects;
# this permits running this script's doctests stand-alone, e.g. from emacs
import onyx
# this refers to the _floatutils.so component of this library
import _floatutils

from _floatutils import FP_NORMAL, FP_ZERO, FP_INFINITE, FP_NAN, FP_SUBNORMAL

def float_to_readable_string(f):
    if not (fpclassify(f) == FP_NORMAL or fpclassify(f) == FP_ZERO or fpclassify(f) == FP_SUBNORMAL):
        f_is_nan = (fpclassify(f) == FP_NAN)
        f_is_inf = (fpclassify(f) == FP_INFINITE)
        assert f_is_nan is not f_is_inf # i.e., xor, since exactly one of these better be True
        f_class = "NaN" if f_is_nan else "infinite"
        raise ValueError("unable to generate readable string for %s value %s" % (f_class, f))
    return _floatutils.float_to_readable_string(f)

def readable_string_to_float(s):
    try:
        import sys
        result = _floatutils.readable_string_to_float(s)
    except ValueError, v:
        problem = _decode_failure_helper(s)
        raise ValueError(("unable to decode string %s  --  " % (s,)) + problem)
    return result

# Try to figure out what went wrong in decoding.
def _decode_failure_helper(s):
    if not len(s) == 23:
        return "bad string length, expected 23 characters but got %d" % (len(s),)
    pm = s[0]
    if not pm == '+' or pm == '-':
        return "bad sign, expected '+' or '-' but got %s" % (pm,)
    if not s[1] == '(' and s[7] == ')':
        return "bad exponent format, expected exponent in '()'s but got %s and %s" % (s[1], s[7])
    e = s[2:7]
    try:
        exponent = int(e, 10) + 1023
    except ValueError, ve:
        return "bad exponent, expected decimal string, but got %s" % e

    if not 0 <= exponent < 0x7ff:
        return "bad exponent value, expected exponent between 0 and 0x7ff but got %x" % exponent
        
    m = s[8:]
    if not m[:2] == '0x':
        return "bad mantissa format, expected leading '0x' but got %s" % m[:2]

    try:
        mantissa = int(m, 16)
    except ValueError, ve:
        return "bad mantissa, expected hexadecimal string, but got %s" % e

def fpclassify(f):
    if not (isinstance(f, float) or isinstance(f, int)):
        raise ValueError("unable to classify object of type %s" % (type(f),))
    return _floatutils.fpclassify(f)

def float_to_hex_string(f):
    return binascii.hexlify(struct.pack('d', f))

def hex_string_to_float(s):
    # unpack returns a list 
    return struct.unpack('d', binascii.unhexlify(s))[0]


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



