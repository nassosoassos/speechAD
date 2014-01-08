###########################################################################
#
# File:         safediv.py
# Date:         Fri 1 Aug 2008 09:51
# Author:       Ken Basye
# Description:  Utility routine for doing safe division of numpy arrays where the denominator array may have some zeros
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008 The Johns Hopkins University
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
Utility routine for doing safe division of Numpy arrays

"""

from __future__ import with_statement
import numpy


def safely_divide_float_array(numerator, denom):
    """
    Divide (element-wise) one Numpy array of dtype float by another, suppressing warnings about
    division by 0.0.  Raises a ValueError in case there is any 0.0 in the denominator whose
    corresponding value in the numerator is not also 0.0.  Converts all NaNs resulting from 0.0/0.0
    operations to 0.0.

    >>> n = numpy.zeros((2,2), dtype=float)
    >>> d = numpy.zeros((2,2), dtype=float)
    >>> r = safely_divide_float_array(n,d)
    >>> r
    array([[ 0.,  0.],
           [ 0.,  0.]])

    >>> n[0,0] = 1.0
    >>> r = safely_divide_float_array(n,d)
    Traceback (most recent call last):
      ...
    ValueError: Unable to safely divide; denominator array has a 0.0 where numerator array has a non-zero value
    >>> d[0,0] = 10.0
    >>> r = safely_divide_float_array(n,d)
    >>> r
    array([[ 0.1,  0. ],
           [ 0. ,  0. ]])

    
    """
    # The following condition ensures that we only divide by 0.0 when the numerator is also 0.0.
    # Note that these logical operations are all ufuncs, so they apply across arrays. The all()
    # function at the end collapses to a single boolean value
    if not ( ((numerator == 0.0) | (denom != 0.0)).all() ):
        raise ValueError("Unable to safely divide; denominator array has a 0.0 where numerator array has a non-zero value")

    # Note that in Numpy, FP division by 0.0 raises the "invalid" error, whereas int division by 0
    # raises the "divide" error.
    with numpy.errstate(invalid='ignore'):
        x = numpy.nan_to_num(numerator / denom)
    return x



if __name__ == '__main__':
    
    from onyx import onyx_mainstartup
    onyx_mainstartup()

