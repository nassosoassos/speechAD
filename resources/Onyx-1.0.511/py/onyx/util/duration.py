###########################################################################
#
# File:         duration.py (./py/onyx/util)
# Date:         5-Jan-2009
# Author:       Hugh Secker-Walker
# Description:  Utilities for getting duration strings
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
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
Utilities for working with floating point durations to reproducibly get string
values.
"""
from __future__ import division

def time_usec_sec_str(duration_sec):
    """
    Format a duration (given as a number in seconds) into strings.  Return a
    pair of strings where the first string gives the duration in microseconds
    and the second string gives the duration as decimal seconds with exactly six
    digits following the decimal point.  Avoids floating-point formatting issues
    by only using integer arithmetic.

    >>> duration = 2478 / 1000000  # 2478 microseconds
    >>> time_usec_sec_str(duration)
    ('2478', '0.002478')
    >>> duration = 2478002478 / 100000  # 24780.02478 microseconds
    >>> time_usec_sec_str(duration)
    ('24780024780', '24780.024780')
    """

    # format time without using floating point formatting such as %g
    USEC_PER_SEC = 1000000
    duration_usec = long(duration_sec * USEC_PER_SEC + 0.5)
    duration_usec_str = '%d' % (duration_usec,)
    duration_digits = list('%07d' % (duration_usec,))
    duration_digits.insert(-6, '.')
    duration_sec_str = ''.join(duration_digits)
    return duration_usec_str, duration_sec_str

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
