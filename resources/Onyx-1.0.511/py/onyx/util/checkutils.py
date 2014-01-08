###########################################################################
#
# File:         checkutils.py (directory: ./py/onyx/util)
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Utilities for verification and checking
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
Utilities for error checking.
"""
from functools import partial

def check_instance(typ, value, qualifier=''):
    """
    Check that 'value' is an instance of 'typ'.  Raises TypeError if
    the check fails.  Optional 'qualifier' is inserted into error
    message immediately preceding the name of the expected type.
    Note: to support use of functools.partial the argument order is
    the opposite of isinstance().

    >>> check_instance(list, list())

    >>> check_instance(list, tuple(), 'non-empty ')
    Traceback (most recent call last):
    ...
    TypeError: expected non-empty 'list', got 'tuple'

    >>> checkint = partial(check_instance, int)

    >>> for i in range(5): checkint(i)

    >>> for i in range(5): checkint(chr(i))
    Traceback (most recent call last):
    ...
    TypeError: expected 'int', got 'str'

    """

    if not isinstance(value, typ):
        raise TypeError("expected %s%r, got %r" % (qualifier, typ.__name__, type(value).__name__,))


def check_positive(value):
    """
    Check that 'value' is positive.  Raises ValueError if it isn't.

    >>> check_positive(1.0)

    >>> check_positive(-3)
    Traceback (most recent call last):
    ...
    ValueError: expected a positive number, got -3

    """

    if value <= 0:
        raise ValueError("expected a positive number, got %s" % (value,))


def check_nonnegative(value):
    """
    Check that 'value' is non-negative.  Raises ValueError if it isn't.

    >>> check_nonnegative(1.0)
    >>> check_nonnegative(0)

    >>> check_nonnegative(-3)
    Traceback (most recent call last):
    ...
    ValueError: expected a non-negative number, got -3

    """

    if value < 0:
        raise ValueError("expected a non-negative number, got %s" % (value,))


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
