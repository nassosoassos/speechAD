###########################################################################
#
# File:         __init__.py (package: onyx.containers)
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Container utilities
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
Container utilities
"""

from itertools import repeat
from onyx.builtin import frozendict

def tuplen(length, initializer=None):
    """
    Return a new tuple with length elements each set to initializer (default None).

    >>> tuplen(3)
    (None, None, None)

    >>> tuplen(4, 0)
    (0, 0, 0, 0)
    """
    return tuple(repeat(initializer, length))

def listn(length, initializer=None):
    """
    Return a new list with length elements each set to initializer (default None).

    >>> listn(3)
    [None, None, None]

    >>> listn(4, 0)
    [0, 0, 0, 0]
    """
    return list(repeat(initializer, length))

def tuplenoflist(length):
    """
    Return a tuple containing length new, empty lists.

    >>> tuplenoflist(3)
    ([], [], [])
    """
    return tuple(list() for x in xrange(length))

def sorteduniquetuple(iterable):
    """
    Return a tuple of the sorted set of unique items it iterable.  Each item in
    iterable must be immutable, otherwise you will get a TypeError about an
    unhashable object.

    Examples

    >>> sorteduniquetuple('hello world')
    (' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w')
    >>> sorteduniquetuple(['world', 'wide', 'hello', 'world'])
    ('hello', 'wide', 'world')
    
    Example of error on mutable items

    >>> sorteduniquetuple(['world', 'wide', 'hello', 'world', ['this item is a mutable list']]) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
        sorteduniquetuple(['world', 'wide', 'hello', 'world', ['this item is a mutable list']])
      File "<stdin>", line ###, in sorteduniquetuple
    TypeError: ...unhashable...

    """
    return tuple(sorted(set(iterable)))

def frozenbijection(iterable):
    """
    Return a bijective pair, (by_id, id_by), where by_id is a sorted tuple of
    the unique items in iterable, and id_by is a frozendict mapping each item to
    its id.  Each item in iterable must be immutable, otherwise you will get a
    TypeError about an unhashable object.

    Examples

    >>> frozenbijection('hello world')
    ((' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'), frozendict({' ': 0, 'e': 2, 'd': 1, 'h': 3, 'l': 4, 'o': 5, 'r': 6, 'w': 7}))
    >>> frozenbijection(['world', 'wide', 'hello', 'world'])
    (('hello', 'wide', 'world'), frozendict({'wide': 1, 'hello': 0, 'world': 2}))

    Example of error on mutable items

    >>> frozenbijection(['world', 'wide', 'hello', 'world', ['this item is a mutable list']]) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
        frozenbijection(('world', 'wide', 'hello', 'world', ['this item is a mutable list']))
      File "<stdin>", line ###, in frozenbijection
      File "<stdin>", line ###, in sorteduniquetuple      
    TypeError: ...unhashable...
    
    """
    by_id = sorteduniquetuple(iterable)
    id_by = frozendict((item, id) for id, item in enumerate(by_id))
    return by_id, id_by

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
