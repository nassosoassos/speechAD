###########################################################################
#
# File:         iterutils.py (directory: ./py/onyx/util)
# Date:         15-Jul-2009
# Author:       Hugh Secker-Walker
# Description:  Tools for working with iterables and generators
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
The objects in this module provide for iterator-based filtering and other stream
operations.

>>> True
True
"""
from __future__ import with_statement
import os
import csv
import operator


def iter_itemgetter(iterable, items):
    """
    Return a generator that constructs and yields a tuple of indexed items from
    each element of the iterable.

    The iterable must yield an item that is indexable by each of the elements of
    items (a finite, iterable sequence).  The returned generator will yield a
    tuple of the indexed elements.

    This function is typically used to select a subset or a permutation of the
    items in each item from iterable.

    Example, showing that the indexing is general enough to use on an iterable
    stream of dicts, and that an indexing item can be repeated
    
    >>> stream = (dict((('a', x), ('b', 3 * x), ('c', 4 * x + 3))) for x in xrange(5))
    >>> tuple(iter_itemgetter(stream, ('c', 'a', 'a')))
    ((3, 0, 0), (7, 1, 1), (11, 2, 2), (15, 3, 3), (19, 4, 4))
    """
    getter = operator.itemgetter(*items)
    for item in iterable:
        yield getter(item)


def numerify(item):
    """
    If item is a string, try to return the int or the float interpretation item,
    otherwise return the item.  If item is not a string, return the item

    >>> numerify('+30')
    30
    >>> numerify('-30.')
    -30.0
    >>> numerify(30.5)
    30.5
    >>> numerify(True)
    True
    >>> numerify(())
    ()
    """
    if isinstance(item, str):
        types = int, float
        for typ in types:
            try:
                return typ(item)
            except ValueError:
                pass
    return item

def iter_numerify(iterable):
    """
    Returns a generator that, when possible, converts sequences of
    strings from iterable into sequences of ints or floats.

    Each item from iterable must be a sequence.  Each string item in the
    sequence is converted to an int if possible, or if that fails, a float, or
    if that fails, is not converted.  A tuple of the results is yielded.

    >>> seq = ['20', 'foo'], ['+30.', ()], ['-50x', None, True, False, 'foobar']
    >>> tuple(iter_numerify(seq))
    ((20, 'foo'), (30.0, ()), ('-50x', None, True, False, 'foobar'))
    """
    for item in iterable:
        yield tuple(numerify(item2) for item2 in item)


def csv_itemgetter(csv_iterable, items):
    """
    Return a generator that will yield a list of the selected items from each
    record of csv_iterable which must yield strings that are in CSV format, the
    first of which must be the header line which names all the possible items.

    >>> import cStringIO
    >>> csv_file = cStringIO.StringIO(
    ... '''name,date,amount
    ... foo,yesterday,+20
    ... bar,today,30
    ... baz,tomorrow,-50
    ... bat,tomorrow etc.,-50.0
    ... 0,0,0
    ... ''')
    >>> items = 'amount', 'name'
    >>> tuple(csv_itemgetter(csv_file, items))
    ((20, 'foo'), (30, 'bar'), (-50, 'baz'), (-50.0, 'bat'), (0, 0))
    """
    return iter_numerify(iter_itemgetter(csv.DictReader(csv_iterable), items))


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
