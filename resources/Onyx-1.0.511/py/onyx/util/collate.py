###########################################################################
#
# File:         collate.py
# Date:         2-Dec-2007
# Author:       Hugh Secker-Walker
# Description:  Simple collating utilities
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
Tools for simple collation tasks.
"""

from collections import defaultdict
from onyx.builtin import frozendict

def collate_pairs(iterable):
    """
    Collate the iterable sequence of pairs, ((key, value), (key,
    value), ...).  Returns a frozendict in which each key maps to the
    sequence of values that appeared with that key in the iteration.
    The collation is stable; that is, the order of the values in the
    list for a given key is the order in which those values appeared
    in the iteration.

    >>> col = collate_pairs(((1, 2), (2, 0), (1, 1), (1, 2), ('a', 'b'), ('a', None)))
    >>> for key in sorted(col.keys()): print key, col[key]
    1 (2, 1, 2)
    2 (0,)
    a ('b', None)
    """
    collator = defaultdict(list)
    for key, value in iterable:
        collator[key].append(value)
    return frozendict((key, tuple(seq)) for key, seq in collator.iteritems())

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
