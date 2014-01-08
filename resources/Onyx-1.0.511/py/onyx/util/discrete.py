###########################################################################
#
# File:         discrete.py
# Date:         9-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Discrete space tools
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
Tools for working with discrete spaces.

A discrete space is a sequence of sequences of immutable items.  It
is often called a feature space.

"""

from onyx.builtin import frozentuple

def iterspace(features, sortem=False):
    """
    Returns a generator that yields each of the points defined by the
    discrete space 'features'.  The 'rightmost' feature is iterated
    most rapidly.  There is no requirement that the items in a feature
    be mutually unique.

    If optional 'sortem' is True, the items in each feature will be
    sorted before the space is enumerated.

    >>> features = (('b', 'a'), (1, 2, 1), ('p', 'n'))
    >>> tuple(iterspace(features))
    (('b', 1, 'p'), ('b', 1, 'n'), ('b', 2, 'p'), ('b', 2, 'n'), ('b', 1, 'p'), ('b', 1, 'n'), ('a', 1, 'p'), ('a', 1, 'n'), ('a', 2, 'p'), ('a', 2, 'n'), ('a', 1, 'p'), ('a', 1, 'n'))

    >>> tuple(iterspace(((3, 1, 2), ('z', 'a')), True))
    ((1, 'a'), (1, 'z'), (2, 'a'), (2, 'z'), (3, 'a'), (3, 'z'))
    """

    # sorted or as is; use frozentuple to verify immutability of feature values
    collect = sorted if sortem else tuple
    features = tuple(frozentuple(collect(feature)) for feature in features)

    ndim = len(features)

    # bootstrap the combinatoric work with the iterator's next for the first dimension
    iterstack = [iter(features[0]).next]
    valstack = [None]
    while True:
        try:
            # advance the last element; can raise StopIteration or IndexError 
            valstack[-1] = iterstack[-1]()
            # fill out remaining dimensions of the iterators and the values
            while len(iterstack) < ndim:
                feature_iter = iter(features[len(iterstack)]).next
                iterstack.append(feature_iter)
                valstack.append(feature_iter())
            yield tuple(valstack)
        except StopIteration:
            # drop back so lower dimension gets advanced
            iterstack.pop()
            valstack.pop()
        except IndexError:
            return

class DiscreteScalarModel(object):
    """
    >>> dsm = DiscreteScalarModel(256)
    >>> for i in xrange(23, 33): dsm.sample(i)
    >>> for i in xrange(23, 43): dsm.sample(i)
    >>> dsm.score(23)
    28
    >>> dsm.score(33)
    29
    """
    def __init__(self, range1, range2=None):
        self.hist = list(0 for i in xrange(range1))
        self.hcount = 0
        self.count = 0

    def score(self, value):
        return self.hcount - self.hist[value]

    def sample(self, value):
        self.count += 1
        self.hcount += 1
        self.hist[value] += 1
        


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
