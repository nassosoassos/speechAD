###########################################################################
#
# File:         nonparametric.py
# Date:         12-Jul-2008
# Author:       Hugh Secker-Walker
# Description:  Logarithmic size non-parametric modeling of unbounded time series
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
The idea here is that you have an unbounded stream of discrete-space
samples.  You are interested in statistics and/or models of the time
series at many time scales.  This module supports the idea of keeping
a number of statistics (sufficient for models) taking up space
proportional to the logarithm of the number of samples seen, and
covering exponentially larger intervals of the past.
"""

from itertools import izip
class NonParametricAccumulator(object):
    """
    Logarithm-space accumulator for non-parametric discrete models.

    >>> a = NonParametricAccumulator()
    >>> a.accum_many(xrange(5))
    >>> a.accum_many(xrange(10))
    >>> a.epochs
    [[(8, 64), (9, 81)], [(13, 85)], [(14, 54)], [(11, 31)]]

    """

    def __init__(self):
        # index into this list is the log base2 of the item count
        # associated with the statistics at that index
        self.epochs = list()
        self.extend()

    def extend(self):
        extension = list()
        extension.append(self.zero_stats())        
        self.epochs.append(extension)

    def zero_stats(self):
        return 0, 0
    def linear_stats(self, datum):
        return datum, datum * datum
    def combine_stats(self, stats_iterable):
        sum0, sum1 = self.zero_stats()
        for s0, s1 in stats_iterable:
            sum0 += s0
            sum1 += s1
        return sum0, sum1

    def accum_one(self, datum):
        linear_stats = self.linear_stats(datum)
        epochs = self.epochs
        epochs[0].append(linear_stats)

        delayed = iter(epochs)
        delayed.next()
        for current, next in izip(epochs, delayed):
            assert 0 < len(current) <= 3
            if len(current) == 3:
                next.append(self.combine_stats(current[:2]))
                del current[:2]
            else:
                break
            
        current = epochs[-1]
            
        assert 0 < len(current) <= 3
        if len(current) == 3:
            extension = list()
            extension.append(self.combine_stats(current[:2]))
            del current[:2]
            epochs.append(extension)

    def accum_many(self, iterable):
        accum_one = self.accum_one
        for item in iterable:
            accum_one(item)

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
