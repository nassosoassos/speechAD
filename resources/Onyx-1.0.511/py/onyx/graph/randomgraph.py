###########################################################################
#
# File:         randomgraph.py (directory: ./py/onyx/graph)
# Date:         19-Jun-2009
# Author:       Hugh Secker-Walker
# Description:  Tools for working with random graphs
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
This module has tools that support the generation of random graphs.
"""
from __future__ import division
import itertools
import numpy

def make_ErdosRenyi(num_nodes, edge_prob, seed=None):
    """
    Creates an Erdos-Renyi random graph with *num_nodes* nodes and independent
    *edge_prob* probability of presence in the graph for each of the binomial
    "*num_nodes* choose two" possible edges in the graph.

    Returns a Numpy array of integers with shape (2, num_edges) specifying the
    resulting random set of edges of the graph.  Each pair of integers on the
    first axis specifies an edge, where each integer is in the inclusive range
    (0, *num_nodes* - 1).  This shape is suitable for use in the constructors of
    sparse matrices in the scipy.sparse package.

    If optional *seed* is anything other than None then the random results will be
    repeatable.

    >>> make_ErdosRenyi(20, 0.05, seed=2)
    array([[ 0,  3,  3,  4,  4,  5,  8,  9, 10, 10, 11, 13, 17, 18],
           [ 2,  5, 17, 12, 16, 12, 11, 13, 13, 17, 14, 17, 19, 19]])

    There are many interesting properties of Erdos-Renyi graphs.  Two of them
    are:

    #. the almost sure probability that an ER graph will have no isolated
       nodes if *edge_prob > log(num_nodes) / num_nodes*

    #. the almost sure probability that an ER graph will have some isolated nodes if
       *edge_prob < log(num_nodes) / num_nodes*.

    >>> num_nodes = 1000
    >>> import math
    >>> isolated_threshold = math.log(num_nodes) / num_nodes
    >>> isolated_threshold #doctest: +ELLIPSIS
    0.0069...

    An *edge_prob* greater than the threshold gives a high probability of zero
    isolated nodes

    >>> er1 = make_ErdosRenyi(num_nodes, 1.125 * isolated_threshold, seed=0)
    >>> er1.shape[-1]
    3918
    >>> len(set(xrange(num_nodes)) - set(er1.flat))
    0

    An *edge_prob* less than the threshold gives a high probability of
    some isolated nodes

    >>> er2 = make_ErdosRenyi(num_nodes, 0.872 * isolated_threshold, seed=0)
    >>> er2.shape[-1]
    3054
    >>> len(set(xrange(num_nodes)) - set(er2.flat))
    2
    """
    num_nodes_1 = num_nodes - 1
    num_edges = (num_nodes * num_nodes_1) // 2

    # generate the pairs of edges
    # XXX there must be an all-Numpy way to generate the indices

    # XXX another approach would be to set the number of edges by sampling
    # once from the distribution of number of edges in a G(n,p) graph, and
    # then randomly constructing that many distinct edges

    # brute-force O(n^2) Python-code generator
    #edge_gen = ((x, y) for x in xrange(num_nodes_1) for y in xrange(x + 1, num_nodes))

    # O(n) in Python, O(n^2) in C
    from_gen = (itertools.repeat(i1, num_nodes_1 - i1) for i1 in xrange(num_nodes_1))
    to_gen = (xrange(i2, num_nodes) for i2 in xrange(1, num_nodes))
    #edge_gen = itertools.izip(itertools.chain(*from_gen), itertools.chain(*to_gen))

    edges = numpy.fromiter(itertools.chain(itertools.chain(*from_gen), itertools.chain(*to_gen)), dtype=numpy.int, count=2*num_edges)
    edges.shape = 2, num_edges

    random = numpy.random if seed is None else numpy.random.RandomState(seed)
    selector = random.random_sample(num_edges) <= edge_prob
    er = numpy.compress(selector, edges, axis=-1)

    return er


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
