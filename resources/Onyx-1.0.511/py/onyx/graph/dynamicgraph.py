###########################################################################
#
# File:         dynamicgraph.py
# Date:         12-Apr-2009
# Author:       Hugh Secker-Walker
# Description:  Work with online, undirected graphs
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
Tools for working with dynamic, or online, graphs.

These graphs have well-defined semantics that they maintain as edges are added
and removed from the graph.

At present, UndirectedOnlineInvariantsGraph is the only such dynamic graph.
"""
from __future__ import with_statement, division
import os
import collections
import itertools
from onyx import builtin
from onyx.util import iterutils
from onyx.graph import randomgraph

class UndirectedOnlineInvariantsGraph(object):
    """
    An undirected graph that maintains a set of invariant features as edges are
    added and removed.

    Create a graph and look at its initial invariant feature set
    
    >>> g = UndirectedOnlineInvariantsGraph()
    >>> g.invariants
    {'mad_lower_k': 0, 'max_degree': 0, 'num_triangles': 0, 'order': 0, 'scan1': 0, 'size': 0}

    Add some edges
    
    >>> for edge in ((0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (1, 'x')):
    ...   _ = g.add_edge(edge)

    Look at the features

    >>> g.invariants
    {'mad_lower_k': 2400, 'max_degree': 3, 'num_triangles': 1, 'order': 6, 'scan1': 4, 'size': 7}

    Add more edges, looking at what gets returned: the normalized edge, and the
    multigraph count

    >>> g.add_edge((0, 3))
    ((0, 3), 1)

    The returned count is one, meaning the edge was a new edge, so the set of
    features has changed.  We see that we've changed most of the features; we
    haven't changed the order because we didn't add a new node.

    >>> g.invariants
    {'mad_lower_k': 2800, 'max_degree': 4, 'num_triangles': 3, 'order': 6, 'scan1': 7, 'size': 8}

    Add the edge again.  Since the return count is greater than one, the edge
    was not new to the graph, and so none of the features has changed.

    >>> g.add_edge((3, 0))
    ((0, 3), 2)
    >>> g.invariants
    {'mad_lower_k': 2800, 'max_degree': 4, 'num_triangles': 3, 'order': 6, 'scan1': 7, 'size': 8}

    The incidence matrix

    >>> incidence, nodes = g.incidence_matrix
    >>> incidence
    array([[0, 1, 1, 1, 0, 0],
           [1, 0, 0, 1, 0, 1],
           [1, 0, 0, 1, 1, 0],
           [1, 1, 1, 0, 1, 0],
           [0, 0, 1, 1, 0, 0],
           [0, 1, 0, 0, 0, 0]], dtype=int8)
    >>> nodes
    [0, 1, 2, 3, 4, 'x']

    >>> import numpy
    >>> eigenvalues = numpy.linalg.eigvalsh(incidence)
    >>> eigenvalues[eigenvalues.argmax()] #doctest: +ELLIPSIS
    2.98093658810182...
    >>> eigenvalues.sort()
    >>> ref = numpy.array([ -1.77960786e+00,  -1.53708297e+00,  -7.06200511e-01, -4.93288006e-16,   1.04195475e+00,   2.98093659e+00])
    >>> ref.sort()
    >>> numpy.allclose(eigenvalues, ref)
    True

    Edge containment

    >>> (1, 3) in g
    True
    >>> (3, 1) in g
    True
    >>> (3, 8) in g
    False

    Self-loop edge is OK for checking containment, even though it isn't a valid edge

    >>> trial_edge = 3, 3
    >>> trial_edge in g
    False
    >>> g.add_edge(trial_edge)
    Traceback (most recent call last):
      ...
    ValueError: invalid attempt to use a self loop on node 3

    Remove copies of an edge, and see that the features only change when the
    edge count drops to zero

    >>> e = 0, 3
    >>> g.add_edge(e); g.invariants
    ((0, 3), 3)
    {'mad_lower_k': 2800, 'max_degree': 4, 'num_triangles': 3, 'order': 6, 'scan1': 7, 'size': 8}
    >>> while e in g: g.remove_edge(e); g.invariants
    ((0, 3), 2)
    {'mad_lower_k': 2800, 'max_degree': 4, 'num_triangles': 3, 'order': 6, 'scan1': 7, 'size': 8}
    ((0, 3), 1)
    {'mad_lower_k': 2800, 'max_degree': 4, 'num_triangles': 3, 'order': 6, 'scan1': 7, 'size': 8}
    ((0, 3), 0)
    {'mad_lower_k': 2400, 'max_degree': 3, 'num_triangles': 1, 'order': 6, 'scan1': 4, 'size': 7}

    >>> g.remove_edge(e)
    Traceback (most recent call last):
      ...
    ValueError: invalid attempt to remove a nonexistent edge (0, 3)


    Create a copy of the graph and then structurally change the copy and see
    that its invariant features have changed

    >>> g2 = UndirectedOnlineInvariantsGraph(g)
    >>> g2.invariants == g.invariants
    True
    >>> g2.add_edge(('a', 'b'))
    (('a', 'b'), 1)
    >>> g2.invariants == g.invariants
    False

    >>> g3 = UndirectedOnlineInvariantsGraph('bogus initializer')
    Traceback (most recent call last):
      ...
    TypeError: expected an instance of UndirectedOnlineInvariantsGraph, got a str


    An edge can be any pair of immutables that support a full ordering.  Note
    that the full-ordering requirement means you can't use a frozenset as a
    node.

    >>> g.add_edge(((20, 30), 'alphabet'))
    (('alphabet', (20, 30)), 1)

    Examples of things an edge cannot be
    
    >>> g.add_edge(1)
    Traceback (most recent call last):
      ...
    TypeError: expected edge to be a container with exactly two nodes, got an instance of int

    >>> g.add_edge([1,2,3])
    Traceback (most recent call last):
      ...
    ValueError: expected edge to have exactly 2 nodes, got 3
    
    >>> g.add_edge(([1,2], set()))
    Traceback (most recent call last):
      ...
    TypeError: expected both nodes in the edge to be immutable types, got a list and a set

    >>> g.add_edge(((1,2), frozenset('abc')))
    Traceback (most recent call last):
      ...
    TypeError: expected both nodes in the edge to be immutable, ordered types, got a tuple and a frozenset


    Look at some internals

    >>> g.invariants['scan1'] == g._scan1_online == g._scan1_brute == 4
    True

    >>> g.invariants['mad_lower_k'] == g._mad_lower_k == g._mad_lower_k_brute == 2400
    True

    >>> edges, adjacencies, k1_scans = g._containers
    >>> for key in sorted(edges.keys()): print key, edges[key]
    (0, 1) 1
    (0, 2) 1
    (1, 3) 1
    (1, 'x') 1
    (2, 3) 1
    (2, 4) 1
    (3, 4) 1
    ('alphabet', (20, 30)) 1

    >>> for key in adjacencies.keys(): print repr(key), sorted(adjacencies[key])
    0 [1, 2]
    1 [0, 3, 'x']
    2 [0, 3, 4]
    3 [1, 2, 4]
    4 [2, 3]
    'alphabet' [(20, 30)]
    (20, 30) ['alphabet']
    'x' [1]


    Remove all the edges

    >>> edges = (0, 1), (0, 2), (1, 3), (1, 'x'), (2, 3), (2, 4), (3, 4), ((20, 30), 'alphabet')
    >>> for edge in edges:
    ...   _ = g.remove_edge(edge)
    >>> g.invariants == UndirectedOnlineInvariantsGraph().invariants == {'mad_lower_k': 0, 'max_degree': 0, 'num_triangles': 0, 'order': 0, 'scan1': 0, 'size': 0}
    True

    Make a fully connected graph

    >>> len(edges)
    8
    >>> for e in ((e1, e2) for e1 in edges for e2 in edges if e1 != e2):
    ...   _ = g.add_edge(e), g.invariants
    >>> g.invariants
    {'mad_lower_k': 7000, 'max_degree': 7, 'num_triangles': 56, 'order': 8, 'scan1': 28, 'size': 28}
    """

    _sentinel = object()
    def __new__(cls, source=_sentinel):
        self = super(UndirectedOnlineInvariantsGraph, cls).__new__(cls)
        if source is UndirectedOnlineInvariantsGraph._sentinel:
            edges, adjacent, k1_scans = self._containers = dict(), collections.defaultdict(set), collections.defaultdict(int)
            self._num_triangles = 0
        else:
            if not isinstance(source, cls):
                raise TypeError("expected an instance of %s, got a %s" % (cls.__name__, type(source).__name__))
            edges, adjacent, k1_scans = source._containers
            assert bool(edges) is bool(adjacent) is bool(k1_scans)
            edges, adjacent, k1_scans = self._containers = dict(edges), collections.defaultdict(set, adjacent), collections.defaultdict(int, k1_scans)
            self._num_triangles = source._num_triangles
        assert bool(edges) is bool(adjacent) is bool(k1_scans)
        return self

    def __init__(self, source=None):
        # None means the invariants are dirty
        self._invariants = None

    @property
    def invariants(self):
        """
        This property returns a dict containing the current values for the
        observable invariant features of the graph.
        """
        invariants = self._invariants
        if invariants is None:
            edges, adjacent, k1_scans = self._containers
            assert bool(edges) is bool(adjacent) is bool(k1_scans)
            # optimization, cache the invariants
            invariants = self._invariants = (
                ('size', len(edges)),
                ('order', len(adjacent)),
                ('max_degree', self._max_degree),
                ('num_triangles', self._num_triangles),
                ('scan1', self._scan1_online),
                ('mad_lower_k', self._mad_lower_k),
                )
        return dict(invariants)

    def get_invariants(self):
        """
        Method to get the invariants.  See invariants property.
        """
        return self.invariants

    def add_edge(self, edge):
        """
        Add the edge to the graph, where edge is a pair of immutables that
        satisfy a full-ordering relation.

        Returns a pair, the normlized edge and the updated multigraph count for
        the edge.  The count will be one if the edge is new to the graph.  In
        this case, the stats for the invariant features will have been updated.
        If the count is greater than one, then the edge was previously in the
        graph and the add_edge() call has not altered the invariant features of
        the graph.
        """
        edges, adjacent, k1_scans = self._containers
        assert bool(edges) is bool(adjacent) is bool(k1_scans)
        edge = self._normal_edge(edge)
        count = edges.get(edge)
        if count is None:
            # it's a new edge
            self._invariants = None
            count = 1
            node1, node2 = edge
            # subtle: adjacent is a defaultdict and automatically adds an empty
            # adjacency set for a previously-unseen node
            adj1 = adjacent[node1]
            adj2 = adjacent[node2]
            assert node1 not in adj2 and node2 not in adj1

            # get this before we update adjacencies
            new_triangle_vertices = (adj1 & adj2) if adj1 and adj2 else ()

            # update adjacencies
            adj1.add(node2)
            adj2.add(node1)

            # update the global triangle count
            num_new_triangles = len(new_triangle_vertices)
            self._num_triangles += num_new_triangles

            # update the local scan counts
            scan_inc = num_new_triangles + 1
            k1_scans[node1] += scan_inc
            k1_scans[node2] += scan_inc
            for node in new_triangle_vertices:
                k1_scans[node] += 1

            # there must not be empty adjacencies or non-positive k1_scans
            assert min(len(adj) for adj in adjacent.itervalues()) >= 1
            assert min(k1_scans.itervalues()) >= 1
        else:
            # update count for an existing edge
            count += 1
        edges[edge] = count
        return edge, count

    def remove_edge(self, edge):
        """
        Decrement by one the multigraph count for the edge in the graph.  Raises
        ValueError if the edge is not in the graph.

        Returns a pair, the normlized edge and the updated multigraph count for
        the edge.  The count will be zero if the edge itself has been removed
        from the graph.  In this case, the features will have been updated.
        Otherwise, the count will be positive and the remove_edge() call will
        not have altered the invariants of the graph.
        """
        edges, adjacent, k1_scans = self._containers
        assert bool(edges) is bool(adjacent) is bool(k1_scans)
        edge = self._normal_edge(edge)
        if edge not in edges:
            raise ValueError("invalid attempt to remove a nonexistent edge %r" % (edge,))
        count = edges[edge]
        assert count > 0, str(count)
        if count == 1:
            self._invariants = None
            count = 0
            del edges[edge]

            node1, node2 = edge
            adj1 = adjacent[node1]
            adj2 = adjacent[node2]
            assert node2 in adj1 and node1 in adj2
            adj1.remove(node2)
            adj2.remove(node1)
            if not adj1:
                del adjacent[node1]
            if not adj2:
                del adjacent[node2]

            triangle_vertices = (adj1 & adj2) if adj1 and adj2 else ()
            num_removed_triangles = len(triangle_vertices)
            self._num_triangles -= num_removed_triangles

            def dec_node(node, dec):
                if k1_scans[node] > dec:
                    k1_scans[node] -= dec
                else:
                    assert k1_scans[node] == dec
                    del k1_scans[node]
            scan_dec = num_removed_triangles + 1
            dec_node(node1, scan_dec)
            dec_node(node2, scan_dec)
            for node in triangle_vertices:
                dec_node(node, 1)

            assert bool(edges) is bool(adjacent) is bool(k1_scans)
            if edges:
                # there must not be empty adjacencies or non-positive k1_scans
                assert min(len(adj) for adj in adjacent.itervalues()) >= 1
                assert min(k1_scans.itervalues()) >= 1
        else:
            count -= 1
            edges[edge] = count
        return edge, count

    @property
    def incidence_matrix(self):
        """
        Property returns a two-item tuple.  The first item is a two-dimensional
        Numpy array of zeros and ones, an incidence matrix for the graph.  The
        second item is a sorted sequence of the nodes of the graph, where the
        index of each node corresponds to the row and column for that node in
        the incidence matrix.
        """
        edges, adjacent, k1_scans = self._containers
        nodes = sorted(adjacent)
        node_map = dict((node, index) for index, node in enumerate(nodes))
        import numpy
        size = len(nodes)
        incidence = numpy.zeros((size, size), dtype=numpy.int8)
        for node1, adj in adjacent.iteritems():
            row = incidence[node_map[node1]]
            row[numpy.fromiter((node_map[node2] for node2 in adj), dtype=numpy.int8, count=len(adj))] = 1
        return incidence, nodes
        

    def __contains__(self, edge):
        """
        Implements containment semantics for edges.
        """
        edges, adjacent, k1_scans = self._containers
        assert bool(edges) is bool(adjacent) is bool(k1_scans)
        edge = self._normal_edge(edge, allow_self_loop=True)
        return edge in edges

    @staticmethod
    def _normal_edge(edge, allow_self_loop=False):    
        # type and value checks
        if not hasattr(edge, '__len__'):
            raise TypeError("expected edge to be a container with exactly two nodes, got an instance of %s" % (type(edge).__name__,))
        if len(edge) != 2:
            raise ValueError("expected edge to have exactly 2 nodes, got %d" % (len(edge),))

        node1, node2 = edge = tuple(edge)
        try:
            hash(edge)
        except TypeError:
            raise TypeError("expected both nodes in the edge to be immutable types, got a %s and a %s" % (type(node1).__name__, type(node2).__name__))
        if node1 == node2 and not allow_self_loop:
            raise ValueError("invalid attempt to use a self loop on node %r" %(node1,))

        try:
            node1 >= None and node2 >= None
        except TypeError:
            raise TypeError("expected both nodes in the edge to be immutable, ordered types, got a %s and a %s" % (type(node1).__name__, type(node2).__name__))

        return edge if node1 <= node2 else (node2, node1)

    @property
    def _max_degree(self):
        """
        Property calculates the maximum degree in the graph.
        """
        edges, adjacent, k1_scans = self._containers
        return max(len(adjacency) for adjacency in adjacent.itervalues()) if adjacent else 0

    @property
    def _mad_lower_k(self):
        """
        Return an integer, 1000 times a lower bound on the maximum average degree.
        """
        edges, adjacent, k1_scans = self._containers
        assert bool(adjacent) is bool(edges)

        # find the lower bound on maxiumum average degree (MAD) using Ullman and
        # Scheinerman's greedy approximation

        # XXX wonder if there is an online algorithm to maintain stats such that
        # calculating a lower bound MAD doesn't require the expensive look at a
        # set of induced subgraphs....  Consider maintaining the largest
        # eigenvalue in an online fashion.

        # make a representation of the graph suitable for the greedy mad work
        # - deep copy the adjancency lists
        # - dict of nodes_by_degree, where each item is the set of nodes of that degree
        adjacent2, nodes_by_degree = dict(), collections.defaultdict(set)        
        for node, adj in adjacent.iteritems():
            adjacent2[node] = set(adj)
            degree = len(adj)
            nodes_by_degree[degree].add(node)
        
        def min_degree_gen():
            # yield a lowest-degree item from nodes_by_degree and its degree;
            # note that client moves some items in nodes_by_degree to the
            # next-lowest indexed slot while this generator is running, although
            # client never puts things in slot zero; we don't bother shrinking
            # nodes_by_degree as this would be more work (and more code) than
            # the one time we increment up to len(nodes_by_degree) and decide
            # we're done
            assert not nodes_by_degree[0]
            degree = 1
            while degree < len(nodes_by_degree):
                assert degree >= 1
                items = nodes_by_degree[degree]
                if items:
                    conjecture = False
                    if conjecture:
                        # conjecture: we would make a better expectation
                        # approximation if we removed a minimum-degree node with
                        # the minimum of some statistic, e.g.  nearest-neighbor
                        # degree, or min triangle participation, or min
                        # scan1... more work

                        # a strict upper-bound on degree, k1_scan
                        min_nnd = len(edges) + 1
                        for item in items:
                            # minimum nearest-neighbor degree
                            #nnd = min(len(adjacent2[item2]) for item2 in adjacent2[item])
                            # note: k1_scans is of the original graph!
                            nnd = min(k1_scans[item2] for item2 in adjacent2[item])
                            if nnd < min_nnd:
                                min_nnd = nnd
                                min_item = item
                        item = min_item
                        items.remove(item)
                    else:
                        # just yield one of the minimum-degree nodes
                        item = items.pop()
                    assert len(adjacent2[item]) == degree
                    yield item, degree
                    if degree >= 2:
                        degree -= 1
                else:
                    degree += 1

        num_edges = len(edges)
        max_average_degree = num_edges / len(adjacent2) if num_edges else 0
        for node, degree in min_degree_gen():
            # do the work to remove the node from the graph
            adj = adjacent2[node]
            del adjacent2[node]
            assert len(adj) == degree
            num_edges -= degree
            for anode in adj:
                # update the adjacent nodes
                adj2 = adjacent2[anode]
                degree2 = len(adj2)
                assert degree2 >= degree
                adj2.remove(node)
                nodes_by_degree[degree2].remove(anode)
                if degree2 > 1:
                    nodes_by_degree[degree2-1].add(anode)
                else:
                    del adjacent2[anode]
            # update the max average degree
            if num_edges > 0:
                assert len(adjacent2) > 0
                average_degree = num_edges / len(adjacent2)
                if average_degree > max_average_degree:
                    max_average_degree = average_degree
        assert num_edges == len(adjacent2) == 0

        # both the factor of two (for average degree), and factor of 1000 (for
        # our integer API)
        return int(2000 * max_average_degree + 0.5)

    @property
    def _mad_lower_k_brute(self):
        # inefficient, looks through all nodes for a minimum-degree node

        edges, adjacent, k1_scans = self._containers
        assert bool(adjacent) == bool(edges)
        
        if not edges:
            return 0

        # find the lower bound on MAD using Ullman and Scheinerman's greedy approximation
        #
        # make a copy of the graph
        # loop:
        #   get average degree and update max degree
        #   remove a lowest degree node
        edges = dict(edges)
        adjacent = dict((node, set(adj)) for node, adj in adjacent.iteritems())
        def ad():
            # average degree
            return 2 * len(edges) / len(adjacent)
        def min_degree_node():
            # exhaustive search finding a min degree node
            # XXX use a heap instead
            # XXX even better, store adjacency by degree and just move the
            # affected nodes lower (see _mad_lower_k)
            adj_iter = adjacent.iteritems()
            min_node, min_adj = adj_iter.next()
            min_deg = len(min_adj)
            for node, adj in adj_iter:
                deg = len(adj)
                if deg < min_deg:
                    min_deg = deg
                    min_node = node
            assert min_deg >= 1
            return min_node, min_deg
        max_average_degree = 0
        while edges:
            average_degree = ad()
            if average_degree > max_average_degree:
                max_average_degree = average_degree
            min_node, min_deg = min_degree_node()        
            for adj in adjacent[min_node]:
                assert adj != min_node
                adjacent[adj].remove(min_node)
                if not adjacent[adj]:
                    del adjacent[adj]
                del edges[(adj, min_node) if adj < min_node else (min_node, adj)]
            del adjacent[min_node]
            assert bool(adjacent) == bool(edges)
            
        return int(1000 * max_average_degree + 0.5)

    @property
    def _scan1_online(self):
        """
        Calculate the k1_scan statistic using the online k1 values.
        """
        edges, adjacent, k1_scans = self._containers
        return max(k1_scans.itervalues()) if k1_scans else 0

    @property
    def _scan1_brute(self):
        """
        Compute and return the global Scan_1 statistic in a brute-force n^2 way.
        """
        edges, adjacent, k1_scans = self._containers

        # find maximum size of explicitly induced subgraphs
        max_size = 0
        for node, adjacency in adjacent.iteritems():
            assert node not in adjacency
            nodes = set(adjacency)
            nodes.add(node)
            subgraph = tuple((n1, n2) for n1 in nodes for n2 in nodes if n1 < n2 and (n1, n2) in edges)
            size = len(subgraph)
            assert len(set(subgraph)) == size
            if size > max_size:
                max_size = size
            
        # verify the online calculation
        assert max_size == self._scan1_online, "_scan1_brute %d  _scan1_online %d" % (max_size, self._scan1_online)

        return max_size

def make_UndirectedOnlineInvariantsGraph(iterable):
    """
    Create an UndirectedOnlineInvariantsGraph and initialize it with items from
    iterable.  Each item is an edge, represented as a pair of distinct, ordered
    invariants specifying the nodes upon which the edge is incident.

    Here's a simple chain of nodes

    >>> g1 = make_UndirectedOnlineInvariantsGraph(itertools.izip(xrange(0, 5), xrange(1, 6)))
    >>> g1.invariants
    {'mad_lower_k': 1667, 'max_degree': 2, 'num_triangles': 0, 'order': 6, 'scan1': 2, 'size': 5}
    >>> incidence, map = g1.incidence_matrix
    >>> incidence
    array([[0, 1, 0, 0, 0, 0],
           [1, 0, 1, 0, 0, 0],
           [0, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 1],
           [0, 0, 0, 0, 1, 0]], dtype=int8)

    Here's an Erdos-Renyi random graph

    >>> g2 = make_UndirectedOnlineInvariantsGraph(randomgraph.make_ErdosRenyi(7, 0.25, seed=1).T)
    >>> g2.invariants
    {'mad_lower_k': 2400, 'max_degree': 3, 'num_triangles': 1, 'order': 7, 'scan1': 4, 'size': 8}
    >>> incidence2, _ = g2.incidence_matrix
    >>> incidence2
    array([[0, 0, 0, 1, 0, 1, 1],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 1, 1],
           [1, 0, 0, 0, 1, 0, 0],
           [1, 0, 1, 0, 1, 0, 0]], dtype=int8)
    """
    graph = UndirectedOnlineInvariantsGraph()
    for edge in iterable:
        graph.add_edge(edge)
    return graph


class PointProcessWindow(object):
    """
    Obsolete: see from onyx.util import pointprocess, pointprocess.PointProcessSamplingWindow

    FIFO window based on timestamps.
    
    >>> secs_per_minute = 60
    >>> secs_per_hour = secs_per_minute * 60
    >>> secs_per_day = secs_per_hour * 24
    >>> secs_per_week = secs_per_day * 7
    >>> secs_per_year = secs_per_day * 365.2425

    >>> g = UndirectedOnlineInvariantsGraph()
    >>> w = PointProcessWindow(secs_per_week, g.add_edge, g.remove_edge)
    >>> w.length
    604800

    Get the invariants at which each of the features is first maximal

    >>> maxen = collections.defaultdict(lambda:collections.defaultdict(int))
    >>> fields = 'employeeIndexDict[From]', 'employeeIndexDict[To]', 'Epoch', 'Topic'
    >>> with open(os.path.join(_module_dir, 'enron_subset.csv'), 'rb') as infile:
    ...   for msg_index, (msg_from, msg_to, msg_epoch, msg_topic) in enumerate(iterutils.csv_itemgetter(infile, fields)):
    ...     w.add((msg_from, msg_to), msg_epoch)
    ...     features = g.invariants
    ...     for key, value in features.iteritems():
    ...       if value > maxen[key][key]:
    ...          assert features['scan1'] == g._scan1_brute
    ...          features2 = dict(features)
    ...          features2['msg_index'] = msg_index
    ...          features2['msg_day'] = int((msg_epoch - w.stats.min_timestamp) / secs_per_day)
    ...          maxen[key] = features2
    >>> g.invariants
    {'mad_lower_k': 5958, 'max_degree': 24, 'num_triangles': 157, 'order': 114, 'scan1': 50, 'size': 247}

    >>> stats = w.stats
    >>> duration_sec = int(stats.max_timestamp - stats.min_timestamp)
    >>> duration_day = duration_sec // secs_per_day
    >>> duration_year = int(duration_sec / secs_per_year)

    >>> print 'duration_year', duration_year, ' duration_day', duration_day, ' duration_sec', duration_sec
    duration_year 0  duration_day 9  duration_sec 850974

    >>> print 'num_added', stats.num_added, ' num_removed', stats.num_removed
    num_added 2500  num_removed 748

    >>> print 'length', stats.length, ' max_queued', stats.max_queued
    length 604800  max_queued 2330

    >>> for key in sorted(maxen): print key, maxen[key][key], ' ', maxen[key]
    mad_lower_k 6690   {'num_triangles': 177, 'msg_day': 6, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'msg_index': 1941, 'max_degree': 52}
    max_degree 52   {'num_triangles': 103, 'msg_day': 4, 'scan1': 81, 'mad_lower_k': 5765, 'size': 213, 'order': 116, 'msg_index': 1329, 'max_degree': 52}
    num_triangles 177   {'num_triangles': 177, 'msg_day': 6, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'msg_index': 1941, 'max_degree': 52}
    order 123   {'num_triangles': 162, 'msg_day': 6, 'scan1': 91, 'mad_lower_k': 6160, 'size': 272, 'order': 123, 'msg_index': 1930, 'max_degree': 52}
    scan1 91   {'num_triangles': 155, 'msg_day': 6, 'scan1': 91, 'mad_lower_k': 6000, 'size': 260, 'order': 122, 'msg_index': 1907, 'max_degree': 52}
    size 283   {'num_triangles': 177, 'msg_day': 6, 'scan1': 91, 'mad_lower_k': 6690, 'size': 283, 'order': 123, 'msg_index': 1941, 'max_degree': 52}
    """
    def __init__(self, length, add, remove):
        stats = builtin.attrdict((
            ('num_added', 0),
            ('num_removed', 0),
            ('max_queued', 0),
            ('min_timestamp', None),
            ('max_timestamp', None),
            ('prev_timestamp', None),
            ))
        self._data = length, collections.deque(), add, remove, stats

    @property
    def length(self):
        length, fifo, add, remove, stats = self._data
        return length

    @property
    def stats(self):
        length, fifo, add, remove, stats = self._data
        stats = builtin.attrdict(stats)
        stats.length = length
        return stats

    def add(self, obj, timestamp):
        length, fifo, add, remove, stats = self._data

        if stats.min_timestamp is None:
            stats.min_timestamp = timestamp
        if timestamp > stats.max_timestamp:
            stats.max_timestamp = timestamp
        assert timestamp >= stats.prev_timestamp
        stats.prev_timestamp = timestamp

        add(obj)
        fifo.append((timestamp, obj))
        stats.num_added += 1

        cutoff = timestamp - length, None
        while fifo[0] <= cutoff:
            ts, ob = fifo.popleft()
            remove(ob)
            stats.num_removed += 1
        if len(fifo) > stats.max_queued:
            stats.max_queued = len(fifo)

        assert stats.num_added == len(fifo) + stats.num_removed

if __name__ == '__main__':
    import os
    import onyx
    _module_dir, _module_name = os.path.split(__file__)
    onyx.onyx_mainstartup()
