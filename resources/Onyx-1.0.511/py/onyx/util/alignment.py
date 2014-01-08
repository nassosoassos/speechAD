###########################################################################
#
# File:         alignment.py
# Date:         Thu 10 Jan 2008 17:14
# Author:       Ken Basye
# Description:  Minimum Edit Distance alignment of lattices and sequences
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
Minimum edit distance alignment of lattices and sequences

Here are some alignment examples.  We begin by showing simple string alignment.
The nbest_align call returns an iterator which will give us alignments of two
sequences in NBest order:

>>> iter = nbest_align('ABCD', 'ABEDF')

Use the function next() to get an alignment:

>>> iter.next()
(2, (('A', 'A', 0), ('B', 'B', 0), ('C', 'E', 1), ('D', 'D', 0), (None, 'F', 1)))

The alignment is a pair of (cost, alignment); here cost = 2.  The alignment is a
sequence of triples which represent the edits.  Each triple consists of a token
from string1 (or None for a deletion), a token from string2 (or None for an
insertion), and the cost of that edit.  See help(nbest_align) for more details.

Mixed sequences may be used:

>>> iter = nbest_align((1, 2, 'foo', True), (3, 2, 'bar', True))
>>> iter.next()
(2, ((1, 3, 1), (2, 2, 0), ('foo', 'bar', 1), (True, True, 0)))

A few more simple examples:

>>> iter = nbest_align('abc', 'adc')
>>> iter.next()
(1, (('a', 'a', 0), ('b', 'd', 1), ('c', 'c', 0)))
>>> iter.next()
(2, (('a', 'a', 0), (None, 'd', 1), ('b', None, 1), ('c', 'c', 0)))
>>> iter.next()
(2, (('a', 'a', 0), ('b', None, 1), (None, 'd', 1), ('c', 'c', 0)))
>>> iter.next()
(3, (('a', None, 1), (None, 'a', 1), ('b', 'd', 1), ('c', 'c', 0)))
>>> iter = nbest_align('abcy', 'xadc')
>>> iter.next()
(3, ((None, 'x', 1), ('a', 'a', 0), ('b', 'd', 1), ('c', 'c', 0), ('y', None, 1)))

Here we use the graphtools interface to construct a double-diamond lattice (see
below).  The first arg is the sequence of node labels, which are not used so
they're set to None.  The remaining three args specify the arcs as a sequence of
start nodes, a sequence of end nodes, and a sequence of arc labels.  See
help(GraphTables) for more information. The source is the node in the upper left
corner, the sink is in the lower right.  ::

 *---A---*---E---*
 |       |       |
 B       D       F
 |       |       |
 *---C---*---G---*

>>> gt3 = GraphTables(((None,)*6, (0,0,1,1,2,3,4), (1,2,3,4,4,5,5), ('A','B','E','D','C','F','G')))
>>> l3 = FrozenGraph(gt3)
>>> l3.is_lattice()
True

Here's another lattice::

 *---A---*
 |       |
 B       C
 |       |
 *---D---*

>>> gt4 = GraphTables(((None,)*4, (0,0,1,2), (1,2,3,3), ('A','B','C','D')))
>>> l4 = FrozenGraph(gt4)
>>> l4.is_lattice()
True

.. Remove this line and uncomment to generate figures for the lattices above
 # >>> dot_globals = ('rankdir=LR','node [label="", shape=circle]')
 # >>> l3.dot_display(globals=dot_globals)
 # >>> l4.dot_display(globals=dot_globals)

Here we align the two lattices.  A lattice alignment is a set of operations
which will transform some path in one lattice into some path in another lattice.

Here's a 1-Best alignment:

>>> align(l3, l4)
(2, (('A', 'A', 0), ('E', 'C', 1), ('F', None, 1)))

Here's an N-Best alignment of the same two lattices:

>>> iter = nbest_align(l3, l4)

Several alignments have cost 2, which is the minimum.  Ties will be broken in a
deterministic but undefined order.

>>> iter.next()
(2, (('A', 'A', 0), ('E', None, 1), ('F', 'C', 1)))
>>> iter.next()
(2, (('A', 'A', 0), ('E', 'C', 1), ('F', None, 1)))
>>> iter.next()
(2, (('A', 'A', 0), ('D', None, 1), ('G', 'C', 1)))

Look Ma, no arcs!

>>> gt5 = GraphTables(((None,), (), (), ()))
>>> l5 = FrozenGraph(gt5)
>>> l5.is_lattice()
True

>>> align(l5, l3)
(3, ((None, 'A', 1), (None, 'E', 1), (None, 'F', 1)))

>>> iter = nbest_align(l5, l3)
>>> iter.next()
(3, ((None, 'A', 1), (None, 'E', 1), (None, 'F', 1)))
>>> iter = nbest_align(l3, l5)
>>> iter.next()
(3, (('A', None, 1), ('E', None, 1), ('F', None, 1)))

Lattices may also be aligned with sequences:

>>> iter = nbest_align(l3, 'ABEDF')
>>> iter.next()
(2, (('A', 'A', 0), (None, 'B', 1), ('E', 'E', 0), (None, 'D', 1), ('F', 'F', 0)))
>>> iter = nbest_align(l3, 'ABCD')
>>> iter.next()
(2, ((None, 'A', 1), ('B', 'B', 0), ('C', 'C', 0), ('G', 'D', 1)))

>>> gt = GraphTables(((None,)*5, (0,0,1,2,3), (1,2,3,3,4), ('A','B','C','D','F')))
>>> g = FrozenGraph(gt)
>>> g
FrozenGraph(GraphTables(((None, None, None, None, None), (0, 0, 1, 2, 3), (1, 2, 3, 3, 4), ('A', 'B', 'C', 'D', 'F'))))
>>> g.is_lattice()
True
>>> iter = nbest_align(l3, g)
>>> iter.next()
(1, (('A', 'A', 0), ('E', 'C', 1), ('F', 'F', 0)))
>>> iter.next()
(2, (('A', 'A', 0), (None, 'C', 1), ('E', None, 1), ('F', 'F', 0)))
>>> iter.next()
(2, (('A', 'A', 0), ('E', None, 1), (None, 'C', 1), ('F', 'F', 0)))
>>> iter.next()
(2, (('A', 'A', 0), ('D', 'C', 1), ('G', 'F', 1)))
>>> iter = nbest_align('ABCD', l3)
>>> iter.next()
(2, (('A', None, 1), ('B', 'B', 0), ('C', 'C', 0), ('D', 'G', 1)))

Specifying substring_cost=True means there is no penalty for deletions from the
second sequence (or lattice) that occur before any characters of the first
sequence are used or after they have all been used.  This provides a way to
search flexibly for a small string or lattice within a larger string or lattice.
Note, e.g., that the 4 final deletions all have cost 0 below.

>>> align('abc', 'abcdefg', substring_cost=True)
(0, (('a', 'a', 0), ('b', 'b', 0), ('c', 'c', 0), (None, 'd', 0), (None, 'e', 0), (None, 'f', 0), (None, 'g', 0)))
>>> iter = nbest_align('abc', 'abcdefg', substring_cost=True)
>>> iter.next()
(0, (('a', 'a', 0), ('b', 'b', 0), ('c', 'c', 0), (None, 'd', 0), (None, 'e', 0), (None, 'f', 0), (None, 'g', 0)))
>>> iter = nbest_align('xadc', 'abcyxabc', substring_cost = True)
>>> iter.next()
(1, ((None, 'a', 0), (None, 'b', 0), (None, 'c', 0), (None, 'y', 0), ('x', 'x', 0), ('a', 'a', 0), ('d', 'b', 1), ('c', 'c', 0)))

>>> iter = nbest_align('ABCD', l3, substring_cost = True)
>>> iter.next()
(2, (('A', None, 1), ('B', 'B', 0), ('C', 'C', 0), ('D', None, 1), (None, 'G', 0)))
>>> iter.next()
(2, (('A', None, 1), ('B', 'B', 0), ('C', 'C', 0), ('D', 'G', 1)))
>>> iter.next()
(2, (('A', 'A', 0), ('B', None, 1), ('C', None, 1), ('D', 'D', 0), (None, 'G', 0)))

Clients may provide their own cost function; see help(nbest_align) for details.

>>> def subst_ok(x,y) : return 1 if x is None or y is None else 0.5 if x != y else 0
>>> align('abcy', 'xadc', subst_ok)
(2.0, (('a', 'x', 0.5), ('b', 'a', 0.5), ('c', 'd', 0.5), ('y', 'c', 0.5)))
>>> iter = nbest_align('abcy', 'xadc', subst_ok)
>>> iter.next()
(2.0, (('a', 'x', 0.5), ('b', 'a', 0.5), ('c', 'd', 0.5), ('y', 'c', 0.5)))
>>> iter.next()
(2.5, ((None, 'x', 1), ('a', 'a', 0), ('b', 'd', 0.5), ('c', 'c', 0), ('y', None, 1)))

The remainder of this test is a caution to clients about NBest alignments.
Although you might think that matching 'xabcp' to the 'abc' right after '234' is
the 'obvious' second choice, this algorithm has other ideas.  You may wish to
consider using some scheme whereby subsequent matches are filtered based on
overlapping with some earlier match.  Alternatively, you might think about
recursively finding matches of a short string within the pieces of the longer
string to the left and right of the first match.

>>> iter = nbest_align('xabcp', '234abcyxadcpqrs', substring_cost = True)
>>> iter.next()
(1, ((None, '2', 0), (None, '3', 0), (None, '4', 0), (None, 'a', 0), (None, 'b', 0), (None, 'c', 0), (None, 'y', 0), ('x', 'x', 0), ('a', 'a', 0), ('b', 'd', 1), ('c', 'c', 0), ('p', 'p', 0), (None, 'q', 0), (None, 'r', 0), (None, 's', 0)))
>>> iter.next()
(2, ((None, '2', 0), (None, '3', 0), (None, '4', 0), (None, 'a', 0), (None, 'b', 0), (None, 'c', 0), (None, 'y', 0), (None, 'x', 0), ('x', None, 1), ('a', 'a', 0), ('b', 'd', 1), ('c', 'c', 0), ('p', 'p', 0), (None, 'q', 0), (None, 'r', 0), (None, 's', 0)))
>>> iter.next()
(2, ((None, '2', 0), (None, '3', 0), (None, '4', 0), (None, 'a', 0), (None, 'b', 0), (None, 'c', 0), (None, 'y', 0), ('x', 'x', 0), ('a', 'a', 0), (None, 'd', 1), ('b', None, 1), ('c', 'c', 0), ('p', 'p', 0), (None, 'q', 0), (None, 'r', 0), (None, 's', 0)))
>>> iter.next()
(2, ((None, '2', 0), (None, '3', 0), (None, '4', 0), (None, 'a', 0), (None, 'b', 0), (None, 'c', 0), (None, 'y', 0), ('x', 'x', 0), ('a', 'a', 0), ('b', None, 1), (None, 'd', 1), ('c', 'c', 0), ('p', 'p', 0), (None, 'q', 0), (None, 'r', 0), (None, 's', 0)))
"""

from itertools import imap
from exceptions import ValueError
from onyx.graph.graphtools import GraphTables, GraphBuilder, FrozenGraph

# Public interface

def align(a1, a2, cost_fn = None, substring_cost = False):             
    """
    Find the best alignment path through a pair of iterables or lattices, a1 and
    a2.  The arguments a1 and a2 may either be iterables or lattices represented
    as FrozenGraph objects.  Mixing of the two argument types is allowed.  The
    alignment is done using either the values returned by the iterables or the
    labels on the arcs of the FrozenGraph.  FrozenGraph arguments must return
    True from is_lattice().

    the alignment path is a sequence of pairs <t1, t2> where the elements of
    each pair are either tokens from a1 and a2, respectively, or None.  Every
    pair will have at least one real token; <None,None> pairs will never occur.

    The path returned will have the following properties:

    0) Taking only the first or only the second element of each pair,
       and removing the Nones, each path is a simple path from the
       start of the corresponding lattice to its end.

    1) Considering each pair as an edit, where a (token,token) pair is
       either an exact match or a substitution, a (token, None) pair
       is an insertion, and a (None, token) pair is a deletion, and
       given a cost function on such pairs, the path returned has
       minimal cost.

    The alignment path returned has the form: (total_cost, (edit1, edit2,
    ... editN)) where each edit is a triple (t1, t2, edit_cost).

    cost_fn may be used to provide costs for substitutions, insertions, and
    deletions, and must take two arguments which are either labels from a1 and
    a2, respectively, or None, and return a cost C >= 0.  If the first argument
    is None, the cost should be that of an insertion.  If the second argument is
    None, the cost should be that of a deletion.  In no case will both arguments
    be None. If cost_fn is not given, the standard function using 0 for an exact
    match and 1 for all other costs will be used.

    If substring_cost is True, align one lattice (a1) within the other (a2).
    This works like ordinary alignment except that any number of elements from
    11 may be freely deleted before the first substitution or insertion and
    after the last substitution or insertion.  Note that insertions from a2 are
    never free.

    Please see the help for this module for a collection of examples.
    """
    if not isinstance(a1, FrozenGraph):
        a1 = _sequence_to_linear_graph(a1)
    elif not a1.is_lattice():
        raise ValueError("First argument is not a lattice")

    if not isinstance(a2, FrozenGraph):
        a2 = _sequence_to_linear_graph(a2)
    elif not a2.is_lattice():
        raise ValueError("Second argument is not a lattice")
    return _lattice_align(a1, a2, cost_fn, substring_cost)


def nbest_align(a1, a2, cost_fn = None, substring_cost = False):             
    """
    Find alignment paths through a pair of iterables or lattices, a1 and a2.
    The arguments a1 and a2 may either be iterables or lattices represented as
    FrozenGraph objects.  Mixing of the two argument types is allowed.  The
    alignment is done using either the values returned by the iterables or the
    labels on the arcs of the FrozenGraph.  FrozenGraph arguments must return
    True from is_lattice().

    An alignment path is a sequence of pairs <t1, t2> where the elements of each
    pair are either tokens from a1 and a2, respectively, or None.  Every pair
    will have at least one real token; <None,None> pairs will never occur.

    The paths found by the algorithm have the following properties:
    
    0) Taking only the first or only the second element of each pair,
       and removing the Nones, each path is a simple path from the
       start of the corresponding lattice to the end.

    1) Considering each pair as an edit, where a (token,token) pair is
       either an exact match or a substitution, a (token, None) pair
       is an insertion, and a (None, token) pair is a deletion, and
       given a cost function on such pairs, the paths are returned in
       order of cost, with minimal cost paths returned first.

    The iterator returned by this function returns alignment paths in the form:
    (total_cost, (edit1, edit2, ... editN)) where each edit is a triple (t1, t2,
    edit_cost).  The paths are returned in non-decreasing order of total cost.

    cost_fn may be used to provide costs for substitutions, insertions, and
    deletions, and must take two arguments which are either labels from a1 and
    a2, respectively, or None, and return a cost C >= 0.  If the first argument
    is None, the cost should be that of an insertion.  If the second argument is
    None, the cost should be that of a deletion.  In no case will both arguments
    be None. If cost_fn is not given, the standard function using 0 for an exact
    match and 1 for all other costs will be used.

    If substring_cost is True, align one lattice (a1) within the other (a2).
    This works like ordinary alignment except that any number of elements from
    11 may be freely deleted before the first substitution or insertion and
    after the last substitution or insertion.  Note that insertions from a2 are
    never free.

    Please see the help for this module for a collection of examples.
    """
    if not isinstance(a1, FrozenGraph):
        a1 = _sequence_to_linear_graph(a1)
    elif not a1.is_lattice():
        raise ValueError("First argument is not a lattice")

    if not isinstance(a2, FrozenGraph):
        a2 = _sequence_to_linear_graph(a2)
    elif not a2.is_lattice():
        raise ValueError("Second argument is not a lattice")

    return _lattice_nbest_align(a1, a2, cost_fn, substring_cost)

# Assorted helpers
def _std_cost(e1, e2):
    return 1 if e1 is None or e2 is None or e1 != e2 else 0

def _make_safe_cost_fn(fn):
    def safe_fn(x,y):
        result = fn(x,y)
        assert result >= 0
        return result
    return safe_fn

def _sequence_to_linear_graph(s):
    gb = GraphBuilder()
    start = gb.new_node()
    for item in s:
        end = gb.new_node()
        gb.new_arc(start, end, item)
        start = end
    return FrozenGraph(gb)


def _find_min_cost_and_pred(mat, arcs1, arcs2):
    if len(arcs1) == 0 and len(arcs2) == 0:
        return (0, (0,0))

    if len(arcs1) == 0:
        assert len(arcs2) != 0
        min_cost = mat[0][arcs2[0]+1]
        min_pred = (0, arcs2[0]+1)
        for a2 in arcs2:
            cost, pred, not_used = mat[0][a2+1]
            if cost < min_cost:
                min_cost, min_pred = cost, (0, a2+1)
        return min_cost, min_pred

    if len(arcs2) == 0:
        assert len(arcs1) != 0
        min_cost = mat[arcs1[0]+1][0]
        min_pred = (arcs1[0]+1, 0)
        for a1 in arcs1:
            cost, pred, not_used = mat[a1+1][0]
            if cost < min_cost:
                min_cost, min_pred = cost, (a1+1, 0)
        return min_cost, min_pred


    min_cost = mat[arcs1[0]+1][arcs2[0]+1]
    min_pred = (arcs1[0]+1, arcs2[0]+1)
    for a1 in arcs1:
        for a2 in arcs2:
            cost, pred, not_used = mat[a1+1][a2+1]
            if cost < min_cost:
                min_cost, min_pred = cost, (a1+1, a2+1)
    return min_cost, min_pred


# 1-Best alignment
#
# Right now this algorithm is substantially faster than the nbest
# algorithm at finding the 1-best result.

def _lattice_align(l1_arg, l2_arg, cost_fn=None, substring_cost=False):
    if cost_fn is None:
        cost_fn = _std_cost
    l1 = l1_arg.get_canonical_DAG()
    l2 = l2_arg.get_canonical_DAG()
    len1 = l1.get_num_arcs()
    len2 = l2.get_num_arcs()

    assert l1.is_lattice()
    assert l2.is_lattice()

    terms = l1.get_terminals()
    lat_start1 = terms[0][0]
    lat_end1 = terms[1][0] 
    terms = l2.get_terminals()
    lat_start2 = terms[0][0]
    lat_end2 = terms[1][0] 


    # initialize cost matrix to empty cells
    mat = [[None for j in range(len2+1)] for i in range(len1+1)]

    mat[0][0] = (0, (None, None), 0)  # real entries are (total cost, predecessor, local cost)
    # initialize first row
    for i in xrange(len1):
        start, end, x = l1.get_arc(i)
        # print "processing l1 arc from %d to %d with label %s" % (start, end, x)
        pred_arcs = l1.get_node_in_arcs(start)
        old_cost, pred = _find_min_cost_and_pred(mat, pred_arcs, (-1,))
        new_cost = cost_fn(x, None)
        mat[i+1][0] = (new_cost + old_cost, pred, new_cost)
        # print "mat[%d][0] is now %s" % (i+1, mat[i+1][0])
    # initialize first column
    for j in xrange(len2):
        start, end, y = l2.get_arc(j)
        # print "processing l2 arc from %d to %d with label %s" % (start, end, y) 
        pred_arcs = l2.get_node_in_arcs(start)
        old_cost, pred = _find_min_cost_and_pred(mat, (-1,), pred_arcs)
        new_cost = 0 if substring_cost else cost_fn(None, y)
        mat[0][j+1] = (new_cost + old_cost, pred, new_cost)
        # print "mat[0][%d] is now %s" % (j+1, mat[0][j+1])
    # fill in remainder of matrix
    for i in xrange(len1):
        for j in xrange(len2):
            start1, end1, x = l1.get_arc(i)
            start2, end2, y = l2.get_arc(j)
            # print "processing l1 arc from %d to %d with label %s" % (start1, end1, x)
            # print "and l2 arc from %d to %d with label %s" % (start2, end2, y)
            pred_arcs1 = l1.get_node_in_arcs(start1)
            pred_arcs2 = l2.get_node_in_arcs(start2)
            old_cost, insert_pred = _find_min_cost_and_pred(mat, pred_arcs1, (j,))
            new_cost = cost_fn(x, None)
            insert_cost = old_cost + new_cost
            best_cost_and_pred = (insert_cost, insert_pred, new_cost)
            old_cost, delete_pred = _find_min_cost_and_pred(mat, (i,), pred_arcs2)
            new_cost = 0 if (substring_cost and i == len1-1) else cost_fn(None, y)
            delete_cost = old_cost + new_cost
            if delete_cost < insert_cost:
                best_cost_and_pred = (delete_cost, delete_pred, new_cost)
            old_cost, subst_pred = _find_min_cost_and_pred(mat, pred_arcs1, pred_arcs2)
            new_cost = cost_fn(x, y)
            subst_cost = old_cost + new_cost
            if subst_cost < best_cost_and_pred[0]:
                best_cost_and_pred = (subst_cost, subst_pred, new_cost)
            mat[i+1][j+1] = best_cost_and_pred
            # print "mat[%d][%d] is now %s" % (i+1, j+1, mat[i+1][j+1])

    # Traceback
    final_arcs1 = l1.get_node_in_arcs(lat_end1)
    final_arcs2 = l2.get_node_in_arcs(lat_end2)
    min_cost, last_pred = _find_min_cost_and_pred(mat, final_arcs1, final_arcs2)

    i,j = last_pred
    traceback = []

    while True:
        next = mat[i][j][1]
        local_cost = mat[i][j][2]
        if next == (None, None):
            assert i == j == 0
            break
        delta_i = next[0] - i
        delta_j = next[1] - j
        assert delta_i <= 0 and delta_j <= 0
        x = l1.get_arc_label(i-1) if delta_i < 0 else None
        y = l2.get_arc_label(j-1) if delta_j < 0 else None
        traceback.append((x,y, local_cost))
        i,j = next
    traceback.reverse()
    return (min_cost, tuple(traceback))        



def _lattice_nbest_align(l1, l2, cost_fn = None, substring_cost = False):
    if not (isinstance(l1, FrozenGraph) and isinstance(l2, FrozenGraph)):
        raise ValueError("lattice_nbest_align needs two FrozenGraph arguments")

    if cost_fn is None:
        cost_fn = _std_cost
    else:
        cost_fn = _make_safe_cost_fn(cost_fn)
    
    l1_canon = l1.get_canonical_DAG()
    l2_canon = l2.get_canonical_DAG()
    len1 = l1_canon.get_num_arcs()
    len2 = l2_canon.get_num_arcs()

    assert l1_canon.is_lattice()
    assert l2_canon.is_lattice()

    terms = l1_canon.get_terminals()
    lat_start1 = terms[0][0]
    lat_end1 = terms[1][0] 
    terms = l2_canon.get_terminals()
    lat_start2 = terms[0][0]
    lat_end2 = terms[1][0] 

    # The lattices to be aligned, l1 and l2, have labels on the arcs;
    # any labels on their nodes will be ignored.  We begin by finding
    # a topological ordering of the lattice arcs, so that each arc is
    # assigned a non-negative index.  We construct a 2-D lattice, g,
    # with nodes representing pairs of arcs in the l1 and l2.  An
    # additional row and column on the top and left of g represent an
    # initial position prior consuming any tokens from l1 and l2,
    # respectively.  The arcs in g are labeled with triples giving the
    # orginal labels from l1 and l2 (or None if the arc's start node
    # is on the left side or top row) and the local cost of the edit
    # represented by the arc.  There are two slightly tricky parts.
    # First, because the alignment is done on lattices, the cost
    # lattice may have links which cross several rows or columns,
    # depending on the adjacencies of l1 and l2.  Second, there's an
    # offset of 1 between the arc numbering for l1 and l2 and the node
    # numbering in g, because of the initial row and column.  Thus,
    # the node in g corresponding to the arc pair <a1, a2> is at
    # indices <a1+1, a2+1>.  One egregious (but very handy) abuse of
    # this arrangement is the occasional use of -1 as an ersatz arcID,
    # which will be converted to a 0 index into g.  Finally, because
    # the graph iterpath function requires a single start and end
    # node, we tie all the potential end nodes in g to a special end
    # node with "ground" arcs.  A node in g is a potential end node if
    # the pair of arcs it represents are each incident on the terminal
    # node of their respective lattices.

    gb = GraphBuilder()
    # Initialize cost lattice nodes 
    node_array = [[gb.new_node() for j in range(len2+1)] for i in range(len1+1)]
    # We use this single node to tie all end nodes together
    end_node = gb.new_node()  

    # Add first row of arcs
    for i in xrange(len1):
        start, end, x = l1_canon.get_arc(i)
        insert_cost = cost_fn(x, None)
        if end == lat_end1 and len2 == 0:
            gb.new_arc(node_array[i+1][0], end_node, (None, None, 0))
            # print "Added ground arc for l1 arc from %d to %d with label %s" % (start, end, x)
        pred_arcs = l1_canon.get_node_in_arcs(start)
        if start == lat_start1:
            pred_arcs.append(-1)
        for arc in pred_arcs:
            gb.new_arc(node_array[arc+1][0], node_array[i+1][0], (x, None, insert_cost))
            # print ("Processed l1 arc from %d to %d with label %s - added %d arcs"
            #         % (start, end, x, len(pred_arcs)))
        
    # Add first column of arcs
    for j in xrange(len2):
        start, end, y = l2_canon.get_arc(j)
        delete_cost = 0 if substring_cost else cost_fn(None, y)
        if end == lat_end2 and len1 == 0:
            gb.new_arc(node_array[0][j+1], end_node, (None, None, 0))
            # print "Added ground arc for l1 arc from %d to %d with label %s" % (start, end, y)
        pred_arcs = l2_canon.get_node_in_arcs(start)
        if start == lat_start2:
            pred_arcs.append(-1)
        for arc in pred_arcs:
            gb.new_arc(node_array[0][arc+1], node_array[0][j+1], (None, y, delete_cost))
            # print ("Processed l1 arc from %d to %d with label %s - added %d arcs"
            #         % (start, end, y, len(pred_arcs)))

    # Construct remainder of cost lattice
    for i in xrange(len1):
        for j in xrange(len2):
            start1, end1, x = l1_canon.get_arc(i)
            start2, end2, y = l2_canon.get_arc(j)
            pred_arcs1 = l1_canon.get_node_in_arcs(start1)
            pred_arcs2 = l2_canon.get_node_in_arcs(start2)
            if start1 == lat_start1:
                pred_arcs1.append(-1)
            if start2 == lat_start2:
                pred_arcs2.append(-1)
            insert_cost = cost_fn(x, None)
            delete_cost = 0 if (substring_cost and i == len1 - 1) else cost_fn(None, y)
            subst_cost = cost_fn(x, y)

            num_added = 0
            if end1 == lat_end1 and end2 == lat_end2:
                gb.new_arc(node_array[i+1][j+1], end_node, (None, None, 0))
                # print "Added ground arc for l1,l1 arcs with labels %s, %s" % (x,y)

            for arc in pred_arcs1:
                gb.new_arc(node_array[arc+1][j+1], node_array[i+1][j+1], (x, None, insert_cost))
                num_added += 1

            for arc in pred_arcs2:
                gb.new_arc(node_array[i+1][arc+1], node_array[i+1][j+1], (None, y, delete_cost))
                num_added += 1

            for arc1 in pred_arcs1: 
                for arc2 in pred_arcs2:
                    gb.new_arc(node_array[arc1+1][arc2+1], node_array[i+1][j+1], (x, y, subst_cost))
                    num_added += 1
                    # print ("Processed l1 arc from %d to %d with label %s "
                    #        "and l1 arc from %d to %d with label %s  - added %d arcs"
                    #          % (start1, end1, x, start2, end2, y, num_added))

    g = FrozenGraph(gb)
    assert g.is_lattice()

    # print "g.get_terminals() = %s, %s" % g.get_terminals()

    def graph_cost_fn(label): return label[2]

    def iter_helper(path):
        arc_labels = [path[2].get_arc_label(arc) for arc in path[1][:-1]]
        return(path[0], tuple(arc_labels))

    return imap(iter_helper,
                g.iterpaths(graph_cost_fn, node_array[0][0], end_node))

                    
if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
