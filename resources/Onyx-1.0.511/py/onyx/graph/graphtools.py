###########################################################################
#
# File:         graphtools.py (directory: ./py/onyx/graph)
# Date:         14-May-2007
# Author:       Hugh Secker-Walker
# Description:  Basic graph tools
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 Hugh Secker-Walker
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
Basic tools for building and using graphs

The classes in this module support common operations using graphs.

They are used to construct graphs, serialize and unserialize graphs, iterate
through the nodes or arcs of a graph, sort nodes and arcs topologically, search
for paths, etc.

A graph consists of a set of nodes and a set of directed arcs.  Each arc
connects two nodes in the graph.  Each node and each arc is identified in the
graph by a non-negative integer ID.  Each node and each arc has an associated
label of user-supplied data.  Many of the supported graph operations depend only
upon the structure presented by the set of nodes and arcs, and do not depend
upon the labels.  For instance, whether a directed graph is acyclic or not
depends only upon the structure of the graph; acyclicity does not depend on the
labels associated with the nodes and arcs.

The factoring here focusses on structural properties of graphs and performing
operations that depend on these properties.  Subclassing of these objects is not
recommended.  Rather, they should be used as members of user-specific classes.
That is, these classes support "has a" usage rather than "is a" usage.

To support user-semantics in a graph, each node and each arc has a user-supplied
label.  These labels permit the user to attach arbitrary data to each node or
arc.  Although it is not required, best practices suggest that each label should
be an immutable object such as an integer, a string, or a tuple of integers and
strings.  For greatest portability of graph-based objects, e.g. in a text
representation, the use of non-negative integers for labels is recommended.

The use of a single label to capture all user-specific semantics of a node or an
arc is a simplifying factorization of the general graph representation problem.
It means that the graph tools do not have to specify or verify semantic
constraints or subtleties on nodes and arcs; the graph tools focus on the
semantics associated with the graph structure.  For the most part graph methods
do not apply any semantics to the labels.  The iteration constructs typically
yield the label associated with each node or arc they visit.  The few methods
that make semantic use of the labels typically do so by calling a user-supplied
callback to generate a numerical metric based on the label.

Furthermore, this use of a single label on each node or arc encourages the
implementor of a specific graph usage to focus on the semantics of what a node
is and what an arc is.  That is, the implementor makes node and arc objects with
whatever properties they need, and the implementor also makes an object to
manage the collection of nodes and arcs.  This management object would also be
responsible for maintaining the graph structure using the graphtools objects,
and it would use the labels in the graph to identifiy the instances of the node
and arc objects in the graph.  E.g. each node label could be a unique unsigned
integer that is an index into the table of data that the management object
maintains for the nodes.

The tools implement several types of graph object.  For instance, one type of a
graph is used to build the graph structure.  Due to its mutability (new nodes
and new arcs are getting added) the builder graph does not support many graph
semantics.  Once the graph is done being built, an immutable (frozen) graph is
generated.  Because it is immutable, this form of the graph can collect global
structural information at construction time and then readily supports read-only
queries and traversals without having to solve any of the difficult semantic
issues that arise while a graph is mutable.

There are of course cases when a graph need to be mutable and to maintain
non-local semantics.  To help simplify the semantic issues in such cases,
support for a streaming model of graph usage has been adopted.  The underlying
constraint imposed by a streaming model is that it should be possible to specify
a graph building and using algorithm that can exist in a bounded amount of
memory while processing an unbounded amount of data.  This implies an algorithm
that will remove nodes and arcs from the graph at the same average rate that
they are being added to the graph.  Such an algorithm thus imposes constraints
on the lifetimes of nodes and arcs.  Adding and removing nodes and arcs
according to a topologically ordered constraint is one simplification that can
help to achieve bounded memory for an unbounded stream.
"""

from __future__ import division
from __future__ import with_statement
import os
import cStringIO, random, operator
from collections import deque, defaultdict
from functools import partial as _partial
from heapq import heappush as _heappush, heappop as _heappop, heapify as _heapify
import itertools
from itertools import izip, chain as _chain, islice as _islice, imap

from onyx import builtin
from onyx.builtin import _wrapped_repr
import onyx.containers
from onyx.containers import listn, tuplenoflist
from onyx.textdata.onyxtext import OnyxTextReader, OnyxTextWriter
from onyx.textdata.yamldata import YamldataReader, YamldataWriter
from onyx.util.checkutils import check_instance


class GraphTables(tuple):
    """
    A fixed-size tuple -- used for exchanging graph data in a simple, tabular
    form.  Subclassing is discouraged.

    With no arguments, constructs an empty set of immutable tables.

    >>> GraphTables()
    GraphTables(((), (), (), ()))

    Otherwise, a single argument is required, and it must be a sequence of four
    sequences.  These four sequences describe the structure of the graph and the
    labels on the elements of the graph.

    Conceptually, there are two tables that make up the description of
    the graph:
    -  a single column table of node labels
    -  a three column table of arcs: arc startnodes, arc endnodes, arc labels

    The identifiers (ids) for the elements of the graph are implicit in these
    tables; they are the indices into the rows of these two tables.  A node id
    is the index into the column of node labels.  An arc id is the index into
    the rows of arc descriptions.

    Each arc startnode and each arc endnode is the index of a node at the end of
    the arc.  That is, each startnode and endnode is a non-negative integer that
    is less than the number of items in the node-labels column.  Each node-label
    and each arc-label is an arbitrary object.

    Consider the following informal description of a graph:
      four nodes with labels:
        node 0  label 1
        node 1  label 2
        node 2  label 'a'
        node 3  label 'b'
      three arcs with labels:
        arc 0  from node 0 to node 1  label 3
        arc 1  from node 1 to node 0  label None
        arc 2  from node 2 to node 2  label 5

    This graph contains two disconnected subgraphs, both of which are
    cyclic.  Here's the GraphTables for that graph

    >>> GraphTables(((1, 2, 'a', 'b'), (0, 1, 2), (1, 0, 2), (3, None, 5)))
    GraphTables(((1, 2, 'a', 'b'), (0, 1, 2), (1, 0, 2), (3, None, 5)))


    The constructor performs consistency checks on the contents of the argument
    to the constructor.

    Constructor takes zero or one arguments, just like tuple.  A
    common mistake is to unpack the argument:

    >>> GraphTables((1, 2, 'a', 'b'), (0, 1, 2), (1, 0, 2), (3, None, 5))
    Traceback (most recent call last):
      ...
    ValueError: GraphTables() expected 0 or 1 constructor arguments, got 4


    The single argument must contain four elements

    >>> GraphTables(((1, 2, 'a', 'b'), (0, 1, 2), (1, 0, 2)))
    Traceback (most recent call last):
      ...
    ValueError: GraphTables(seq) expected 4 items in seq, got 3

    The lengths of the columns for the arcs must all be the same:

    >>> GraphTables(((1, 2, 3), (0, 1), (1, 0), (3, 4, 4)))
    Traceback (most recent call last):
      ...
    ValueError: expected equal lengths for arcstartnodes 2, andarcendnodes 2, and arclabels 3

    >>> GraphTables(((1, 2, 3), (0,), (1, 0), (3, 4)))
    Traceback (most recent call last):
      ...
    ValueError: expected equal lengths for arcstartnodes 1, andarcendnodes 2, and arclabels 2

    >>> GraphTables(((1, 2, 3), (0, 1), (1, 0, 2), (3, 4)))
    Traceback (most recent call last):
      ...
    ValueError: expected equal lengths for arcstartnodes 2, andarcendnodes 3, and arclabels 2

    The node indices for the arcs must be valid indices into the node-labels
    column:

    >>> GraphTables(((1, 2, 3, 4), (0, 1, 4), ('a', -1, 5), (3, 4, 5)))
    Traceback (most recent call last):
      ...
    ValueError: expected arcstartnodes and arcendodes to contain non-negative ids less than 4, but also got (-1, 4, 5, 'a')
    """
    _required_len = 4
    __slots__ = tuple()
    def __new__(cls, seq=tuple(tuple() for i in xrange(_required_len)), *_sentinel):
        if _sentinel:
            raise ValueError("GraphTables() expected 0 or 1 constructor arguments, got %d" % (len(_sentinel) + 1,))
        return super(GraphTables, cls).__new__(cls, seq)
    def __init__(self, *args):
        if len(self) != self._required_len:
            raise ValueError("GraphTables(seq) expected %d items in seq, got %d" % (self._required_len, len(self),))
        super(GraphTables, self).__init__()
        self.verify()
    def __repr__(self):
        return _wrapped_repr(GraphTables, self)
    def verify(self):
        len_nodelabels, len_arcstartnodes, len_arcendnodes, len_arclabels = (len(column) for column in self)
        if not (len_arcstartnodes == len_arcendnodes == len_arclabels):
            raise ValueError("expected equal lengths for arcstartnodes %d, andarcendnodes %d, and arclabels %d"
                             % (len_arcstartnodes, len_arcendnodes, len_arclabels))
        validnodeids = set(xrange(len_nodelabels))
        nodelabels, arcstartnodes, arcendnodes, arclabels = self
        arcnodeids = set(_chain(arcstartnodes, arcendnodes))
        if not (arcnodeids <= validnodeids):
            raise ValueError("expected arcstartnodes and arcendodes to contain non-negative ids less than %d, but also got %r"
                             % (len_nodelabels, tuple(sorted(arcnodeids - validnodeids)),))


class SerializedGraphTables(GraphTables):
    """
    GraphTables object built from the Yaml representation in a stream.

    >>> doc = '''
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : SerializedGraphTables
    ...   __onyx_yaml__stream_version : '0'
    ...
    ... -
    ...   # canonic format for SerializedGraphTables
    ...   - num_nodes 3
    ...   # fields are: node-id node-label
    ...   - 0  -1
    ...   - 1  nil
    ...   - 2  end
    ...   - num_arcs 3
    ...   # fields are: arc-id start-node-id end-node-id arc-label
    ...   - 0  0 1  -1
    ...   - 1  0 2  -2
    ...   - 2  1 2  -3
    ... '''

    >>> SerializedGraphTables(doc)
    GraphTables((('-1', 'nil', 'end'), (0, 0, 1), (1, 2, 2), ('-1', '-2', '-3')))

    >>> # FrozenGraph(SerializedGraphTables(doc)).dot_display(globals=['rankdir=LR'])

    >>> SerializedGraphTables(doc) == GraphTables((('-1', 'nil', 'end'), (0, 0, 1), (1, 2, 2), ('-1', '-2', '-3')))
    True
    """
    STREAM_TYPE = 'SerializedGraphTables'
    STREAM_VERSION = '0'

    __slots__ = tuple()
    def __new__(cls, stream):

        reader = YamldataReader(stream,
                                stream_type=SerializedGraphTables.STREAM_TYPE,
                                stream_version=SerializedGraphTables.STREAM_VERSION,
                                no_stream_options=True)
        # parse
        #
        # XXX YamldataReader should have tools to help with indexed tabular parsing;
        # here's a start....
        def speccer(spec):
            # return a function that checks a match to the specification
            if type(spec) is type:
                def check(item):
                    # coerce
                    try:
                        return spec(item)
                    except (TypeError, ValueError):
                        raise TypeError("expected a valid %s constructor argument, got %r" % (spec.__name__, item))
            else:
                def check(item):
                    # match
                    if item != spec:
                        raise ValueError("expected %r, got %r" % (spec, item))
                    return item
            return check

        def checked_iter(iterable):
            nxt = iter(iterable).next
            def next(specs=None):
                items = nxt()
                if specs is not None:
                    count = specs if type(specs) is int else len(specs)
                    if not hasattr(items, '__len__'):
                        raise ValueError("expected %d items, but got a non-sequence %s" % (count, type(items).__name__,))
                    if len(items) != count:
                        raise ValueError("expected %d items, but got %d" % (count, len(items),))
                    if type(specs) is not int:
                        items = tuple(speccer(spec)(item) for spec, item in itertools.izip(specs, items))
                return items
            return next

        # XXX asserts should become errors
        itr = iter(reader)
        next = checked_iter(itr)

        field, num_nodes = next(2)
        assert field == 'num_nodes'
        num_nodes = int(num_nodes)

        node_labels = list()
        for enum, (node_id, node_label) in _islice(enumerate(itr), num_nodes):
            node_id = int(node_id)
            assert node_id == enum
            node_labels.append(node_label)
        node_labels = tuple(node_labels)

        field, num_arcs = next(('num_arcs', int))

        start_nodes = list()
        end_nodes = list()
        arc_labels = list()
        for enum, (arc_id, start_node, end_node, arc_label) in _islice(enumerate(itr), num_arcs):
            arc_id = int(arc_id)
            assert arc_id == enum
            start_nodes.append(int(start_node))
            end_nodes.append(int(end_node))
            arc_labels.append(arc_label)
        start_nodes = tuple(start_nodes)
        end_nodes = tuple(end_nodes)
        arc_labels = tuple(arc_labels)

        return super(SerializedGraphTables, cls).__new__(cls, (node_labels, start_nodes, end_nodes, arc_labels))


class SerializedGraphTables2(GraphTables):
    """
    GraphTables object built from the Yaml representation in a stream.

    >>> doc = '''
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : OnyxText
    ...   __onyx_yaml__stream_version : '0'
    ...
    ... -
    ...   # canonic OnyxText format for SerializedGraphTables2
    ...   - stream_type OnyxText stream_version 0 data_type SerializedGraphTables2 data_version 0
    ...   - num_nodes 3
    ...   - Nodes IndexedCollection Node 3
    ...   # fields are: 'Node' node-id node-label
    ...   - Node 0  -1
    ...   - Node 1  nil
    ...   - Node 2  end
    ...   - num_arcs 3
    ...   - Arcs IndexedCollection Arc 3
    ...   # fields are: 'Arc' arc-id start-node-id end-node-id arc-label
    ...   - Arc 0  0 1  -1
    ...   - Arc 1  0 2  -2
    ...   - Arc 2  1 2  -3
    ... '''

    >>> sgt0 = SerializedGraphTables2(doc)
    >>> sgt0
    GraphTables((('-1', 'nil', 'end'), (0, 0, 1), (1, 2, 2), ('-1', '-2', '-3')))

    >>> SerializedGraphTables2(doc) == GraphTables((('-1', 'nil', 'end'), (0, 0, 1), (1, 2, 2), ('-1', '-2', '-3')))
    True

    >>> out_stream = cStringIO.StringIO()
    >>> sgt0.serialize(out_stream)
    >>> print(out_stream.getvalue())
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "OnyxText"
    - - stream_type OnyxText stream_version 0 data_type SerializedGraphTables2 data_version 0
      - num_nodes 3
      - Nodes IndexedCollection Node 3
      - Node 0 -1
      - Node 1 nil
      - Node 2 end
      - num_arcs 3
      - Arcs IndexedCollection Arc 3
      - Arc 0 0 1 -1
      - Arc 1 0 2 -2
      - Arc 2 1 2 -3
    <BLANKLINE>

    >>> out_stream.seek(0)
    >>> sgt1 = SerializedGraphTables2(out_stream)
    >>> sgt1 == sgt0 == GraphTables((('-1', 'nil', 'end'), (0, 0, 1), (1, 2, 2), ('-1', '-2', '-3')))
    True

    """
    DATA_TYPE = 'SerializedGraphTables2'
    DATA_VERSION = '0'

    __slots__ = tuple()
    def __new__(cls, stream):

        reader = YamldataReader(stream, stream_type=OnyxTextReader.STREAM_TYPE,
                                stream_version=OnyxTextReader.STREAM_VERSION, no_stream_options=True)
        ctr = OnyxTextReader(reader, data_type=SerializedGraphTables2.DATA_TYPE, data_version=SerializedGraphTables2.DATA_VERSION)

        # Read nodes
        name, num_nodes = ctr.read_scalar(name="num_nodes", rtype=int)
        def node_reader(stream, user_data, tokens):
            assert len(tokens) == 3  # Checked already by read_indexed_collection, see below
            return tokens[2] # node label
        name, node_labels = ctr.read_indexed_collection(node_reader, None, name="Nodes", header_token_count=3)

        # Read arcs
        name, num_arcs = ctr.read_scalar(name="num_arcs", rtype=int)
        def arc_reader(stream, user_data, tokens):
            assert len(tokens) == 5
            return (int(tokens[2]), int(tokens[3]), tokens[4])  # (start_node, end_node, arc_label)
        name, arcs = ctr.read_indexed_collection(arc_reader, None, name="Arcs", header_token_count=5)
        start_nodes, end_nodes, arc_labels = zip(*arcs)

        return super(SerializedGraphTables2, cls).__new__(cls, (node_labels, start_nodes, end_nodes, arc_labels))


    # Not clear how useful this is as a standalone function, maybe
    # just fold this code into serialize?  Also note the writing
    # functionality probably belongs elsewhere, since these objects
    # can only be created from streams in the first place.
    def tuple_iter(self):
        """
        Returns a generator that yields tuples of tokens.  The tuples
        represent the graph in a form serializable by YamldataWriter.
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = self
        arcs = itertools.izip(arcstartnodes, arcendnodes, arclabels)
        ctw = OnyxTextWriter()

        def node_writer(stream, label):
            yield (label,)
        def arc_writer(stream, arc):
            yield tuple(arc)

        return _chain(
            ctw.gen_header(data_type=SerializedGraphTables2.DATA_TYPE, data_version=SerializedGraphTables2.DATA_VERSION),
            ctw.gen_scalar("num_nodes", len(nodelabels)),
            ctw.gen_indexed_collection("Nodes", "Node", nodelabels, node_writer),
            ctw.gen_scalar("num_arcs", len(arclabels)),
            ctw.gen_indexed_collection("Arcs", "Arc", arcs, arc_writer))


    def serialize(self, stream):
        """
        Serialize this graph into a YAML stream.  The resulting text is
        human-readable and matches the format used to create a
        SerializedGraphTables2.
        """
        writer = YamldataWriter(stream, stream_type=OnyxTextWriter.STREAM_TYPE,
                                stream_version=OnyxTextWriter.STREAM_VERSION)
        writer.write_document(self.tuple_iter())



class GraphBase(object):
    """
    Base class for graph work, contains utility and verification mixins.
    """

    def __init__(self, seqs):
        if not isinstance(seqs, GraphTables):
            raise TypeError("expected %s, got %s" % (GraphTables, type(seqs),))
        super(GraphBase, self).__init__()
        self.nodelabels, self.arcstartnodes, self.arcendnodes, self.arclabels = self.graphseqs = seqs
        GraphBase._verify(self)

    @property
    def num_nodes(self):
        """
        The number of nodes in the graph.
        """
        return len(self.nodelabels)

    def get_num_nodes(self):
        """
        Return the number of nodes in the graph.
        """
        return self.num_nodes

    @property
    def num_arcs(self):
        """
        The number of arcs in the graph.
        """
        return len(self.arclabels)

    def get_num_arcs(self):
        """
        Return the number of arcs in the graph.
        """
        return self.num_arcs

    def get_node_label(self, node_id):
        """
        Return the node label associated with the node_id.
        """
        self._check_node_id(node_id, "node_id")
        return self.nodelabels[node_id]

    def get_arc_label(self, arc_id):
        """
        Return the arc label associated with the arc_id.
        """
        self._check_arc_id(arc_id, "arc_id")
        return self.arclabels[arc_id]

    def get_arc(self, arc_id):
        """
        Return a tuple of the (start_node, end_node, arc_label) for
        the arc associated with the arc_id.
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        self._check_arc_id(arc_id, "arc_id")
        return arcstartnodes[arc_id], arcendnodes[arc_id], arclabels[arc_id]

    def __eq__(self, other):
        if self is other:
            return True
        if type(self) != type(other):
            return False
        return self.graphseqs == other.graphseqs

    def __nq__(self, other):
        return not (self == other)

    def text_iter(self):
        """
        Returns a generator that yields lines of text, including
        newlines.  The text represents the graph in a human-readable,
        serializable form.
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        yield "num_nodes %d\n" % (len(nodelabels),)
        for node_info in itertools.izip(itertools.count(), nodelabels):
            yield "%-2d  %s\n" % node_info

        yield "num_arcs %d\n" % (len(arclabels),)
        for arc_info in itertools.izip(itertools.count(), arcstartnodes, arcendnodes, arclabels):
            yield "%-2d  %-2d %-2d  %s\n" % arc_info

##     _default_label = lambda label, *_: '' if label is None or not str(label) else ('label="%s"' % (str(label),))

    _default_label_callback = lambda label, *_: '' if label is None else str(label)

    def dot_iter(self, graph_label='', graph_type='digraph', globals=(),
                 node_label_callback=_default_label_callback, arc_label_callback=_default_label_callback,
                 node_attributes_callback=None, arc_attributes_callback=None):
        """
        Returns a generator that yields lines of text, including newlines.  The
        text represents the graph in the DOT language for which there are many
        displayers.  See also dot_display().

        Optional argument graph_label is a string label for the graph.  Optional
        argument graph_type defaults to 'digraph', can also be 'graph'.

        Optional globals is an iterable that yields strings to put in the
        globals section of the DOT file.

        Optional node_label_callback is a function that will be called with
        three arguments for each node: (node_label, is_start, is_end), where
        node_label is the node's label, is_start is True if the node is a start
        node (has no incoming arcs), and is_end is True if the node is an end
        node (has no outgoing arcs).  The callback function should return a
        string which will be used as the text label for the node in the figure,
        or ``None`` or the empty string for no label.  If this argument is not
        given, each node with a non-``None`` label will be labelled with the str
        value of its label.  If this argument is ``None``, nodes will not be
        labelled.

        Optional arc_label_callback is a function that will be called with the
        arc label for each arc.  The callback function should return a string
        which will be used as the text label for the arc in the figure, or ``None``
        or the empty string for no label.  If this argument is not given, each
        arc with a non-``None`` label will be labelled with the str value of its
        label.  If this argument is ``None``, arcs will not be labelled.

        Optional node_attributes_callback is a function that returns an iterable
        that will yield a string for each attribute assignment for the node.
        It will be called with three arguments for each node: (node_label,
        is_start, is_end), where node_label is the node's label, is_start is
        True if the node is a start node (has no incoming arcs), and is_end is
        True if the node is an end node (has no outgoing arcs).

        Optional arc_attributes_callback is a function that returns an iterable
        that will yield a string for each attribute assignment for the arc.
        It will be called with the arc label for each arc in the graph.

        >>> graph = FrozenGraph(GraphTables(((1, 2, 'a'), (0, 1, 2), (1, 0, 2), (3, None, 5))))
        >>> print ''.join(graph.dot_iter())
        digraph  { 
          n0  [label="1"];
          n1  [label="2"];
          n2  [label="a"];
          n0 -> n1  [label="3"];
          n1 -> n0;
          n2 -> n2  [label="5"];
        }
        <BLANKLINE>

        >>> print ''.join(graph.dot_iter(graph_label='Foo',
        ...                              globals=['size="5,6.125";', 'ordering=out;'],
        ...                              node_label_callback=lambda label, is_start, is_end: ('<' if is_start else '') + str(label) + ('>' if is_end else ''),
        ...                              arc_label_callback=str,
        ...                              node_attributes_callback=lambda label, is_start, is_end: ['shape=%s' % ('octagon' if is_start or is_end else 'ellipse',), 'style=%s' % ('bold' if isinstance(label, int) else 'normal',)],
        ...                              arc_attributes_callback=lambda label: ['style=%s' % ('bold' if label is None else 'normal',)]))
        digraph Foo { 
          label="Foo";
          size="5,6.125";
          ordering=out;
          n0  [label="1", shape=ellipse, style=bold];
          n1  [label="2", shape=ellipse, style=bold];
          n2  [label="<a>", shape=octagon, style=normal];
          n0 -> n1  [label="3", style=normal];
          n1 -> n0  [label="None", style=bold];
          n2 -> n2  [label="5", style=normal];
        }
        <BLANKLINE>
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        node_name_num_digits = len(str(len(nodelabels)))

        # opening
        yield "%s %s { \n" % (graph_type, graph_label)

        if graph_label:
            yield '  label="%s";\n' % (graph_label,)

        # user globals

        # By default we use a left-to-right layout, but we prepend this so that if the caller's
        # globals includes a different rankdir attribute it will be effective.
        # XXX I'd like to turn this on when the dust settles - KJB 11-Jan-2009
        # globals = ('rankdir=LR',) + tuple(globals)

        for line in globals:
            yield '  %s\n' % (line,)

        starts = frozenset(self.startnodes)
        ends = frozenset(self.endnodes)

        def get_attributes_string(label_callback, attributes_callback, args):
            attributes = list()
            if label_callback is None:
                attributes.append('label=""')
            else:
                label = label_callback(*args)
                if label:
                    attributes.append('label="%s"' % (label,))
            if attributes_callback is not None:
                attributes.extend(attributes_callback(*args))
            while None in attributes:
                attributes.remove(None)
            attr_string = ('  [' + ', '.join(attributes) + ']') if attributes else ''
            return attr_string

        # nodes
        for id, label in itertools.izip(itertools.count(), nodelabels):
            attr_string = get_attributes_string(node_label_callback, node_attributes_callback, (label, id in starts, id in ends))
            yield '  n%0*d%s;\n' % (node_name_num_digits, id, attr_string)

        # arcs
        for start, end, label in itertools.izip(arcstartnodes, arcendnodes, arclabels):
            attr_string = get_attributes_string(arc_label_callback, arc_attributes_callback, (label, ))
            yield '  n%0*d -> n%0*d%s;\n' % (node_name_num_digits, start, node_name_num_digits, end, attr_string)

        # closing
        yield "}\n"

    def dot_display(self, temp_file_prefix='graphtools_',
                    display_tool_format="open -a /Applications/Graphviz.app %s", **kwargs):
        """
        Display a dot-generated representation of the graph.  Returns the name
        of the temporary file that the display tool is working from.  The caller
        is responsible for removing this file.  See also dot_iter().

        Optional temp_file_prefix is the prefix used for the temporary filename.

        Optional display_tool_format is formatting string, with %s where the
        filename goes, used to generate the command that will display the file.
        By default it assumes you're on a Mac and have Graphviz.app installed in
        the /Applications directory.

        Remaining keyword arguments are handed to the dot_iter function.
        """
        from onyx.util import opentemp
        with opentemp('wb', suffix='.dot', prefix=temp_file_prefix) as (name, outfile):
            for line in self.dot_iter(**kwargs):
                outfile.write(line)
        os.system(display_tool_format % (name,))
        return name

    @staticmethod
    def _verify(self):
        self.graphseqs.verify()

    def verify(self):
        GraphBase._verify(self)

    @staticmethod
    def _check_id(seq, id, label):
        len_seq = len(seq)
        if len_seq == 0:
            raise ValueError("invalid attempt to access empty sequence using %s = %d" % (label, id))
        if not (0 <= id < len_seq):
            raise ValueError("expected 0 <= %s < %d, got %s = %s" % (label, len(seq), label, id))

    def _check_node_id(self, id, label):
        self._check_id(self.nodelabels, id, label)

    def _check_arc_id(self, id, label):
        self._check_id(self.arclabels, id, label)

    @staticmethod
    def _check_label(label, label_name):
        try:
            hash(label)
        except TypeError:
            raise ValueError("%s must be immutable (hashable), got a %s" %(label_name, type(label).__name__))

    def __repr__(self):
        seqs = GraphTables(self.graphseqs)
        return "%s(%r)" % (type(self).__name__, seqs,)


class GraphArcStructure(object):
    """
    A baseclass for graphical objects.  It provides property access to the
    lowest-level structural information about the graph.

    Subclasses must initialize *self._num_nodes* with the number of nodes, and
    *self._arc_data* with a pair of sequences representing the starting and
    ending nodes of the arcs in the graph.  This baseclass is agnostic as to the
    types of the sequences.  However, if the subclass is mutable, e.g. the
    sequences are lists, it must maintain the invariants involving
    *self._num_nodes* and *self._arc_data* that the private ``_verify`` method
    successfully checks the invariants.
    """

    # XXX there's more work to be done, treating this much more like view of a
    # container of arcs

    def __init__(self):
        GraphArcStructure._verify(self)

    @staticmethod
    def create(num_nodes=None, arc_data=None):
        assert (num_nodes is None) is (arc_data is None)
        if num_nodes is None:
            num_nodes = 0
            arc_data = (), ()
        arcstartnodes, arcendnodes = arc_data
        return num_nodes, (list(arcstartnodes), list(arcendnodes))

    def verify(self):
        """
        Verify the object's invariants.
        """
        GraphArcStructure._verify(self)
        return True
        
    @staticmethod
    def _verify(self):
        arcstartnodes, arcendnodes = self._arc_data
        assert len(arcstartnodes) == len(arcendnodes) == self.num_arcs
        nodes = tuple(self.iter_node_ids)
        assert nodes == tuple(xrange(self.num_nodes))
        nodes = set(nodes)
        assert set(arcstartnodes) <= nodes
        assert set(arcendnodes) <= nodes
    
    @property
    def num_nodes(self):
        """
        The number of nodes in the graph.
        """
        return self._num_nodes
    @property
    def iter_node_ids(self):
        """
        An iterator that enumerates valid node identifiers in the graph.  It's
        like ``xrange(graph.num_nodes)``, except that for mutable subclasses,
        the iterator correctly iterates over identifiers for nodes that are
        added while the iteration is active.
        """
        for node in itertools.count():
            if node >= self.num_nodes:
                return
            yield node

    @property
    def num_arcs(self):
        """
        The number of arcs in the graph.
        """
        arcstartnodes, arcendnodes = self._arc_data
        return len(arcstartnodes)
    @property
    def iter_arc_ids(self):
        """
        An iterator that enumerates valid arc identifiers in the graph.  It's
        like ``xrange(graph.num_arcs)``, except that for mutable subclasses, the
        iterator correctly iterates over identifiers for arcs that are added
        while the iteration is active.
        """
        for arc in itertools.count():
            if arc >= self.num_arcs:
                return
            yield arc
    @property
    def iter_arcs(self):
        """
        An iterator that enumerates the arcs in the graph in their arc
        identifier order.  It yields a pair, (start_node_id, end_node_id), for
        each arc in the graph.  For mutable subclasses, the iterator correctly
        iterates over arcs that are added while the iteration is active.
        """
        return itertools.izip(*self._arc_data)


class GraphArcNodeStructure(GraphArcStructure):
    """
    A baseclass for graphical objects.  In addition to being a
    :class:`GraphArcStructure` it provides property access to dervied
    information about distinguished nodes in the graph.

    Subclasses must initialize *self._node_data* with three sets: a set of the
    identifiers for nodes that have self loops, a set of the identifiers for the
    starting nodes, and a set of the identifiers for the ending nodes of the
    graph.  If the subclass is mutable, it must maintain these three sets.
    """
    def __init__(self):
        super(GraphArcNodeStructure, self).__init__()
        GraphArcNodeStructure._verify(self)

    @staticmethod
    def create(num_nodes, arc_data):
        all_nodes = set(xrange(num_nodes))
        arcstartnodes, arcendnodes = arc_data
        assert len(arcstartnodes) == len(arcendnodes)
        assert set(arcstartnodes) <= all_nodes
        assert set(arcendnodes) <= all_nodes

        self_nodes = set(arcstart for arcstart, arcend in itertools.izip(arcstartnodes, arcendnodes) if arcstart == arcend)
        # subtle: a self loop must not prevent a node from being a graphstart or graphend
        non_self_start = set(arcstart for arcstart, arcend in itertools.izip(arcstartnodes, arcendnodes) if arcstart != arcend)
        non_self_end = set(arcend for arcstart, arcend in itertools.izip(arcstartnodes, arcendnodes) if arcstart != arcend)
        startnodes = all_nodes - non_self_end
        endnodes = all_nodes - non_self_start

        return self_nodes, startnodes, endnodes

    def verify(self):
        """
        Verify the object's invariants.
        """
        super(GraphArcNodeStructure, self).verify()
        GraphArcNodeStructure._verify(self)
        return True
        
    @staticmethod
    def _verify(self):
        # no point in duplicating the invariant checking that the create staticmethod does
        # subtle: a set and a frozenset with the same contents will compare equal!
        assert self._node_data == GraphArcNodeStructure.create(self._num_nodes, self._arc_data)        
        
    @property
    def num_start_nodes(self):
        """
        Number of start nodes in the graph.  A start node is a node that has no
        arc from another node ending on it.
        """
        loopnodes, startnodes, endnodes = self._node_data
        return len(startnodes)
    @property
    def start_nodes(self):
        """
        A sorted tuple of the start nodes in the graph.  A start node is a node
        that has no arc from another node ending on it.
        """
        loopnodes, startnodes, endnodes = self._node_data
        return tuple(sorted(startnodes))
        
    @property
    def num_end_nodes(self):
        """
        Number of end nodes in the graph.  A end node is a node that has no
        arc leading from it to another node.
        """
        loopnodes, startnodes, endnodes = self._node_data
        return len(endnodes)
    @property
    def end_nodes(self):
        """
        A sorted tuple of the of end nodes in the graph.  A end node is a node
        that has no arc leading from it to another node.
        """
        loopnodes, startnodes, endnodes = self._node_data
        return tuple(sorted(endnodes))

    @property
    def has_self_loop(self):
        """
        ``True`` if there is any node in the graph with an arc to itself,
        ``False`` if there is no node in the graph with an arc to itself.
        """
        loopnodes, startnodes, endnodes = self._node_data
        return bool(loopnodes)
    @property
    def self_loops(self):
        """
        A sorted tuple of the nodes in the graph, each of which has one or more
        arcs to itself.
        """
        loopnodes, startnodes, endnodes = self._node_data
        return tuple(sorted(loopnodes))

class FrozenGraphArcNodeStructure(GraphArcNodeStructure):
    def __init__(self, num_nodes=None, arc_data=None):
        # for GraphArcStructure
        self._num_nodes, arc_data = GraphArcStructure.create(num_nodes, arc_data)
        self._arc_data = tuple(tuple(datum) for datum in arc_data)
        # for GraphNodeStructure
        self._node_data = tuple(frozenset(datum) for datum in GraphArcNodeStructure.create(self._num_nodes, self._arc_data))
        FrozenGraphArcNodeStructure._verify(self)
        super(FrozenGraphArcNodeStructure, self).__init__()
        
    @staticmethod
    def _verify(self):
        hash((self._num_nodes, self._arc_data))
        hash(self._node_data)
        
    def verify(self):
        super(FrozenGraphArcNodeStructure, self).verify()
        FrozenGraphArcNodeStructure._verify(self)
        

class GraphStructureBuilder(GraphArcNodeStructure):
    """
    Low level builder for the core structure of a graph.

    >>> builder = GraphStructureBuilder()
    >>> builder.num_nodes, builder.num_arcs, builder.num_start_nodes, builder.start_nodes, builder.num_end_nodes, builder.end_nodes
    (0, 0, 0, (), 0, ())
    >>> n0 = builder.new_node()
    >>> n1, n2, n3 = builder.new_nodes(3)
    >>> builder.num_nodes, builder.num_arcs, builder.num_start_nodes, builder.start_nodes, builder.num_end_nodes, builder.end_nodes
    (4, 0, 4, (0, 1, 2, 3), 4, (0, 1, 2, 3))
    >>> n0, n2
    (0, 2)
    >>> a0 = builder.new_arc(n0, n2)
    >>> builder.num_nodes, builder.num_arcs, builder.num_start_nodes, builder.start_nodes, builder.num_end_nodes, builder.end_nodes
    (4, 1, 3, (0, 1, 3), 3, (1, 2, 3))
    >>> a1 = builder.new_arc_safe(n0, 5)
    >>> builder.num_nodes, builder.num_arcs, builder.num_start_nodes, builder.start_nodes, builder.num_end_nodes, builder.end_nodes, builder.has_self_loop, builder.self_loops
    (6, 2, 4, (0, 1, 3, 4), 5, (1, 2, 3, 4, 5), False, ())
    >>> for node in builder.iter_node_ids:
    ...   print 'arc', n1, '->', node, '', builder.new_arc(n1, node)
    ...   if node < 3: print 'node', builder.new_node()
    arc 1 -> 0  2
    node 6
    arc 1 -> 1  3
    node 7
    arc 1 -> 2  4
    node 8
    arc 1 -> 3  5
    arc 1 -> 4  6
    arc 1 -> 5  7
    arc 1 -> 6  8
    arc 1 -> 7  9
    arc 1 -> 8  10
    >>> builder.num_nodes, builder.num_arcs, builder.num_start_nodes, builder.start_nodes, builder.num_end_nodes, builder.end_nodes, builder.has_self_loop, builder.self_loops
    (9, 11, 1, (1,), 7, (2, 3, 4, 5, 6, 7, 8), True, (1,))
    >>> assert len(tuple(builder.iter_arc_ids)) == builder.num_arcs
    >>> for index, (start_node, end_node) in enumerate(builder.iter_arcs):
    ...   print 'arc', index, '', start_node, '->', end_node
    ...   if end_node == 5: print 'arc', builder.new_arc(start_node, end_node - 1)
    arc 0  0 -> 2
    arc 1  0 -> 5
    arc 11
    arc 2  1 -> 0
    arc 3  1 -> 1
    arc 4  1 -> 2
    arc 5  1 -> 3
    arc 6  1 -> 4
    arc 7  1 -> 5
    arc 12
    arc 8  1 -> 6
    arc 9  1 -> 7
    arc 10  1 -> 8
    arc 11  0 -> 4
    arc 12  1 -> 4
    >>> builder.verify()
    True
    >>> builder.start_nodes, builder.end_nodes
    ((1,), (2, 3, 4, 5, 6, 7, 8))
    >>> builder.num_nodes
    9
    >>> new_starts, new_ends = builder.add_graph(builder)
    >>> tuple(sorted(new_starts)), tuple(sorted(new_ends))
    ((10,), (11, 12, 13, 14, 15, 16, 17))
    >>> builder.num_nodes
    18
    """
    def __init__(self):
        # for GraphArcStructure
        self._num_nodes, self._arc_data = GraphArcStructure.create()
        # for GraphNodeStructure
        #self._node_loops, self._node_data = GraphArcNodeStructure.create(self._num_nodes, self._arc_data)
        self._node_data = GraphArcNodeStructure.create(self._num_nodes, self._arc_data)
        super(GraphStructureBuilder, self).__init__()

    def new_node(self):
        """
        Adds a new node to the graph.

        Returns an identifier for the new node, a non-negative integer.
        """
        node_id, = self.new_nodes(1)
        return node_id

    def new_nodes(self, count):
        """
        Adds *count* new nodes to the graph.

        Returns a tuple of identifiers for the new nodes, where each identifier
        is a non-negative integer.
        """
        loopnodes, startnodes, endnodes = self._node_data
        new_ids = tuple(xrange(self.num_nodes, self.num_nodes + count))
        startnodes.update(new_ids)
        endnodes.update(new_ids)
        self._num_nodes += count
        return new_ids

    def new_arc(self, start_node, end_node):
        """
        Adds a new arc to the graph from the node identified by *start_node* to
        the node identified by *end_node*.

        Returns an identifier for the new arc, a non-negative integer.

        Raises ``ValueError`` if either *start_node* or *end_node* is not a
        valid node identifier for the graph.
        """
        num_nodes = self.num_nodes
        if num_nodes == 0:
            raise ValueError("attempt to add arc (%d -> %d) to a graph with no nodes" % (start_node, end_node))
        if min(start_node, end_node) < 0 or max(start_node, end_node) >= num_nodes:
            raise ValueError("expected 0 <= start_node, end_node < %d, but got start_node = %d and end_node = %d" % (num_nodes, start_node, end_node))
        arc_id = self.num_arcs
        arcstartnodes, arcendnodes = self._arc_data
        arcstartnodes.append(start_node)
        arcendnodes.append(end_node)
        loopnodes, startnodes, endnodes = self._node_data
        if start_node == end_node:
            loopnodes.add(start_node)
        else:
            startnodes.discard(end_node)
            endnodes.discard(start_node)
        return arc_id

    def new_arc_safe(self, start_node, end_node):
        """
        Adds a new arc to the graph from the node identified by *start_node* to
        the node identified by *end_node*.

        Automatically adds nodes to the graph in order that non-negative
        *start_node* and *end_node* are valid node identifiers.

        Returns an identifier for the new arc, a non-negative integer.

        Raises ``ValueError`` if either of *start_node* or *end_node* is not a
        non-negative integer.
        """
        num_needed = max(start_node, end_node) + 1 - self.num_nodes 
        if num_needed > 0:
            self.new_nodes(num_needed)
        return self.new_arc(start_node, end_node)

    def add_graph(self, other):
        """
        Adds another graph, *other*, to this graph.

        Returns a pair, the sets of node_ids in this graph of the start_nodes
        and end_nodes of *other*.

        Raises ``TypeError`` if *other* is not an instace of
        :class:`GraphArcNodeStructure`.
        """
        if isinstance(other, GraphBase):
            nodelabels, arcstartnodes, arcendnodes, arclabels = other.graphseqs
            assert tuple(nodelabels) == tuple(xrange(other.num_nodes))
            assert tuple(arclabels) == tuple(xrange(other.num_arcs))
            other = FrozenGraphArcNodeStructure(other.num_nodes, (arcstartnodes, arcendnodes))

        check_instance(GraphArcNodeStructure, other)        

        offset = self.num_nodes
        def iter_add_offset(seq, copy=False):
            if copy:
                # used to prevent infinite recursion when adding ourself
                seq = tuple(seq)
            for item in seq:
                yield item + offset

        self._num_nodes += other._num_nodes

        arcstartnodes, arcendnodes = self._arc_data
        otherarcstartnodes, otherarcendnodes = other._arc_data
        arcstartnodes.extend(iter_add_offset(otherarcstartnodes, copy=other is self))
        arcendnodes.extend(iter_add_offset(otherarcendnodes, copy=other is self))

        loopnodes, startnodes, endnodes = self._node_data
        otherloopnodes, otherstartnodes, otherendnodes = other._node_data
        loopnodes.update(iter_add_offset(otherloopnodes, copy=other is self))
        new_starts = set(iter_add_offset(otherstartnodes))
        new_ends = set(iter_add_offset(otherendnodes))
        startnodes.update(new_starts)
        endnodes.update(new_ends)

        self.verify()
        return new_starts, new_ends

class GraphBuilder(GraphBase):
    """
    A lightweight object used to build a directed graph.  An instance of
    GraphBuilder is used to build a representation of a graph by creating nodes
    and by creating arcs between these nodes.

    Each node and each arc that is added to the graph can have a label.  These
    labels allow the client to arrange the semantic content graphically.  Each
    label can be any immutable (i.e. hashable) Python object.  The user of the
    GraphBuilder applies semantics to the labels, often using the labels as
    indices or keys into other client data structures.  Best practices in data
    portability are to use either a string or an integer for each label.

    If optional *init_graph* is given its structure and labels will be used to
    initialize the builder.  It must be an instance of :class:`GraphBase` or
    :class:`GraphTables`, otherwise a ``TypeError`` is raised.

    Once a graph has been built, it is typically used to initialize a
    :class:`FrozenGraph` which has many methods for examining structural
    properties of the graph and for performing iterations and transformations.

    >>> gb = GraphBuilder()
    >>> n1 = gb.new_node('A')
    >>> n2 = gb.new_node('B')
    >>> n3 = gb.new_node('C')
    >>> a1 = gb.new_arc(n1, n2, 'X')
    >>> a2 = gb.new_arc(n2, n3, 'Y')
    >>> fg0 = FrozenGraph(gb)
    >>> fg0
    FrozenGraph(GraphTables((('A', 'B', 'C'), (0, 1), (1, 2), ('X', 'Y'))))

    >>> all_nodes, starts, ends = mapped_nodes = gb.add_graph(fg0, lambda x: x+x, lambda x: 3*x)
    >>> all_nodes
    (3, 4, 5)
    >>> starts, ends
    ((3,), (5,))

    >>> fg1 = FrozenGraph(gb)
    >>> fg1
    FrozenGraph(GraphTables((('A', 'B', 'C', 'AA', 'BB', 'CC'), (0, 1, 3, 4), (1, 2, 4, 5), ('X', 'Y', 'XXX', 'YYY'))))
    """
    def __init__(self, init_graph=GraphTables()):
        if isinstance(init_graph, GraphTables):
            initseq = init_graph
        elif isinstance(init_graph, GraphBase):
            init_graph.verify()
            initseq = init_graph.graphseqs
        else:
            raise TypeError("expected %s or %s, got %s" % (GraphTables, GraphBase, type(init_graph).__name__))

        super(GraphBuilder, self).__init__(GraphTables(list(seq) for seq in initseq))

    def new_node(self, nodelabel=None):
        """
        Creates a new node in the graph, with optional nodelabel.  Returns the
        id of the node, a non-negative integer.
        """
        self._check_label(nodelabel, "nodelabel")
        nodelabels = self.nodelabels
        id = len(nodelabels)
        nodelabels.append(nodelabel)
        return id

    def new_node_label_is_id(self):
        """
        Creates a new node in the graph, with nodelabel being the id of the
        node.  Returns the id of the node, a non-negative integer.
        """
        return self.new_node(len(self.nodelabels))

    def new_arc(self, start_node, end_node, arclabel=None):
        """
        Creates a new arc in the graph connecting the start_node with the
        end_node, and with an optional arclabel.  It is an error if start_node
        or end_node are not valid node ids created by new_node().  Returns the
        id of the arc, a non-negative integer.
        """
        self._check_node_id(start_node, "start_node")
        self._check_node_id(end_node, "end_node")
        self._check_label(arclabel, "arclabel")

        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs

        arcstartnodes.append(start_node)
        arcendnodes.append(end_node)
        id = len(arclabels)
        arclabels.append(arclabel)
        return id

    def new_arc_label_is_id(self, start_node, end_node):
        """
        Creates a new arc in the graph, with arclabel being the id of the arc.
        Returns the id of the arc, a non-negative integer.
        """
        return self.new_arc(start_node, end_node, len(self.arclabels))

    def add_graph(self, subgraph, node_label_callback=lambda x:x, arc_label_callback=lambda x:x):
        """
        Add an entire subgraph to this builder's graph.

        Add all the nodes and edges and their respective labels from *subgraph*
        into the builder's graph.

        Returns a triple of tuples representing the nodes of the subgraph within
        the builder's graph.  The items in the triple are the node indices in
        the builder of all the nodes from *subgraph*, the indices of the
        subgraph's startnodes, and the indices of the subgraph's endnodes
        respectively.  Note that these node indices are not necessarily in the
        same order as they are in *subgraph*.
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = subgraph.graphseqs
        node_map = tuple(self.new_node(node_label_callback(label)) for label in nodelabels)
        for start, end, label in itertools.izip(arcstartnodes, arcendnodes, arclabels):
            self.new_arc(node_map[start], node_map[end], arc_label_callback(label))
        new_starts = tuple(node_map[start] for start in subgraph.startnodes)
        new_ends = tuple(node_map[end] for end in subgraph.endnodes)

        return node_map, new_starts, new_ends

    def add_graph_label_is_id(self, subgraph):
        def assert_label_is_id(seq):
            assert tuple(seq) == tuple(xrange(len(seq))), '%s' % (seq,)
            return True
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        assert len(arcstartnodes) == len(arcendnodes) == len(arclabels)
        assert assert_label_is_id(nodelabels)
        assert assert_label_is_id(arclabels)
        assert set(arcstartnodes) | set(arcendnodes) <= set(xrange(self.num_nodes))
        subgraph_nodelabels, subgraph_arcstartnodes, subgraph_arcendnodes, subgraph_arclabels = subgraph.graphseqs
        assert len(subgraph_arcstartnodes) == len(subgraph_arcendnodes) == len(subgraph_arclabels)
        assert assert_label_is_id(subgraph_nodelabels)
        assert assert_label_is_id(subgraph_arclabels)
        assert set(subgraph_arcstartnodes) | set(subgraph_arcendnodes) <= set(xrange(subgraph.num_nodes))

        node_offset = self.num_nodes
        nodelabels.extend(xrange(node_offset, node_offset + subgraph.num_nodes))
        arc_offset = self.num_arcs
        arclabels.extend(xrange(arc_offset, arc_offset + subgraph.num_arcs))

        arcstartnodes.extend(startnode+node_offset for startnode in subgraph_arcstartnodes)
        arcendnodes.extend(endnode+node_offset for endnode in subgraph_arcendnodes)
        assert len(arcstartnodes) == len(arcendnodes) == len(arclabels)
        assert set(arcstartnodes) | set(arcendnodes) <= set(xrange(self.num_nodes))

        new_starts = tuple(start+node_offset for start in subgraph.startnodes)
        new_ends = tuple(end+node_offset for end in subgraph.endnodes)
        return new_starts, new_ends
        
##     XXX It might be nice to return instead a mapping from the internal
##     indices of the original graph to the internal indices of the GB, perhaps
##     restricted to just the terminal nodes, but this isn't needed right now.

##     A somewhat deeper matter is that it really seems like it should be
##     possible to implement this function outside either class, but I don't see
##     any way to do that given the current interfaces.  HMMMM....

##         #map = dict()
##         node_map = list()
##         new_starts = list()
##         new_ends = list()
##         startnodes, endnodes = tuple(set(nodes) for nodes in graph.terminals)
##         node_map, new_starts, new_ends = tuplenoflist(3)

##         node_map = list()
##         for old_node, label in enumerate(nodelabels):
##             new_node = self.new_node(label)
##             node_map.append(new_node)


##             if old_node in startnodes:
##                 new_starts.append(new_node)
##             if old_node in endnodes:
##                 new_ends.append(new_node)

class _SubgraphsGetter(object):
    def get_subgraph(self, index):
        """
        Returns a triple of tuples: (nodes, starts, ends) representing the
        subgraph at *index*.

        Raises ``ValueError`` if *index* is not a valid indentifier for a subgraph.
        """
        if not (0 <= index < len(self.subgraphs)):
            raise ValueError("expected 0 <= index < %d, got index = %d" % (len(self.subgraphs), index))
        return self.subgraphs[index]

class SubgraphsBuilder(_SubgraphsGetter):
    """
    Lightweight object for keeping track of subgraphs of a graph.

    A given subgraph is specified by a triple of sets, (nodes, starts, ends):
    where nodes is the set of immutable identifiers for all the nodes in the
    subgraph, starts is the set identifiers for what are to be considered the
    starting nodes of the subgraph, and ends is the set identifiers for what are
    to be considered the ending nodes of the subgraph.  An instance of
    SubgraphsBuilder manages a growable collection of such triples, making sure
    for each subgraph that starts and ends are subsets of nodes.

    If optional *init_subgraphs* is provided it must either be an instance of
    SubgraphsBuilder or an iterable that yields subgraph-specifying triples.  It
    will be used to initialize the subgraphs in this instance.

    >>> sb = SubgraphsBuilder()
    >>> sb.add_subgraph(xrange(10), xrange(2), xrange(3, 5))
    0
    >>> sb.add_subgraph((47, 49, 51, 55), (47,), ())
    1
    >>> sb.get_subgraph(0)
    ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1), (3, 4))

    >>> sb2 = SubgraphsBuilder(sb)
    >>> id = sb2.add_subgraph('abcdef', 'ac', 'de')
    >>> id + 1 == len(sb2)
    True
    >>> sb2.get_subgraph(id)
    (('a', 'b', 'c', 'd', 'e', 'f'), ('a', 'c'), ('d', 'e'))

    >>> sb.add_subgraph(xrange(10), xrange(2), xrange(8, 12))
    Traceback (most recent call last):
      ...
    ValueError: at index 2, expected starts and ends to be subsets of nodes, but got these too: 10 11
    """
    def __init__(self, init_subgraphs=()):
        if isinstance(init_subgraphs, type(self)):
            self.subgraphs = list(init_subgraphs.subgraphs)
        else:
            self.subgraphs = list()
            for x in init_subgraphs:
                self.add_subgraph(*x)

    def __len__(self):
        return len(self.subgraphs)

    def add_subgraph(self, nodes, starts, ends):
        """
        Add a new subgraph specification.  Each of *nodes*, *starts*, and *ends*
        is an iterable collection of immutable node specifiers for,
        respectively, all the nodes in the subgraph, the starting nodes of the
        subgraph, and the ending nodes of the subgraph.

        Returns an identifier that can be used in a call to
        :meth:`get_subgraph` to retrieve the subgraph specification.

        Raises ``TypeError`` if any item in *starts*, *ends*, or *nodes* is a
        mutable object.
        Raises ``ValueError`` if *starts* or *ends* are not a subset of *nodes*.
        """
        id = len(self.subgraphs)
        nodes, starts, ends = (set(x) for x in (nodes, starts, ends))
        extras = (starts | ends) - nodes
        if extras:
            raise ValueError("at index %d, expected starts and ends to be subsets of nodes, but got these too: %s" %(id, ' '.join(repr(x) for x in sorted(extras))))
        self.subgraphs.append(tuple(tuple(sorted(spec)) for spec in (nodes, starts, ends)))
        return id

class FrozenSubgraphs(_SubgraphsGetter):
    """
    An immutable collection of subgraph specifiers.

    >>> sb = SubgraphsBuilder()
    >>> sb.add_subgraph(xrange(10), xrange(2), xrange(3, 5))
    0
    >>> sb.add_subgraph((47, 49, 51, 55), (47,), ())
    1
    >>> FrozenSubgraphs(sb).get_subgraph(0)
    ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (0, 1), (3, 4))
    """
    def __init__(self, subgraphs):
        check_instance(SubgraphsBuilder, subgraphs)
        self.subgraphs = tuple(subgraphs.subgraphs)
        # ensure immutability
        hash(self.subgraphs)

class SetGraphBuilder(GraphBuilder):
    """
    GraphBuilder with set semantics.  A Node is unique based on its label.  An
    Arc is unique based on its end nodes and its label.  As with set objects,
    adding a node or arc more than once does not change the graph.

    >>> b = SetGraphBuilder()
    >>> b.add_node('A')
    'A'
    >>> b.add_node('B')
    'B'
    >>> b.num_nodes
    2
    >>> b.add_node('A')
    'A'
    >>> b.num_nodes
    2

    >>> b.add_arc('A', 'B', 'X')
    0
    >>> b.add_arc('A', 'B', 'Y')
    1
    >>> b.num_arcs
    2
    >>> b.add_arc('A', 'B', 'X')
    0
    >>> b.num_arcs
    2

    Nodes are added automatically by add_arc.

    >>> b.add_arc('AA', 'BB', 'XX')
    2
    >>> b.num_nodes
    4
    >>> b.num_arcs
    3

    >>> g = FrozenGraph(b)

    >>> #g.dot_display(globals=('label="SetGraphBuilder";', 'labelloc=top;',))

    >>> g.get_forward_transitive_closures(['A'])
    (frozenset(['B']), frozenset([0, 1]))
    >>> g.get_backward_transitive_closures(['A'], True)
    (frozenset(['A']), frozenset([]))

    You can initialize a SetGraphBuilder from a FrozenGraph and continue building

    >>> b1 = SetGraphBuilder(FrozenGraph(g))
    >>> b == b1
    True

    But trying to initialize a SetGraphBuilder from a graph with non-unique node
    labels is an error:

    >>> gb = GraphBuilder(g)
    >>> id = gb.new_node('A')
    >>> b = SetGraphBuilder(FrozenGraph(gb))
    Traceback (most recent call last):
    ...
    ValueError: SetGraphBuilder must be initialized with a graph having unique node labels

    As is trying to initialize from a graph with non-unique arc labels between the same two nodes:

    >>> gb = GraphBuilder(g)
    >>> id = gb.new_arc(2, 3, 'XX')
    >>> b = SetGraphBuilder(FrozenGraph(gb))
    Traceback (most recent call last):
    ...
    ValueError: SetGraphBuilder must be initialized with a graph having unique arc labels between two nodes

    """
    def __init__(self, *_):
        super(SetGraphBuilder, self).__init__(*_)
        self.node_id_by_label = dict(itertools.izip(self.nodelabels, itertools.count()))
        if len(self.nodelabels) != len(self.node_id_by_label):
            raise ValueError("%s must be initialized with a graph having unique node labels" % (type(self).__name__,))
        arc_spec_iter = ((self.nodelabels[s], self.nodelabels[e], arc_label) for
                         (s, e, arc_label) in itertools.izip(self.arcstartnodes, self.arcendnodes, self.arclabels))
        self.arc_id_by_arc_spec = dict(itertools.izip(arc_spec_iter, itertools.count()))
        if len(self.arclabels) != len(self.arc_id_by_arc_spec):
            raise ValueError("%s must be initialized with a graph having unique arc labels between two nodes" % (type(self).__name__,))


    def new_node(self, node_label=None):
        raise NotImplementedError("%s requires use of add_node(node_label) instead of new_node()" % (type(self).__name__,))

    def new_arc(self, start_node, end_node, arclabel=None):
        raise NotImplementedError("%s requires use of add_arc(start_node, end_node, node_label=None) instead of new_arc()" % (type(self).__name__,))

    def _new_node(self, node_label):
        # helper to create new nodes for add_node and add_arc
        node_id_by_label = self.node_id_by_label
        assert node_label not in node_id_by_label
        node_id = super(SetGraphBuilder, self).new_node(node_label)
        assert node_id == len(node_id_by_label)
        node_id_by_label[node_label] = node_id

    def add_node(self, node_label):
        """
        Adds a node in the graph, with node_label, which must be immutable.

        Returns the label as the node id for reference to the node in calls to
        add_arc().  The nodes in the graph are a set, so add_node() is
        idempotent; additional calls to add_node() with the same node_label will
        not create additional nodes.
        """
        # note: this check is redundant with what happens in the baseclass's
        # new_node(), but we perform the check here to prevent a less intellible
        # KeyError from node_id_by_label
        self._check_label(node_label, "node_label")
        if node_label not in self.node_id_by_label:
            self._new_node(node_label)
        return node_label

    def add_arc(self, start_node, end_node, arclabel=None):
        """
        Add an arc to the graph connecting the start_node with the end_node, and
        with an optional arclabel.  If not already present in the graph, each of
        start_node or end_node will be added to the graph as if add_node() had
        been called.

        Returns the id of the arc, a non-negative integer.  The arcs in the
        graph are a set, so add_arc() is idempotent; additional calls to
        add_arc() with the same arguments will not create additional arcs.
        """
        # note: these check are redundant with what happens in the baseclass's
        # new_node(), but we perform the check here to prevent a less intellible
        # KeyError from node_id_by_label
        self._check_label(start_node, "start node label")
        self._check_label(end_node, "end node label")
        # ensure that the nodes are present
        node_id_by_label = self.node_id_by_label
        for node_label in (start_node, end_node):
            if node_label not in node_id_by_label:
                self._new_node(node_label)
        # see about the arc
        arc_spec = start_node, end_node, arclabel
        arc_id_by_arc_spec = self.arc_id_by_arc_spec
        arc_id = arc_id_by_arc_spec.get(arc_spec)
        if arc_id is None:
            # create the new arc
            arc_id = super(SetGraphBuilder, self).new_arc(node_id_by_label[start_node], node_id_by_label[end_node], arclabel)
            assert arc_id == len(arc_id_by_arc_spec)
            arc_id_by_arc_spec[arc_spec] = arc_id
        return arc_id


def make_initialized_set_graph_builder(arc_iterator):
    """
    Make a new SetGraphBuilder and populate it with arcs specified by items in
    arc_iterator.  Each item in arc_iterator is either a pair or a triple of
    immutable objects.  The first and second objects are the labels for the
    start and end nodes of the arc.  The optional third object (default
    ``None``) is the arc label.  The set-based semantics of SetGraphBuilder hold
    for the nodes and arcs.

    Returns the SetGraphBuilder that has been initialized with the arc
    specifications in arc_iterator.

    Function returns an iterator that yields a multigraph that also shows the set semantics:

    >>> def arc_yielder():
    ...   yield 'A1', 'A2'
    ...   yield 'A1', 'A2', 'a'
    ...   yield 'A2', 'B1', 'ab'
    ...   yield 'A2', 'B1', 'ab'
    >>> b = make_initialized_set_graph_builder(arc_yielder())
    >>> for line in b.text_iter():
    ...   print line,
    num_nodes 3
    0   A1
    1   A2
    2   B1
    num_arcs 3
    0   0  1   None
    1   0  1   a
    2   1  2   ab

    Add a new arc:
    >>> b.add_arc('X1', 'X2', 'x')
    3
    >>> for line in b.text_iter():
    ...   print line,
    num_nodes 5
    0   A1
    1   A2
    2   B1
    3   X1
    4   X2
    num_arcs 4
    0   0  1   None
    1   0  1   a
    2   1  2   ab
    3   3  4   x

    >>> g = FrozenGraph(b)
    >>> #g.dot_display(globals=['label="make_initialized_set_graph_builder";', 'labelloc=top;', 'rankdir=LR;'])
    """
    def arc_normalize(start, end, arc_label=None):
        return start, end, arc_label
    builder = SetGraphBuilder()
    for item in arc_iterator:
        builder.add_arc(*arc_normalize(*item))
    return builder


class NodeLabelIdGraphBuilder(GraphBuilder):
    """
    Graph builder that uses node labels as node ids.  This supports the common
    situation of building a graph in which each node has a unique label that is
    easy for the client to use when creating arcs.

    Each node label will be a tuple of a single integer

    >>> builder = NodeLabelIdGraphBuilder()
    >>> nodes = tuple(builder.new_node((x,)) for x in xrange(10))
    >>> nodes
    ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,))

    Link the nodes into a chain

    >>> i1 = iter(nodes); i1.next()
    (0,)
    >>> arcs = tuple(builder.new_arc(x, y) for x, y in itertools.izip(nodes, i1))
    >>> arcs
    (0, 1, 2, 3, 4, 5, 6, 7, 8)

    Exercise some error messages

    >>> builder.new_node((1,))
    Traceback (most recent call last):
      ...
    ValueError: NodeLabelIdGraphBuilder: unexpected duplicate node label: (1,)

    >>> builder.new_arc((1,), 2)
    Traceback (most recent call last):
      ...
    ValueError: NodeLabelIdGraphBuilder: expected end_node argument to be an existing node label, got 2
    """
    def __init__(self, *_):
        super(NodeLabelIdGraphBuilder, self).__init__(*_)
        self.node_id_by_label = dict()

    def new_node(self, nodelabel):
        """
        Creates a new node in the graph, with nodelabel, which must be immutable
        and unique across the labels of all nodes.  Returns the label.
        """
        # note: this check is redundant with what happens in the
        # baseclass's new_node(), but we need to perform the check
        # before checking if nodelabel is in node_id_by_label in order
        # to prevent a less intellible KeyError
        self._check_label(nodelabel, "nodelabel")

        node_id_by_label = self.node_id_by_label
        if nodelabel in node_id_by_label:
            raise ValueError("%s: unexpected duplicate node label: %r" % (type(self).__name__, nodelabel,))
        node_id_by_label[nodelabel] = super(NodeLabelIdGraphBuilder, self).new_node(nodelabel)
        return nodelabel

    def new_arc(self, start_node, end_node, arclabel=None):
        """
        Creates a new arc in the graph connecting the start_node with the
        end_node, and with an optional arclabel.  It is an error if start_node
        or end_node are not valid node labels for nodes created by new_node().
        Returns the id of the arc, a non-negative integer.
        """
        node_id_by_label = self.node_id_by_label
        for label, name in ((start_node, 'start_node'), (end_node, 'end_node')):
            if label not in node_id_by_label:
                raise ValueError("%s: expected %s argument to be an existing node label, got %r" % (type(self).__name__, name, label))
        start_id = node_id_by_label[start_node]
        end_id = node_id_by_label[end_node]
        return super(NodeLabelIdGraphBuilder, self).new_arc(start_id, end_id, arclabel)

    def new_node_label_is_id(self):
        raise NotImplementedError("the functionality does not make sense for this subclass")

    def new_arc_label_is_id(self):
        raise NotImplementedError("the functionality does not make sense for this subclass")


def make_random_graph(numnodes, numarcs, seed=None):
    """
    Return a FrozenGraph with numnodes nodes and numarcs arcs.  Each arc's start
    node and end node are chosen at random (with replacement) from the set of
    nodes.  Optional seed is used to seed the randomness for repeatable
    behavior.

    Each arc label will be the arc index, and each node label will be the node
    index.  This facilitates using these labels to index into external sequences
    of node-specific or arc-specific information.

    With this particular set of 100 nodes and 120 random arcs, most of the nodes
    are in a large weakly-connected set, but there are a few other isolated
    nodes and a couple of very small weakly-connected sets.  There are no self
    loops.

    >>> a = make_random_graph(100, 120, seed=6)
    >>> a.has_self_loop(), a.has_cycle(), a.has_multiarcs(), a.is_connected(), a.is_lattice(), a.is_strict_dag(), a.has_join
    (False, True, True, False, False, False, True)
    >>> tuple(len(x) for x in a.connected_sets)
    (89, 1, 1, 1, 3, 1, 2, 1, 1)

    This holds by construction, regardless of the randomness

    >>> a.node_labels_are_node_ids(), a.arc_labels_are_arc_ids()
    (True, True)

    Using a different randomness, show that the nodes are indeed chosen with
    replacement, giving rise to self loops in this case.

    >>> b = make_random_graph(100, 120, seed=0)
    >>> b.has_self_loop()
    True

    This holds by construction, regardless of the randomness

    >>> b.node_labels_are_node_ids(), b.arc_labels_are_arc_ids()
    (True, True)

    Example of plotting a graph.  Uncomment it to display it with Graphviz

    >>> # _ = b.dot_display()
    """
    from random import Random
    rand = Random()
    rand.seed(seed)
    randint = _partial(rand.randint, 0, numnodes - 1)

    builder = GraphBuilder()
    new_node = builder.new_node_label_is_id
    new_arc = builder.new_arc_label_is_id
    for i in xrange(numnodes):
        new_node()
    for i in xrange(numarcs):
        new_arc(randint(), randint())
    return FrozenGraph(builder)


class TopologicalGraphBuilder(GraphBuilder):
    """
    Used to build an directed acyclic graph (DAG) by requiring a topologically
    ordered creation of the arcs in the graph.

    The user is responsible for ensuring that the arcs are created according to
    a topological ordering relation for each of the nodes and for each of the
    arcs.  For nodes this means that the start node for an arc must have been
    created prior to the creation of the end node for an arc.  For arcs this
    means that all of the arcs that end on a given node must have been created
    prior to the creation of any arc that leaves that node.

    Each of these constraints means that there will be no self loops.

    This is a subclass of GraphBuilder and has the same interface.

    Raises ``ValueError`` if an attempt is made to create an arc that violates
    the topological ordering constraints for nodes or arcs.

    >>> builder = TopologicalGraphBuilder()

    Do stuff topologically

    >>> start = builder.new_node()
    >>> mid1 = builder.new_node()
    >>> mid2 = builder.new_node()
    >>> end = builder.new_node()
    >>> assert start < mid1 < mid2 < end
    >>> l1 = builder.new_arc(start, mid2)
    >>> l2 = builder.new_arc(start, end)
    >>> l3 = builder.new_arc(mid2, end)
    >>> assert l1 < l2 < l3

    >>> g = FrozenGraph(builder)
    >>> g.has_self_loop(), g.has_cycle(), g.is_strict_dag()
    (False, False, True)


    Copy construction examples.  These verify the constraints on the
    constructor argument.

    >>> TopologicalGraphBuilder(builder)
    TopologicalGraphBuilder(GraphTables(([None, None, None, None], [0, 0, 2], [2, 3, 3], [None, None, None])))

    >>> TopologicalGraphBuilder(GraphTables(([None, None, None, None], [0, 0, 2], [2, 3, 3], [None, None, None])))
    TopologicalGraphBuilder(GraphTables(([None, None, None, None], [0, 0, 2], [2, 3, 3], [None, None, None])))


    Allow self-loops.

    >>> builder = TopologicalGraphBuilder(allow_self_loops=True)
    >>> num_nodes = 5
    >>> nodes = tuple(builder.new_node_label_is_id() for i in xrange(num_nodes))

    Self loop must be the first outgoing arc on a node.  These are OK.

    >>> tuple(builder.new_arc_label_is_id(node, node) for node in nodes)
    (0, 1, 2, 3, 4)

    This one isn't.

    >>> builder.new_arc_label_is_id(0, 0)
    Traceback (most recent call last):
      ...
    ValueError: arc violates the topological ordering because the end node, 0, already has at least one outgoing arc: failed arc (0, 0, <int>)

    Another example showing the failure if the self loop is not the
    first outgoing arc.

    >>> node1, node2 = builder.new_node_label_is_id(), builder.new_node_label_is_id()
    >>> builder.new_arc_label_is_id(node1, node2)
    5
    >>> builder.new_arc_label_is_id(node2, node2)
    6
    >>> builder.new_arc_label_is_id(node1, node1)
    Traceback (most recent call last):
      ...
    ValueError: arc violates the topological ordering because the end node, 5, already has at least one outgoing arc: failed arc (5, 5, <int>)


    Try to violate the constraint with an arc from a higher to a lower node

    >>> builder.new_arc(end, mid1)
    Traceback (most recent call last):
      ...
    ValueError: arc violates the topological ordering because the start node is greater than or equal to the end node: failed arc (3, 1, <NoneType>)

    Try to violate the constraint with an arc that ends on a node that has an
    outgoing arc

    >>> builder.new_arc(mid1, mid2, 'foo')
    Traceback (most recent call last):
      ...
    ValueError: arc violates the topological ordering because the end node, 2, already has at least one outgoing arc: failed arc (1, 2, <str>)


    Verify that the topological constraints are checked on copy construction

    >>> TopologicalGraphBuilder(GraphTables(([None, None, None, None], [0, 0, 2, 3], [2, 3, 3, 3], [None, None, None, None])))
    Traceback (most recent call last):
      ...
    ValueError: arc violates the topological ordering because the start node is greater than or equal to the end node: failed arc (3, 3, <NoneType>)

    >>> TopologicalGraphBuilder(GraphTables(([None, None, None, None], [0, 0, 2, 1], [2, 3, 3, 2], [None, None, None, 'bar'])))
    Traceback (most recent call last):
      ...
    ValueError: arc violates the topological ordering because the end node, 2, already has at least one outgoing arc: failed arc (1, 2, <str>)
    """

    # Note on future work
    #
    # In a topologically constructed graph, a node moves sequentially
    # through four states: new, not connected; incoming arcs being
    # added; outgoing arcs being added; done with all connections.
    # Either or both of the middle two states can be skipped.
    #
    # At present, the nodes in TopologicalGraphBuilder honor the
    # constraints of the first three of these states, but only
    # implicitly; and the fourth state is implicitly achieved by every
    # node when the graph gets frozen.  Adding a single
    # finish_arc(arc_id) method would make the finishing explicit.
    # However, in this object no fundamentally new purpose would be
    # served by doing that.
    #
    # We may want an alternate topological graph builder that exposes
    # these states more explicitly.  The new functionality would be
    # some sort of semantics associated with that part of the graph
    # that is fixed (that subgraph consisting of all nodes that are in
    # the 'done with all connections' state).  Something like this
    # will be necessary for stream-processing graph builders....

    def __init__(self, init=GraphTables(), allow_self_loops=False):
        """
        Creates a buildable graph that requires topologically ordered
        construction.  If allow_self_loops is True, then a single self loop is
        allowed on a node.  The self loop must be the first outgoing arc for the
        node.
        """
        super(TopologicalGraphBuilder, self).__init__(init)
        self.allow_self_loops = allow_self_loops
        # the outgoing set is those nodes that have at least one
        # outgoing arc, so they can never have a new incoming arc
        # added to them
        self.outgoing = TopologicalGraphBuilder._verify(self)

    @staticmethod
    def _verify(self):
        # the verification step is also used for initialization of the
        # outgoing set
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        outgoing = set()
        check = _partial(self._ordered_check_arc, outgoing)
        for args in itertools.izip(arcstartnodes, arcendnodes, arclabels):
            check(*args)
        return outgoing

    def verify(self):
        super(TopologicalGraphBuilder, self).verify()
        outgoing = TopologicalGraphBuilder._verify(self)
        assert outgoing == self.outgoing

    def _ordered_check_arc(self, outgoing, start_node, end_node, arclabel):
        # checks the topological validity of the proposed arc, updates
        # outgoing if the check succeeds
        if start_node > end_node or (start_node == end_node and not self.allow_self_loops):
            raise ValueError("arc violates the topological ordering because the start node is greater than or equal to the end node: "
                             "failed arc (%d, %d, <%s>)" % (start_node, end_node, type(arclabel).__name__))
        if end_node in outgoing:
            raise ValueError("arc violates the topological ordering because the end node, %d, "
                             "already has at least one outgoing arc: failed arc (%d, %d, <%s>)" % (end_node, start_node, end_node, type(arclabel).__name__))
        outgoing.add(start_node)

    def new_arc(self, start_node, end_node, arclabel=None):
        """
        Creates a new arc in the graph connecting the start_node with
        the end_node, and with an optional arclabel.  It is an error
        if start_node or end_node are not valid node ids created by
        new_node() or if the topological construction constraints are
        violated.  Returns the id of the arc, a non-negative integer.
        """
        self._ordered_check_arc(self.outgoing, start_node, end_node, arclabel)
        return super(TopologicalGraphBuilder, self).new_arc(start_node, end_node, arclabel)

def make_random_DAG(maxnumstarts, endtime, averageoutorder, averagearclength, numnodetokens, averagearcsuccesspermil, seed=None):
    """
    Generate a DAG that has nice structural properties for streaming models of
    speech-centric graph processing.

    Returns the FrozenGraph corresponding to the generated DAG.

    Raises ``ValueError`` if the lattice dies out, which can happen if
    averagearcsuccesspermil and averageoutorder are both small.


    >>> def printit(g): print 'numnodes', g.num_nodes, 'numarcs', g.num_arcs, 'self_loop', g.has_self_loop(), 'cyclic', g.has_cycle(), 'connected', g.is_connected(), 'strict_dag', g.is_strict_dag()

    >>> a = make_random_DAG(5, 20, 7, 10, 4, 800, seed=0)
    >>> printit(a)
    numnodes 136 numarcs 411 self_loop False cyclic False connected True strict_dag True

    >>> assert not a.has_cycle()
    >>> a._verify_topologicals()

    The transpose has same acyclic stats

    >>> nodelabels, starts, ends, arclabels = a.graphseqs
    >>> b = FrozenGraph(GraphTables((nodelabels, ends, starts, arclabels)))
    >>> printit(b)
    numnodes 136 numarcs 411 self_loop False cyclic False connected True strict_dag True

    >>> assert not b.has_cycle()
    >>> a.graphseqs == FrozenGraph(GraphTables((nodelabels, starts, ends, arclabels))).graphseqs
    True

    A larger example

    >>> a = make_random_DAG(1, 250, 6, 25, 50, 600, seed=0)
    >>> printit(a)
    numnodes 10006 numarcs 31767 self_loop False cyclic False connected True strict_dag True

    Some decent ones to plot.  Uncomment to display it with Graphviz

    >>> # _ = make_random_DAG(1, 100, 3, 5, 2, 500, seed=0).dot_display(globals=('rankdir=LR;',))
    >>> # _ = make_random_DAG(1, 60, 2, 3, 2, 700, seed=0).dot_display(globals=('rankdir=LR;',))
    >>> # _ = make_random_DAG(1, 60, 2, 3, 2, 700, seed=0)
    >>> # _ = _.random_dag_lattice().dot_display(globals=('rankdir=LR;',), arc_label_callback=None)

    # work in progress

    >>> endtime = 80
    >>> make_random_DAG(1, endtime, 2, 4, 2, 600, seed=0).lattice_work(endtime) and None

    A graph that dies out due to low-branching factor (averageoutorder) and low
    arc success rate (averagearcsuccesspermil)

    >>> make_random_DAG(1, 320, 3, 10, 5, 300, seed=0)
    Traceback (most recent call last):
     ...
    ValueError: lattice died out: make_random_DAG(1, 320, 3, 10, 5, 300, 0)
    """
    from random import Random
    rand = Random()
    rand.seed(seed)
    randint = _partial(rand.randint, 1)

    outupper = 2 * averageoutorder
    lengthupper = 2 * averagearclength

    # we build the lattice topologically; we do this by having a heap
    # of pending arcs (not yet created in the graph) with the ordering
    # based on their end time
    activeheap = list()
    ending = [False]
    def push_pending_arcs(start):
        starttime, _startnodetoken = start
        numarcs = randint(outupper)
        for i in xrange(numarcs):
            if not (averagearcsuccesspermil >= randint(1000)):
                continue
            endnodetime = starttime + randint(lengthupper)
            endnodetoken = randint(numnodetokens)
            end = endnodetime, endnodetoken
            # note that start goes into the heap too
            _heappush(activeheap, (end, start))

    builder = TopologicalGraphBuilder()
    # new_node = builder.new_node_label_is_id
    def new_node(key):
        # return builder.new_node_label_is_id()
        assert len(key) == 2
        return builder.new_node(key)
        # return builder.new_node("%s, %s" % key)
    new_arc = builder.new_arc_label_is_id
    nodesbytimetoken = dict()

    # prime the heap
    starttime = 0
    for start in ((starttime, randint(numnodetokens)) for i in xrange(maxnumstarts)):
        # we push these pending create arcs regardless of whether the
        # endnode will be new or not (unlike for the rest of the lattice)
        push_pending_arcs(start)
        if start not in nodesbytimetoken:
            # use start, so that the node token is part of a node's identity
            nodesbytimetoken[start] = new_node(start)

    while activeheap:
        # specs for the arc we're going to make
        end, start = _heappop(activeheap)
        starttime, _starttoken = start
        if starttime >= endtime:
            ending[0] = True
            continue
        endnode = nodesbytimetoken.get(end)
        if endnode is None:
            # create the new end node
            endnode = nodesbytimetoken[end] = new_node(end)
            # put specs for successors to end node in the heap
            push_pending_arcs(end)
        startnode = nodesbytimetoken[start]
        arc = new_arc(startnode, endnode)

    if not ending[0]:
        raise ValueError("lattice died out: make_random_DAG(%d, %d, %d, %d, %d, %d, %d)"
                         % (maxnumstarts, endtime, averageoutorder, averagearclength, numnodetokens, averagearcsuccesspermil, seed))

    # we're done
    dag = FrozenGraph(builder)
    assert dag.is_strict_dag()
    # assert dag.node_labels_are_node_ids() and dag.arc_labels_are_arc_ids()
    return dag

def make_random_confusion_network(min_links, max_links, min_arcs_per_link, max_arcs_per_link, num_arc_labels, seed=None):
    """
    >>> g = make_random_confusion_network(3, 20, 1, 10, 50)
    """
    pass


# handy callbacks for the depth-first iterator
def _pre_callback(self, id):
    print '(%d' % (id,),

def _post_callback(self, id):
    print '%d)' % (id,),

def _cycle_callback(self, parent, child):
    if parent == child:
        # self loop
        print '(^%d)' % (child,),
    else:
        print '(c%d)' % (child,),
        
    #print ('%d' + ('^' if parent == child else '()') + '%d') % (parent, child),

def _join_callback(self, parent, child):
    assert parent != child
    # a join
    print '(%dj%d)' % (parent, child,),


class FrozenGraph(GraphBase):
    """
    An immutable graph object suitable for graph operations that depend on
    global structural properties of the graph.

    This graph contains two disconnected subgraphs, both of which are cyclic,
    one with a self loop.

    >>> a = FrozenGraph(GraphTables(((0, 1, 2, 3), (0, 1, 2), (1, 0, 2), (3, None, 5))))
    >>> a.has_cycle(), a.is_connected(), a.has_self_loop(), a.has_multiarcs(), a.is_lattice(), a.node_labels_are_node_ids(), a.arc_labels_are_arc_ids()
    (True, False, True, False, False, True, False)

    Reverse each of the conditions

    >>> b = FrozenGraph(GraphTables(((-1, -2), (0, 0), (1, 1), (0, 1))))
    >>> b.has_cycle(), b.is_connected(), b.has_self_loop(), b.has_multiarcs(), b.is_lattice(), b.node_labels_are_node_ids(), b.arc_labels_are_arc_ids()
    (False, True, False, True, True, False, True)

    XXX [Fix this documentation] Case of a cyclic graph that is not
    bakis_cyclic, i.e. the cyclicity is only due to self loops

    >>> c = FrozenGraph(GraphTables((('a', 'b'), (0, 1), (1, 1), ('c', 'd'))))
    >>> c.has_cycle(), c.has_self_loop(), c.strict_is_cyclic()
    (False, True, True)

    >>> b = SetGraphBuilder()
    >>> _ = b.add_arc(0, 1)
    >>> _ = b.add_arc(0, 2)
    >>> d = FrozenGraph(b)
    >>> d.has_join, d.is_tree, d.is_tree_strict
    (False, True, True)

    >>> _ = b.add_arc(2, 2)
    >>> e = FrozenGraph(b)
    >>> e.has_join, e.is_tree, e.is_tree_strict
    (False, True, False)

    >>> _ = b.add_arc(3, 2)
    >>> f = FrozenGraph(b)
    >>> f.has_join, f.is_tree, f.is_tree_strict
    (True, False, False)
    """

    def __init__(self, source):
        """
        Creates a frozen graph which is a snapshot of the current state of
        source.  source must be an instance of GraphTables or GraphBase.
        """
        node_id_by_label = None
        if isinstance(source, GraphTables):
            sourceseq = source
        elif isinstance(source, GraphBase):
            source.verify()
            sourceseq = source.graphseqs
            if isinstance(source, SetGraphBuilder):
                node_id_by_label = source.node_id_by_label
        elif isinstance(source, GraphArcNodeStructure):
            source.verify()
            node_labels = xrange(source.num_nodes)
            arcstartnodes, arcendnodes = source._arc_data
            arc_labels = xrange(source.num_arcs)
            sourceseq = node_labels, arcstartnodes, arcendnodes, arc_labels
        else:
            raise TypeError("expected %s or %s, got %s" % (GraphTables, GraphBase, type(source)))

        seqs = GraphTables(tuple(seq) for seq in sourceseq)
        hash(seqs)
        super(FrozenGraph, self).__init__(seqs)

        if node_id_by_label is None:
            self.node_id_by_user_id = self.user_id_by_node_id = lambda x: x
        else:
            self.node_id_by_user_id = lambda x: node_id_by_label[x]
            node_labels = seqs[0]
            self.user_id_by_node_id = lambda x: node_labels[x]

        # create some non-local structural info
        self._make_node_properties()
        self._make_arc_properties()
        self._make_adjacency_lists()
        self._make_terminal_lists()
        self._make_topological_lists()
        self._make_connected_sets()

        self._is_confusion_network = None

        FrozenGraph._verify(self)

        assert self.is_confusion_network() in (False, True)
        #assert self._has_cycle == self._is_cyclic == self.has_cycle() == self.is_cyclic()

    @staticmethod
    def _verify(self):
        self._verify_adjacency()
        self._verify_terminals()
        self._verify_topologicals()
        self._verify_connecteds()

        assert self._is_confusion_network in (None, False, True)

    def verify(self):
        super(FrozenGraph, self).verify()
        FrozenGraph._verify(self)


    # Some graph theoretic properties

    def has_self_loop(self):
        """
        Return ``True`` if there is any node with a self loop, ``False``
        otherwise.
        """
        return self._has_self_loop

    def has_cycle(self):
        """
        Return ``True`` if there are is any multi-node cycle in the graph,
        ``False`` otherwise.
        """
        return self._has_cycle

    def strict_is_cyclic(self):
        """
        Return ``True`` if the graph contains one or more cycle of any kind,
        ``False`` if the graph is strictly acyclic (strictly a DAG).  For the
        purposes of this function, a multinode cycle or a self loop constitutes
        cyclicity.  See also :meth:`has_self_loop` and :meth:`has_cycle`.
        """
        return self.has_cycle() or self.has_self_loop()

    def has_multiarcs(self):
        """
        Return True if there are any two distinct nodes with parallel arcs
        between them, False if there is either zero or one arc between each pair
        of distinct nodes.  This measure is of more interest for undirected
        graphs than directed graphs.
        """
        return self._has_multiarcs

    def is_connected(self):
        """
        Return True if the nodes of the graph are weakly connected, False if
        there are disconnected subsets of nodes.
        """
        return len(self.connected_sets) == 1

    def is_lattice(self):
        """
        Return True if the graph has no cycles and has a single start node and a
        single end node.
        """
        return not self.has_cycle() and len(self.startnodes) == len(self.endnodes) == 1

    @property
    def has_join(self):
        """
        ``True`` if there are any joins in the graph, ``False`` otherwise.  A
        join occurs when two or more arcs end on the same node.
        """
        return self._has_join

    @property
    def is_tree(self):
        """
        ``True`` if the graph is a tree, ``False`` otherwise.  There are many
        ways to decide if a graph is a tree.  For our purposes, a graph is a
        tree if it has a single startnode and no joins.
        """
        return len(self.startnodes) == 1 and not self.has_join

    @property
    def is_tree_strict(self):
        """
        ``True`` if the graph is strictly a tree, ``False`` otherwise.  There
        are many ways to define a strict a tree.  For our purposes, a graph is a
        strict tree if it has a single startnode, no joins, and no self-loops.
        """
        return len(self.startnodes) == 1 and not self.has_join and not self.has_self_loop()

    def is_strict_dag(self):
        """
        Return True if the graph has no self loops and no cycles.
        """
        return not self.has_self_loop() and not self.has_cycle()

    def is_confusion_network(self):
        """
        Returns True if the graph has the structure of a confusion network (also
        called a consensus network, or a sausage, or a linear grammar).

        A confusion network is a strict DAG (no cycles or self loops) with a
        single start node and a single end node.  The essential structural
        property is that for each node in the graph, u, all the arcs that start
        from u end on a single node, v, that is distinct from u.

        >>> FrozenGraph(GraphTables((tuple(xrange(3)), (0,0,1,1,1,1), (1,1,2,2,2,2), tuple(xrange(6))))).is_confusion_network()
        True
        >>> print FrozenGraph(GraphTables((tuple(xrange(3)), (0,0,1,1,1,1), (1,2,2,2,2,2), tuple(xrange(6))))).is_confusion_network()
        False
        """
        # a lazy, memoizing implementation
        confusion = self._is_confusion_network
        if confusion is None:
            def is_confusion():
                if not self.is_lattice() or self.has_self_loop():
                    return False
                assert self.is_strict_dag()
                assert not self.has_cycle()
                #assert not self.has_self_loop()
                endnode, = self.endnodes
                nodeadjout = self.nodeadjout
                for node in xrange(self.num_nodes):
                    if node == endnode:
                        continue
                    num_end_nodes = len(set(end for end, _ in nodeadjout[node]))
                    if num_end_nodes != 1:
                        return False
                return True
            confusion = self._is_confusion_network = is_confusion()
            # XXX this assert will fail on an empty graph...
            # XXX revisit whether this a simpler way to set _is_confusion_network, e.g. from __init__
            if self.is_lattice() and not self.has_self_loop():
                if self.num_nodes > 0:
                    quickie = len(set((start, end) for start, end in itertools.izip(self.arcstartnodes, self.arcendnodes))) == self.num_nodes - 1
                    if confusion != quickie:
                        self.dot_display(globals=('rankdir=LR;',))
                    assert confusion == quickie
                else:
                    assert confusion == True
        assert confusion in (True, False)
        return confusion


    def node_labels_are_node_ids(self):
        """
        Return True if each node label is the index of the node.
        """
        return self._node_labels_are_node_ids

    def arc_labels_are_arc_ids(self):
        """
        Return True if each arc label is the index of the arc.
        """
        return self._arc_labels_are_arc_ids

    @property
    def startnodes(self):
        """
        A tuple of the indices of the starting-node ids.  These are nodes with
        no incoming arcs.
        """
        return self._startnodes

    @property
    def endnodes(self):
        """
        A tuple of the indices of the ending-node ids.  These are nodes with
        no outgoing arcs.
        """
        return self._endnodes

    @property
    def terminals(self):
        """
        A pair of tuples, the :attr:`startnodes` and the :attr:`endnodes`.
        """
        return self.startnodes, self.endnodes

    def get_terminals(self):
        """
        Returns a two-item tuple, (startnodes, endnodes), where startnodes is a
        tuple of the start node ids (nodes with no incoming arcs) and endnodes
        is a tuple of the end node ids (nodes with no outgoing arcs).
        """
        return self.startnodes, self.endnodes

    def get_reversed(self):
        """
        Returns a new FrozenGraph based on the contents of self.  In
        the new graph the directions of all the arcs are reversed.

        >>> g = FrozenGraph(GraphTables(((-10, -11, -12), (2, 1, 1), (0, 2, 0,), (-20, -21, -22))))
        >>> g.get_reversed()
        FrozenGraph(GraphTables(((-10, -11, -12), (0, 2, 0), (2, 1, 1), (-20, -21, -22))))
        >>> g.get_reversed().get_reversed() == g
        True
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        return FrozenGraph(GraphTables((nodelabels, arcendnodes, arcstartnodes, arclabels)))

    def get_nocycles(self):
        """
        Returns a new FrozenGraph based on the contents of self.  The set of
        nodes is unchanged.  Cycles appearing in the original graph are broken
        in the new graph by removing cycle-completing arcs.  The set of
        cycle-completing arcs is not canonical; that is, a different ordering of
        nodes and arcs in the original graph can result in different set of arcs
        being removed.

        XXX note: an alternative algorithm would just reverse each of
        the cycle-completing arcs

        >>> g = FrozenGraph(GraphTables(((-10, -11, -12), (2, 1, 0, 1), (0, 2, 2, 0), (-20, -21, -22, None))))
        >>> g.get_nocycles()
        FrozenGraph(GraphTables(((-10, -11, -12), (2, 1, 1), (0, 2, 0), (-20, -21, None))))
        """
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        cycle_breakers = self._cycle_breakers
        new_arcs = tuple(list(x) for x in (arcstartnodes, arcendnodes, arclabels))
        for drop_id in reversed(tuple(id for id, ends in enumerate(itertools.izip(arcstartnodes, arcendnodes)) if ends in cycle_breakers)):
            for x in new_arcs:
                del x[drop_id]
        return FrozenGraph(GraphTables((nodelabels,) + new_arcs))

    def get_closure_lattice(self, start_node_iter, end_node_iter):
        start_closure = self.get_node_forward_transitive_closure(start_node_iter, True)
        end_closure = self.get_node_backward_transitive_closure(end_node_iter, True)
        # the set of nodes
        pruned_node_closure = start_closure & end_closure
        keep_nodes = tuple(sorted(pruned_node_closure))
        def invert(seq):
            map = listn(keep_nodes[-1] + 1)
            for index, item in enumerate(seq):
                map[item] = index
            return tuple(map)
        keep_nodes_map = invert(keep_nodes)

        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        new_node_labels = tuple(label for id, label in enumerate(nodelabels) if id in pruned_node_closure)
        keep_arcs = tuple((keep_nodes_map[start], keep_nodes_map[end], label) for start, end, label in itertools.izip(arcstartnodes, arcendnodes, arclabels) if start in pruned_node_closure and end in pruned_node_closure)

        new_arcs = tuplenoflist(3)
        for arc in keep_arcs:
            for index in xrange(3):
                new_arcs[index].append(arc[index])

        return FrozenGraph(GraphTables(((new_node_labels,) + new_arcs)))

    def random_dag_lattice(self):
        nodelabels = self.nodelabels

        # XXX special case for development only!
        start_nodes = (0,)
        end_nodes = len(nodelabels) - 2, len(nodelabels) - 1
        print 'start_nodes:', start_nodes, 'end_nodes:', end_nodes
        return self.get_closure_lattice(start_nodes, end_nodes)

    def get_canonical_DAG(self):
        """
        Returns a new FrozenGraph based on the contents of self.  In the new
        graph the nodes and arcs are rearranged to be indexed in a topological
        order.

        Raises a ``ValueError`` if has_cycle() is True.

        >>> FrozenGraph(GraphTables(((-10, -11, -12), (2, 1, 1), (0, 2, 0,), (-20, -21, -22)))).get_canonical_DAG()
        FrozenGraph(GraphTables(((-11, -12, -10), (0, 1, 0), (1, 2, 2), (-21, -20, -22))))
        """
        if self.has_cycle():
            raise ValueError("expected has_cycle() to be False")
        return FrozenGraph(self.get_mapped_tables((self.toponodes, self.topoarcs)))

    def get_line_graph(self, node_labeler=None, arc_labeler=None):
        """
        Returns a new FrozenGraph based on the contents of self.  The new graph
        will be the line-graph of the original, also called the interchange or
        edge-dual graph.

        The line-graph (V*, E*) of a directed graph (V,E) has nodes V* which
        correspond to the arcs E of the original graph.  Two nodes v1 and v2 in
        V* have an arc between them in E* if and only if the arcs e1 and e2 in E
        that they correspond to form a path of length two in V, that is, if and
        only if there is a node v in the original graph such that e1 is an
        in-arc of v and e2 is an out-arc of v.

        By default, nodes in the resulting line-graph will have labels which are
        triples (pred_node_label, arc_label, succ_node_label) where these labels
        come from the original graph; arcs in the resulting graph will have
        labels which are triples (in_arc_label, node_label, out_arc_label),
        again with labels from the original graph.  Callers may provide their
        own labeling functions for either nodes or arcs, each will be called
        with three arguments which are exactly the respective triples shown
        above.

        >>> g0 = FrozenGraph(GraphTables(((0, 1, 2, 3), (0, 1, 2), (1, 2, 3), ('A', 'B', 'C'))))

        >>> lg0 = g0.get_line_graph()
        >>> lg0
        FrozenGraph(GraphTables((((0, 'A', 1), (1, 'B', 2), (2, 'C', 3)), (0, 1), (1, 2), (('A', 1, 'B'), ('B', 2, 'C')))))


        Here's an example with a simpler labeler, showing that a diamond-shaped
        lattice is converted to a disconnected graph.  Repeated applications of
        the line-graph operator result in the empty graph.

        >>> diamond = FrozenGraph(GraphTables((('w', 'x', 'y', 'z'), (0, 0, 1, 2), (1, 2, 3, 3), ('A', 'B', 'C', 'D'))))
        >>> labeler = lambda left, middle, right: middle
        >>> lg1 = diamond.get_line_graph(labeler, labeler)
        >>> lg1
        FrozenGraph(GraphTables((('A', 'B', 'C', 'D'), (0, 1), (2, 3), ('x', 'y'))))

        >>> lg2 = lg1.get_line_graph(labeler, labeler)
        >>> lg2
        FrozenGraph(GraphTables((('x', 'y'), (), (), ())))

        >>> lg3 = lg2.get_line_graph(labeler, labeler)
        >>> lg3
        FrozenGraph(GraphTables(((), (), (), ())))

        Cycle graphs are idempotent under two applications of the line-graph operator.

        >>> cycle = FrozenGraph(GraphTables((('a', 'b', 'c', 'd'), (0, 1, 2, 3), (1, 2, 3, 0), ('A', 'B', 'C', 'D'))))
        >>> cycle_lg1 = cycle.get_line_graph(labeler, labeler)

        >>> cycle_lg1
        FrozenGraph(GraphTables((('A', 'B', 'C', 'D'), (3, 0, 1, 2), (0, 1, 2, 3), ('a', 'b', 'c', 'd'))))

        >>> cycle_lg2 = cycle_lg1.get_line_graph(labeler, labeler)
        >>> cycle_lg2 == cycle
        True

        Here's an example with two unconnected cycles of circumferences 3 and 4

        >>> two_cycles = FrozenGraph(GraphTables((('a', 'b', 'c', 'd', 'x', 'y', 'z'),
        ...                                       (0, 1, 2, 3, 4, 5, 6),
        ...                                       (1, 2, 3, 0, 5, 6, 4),
        ...                                       ('A', 'B', 'C', 'D', 'X', 'Y', 'Z'))))
        >>> two_cycles_lg1 = two_cycles.get_line_graph(labeler, labeler)
        >>> two_cycles_lg2 = two_cycles_lg1.get_line_graph(labeler, labeler)
        >>> two_cycles_lg2 == two_cycles
        True

        >>> debruijn_labeler = lambda left, middle, right: str(left) + str(right)[-1]
        >>> empty_labeler = lambda a,b,c: None
        >>> debruijn0 = FrozenGraph(GraphTables((('0', '1'), (0, 1, 0, 1), (0, 0, 1, 1), (None, None, None, None))))
        >>> debruijn1 = debruijn0.get_line_graph(debruijn_labeler, empty_labeler)
        >>> debruijn2 = debruijn1.get_line_graph(debruijn_labeler, empty_labeler)
        >>> debruijn3 = debruijn2.get_line_graph(debruijn_labeler, empty_labeler)
        >>> debruijn3.num_nodes
        16

        """

        def default_node_labeler(pred_node_label, arc_label, succ_node_label):
            return (pred_node_label, arc_label, succ_node_label)

        def default_arc_labeler(in_arc_label, node_label, out_arc_label):
            return (in_arc_label, node_label, out_arc_label)

        if node_labeler is None:
            node_labeler = default_node_labeler
        if arc_labeler is None:
            arc_labeler = default_arc_labeler

        gb = GraphBuilder()

        # Pass 1, create all the nodes in the result by iterating over the arcs
        # in self, build an arc-index to new_node_id table
        arc_index_to_new_node_id = dict()
        for (s, e, arc_label, arc_index) in itertools.izip(self.arcstartnodes, self.arcendnodes, self.arclabels, itertools.count()):
            new_node_label = node_labeler(self.nodelabels[s], arc_label, self.nodelabels[e])
            new_node_id = gb.new_node(new_node_label)
            arc_index_to_new_node_id[arc_index] = new_node_id

        # Pass 2, create all the arcs in the result by iterating over the
        # arc-pairs in self.  An arc-pair is any two arcs where the first goes
        # into some node and the second leaves that same node.
        for id, node_label in enumerate(self.nodelabels):
            for pred_node_id, in_arc_id in self.nodeadjin[id]:
                for succ_node_id, out_arc_id in self.nodeadjout[id]:
                    new_arc_label = arc_labeler(self.arclabels[in_arc_id], node_label, self.arclabels[out_arc_id])
                    new_start_id = arc_index_to_new_node_id[in_arc_id]
                    new_end_id = arc_index_to_new_node_id[out_arc_id]
                    gb.new_arc(new_start_id, new_end_id, new_arc_label)

        return FrozenGraph(gb)


    def get_node_in_arcs(self, node_id):
        """
        Returns a list of the arc_ids of the incident arcs on node_id.

        Query a graph with a single node with two self loops
        >>> FrozenGraph(GraphTables(((-10,), (0, 0), (0, 0,), (-20, -21)))).get_node_in_arcs(0)
        [0, 1]
        """
        node_id = self.node_id_by_user_id(node_id)
        self._check_node_id(node_id, 'node_id')
        return sorted(arcid for nodeid, arcid in self.nodeadjin[node_id])

    def lattice_work(self, endtime):
        nodelabels = self.nodelabels

        startnodes = self.startnodes
        endnodes = tuple(id for id, (time, token) in enumerate(nodelabels) if time >= endtime)

        node_closure_forward, arc_closure_forward = self.get_forward_transitive_closures(startnodes, force_reflexive=True)
        node_closure_backward, arc_closure_backward = self.get_backward_transitive_closures(endnodes, force_reflexive=True)

        node_closure = node_closure_forward & node_closure_backward
        arc_closure = arc_closure_forward & arc_closure_backward

        node_closure_labels = frozenset(nodelabels[node_id] for node_id in node_closure)

        end_closures = tuple(self.get_backward_transitive_closures((endnode,), force_reflexive=False) for endnode in endnodes)

        fixed_history = end_closures[0][1]
        for nc, ac in end_closures[1:]:
            fixed_history &= ac

        def node_attributes_callback(label, is_start, is_end):
            return ('style=%s' % ('normail' if label in node_closure_labels else 'dotted',),)
        def arc_attributes_callback(label):
            if label in fixed_history:
                return ('style=bold',)
            if label not in arc_closure:
                return ('style=dotted',)
            return None

        return
        self.dot_display(globals=('rankdir=LR;',), node_attributes_callback=node_attributes_callback, arc_attributes_callback=arc_attributes_callback)
        # self.random_dag_lattice().dot_display(globals=('rankdir=LR;',), arc_label_callback=None)


    def get_forward_transitive_closures(self, seed_node_id_iter, force_reflexive=False):
        """
        Find the part of the graph that is reachable going forward from the set
        of nodes in seed_node_id_iter.

        Return a tuple of two frozensets: the ids of the nodes and the ids of
        the arcs in the forward-reachable part of the graph.  If optional
        force_reflexive is True then include all the nodes from
        seed_node_id_iter in the set of returned node ids regardless of whether
        the graph has cycles that include them.

        >>> FrozenGraph(GraphTables(((-10, -11, -12), (2, 1, 1), (0, 2, 0,), (-20, -21, -22)))).get_forward_transitive_closures((1,))
        (frozenset([0, 2]), frozenset([0, 1, 2]))
        >>> FrozenGraph(GraphTables(((10, 20), (0, 0, 1, 1), (0, 1, 0, 1), ('00', '01', '10', '11')))).get_forward_transitive_closures((0,))
        (frozenset([0, 1]), frozenset([0, 1, 2, 3]))
        """
        return self._get_transitive_closures(self.node_id_by_user_id, self.user_id_by_node_id, self.nodeadjout, seed_node_id_iter, force_reflexive)

    def get_backward_transitive_closures(self, seed_node_id_iter, force_reflexive=False):
        """
        Find the part of the graph that is reachable going backward from the set
        of nodes in seed_node_id_iter.  I.e. find the part of the graph that can
        reach the nodes in seed_node_id_iter when going forward.

        Return a tuple of two frozensets: the ids of the nodes and the ids of
        the arcs in the backward-reachable part of the graph.  If optional
        force_reflexive is True then include all the nodes from
        seed_node_id_iter in the set of returned node ids regardless of whether
        the graph has cycles that include them.

        >>> FrozenGraph(GraphTables(((-10, -11, -12), (2, 1, 1), (0, 2, 0,), (-20, -21, -22)))).get_backward_transitive_closures((2,))
        (frozenset([1]), frozenset([1]))
        >>> FrozenGraph(GraphTables(((10, 20), (0, 0, 1, 1), (0, 1, 0, 1), ('00', '01', '10', '11')))).get_backward_transitive_closures((0,))
        (frozenset([0, 1]), frozenset([0, 1, 2, 3]))
        """
        return self._get_transitive_closures(self.node_id_by_user_id, self.user_id_by_node_id, self.nodeadjin, seed_node_id_iter, force_reflexive)

    @staticmethod
    def _get_transitive_closures(node_id_by_user_id, user_id_by_node_id, nodeadj, seed_node_id_iter, force_reflexive):
        # worker for building sets of transitive closures
        seeds = tuple(node_id_by_user_id(node_id) for node_id in seed_node_id_iter)
        workset = set(seeds)
        node_closure = set()
        arc_closure = set()
        while workset:
            for node_id, arc_id in nodeadj[workset.pop()]:
                arc_closure.add(arc_id)
                if node_id not in node_closure:
                    node_closure.add(node_id)
                    assert node_id not in workset
                    workset.add(node_id)
        if force_reflexive:
            node_closure.update(seeds)
        return frozenset(user_id_by_node_id(node_id) for node_id in node_closure), frozenset(arc_closure)


    def iter_nodes(self, reverse_order=False):
        """
        Returns a generator over the nodes in the graph.  The generator yields a
        three-item tuple: the first item is the label of the node, the second
        item is a generator over the in-arcs of the node, and the third item is
        a generator over the out-arcs of the node.  Each of these arc generators
        yields a pair (arc_label, node_label) where the node label is the label
        of the node at the other end of the arc relative to the node currently
        being yielded.  When a topological ordering of nodes is possible, the
        nodes will be iterated in forward topological order by default, and in
        reverse topological order if reverse_order is True.  If there is no
        topological ordering because the graph has cycles, then the order of
        iteration is undefined.
        """

        toponodes = self.toponodes if not reverse_order else reversed(self.toponodes)
        nodeadjin = self.nodeadjin
        nodeadjout = self.nodeadjout

        def node_iter():
            for node_id in toponodes:
                node_label = self.get_node_label(node_id)
                in_arc_iter = ((self.get_arc_label(arc_id),
                                self.get_node_label(other_node_id)) for (other_node_id, arc_id) in nodeadjin[node_id])
                out_arc_iter = ((self.get_arc_label(arc_id),
                                 self.get_node_label(other_node_id)) for (other_node_id, arc_id) in nodeadjout[node_id])
                yield node_label, in_arc_iter, out_arc_iter

        return node_iter()

    def iterpaths(self, cost_func, start_node, end_node):
        """
        Returns a generator that yields paths between the start_node and
        end_node based on the summed cost of the arcs on the path.  The
        cost_func will be called with a single argument, an arc label, and it
        should return the cost for the arc.  See GraphBuilder.new_arc().

        The generator yields three-item tuples.  The first item is the cost for
        the path, the second item is a tuple of arc-ids for the path, the third
        item is the graph object itself.  The costs are non-decreasing for
        successive yields.  See GraphBase.get_arc() for getting the information
        associated with an arc id.

        It is an error to call this if the graph contains cycles.  See
        strict_is_cyclic().
        """

        self._check_node_id(start_node, "start_node")
        self._check_node_id(end_node, "end_node")

        # this is an overly conservative check because there is not
        # necessarily a cycle that's reachable from start_node, but
        # until we have more precise cycle checking we'll bail out
        # here
        self._strict_check_acyclic()

        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        nodeoutarcs = self.nodeoutarcs

        # heap for candidate paths
        pathheap = list()
        push = _partial(_heappush, pathheap)
        pop = _partial(_heappop, pathheap)

        # There are of course numerous optimizations that could be
        # added to this simple implementation of Djikstra's algorithm.
        # Using a minimum spanning tree to find the best path would be
        # a good start.  Some fairly recent (circa 2002)
        # speech-recognition nbest algorithms are much better.

        # a path is a tuple of a score and a tuple of arc-ids
        # note: each path is a separate object; this is simple, tho inefficient
        def extend(score, path, nodeid):
            for outarc in nodeoutarcs[nodeid]:
                arccost = cost_func(arclabels[outarc])
                newscore = score + arccost
                if newscore < score:
                    raise ValueError("expected non-decreasing costs, but arc %d "
                                     "leaving node %d has a cost of %s which, when "
                                     "added to path-score %s, gives newscore %s"
                                     % (outarc, nodeid, arccost, score, newscore,))
                push((newscore, path + (outarc,),))

        # initialize
        extend(0, (), start_node)
        # search
        while pathheap:
            topscore, toppath = pop()
            topendnode = arcendnodes[toppath[-1]]
            if topendnode == end_node:
                yield topscore, toppath, self
            else:
                extend(topscore, toppath, topendnode)

    def iter_arc_ngrams(self, ngram_order):
        """
        Returns a generator that yields information about the
        arc-ngram-sequences of length *ngram_order* in the graph.

        Each iteration yields a pair of tuples for an ngram in the graph.  The
        first tuple in the pair is the sequence of *ngram_order + 1* node labels
        for the nodes that define the ngram.  The second tuple in the pair is
        the sequence of *ngram_order* arc labels for the arcs in the ngram.

        If self.is_strict_dag() is not ``True`` a ``ValueError`` is raised.

        An example where the tri-grams reveal a four-gram.

        >>> g = make_random_DAG(1, 15, 2, 4, 2, 600, seed=0)

        For display purposes, convert the node labels into node
        indicies.

        >>> nodelabels, arcstartnodes, arcendnodes, arclabels = g.graphseqs
        >>> g = FrozenGraph(GraphTables((tuple(xrange(len(nodelabels))), arcstartnodes, arcendnodes, arclabels)))
        >>> for ngram in g.iter_arc_ngrams(3): print ngram
        ((0, 1, 6, 9), (0, 6, 10))
        ((0, 1, 6, 9), (1, 6, 10))
        ((0, 3, 4, 5), (3, 4, 5))
        ((0, 3, 4, 7), (3, 4, 7))
        ((0, 3, 4, 8), (3, 4, 8))
        ((3, 4, 5, 9), (4, 5, 9))
        >>> for ngram in g.iter_arc_ngrams(4): print ngram
        ((0, 3, 4, 5, 9), (3, 4, 5, 9))
        """

        # self.dot_display(globals=('rankdir=LR;',))
        if not self.is_strict_dag():
            raise ValueError("expected is_strict_dag() to be True, got %s" % (self.is_strict_dag(),))

        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        nodeadjout = self.nodeadjout

        out_stack, node_stack, arc_stack = tuplenoflist(3)

        for startnode in xrange(self.num_nodes):
            node_stack.append((startnode, nodelabels[startnode]))
            out_stack.append(iter(nodeadjout[startnode]))
            while out_stack:
                assert len(out_stack) == len(node_stack) == len(arc_stack) + 1
                try:
                    endnode, arc = out_stack[-1].next()
                except StopIteration:
                    startnode, _ = node_stack.pop()
                    out_stack.pop()
                    if out_stack:
                        arc_stack.pop()
                else:
                    node_stack.append((endnode, nodelabels[endnode]))
                    arc_stack.append(arclabels[arc])
                    if len(out_stack) < ngram_order:
                        startnode = endnode
                        out_stack.append(iter(nodeadjout[startnode]))
                    else:
                        yield tuple(nodelabel for _, nodelabel in node_stack), tuple(arc_stack)
                        node_stack.pop()
                        arc_stack.pop()

    def get_topological_reorderings(self):
        """
        Find a topological ordering for the nodes and arcs in an acyclic graph.

        Return two sequences, each of which maps old ids to new ids.  The first
        is the map of node ids, the second is the map of arc ids.

        XXX the funcgtionality of this method and get_mapped_tables() are
        largely obsoleted by get_canonical_DAG()
        """
        self._strict_check_acyclic()

        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        nodeoutarcs = self.nodeoutarcs

        unset = -1
        # the max depth at which we see the item
        max_depth_by_node = list(unset for x in xrange(self.num_nodes))
        max_depth_by_arc =  list(unset for x in xrange(self.num_arcs))

        # breadth-first traversal of graph for filling in the max depths
        # XXX make iterators for traversals
        nodes_this_depth = self.startnodes
        for depth in itertools.count():
            if not nodes_this_depth:
                break
            # print "depthx", depth, '',
            nodes_next_depth = set()
            for node in nodes_this_depth:
                max_depth_by_node[node] = depth
                for arc in nodeoutarcs[node]:
                    max_depth_by_arc[arc] = depth
                    nodes_next_depth.add(arcendnodes[arc])
            nodes_this_depth = nodes_next_depth

        assert unset not in max_depth_by_node
        assert unset not in max_depth_by_arc

        nodes_by_depth = tuple(set() for x in xrange(depth))
        for node_id, node_depth in itertools.izip(itertools.count(), max_depth_by_node):
            nodes_by_depth[node_depth].add(node_id)

        # note: arcs only go to depth - 1
        arcs_by_depth = tuple(set() for x in xrange(depth - 1))
        for arc_id, arc_depth in itertools.izip(itertools.count(), max_depth_by_arc):
            arcs_by_depth[arc_depth].add(arc_id)

        # XXX use labels-based ordering to give more canonical representation
        node_map = tuple(nodeid for nodeset in nodes_by_depth for nodeid in sorted(nodeset))
        arc_map = tuple(arcid for arcset in arcs_by_depth for arcid in sorted(arcset))

        return node_map, arc_map


    def get_mapped_tables(self, maps):
        """
        Using the supplied id maps, return a GraphTables object with mapped
        graph tables.

        XXX the funcgtionality of this method and get_topological_reorderings()
        are largely obsoleted by get_canonical_DAG()
        """
        def invert(seq):
            map = listn(len(seq))
            for index, item in enumerate(seq):
                map[item] = index
            assert None not in map
            return tuple(map)

        node_map, arc_map = maps
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs

        # the permutations are used to directly place the labels to their new indices
        new_nodelabels = tuple(nodelabels[mapped_node_id] for mapped_node_id in node_map)
        new_arclabels = tuple(arclabels[mapped_arc_id] for mapped_arc_id in arc_map)

        # the arcstartnodes and arcendnodes must get inverse-mapped to reflect their new indices
        inv_node_map = invert(node_map)
        new_arcstartnodes = tuple(inv_node_map[arcstartnodes[mapped_arc_id]] for mapped_arc_id in arc_map)
        new_arcendnodes = tuple(inv_node_map[arcendnodes[mapped_arc_id]] for mapped_arc_id in arc_map)

        return GraphTables((new_nodelabels, new_arcstartnodes, new_arcendnodes, new_arclabels,))


    def _make_node_properties(self):
        nodelabels = self.nodelabels
        self._node_labels_are_node_ids = (nodelabels == tuple(xrange(len(nodelabels))))

    # some arc-based properties
    def _make_arc_properties(self):
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs

        self._arc_labels_are_arc_ids = (arclabels == tuple(xrange(len(arclabels))))

        # self loops
        self_loops = set(startnode for startnode, endnode in itertools.izip(arcstartnodes, arcendnodes) if startnode == endnode)
        self._has_self_loop = True if self_loops else False

        non_self_arcs = tuple((startnode, endnode) for startnode, endnode in itertools.izip(arcstartnodes, arcendnodes) if startnode != endnode)
        # inequality will be True if there is more than one instance of any (startnode, endnode) in non_self_arcs
        self._has_multiarcs = (len(set(non_self_arcs)) < len(non_self_arcs))

    # adjacency list support
    def _make_adjacency_lists(self):
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        numnodes = len(nodelabels)

        # each item in these sequences is a pair, (node_index,
        # arc_index)
        nodeadjin = tuplenoflist(numnodes)
        nodeadjout = tuplenoflist(numnodes)
        for arcid, startnode, endnode in itertools.izip(itertools.count(), arcstartnodes, arcendnodes):
            nodeadjin[endnode].append((startnode, arcid))
            nodeadjout[startnode].append((endnode, arcid))
        self.nodeadjin = tuple(tuple(sorted(adjacency)) for adjacency in nodeadjin)
        self.nodeadjout = tuple(tuple(sorted(adjacency)) for adjacency in nodeadjout)

        # the nodeoutarcs member is useful for now, but it would be
        # nice to use only the adjacency lists
        self.nodeoutarcs = tuple(tuple(sorted(arcid for nodeid, arcid in adjout)) for adjout in self.nodeadjout)

    def _verify_adjacency(self):
        # verify adjacency invariants
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs
        nodeadjout, nodeadjin = self.nodeadjout, self.nodeadjin
        assert len(nodelabels) == len(nodeadjout) == len(nodeadjin)
        for nodeid, adjout, adjin in itertools.izip(itertools.count(), nodeadjout, nodeadjin):
            assert adjout == tuple(sorted(adjout))
            assert adjin == tuple(sorted(adjin))
            for (outnode, outarc), (innode, inarc) in itertools.izip(adjout, adjin):
                assert arcstartnodes[outarc] == nodeid
                assert arcendnodes[outarc] == outnode
                assert arcendnodes[inarc] == nodeid
                assert arcstartnodes[inarc] == innode

    @staticmethod
    def _filter_adj(filter_node_index, adj):
        """
        Generator filter for an adjacency list that skips the given node index.
        Typical usage is to ignore self loops in the adjacency list.
        """
        for info in adj:
            node_index, arc_index = info
            if node_index != filter_node_index:
                yield info

    # terminal nodes support
    def _make_terminal_lists(self):
        # ignoring self loops entirely: start nodes are those for
        # which there are no in arcs, end nodes are those for which
        # there are no out arcs; non-empty startnodes and endnodes are
        # necessary for DAG, but not sufficient
        filter_adj = self._filter_adj
        self._startnodes = tuple(index for index, adj in enumerate(self.nodeadjin) if not tuple(filter_adj(index, adj)))
        self._endnodes = tuple(index for index, adj in enumerate(self.nodeadjout) if not tuple(filter_adj(index, adj)))

    def _verify_terminals(self):
        assert self.startnodes == tuple(sorted(self.startnodes))
        assert self.endnodes == tuple(sorted(self.endnodes))
        filter_adj = self._filter_adj
        startnodes = set(self.startnodes)
        endnodes = set(self.endnodes)
        for nodeid, adjout, adjin in itertools.izip(itertools.count(), self.nodeadjout, self.nodeadjin):
            assert (nodeid in startnodes) == (not tuple(filter_adj(nodeid, adjin)))
            assert (nodeid in endnodes) == (not tuple(filter_adj(nodeid, adjout)))


    def _check_depth_first(self):
        toponodes = list()
        topoarcs = list()
        flags = builtin.attrdict()
        flags.has_cycle = flags.has_self_loop = False        

        def pre_callback(obj, node):
            pass
        def post_callback(obj, node):

            # extend with all incident arc ids
            topoarcs.extend(reversed(sorted(arcid for nodeid, arcid in obj.nodeadjin[node])))
        def cycle_callback(obj, parent, child):
            if parent == child:
                flags.has_self_loop = True
            else:
                flags.has_cycle = True
        self.depth_first_callback(pre_callback=pre_callback, post_callback=post_callback, cycle_callback=cycle_callback)

        toponodes.reverse()
        assert tuple(toponodes) == self.toponodes
        topoarcs.reverse()
        assert tuple(topoarcs) == self.topoarcs

        assert flags.has_cycle == self._has_cycle
        assert flags.has_self_loop == self._has_self_loop

    def depth_first_callback(self, pre_callback=None, post_callback=None, join_callback=None, cycle_callback=None):
        """
        Performs a depth-first traversal of the graph, making callbacks for all
        structural events involving the nodes in the graph.  The sequence of
        callbacks represent a structure-revealing serialization of events from
        which the graph could be constructed.

        The optional *pre_callback* and *post_callback* arguments are functions
        of two arguments, (*graph*, *node_id*), where *graph* is the FrozenGraph
        instance making the callback and *node_id* is the identifier of the node
        in question.  These two callbacks are each called exactly once with the
        node identifier of each node in the graph.  The interleaving of these
        calls reveals the topological, tree, and spanning-tree structure in the
        graph.

        The *pre_callback* function is called each time a new *node_id* is first
        encountered in the depth traversal.  It is guaranteed to be called with
        a given value for *node_id* prior to being called for any node that is
        reachable from the given value of *node_id*.

        The *post_callback* function is called after all nodes reachable from
        *node_id* have been encountered and processed by *pre_callback* and
        *post_callback*.  For DAGs, the reverse of the sequence of *node_id* in
        the calls to *post_callback* is a topological ordering of the nodes.
        For example, :attr:`toponodes` is constructed this way.  Also, for any
        subtree in the graph, the *node_id* values appearing in the
        *pre_callback* and *post_callback* calls are nested.
          
        The *cycle_callback* and *join_callback* arguments are functions of
        three arguments, (*graph*, *start_node_id*, *end_node_id*), where
        *graph* is the FrozenGraph instance making the callback and
        *start_node_id* and *end_node_id* are two nodes that are connected by an
        arc.  These two callbacks reveal the joins and cycles in the graph.

        The *join_callback* is called whenever a join is encountered that does
        not complete a cycle.  That is, *end_node_id* will already have been
        used in calls to both *pre_callback* and *post_callback*.  In each call
        to *join_callback*, it is guaranteed that *start_node_id* and
        *end_node_id* are distinct.

        The *cycle_callback* is called whenever a join is encountered that does
        complete a cycle.  In this case, *end_node_id* will already have been
        used in a *pre_callback*, but it will not have been used in a
        *post_callback*.  If *start_node_id* and *end_node_id* are the same, the
        cycle is a self loop.  Otherwise, *start_node_id* and *end_node_id* are
        distinct and the cycle includes multiple nodes.  The set of arcs
        appearing in the calls to *cycle_callback* is a set of arcs which, if
        removed or reversed, would convert the graph a DAG.  Put another way, if
        *cycle_callback* is not ever called, the graph is a DAG.
        """
        def make_ungen(func):
            def ungen(*args):
                func(*args)
                return ()
            return ungen
        if pre_callback is not None: pre_callback = make_ungen(pre_callback)
        if post_callback is not None: post_callback = make_ungen(post_callback)
        if join_callback is not None: join_callback = make_ungen(join_callback)
        if cycle_callback is not None: cycle_callback = make_ungen(cycle_callback)
        for _ in self.depth_first_iter(pre_callback=pre_callback, post_callback=post_callback, join_callback=join_callback, cycle_callback=cycle_callback):
            pass

    def depth_first_iter(self, pre_callback=None, post_callback=None, join_callback=None, cycle_callback=None):
        """
        Performs a depth-first traversal of the graph, making callbacks to
        generator functions for all structural events involving the nodes in the
        graph.  The sequence of callbacks represent a structure-revealing
        serialization of events from which the graph could be constructed.  The
        iterator yields the items from each of the iterables/generators that
        each generator callback returns.

        The optional *pre_callback* and *post_callback* arguments are generator
        functions of two arguments, (*graph*, *node_id*), where *graph* is the
        FrozenGraph instance making the callback and *node_id* is the identifier
        of the node in question.  These two callbacks are each called exactly
        once with the node identifier of each node in the graph.  The
        interleaving of these calls reveals the topological, tree, and
        spanning-tree structure in the graph.

        The *pre_callback* generator function is called each time a new
        *node_id* is first encountered in the depth traversal.  It is guaranteed
        to be called with a given value for *node_id* prior to being called for
        any node that is reachable from the given value of *node_id*.

        The *post_callback* generator function is called after all nodes
        reachable from *node_id* have been encountered and processed by
        *pre_callback* and *post_callback*.  For DAGs, the reverse of the
        sequence of *node_id* in the calls to *post_callback* is a topological
        ordering of the nodes.  For example, :attr:`toponodes` is constructed
        this way.  Also, for any subtree in the graph, the *node_id* values
        appearing in the *pre_callback* and *post_callback* calls are nested.
          
        The *cycle_callback* and *join_callback* arguments are generator
        functions of three arguments, (*graph*, *start_node_id*, *end_node_id*),
        where *graph* is the FrozenGraph instance making the callback and
        *start_node_id* and *end_node_id* are two nodes that are connected by an
        arc.  These two callbacks reveal the joins and cycles in the graph.

        The *join_callback* is called whenever a join is encountered that does
        not complete a cycle.  That is, *end_node_id* will already have been
        used in calls to both *pre_callback* and *post_callback*.  In each call
        to *join_callback*, it is guaranteed that *start_node_id* and
        *end_node_id* are distinct.

        The *cycle_callback* is called whenever a join is encountered that does
        complete a cycle.  In this case, *end_node_id* will already have been
        used in a *pre_callback*, but it will not have been used in a
        *post_callback*.  If *start_node_id* and *end_node_id* are the same, the
        cycle is a self loop.  Otherwise, *start_node_id* and *end_node_id* are
        distinct and the cycle includes multiple nodes.  The set of arcs
        appearing in the calls to *cycle_callback* is a set of arcs which, if
        removed or reversed, would convert the graph a DAG.  Put another way, if
        *cycle_callback* is not ever called, the graph is a DAG.
        """
        nodeadjout = self.nodeadjout
        metaroot = numnodes = len(nodeadjout)

        # not visited, active, done
        white, grey, black = None, True, False
        color = listn(numnodes + 1, white)
        visitstack = list()
        balanced_stack = __debug__ and list()

        # standard, depth-first coloring of nodes; each item on visitstack is a
        # pair, (node id, next() method of out arc iterator for the node);
        # frontloading with the startnodes, if any, gives more canonical results
        # for DAGs; but, the result of this algorithm is not canonical
        #
        # note: the reversals done on the iterators make the topological
        # orderings 'stable' in the sense of a stable sort -- ordered ids that
        # are topologically correct will not be reordered;
        #
        # note: have to include all nodes in metaroot's iterator due to
        # possibility of cycles that aren't weakly connected to any start node
        color[metaroot] = grey
        visitstack.append((metaroot, _chain(reversed(self.startnodes), xrange(numnodes - 1, -1, -1)).next))
        __debug__ and balanced_stack.append(metaroot)
        while visitstack:
            parent, next = visitstack[-1]
            assert parent == balanced_stack[-1] and color[parent] is grey
            try:
                child = next()
            except StopIteration:
                # we're done with the child(ren), if any, of parent
                color[parent] = black
                visitstack.pop()
                assert balanced_stack.pop() == parent
                if parent is not metaroot and post_callback is not None:
                    for y in post_callback(self, parent):
                        yield y
            else:
                if parent is metaroot and color[child] is black:
                    # nothing to do for finished nodes in cycles that aren't reachable from a startnode
                    continue
                if color[child] is white:
                    # first time visiting the child node: grey it, and put it
                    # and its children iterator on the stack
                    color[child] = grey
                    visitstack.append((child, reversed(sorted(nodeid for nodeid, arcid in nodeadjout[child])).next))
                    __debug__ and balanced_stack.append(child)
                    if pre_callback is not None:
                        for y in pre_callback(self, child):
                            yield y
                elif color[child] is grey:
                    assert parent is not metaroot
                    # we're returning to an active node, a cycle
                    if cycle_callback is not None:
                        for y in cycle_callback(self, parent, child):
                            yield y
                else:
                    # already did the child node, a join
                    assert color[child] is black
                    assert parent is not metaroot
                    assert parent != child
                    if join_callback is not None:
                        for y in join_callback(self, parent, child):
                            yield y

        assert not balanced_stack
        assert tuple(set(color)) == (black,)

    def _make_topological_lists(self):
        toponodes = list()
        topoarcs = list()
        cycle_breakers = set()
        flags = builtin.attrdict()
        flags.has_join = flags.has_cycle = flags.has_self_loop = False

        pre_callback = None
        def post_callback(obj, node):
            toponodes.append(node)
            # extend with all incident arc ids
            topoarcs.extend(reversed(sorted(arcid for nodeid, arcid in obj.nodeadjin[node])))
        def cycle_callback(obj, parent, child):
            cycle_breakers.add((parent, child))
            if parent == child:
                flags.has_self_loop = True
            else:
                flags.has_cycle = True
                flags.has_join = True
        def join_callback(obj, parent, child):
            flags.has_join = True
        
        self.depth_first_callback(pre_callback=pre_callback, post_callback=post_callback, cycle_callback=cycle_callback, join_callback=join_callback)

        self.toponodes = tuple(reversed(toponodes))
        self.topoarcs = tuple(reversed(topoarcs))
        self._cycle_breakers = frozenset(cycle_breakers)
        assert self._has_self_loop == flags.has_self_loop
        self._has_cycle = flags.has_cycle
        self._has_join = flags.has_join

    def _verify_topologicals(self):
        nodelabels, arcstartnodes, arcendnodes, arclabels = self.graphseqs

        # topological nodes
        numnodes = len(nodelabels)
        toponodes = self.toponodes
        nodeadjin = self.nodeadjin
        nodeadjout = self.nodeadjout
        # verify lengths
        assert len(nodeadjin) == len(nodeadjout) == len(toponodes) == numnodes
        # verify permutation of nodeids
        assert set(toponodes) == set(xrange(numnodes))
        seennodes = set()
        for node in toponodes:
            assert node not in seennodes
            for prednode, predarc in nodeadjin[node]:
                assert prednode in seennodes or self.strict_is_cyclic()
            for succnode, succarc in nodeadjout[node]:
                assert succnode not in seennodes or self.strict_is_cyclic()
            seennodes.add(node)

        # topological arcs
        numarcs = len(arclabels)
        topoarcs = self.topoarcs
        assert set(topoarcs) == set(xrange(numarcs))
        # verify permutation of arcids
        seenstarts = set(self.startnodes)
        for startnode, endnode in ((arcstartnodes[arc], arcendnodes[arc]) for arc in topoarcs):
            assert endnode not in seenstarts or self.strict_is_cyclic()
            seenstarts.add(startnode)

    def _make_connected_sets(self):
        # create a sequence of the sets of weakly connected nodes
        nodeadjout = self.nodeadjout
        nodeadjin = self.nodeadjin
        numnodes = self.num_nodes

        # We do this by having a set of nodes, toponodes, each of which we
        # haven't put into any weakly connected subgraph.  We take one of the
        # nodes in toponodes and we find every node we can reach from this node
        # (both in and out, thus the "weakly").  To create the next connected
        # set, we remove all of these nodes from topnodes and put them in the
        # next connected set.  Loop until topnodes is empty.
        connected_sets = list()
        topnodes = set(xrange(numnodes))
        while topnodes:
            connected = set()
            nodeset = set()

            nodeid1 = topnodes.pop()
            nodeset.add(nodeid1)
            while nodeset:
                nodeid = nodeset.pop()
                assert nodeid not in connected
                assert nodeid not in topnodes
                connected.add(nodeid)
                next = set(node for node, arc in _chain(nodeadjin[nodeid], nodeadjout[nodeid])) - connected
                nodeset.update(next)
                topnodes -= next
            connected_sets.append(frozenset(connected))

        assert set(node for nodeset in connected_sets for node in nodeset) == set(xrange(numnodes))
        self.connected_sets = tuple(sorted(connected_sets))

    def _verify_connecteds(self):
        numnodes = len(self.nodelabels)
        nodeadjout = self.nodeadjout
        nodeadjin = self.nodeadjin
        connected_sets = self.connected_sets
        assert tuple(sorted(connected_sets)) == connected_sets
        assert set(node for nodeset in connected_sets for node in nodeset) == set(xrange(numnodes))
        for connected_set in connected_sets:
            assert connected_set
            connected_set2 = set()
            for nodeid in connected_set:
                connected_set2.add(nodeid)
                connects = set(node for node, arc in _chain(nodeadjin[nodeid], nodeadjout[nodeid]))
                assert connects <= connected_set
                connected_set2.update(connects)
            assert connected_set2 == connected_set

    def _strict_check_acyclic(self):
        if self.strict_is_cyclic():
            raise ValueError("expected graph to be acyclic")

    def __add__(self, other):
        """
        >>> a = FrozenGraph(GraphTables(((1, 2, 'a', 'b'), (0, 1, 2), (1, 0, 2), (3, None, 5))))
        >>> b = FrozenGraph(GraphTables(((1, 2, 'A', 'B', 'C'), (0, 2, 1, 3), (1, 0, 2, 3), ('X', 'Y', 'Z', 'ZZ'))))
        >>> a + b
        FrozenGraph(GraphTables(((1, 2, 'A', 'B', 'C', 1, 2, 'a', 'b'), (0, 2, 1, 3, 5, 6, 7), (1, 0, 2, 3, 6, 5, 7), ('X', 'Y', 'Z', 'ZZ', 3, None, 5))))
        """
        check_instance(GraphBase, other)
        return _graph_disjoint_union(self, other)


def _graph_disjoint_union(g1, g2):
    # returns the FrozenGraph of the disjoint union of the two
    # arguments which must be instances of GraphBase
    assert isinstance(g1, GraphBase), type(g1).__name__
    assert isinstance(g2, GraphBase), type(g2).__name__
    if len(g1.arcstartnodes) < len(g2.arcstartnodes):
        # a minor optimization so we iterate over the smaller sized graph
        g1, g2 = g2, g1
    nodelabels1, arcstartnodes1, arcendnodes1, arclabels1 = g1.graphseqs
    nodelabels2, arcstartnodes2, arcendnodes2, arclabels2 = g2.graphseqs

    # new labels
    nodelabels = nodelabels1 + nodelabels2
    arclabels = arclabels1 + arclabels2

    # create the new arc indicies
    offset = len(nodelabels1)
    arcstartnodes = arcstartnodes1 + tuple(id + offset for id in arcstartnodes2)
    arcendnodes = arcendnodes1 + tuple(id + offset for id in arcendnodes2)

    return FrozenGraph(GraphTables((nodelabels, arcstartnodes, arcendnodes, arclabels)))

def _graph_products(g1, g2, node_label_callback=None, arc_label_callback=None, commutative=False):
    """
    >>> a = GraphBase(GraphTables((('a', 'b'), (0, 1, 1), (1, 0, 1), ('A', 'B', 'C'))))
    >>> b = GraphBase(GraphTables(((1, 2, 3), (0, 1, 2, 1), (1, 2, 0, 0), (1, 2, 3, 4))))
    >>> _graph_products(a, b, commutative=True)
    order: 6
    conormal: 31
    lexicographic: 25
    normal: 25
    cartesian: 17
    tensor: 12

    >>> a = make_random_DAG(1, 20, 2, 3, 2, 700, seed=0)
    >>> b = make_random_DAG(1, 10, 3, 4, 3, 700, seed=1)
    >>> _graph_products(a, b)
    order: 510
    conormal: 26724
    lexicographic: 12648
    normal: 1938
    cartesian: 1224
    tensor: 714
    """

    assert isinstance(g1, GraphBase), type(g1).__name__
    assert isinstance(g2, GraphBase), type(g2).__name__
    if commutative and len(g1.arcstartnodes) < len(g2.arcstartnodes):
        # a minor optimization so g2 is the smaller sized graph
        g1, g2 = g2, g1
    nodelabels1, arcstartnodes1, arcendnodes1, arclabels1 = g1.graphseqs
    nodelabels2, arcstartnodes2, arcendnodes2, arclabels2 = g2.graphseqs

    # arc dictionaries
    # XXX this loses multiarcs!
    adjacency1 = dict(((v1, v2), label) for v1, v2, label in itertools.izip(arcstartnodes1, arcendnodes1, arclabels1))
    adjacency2 = dict(((v1, v2), label) for v1, v2, label in itertools.izip(arcstartnodes2, arcendnodes2, arclabels2))

    # Cartesian set of nodes
    nodes = dict(((id1, id2), (label1, label2)) for id1, label1 in enumerate(nodelabels1) for id2, label2 in enumerate(nodelabels2))
    #print 'nodes:', nodes
    print 'order:', len(nodes)

    # nothing efficient here, we do the n^2 work on the resulting set of nodes

    # product types
    tensor = dict()
    cartesian = dict()
    lexicographic = dict()
    normal = dict()
    conormal = dict()

    sentinel = object()
    for u in nodes.iterkeys():
        for v in nodes.iterkeys():
            # u, v are nodes in the result graph
            # u1, u2 are factors of u from g1 and g2 respectively
            u1, u2 = u
            # v1, v2 are factors of v from g1 and g2 respectively
            v1, v2 = v
            # a1, a2 are possibly-adjacent nodes (arc) in g1 and g2 respectively
            a1 = u1, v1
            a2 = u2, v2
            # arc labels in g1 and g2, or sentinel if no such arc
            t1 = adjacency1.get(a1, sentinel)
            t2 = adjacency2.get(a2, sentinel)

            #print u, '->', v, ':', (t1, t2)

            # conormal
            if (t1 is not sentinel) or (t2 is not sentinel):
                #print ' ', 'conormal'
                conormal[u, v] = t1, t2

            # lexicographic: not commutative
            if (t1 is not sentinel) or (u1 == v1 and (t2 is not sentinel)):
                #print ' ', 'lexicographic'
                lexicographic[u, v] = t1, t2

            # normal
            if (u1 == v1 and (t2 is not sentinel)) or (u2 == v2 and (t1 is not sentinel)) or ((t1 is not sentinel) and (t2 is not sentinel)):
                #print ' ', 'normal'
                normal[u, v] = t1, t2

            # cartesian
            if (u1 == v1 and (t2 is not sentinel)) or (u2 == v2 and (t1 is not sentinel)):
                #print ' ', 'cartesian'
                cartesian[u, v] = t1, t2

            # tensor
            if (t1 is not sentinel) and (t2 is not sentinel):
                #print ' ', 'tensor'
                tensor[u, v] = t1, t2

    print 'conormal:', len(conormal)
    print 'lexicographic:', len(lexicographic)
    print 'normal:', len(normal)
    print 'cartesian:', len(cartesian)
    print 'tensor:', len(tensor)

    return
    nodelabels = list()
    for label1 in nodelabels1:
        for label2 in nodelabels2:
            label = label1, label2
            if node_label_callback is not None:
                label = node_label_callback(label)
            nodelabels.append(label)
    len_nodelabels = len(nodelabels)
    assert len_nodelabels == len(nodelabels1) * len(nodelabels2)

    #print 'adjacency1:', adjacency1
    #print 'adjacency2:', adjacency2

    xrange1 = xrange(len(nodelabels1))
    xrange2 = xrange(len(nodelabels2))
    for id1 in xrange1:
        for id2 in xrange2:
            pass

    return

    gnodelabels = nodelabels1 + nodelabels2
    garclabels = arclabels1 + arclabels2

    offset = len(arcstartnodes1)
    assert offset >= len(arcstartnodes2)
    garcstartnodes = arcstartnodes1 + tuple(id + offset for id in arcstartnodes2)
    garcendnodes = arcendnodes1 + tuple(id + offset for id in arcendnodes2)

    return FrozenGraph(GraphTables((gnodelabels, garcstartnodes, garcendnodes, garclabels)))


def _graph_cartesian_product(g1, g2, node_label_callback=None, arc_label_callback=None):
    """
    Returns the FrozenGraph of the Cartesian product of the two
    arguments, each of which must be instances of GraphBase.

    >>> a = FrozenGraph(GraphTables((('a', 'b'), (0, 1, 1), (1, 0, 1), ('A', 'B', 'C'))))
    >>> b = FrozenGraph(GraphTables(((1, 2, 3), (0, 1, 2, 1), (1, 2, 0, 0), (1, 2, 3, 4))))
    >>> _graph_cartesian_product(a, b)
    """

    assert isinstance(g1, GraphBase), type(g1).__name__
    assert isinstance(g2, GraphBase), type(g2).__name__
    if len(g1.arcstartnodes) < len(g2.arcstartnodes):
        # a minor optimization so we iterate over the smaller sized graph
        g1, g2 = g2, g1
    nodelabels1, arcstartnodes1, arcendnodes1, arclabels1 = g1.graphseqs
    nodelabels2, arcstartnodes2, arcendnodes2, arclabels2 = g2.graphseqs

    nodelabels = list()
    for label1 in nodelabels1:
        for label2 in nodelabels2:
            label = label1, label2
            if node_label_callback is not None:
                label = node_label_callback(label)
            nodelabels.append(label)
    len_nodelabels = len(nodelabels)
    assert len_nodelabels == len(nodelabels1) * len(nodelabels2)

    adjacency1 = frozenset(itertools.izip(arcstartnodes1, arcendnodes1))
    adjacency2 = frozenset(itertools.izip(arcstartnodes2, arcendnodes2))
    #print 'adjacency1:', adjacency1
    #print 'adjacency2:', adjacency2

    xrange1 = xrange(len(nodelabels1))
    xrange2 = xrange(len(nodelabels2))
    for id1 in xrange1:
        for id2 in xrange2:
            pass

    return

    gnodelabels = nodelabels1 + nodelabels2
    garclabels = arclabels1 + arclabels2

    offset = len(arcstartnodes1)
    assert offset >= len(arcstartnodes2)
    garcstartnodes = arcstartnodes1 + tuple(id + offset for id in arcstartnodes2)
    garcendnodes = arcendnodes1 + tuple(id + offset for id in arcendnodes2)

    return FrozenGraph(GraphTables((gnodelabels, garcstartnodes, garcendnodes, garclabels)))

def compose_DAGs(dag1, dag2):
    """
    Composes two DAGs into a new graph.  Returns a FrozenGraph of the
    composition.  Each node label and each arc label in the new graph is a tuple
    of (label1, label2) from the corresponding nodes or arcs being composed.

    >>> dag1 = FrozenGraph(GraphTables(((-11, -12, -10), (0, 1, 0), (1, 2, 2), (-21, -20, -22))))
    >>> dag2 = FrozenGraph(GraphTables(((-111, -112, -110), (0, 1, 0), (1, 2, 2), (-121, -120, -122))))
    >>> compose_DAGs(dag1, dag2)
    Traceback (most recent call last):
      ...
    NotImplementedError: early stages of work in progress

    FrozenGraph(GraphTables(((), (), (), ())))

    Must use cycle-free graphs

    >>> compose_DAGs(dag1, FrozenGraph(GraphTables(((-1,), (0,), (0,), (-2,)))))
    Traceback (most recent call last):
      ...
    ValueError: expected strict cycle checks to be (False, False), got (False, True)
    """
    if not isinstance(dag1, FrozenGraph) or not isinstance(dag2, FrozenGraph):
        raise ValueError("expected two FrozenGraph, got %s and %s" % (type(dag1).__name__, type(dag2).__name__))
    ref = False, False
    cyclicp = tuple(dag.has_cycle() or dag.has_self_loop() for dag in (dag1, dag2))
    if cyclicp != ref:
        raise ValueError("expected strict cycle checks to be %r, got %r" % (ref, cyclicp,))

    builder = TopologicalGraphBuilder()

    raise NotImplementedError("early stages of work in progress")
    return FrozenGraph(builder)


def _test():

    counts = range(10)
    numcounts = len(counts)

    # GraphBuilder
    g = GraphBuilder()
    assert g.num_nodes == g.num_arcs == 0

    # make some nodes
    assert list(g.new_node(~x) for x in counts) == counts
    # add one for our arc building
    assert g.num_nodes == numcounts
    assert g.num_arcs == 0
    for node_id in counts:
        assert g.get_node_label(node_id) == ~node_id

    # make some arcs: linear, acyclic
    g.new_node()
    assert list(g.new_arc(x, x+1, ~x) for x in counts) == counts
    assert g.num_nodes == numcounts + 1
    assert g.num_arcs == numcounts
    for arc_id in counts:
        assert g.get_arc(arc_id) == (arc_id, arc_id+1, ~arc_id,)

    # make some more arcs: acyclic
    tuple(g.new_arc(0, x+1, ~x) for x in counts)
    assert g.num_nodes == numcounts + 1
    assert g.num_arcs == 2 * numcounts

    g.verify()


    # FrozenGraph
    fg = FrozenGraph(g)
    assert not fg.has_cycle()

    # verification of singleton terminals
    (startnode,), (endnode,) = fg.get_terminals()
    print "score ((arc_id (arc_start_node, arc_end_node, arc_label)), ...)"
    for path_score, path, _graph in fg.iterpaths(lambda x: 1, startnode, endnode):
        # print everything to show that each path is unique
        print path_score, tuple((arcid, fg.get_arc(arcid),) for arcid in path)

    node_map, arc_map = fg.get_topological_reorderings()
    print "node_map:", node_map
    print "arc_map:", arc_map
    # evidence that get_topological_reorderings isn't necessary
    assert node_map == fg.toponodes

    # make some more arcs, still acyclic
    tuple(g.new_arc(x, numcounts, ~x) for x in counts)
    fg2 = FrozenGraph(g)
    assert not fg2.has_cycle()

    # verification of singleton terminals
    (startnode,), (endnode,) = fg2.get_terminals()
    print
    print "score ((arc_id (arc_start_node, arc_end_node, arc_label)), ...)"
    for path_score, path, _graph in fg2.iterpaths(lambda x: 1, startnode, endnode):
        # print everything to show that each path is unique
        print path_score, tuple((arcid, fg2.get_arc(arcid),) for arcid in path)

    # add a second end node and more skipping arcs
    g.new_node()
    tuple(g.new_arc(x, x+2, ~x) for x in counts)
    assert g.num_nodes == numcounts + 2
    assert g.num_arcs == 4 * numcounts
    fg3 = FrozenGraph(g)
    # verification of terminal sets
    (startnode,), (endnode1, endnode2,) = fg3.get_terminals()
    print
    print "score ((arc_id (arc_start_node, arc_end_node, arc_label)), ...)"
    # verify path finding between non-terminal nodes
    for path_score, path, _graph in fg3.iterpaths(lambda x: 1, 3, 7):
        # print everything to show that each path is unique
        print path_score, tuple((arcid, fg3.get_arc(arcid),) for arcid in path)


    # make a topologically sorted version: note that the labels on the
    # paths are the same as for fg3, but the node and arc ids differ
    maps = fg3.get_topological_reorderings()
    fg4 = FrozenGraph(fg3.get_mapped_tables(maps))
    # verification of terminal sets
    (startnode,), (endnode1, endnode2,) = fg4.get_terminals()
    print
    print "score ((arc_id (arc_start_node, arc_end_node, arc_label)), ...)"
    # verify path finding between non-terminal nodes
    for path_score, path, _graph in fg4.iterpaths(lambda x: 1, 3, 7):
        # print everything to show that each path is unique
        print path_score, tuple((arcid, fg4.get_arc(arcid),) for arcid in path)


    # now make it cyclic; actually, going further, make it (strongly)
    # connected, so no terminals
    tuple(g.new_arc(x+1, 0, ~x) for x in counts)
    tuple(g.new_arc(x+2, 0, ~x) for x in counts)
    assert g.num_nodes == numcounts + 2
    assert g.num_arcs == 6 * numcounts
    assert FrozenGraph(g).has_cycle()
    assert FrozenGraph(g).get_terminals() == ((), (),)


def A_tutorial():
    """
This is a tutorial on using graphtools.  It's also a doctest-based
test of the graphtools module.

This tutorial demonstrates use of the graphtools module.  It shows how to build
and access a graph using GraphBuilder.  It shows how to create a FrozenGraph and
the operations and invariants associatated with a FrozenGraph.

Play around with points on the compass and connections between them.

Let's build this graph::

        --------------------\
      /                      V
 West -------> North ------> South --------> East
   \              \                          ^ ^
    \              \------------------------/  /
     \                                        /
      \--------------------------------------/


First some points:

>>> builder = GraphBuilder()
>>> north = builder.new_node("North")
>>> east =  builder.new_node("East")
>>> south =  builder.new_node("South")
>>> west =  builder.new_node("West")
>>> builder.num_nodes
4

Now make some directed connections:

>>> for start_node, end_node in (
...     (west, east),
...     (north, south),
...     (west, south),
...     (south, east),
...     (north, east),
...     (west, north)):
...   _ = builder.new_arc(start_node, end_node, builder.get_node_label(start_node) + 'To' + builder.get_node_label(end_node))
>>> builder.num_arcs
6

Get an immutable version of the graph and print it out:

>>> frozen = FrozenGraph(builder)
>>> for text in frozen.text_iter():
...   print text,
num_nodes 4
0   North
1   East
2   South
3   West
num_arcs 6
0   3  1   WestToEast
1   0  2   NorthToSouth
2   3  2   WestToSouth
3   2  1   SouthToEast
4   0  1   NorthToEast
5   3  0   WestToNorth

It's a directed acyclic graph (DAG).

>>> frozen.has_cycle()
False

See which nodes are global start nodes (have no arcs that end on them)
and global end nodes (have no arcs that start on them):

>>> frozen.get_terminals()
((3,), (1,))

Note:

>>> frozen.get_terminals() == ((west,), (east,))
True

Iteration by node label with secondary iteration of in- and out-arcs, done in topological order
since that's well defined for this graph:

>>> for (node_label, in_arc_iter, out_arc_iter) in frozen.iter_nodes():
...    print('For node with label %s:' % (node_label,))
...    print('In-arcs are:')
...    for arc_label, other_node_label in in_arc_iter:
...        print('%s in from node %s' % (arc_label, other_node_label))
...    print('Out-arcs are:')
...    for arc_label, other_node_label in out_arc_iter:
...        print('%s out to node %s' % (arc_label, other_node_label))
For node with label West:
In-arcs are:
Out-arcs are:
WestToNorth out to node North
WestToEast out to node East
WestToSouth out to node South
For node with label North:
In-arcs are:
WestToNorth in from node West
Out-arcs are:
NorthToEast out to node East
NorthToSouth out to node South
For node with label South:
In-arcs are:
NorthToSouth in from node North
WestToSouth in from node West
Out-arcs are:
SouthToEast out to node East
For node with label East:
In-arcs are:
NorthToEast in from node North
SouthToEast in from node South
WestToEast in from node West
Out-arcs are:

And the same thing, but in reverse topological order

>>> for (node_label, in_arc_iter, out_arc_iter) in frozen.iter_nodes(reverse_order=True):
...    print('For node with label %s:' % (node_label,))
...    print('In-arcs are:')
...    for arc_label, other_node_label in in_arc_iter:
...        print('%s in from node %s' % (arc_label, other_node_label))
...    print('Out-arcs are:')
...    for arc_label, other_node_label in out_arc_iter:
...        print('%s out to node %s' % (arc_label, other_node_label))
For node with label East:
In-arcs are:
NorthToEast in from node North
SouthToEast in from node South
WestToEast in from node West
Out-arcs are:
For node with label South:
In-arcs are:
NorthToSouth in from node North
WestToSouth in from node West
Out-arcs are:
SouthToEast out to node East
For node with label North:
In-arcs are:
WestToNorth in from node West
Out-arcs are:
NorthToEast out to node East
NorthToSouth out to node South
For node with label West:
In-arcs are:
Out-arcs are:
WestToNorth out to node North
WestToEast out to node East
WestToSouth out to node South


A helper for looking at paths between nodes, ranked by cost.

>>> def printpaths(pathiter, verbose=False):
...   print "score",  "((arc_id, (arc_start_node, arc_end_node, arc_label)), ...)" if verbose else "(arc_label, ...)"
...   for path_score, path, graph in pathiter:
...     print path_score, tuple( ((arcid, graph.get_arc(arcid),) if verbose else graph.get_arc_label(arcid)) for arcid in path)

There are two paths from north to east.  Note that by using len as the arc label
costing function, the score is just the sum of the lengths of the arcs' label
strings.

>>> printpaths(frozen.iterpaths(len, north, east))
score (arc_label, ...)
11 ('NorthToEast',)
23 ('NorthToSouth', 'SouthToEast')

Given that it's a DAG and there are paths from north to east, we should expect
no path from east to north.

>>> printpaths(frozen.iterpaths(len, east, north))
score (arc_label, ...)

Use the two terminals to get all paths through the graph.  Verbose setting shows
details of arc and node ids on each path.

>>> printpaths(frozen.iterpaths(len, west, east), True)
score ((arc_id, (arc_start_node, arc_end_node, arc_label)), ...)
10 ((0, (3, 1, 'WestToEast')),)
22 ((2, (3, 2, 'WestToSouth')), (3, (2, 1, 'SouthToEast')))
22 ((5, (3, 0, 'WestToNorth')), (4, (0, 1, 'NorthToEast')))
34 ((5, (3, 0, 'WestToNorth')), (1, (0, 2, 'NorthToSouth')), (3, (2, 1, 'SouthToEast')))

Get a permutation of node ids and arc ids maps node and arc ids into a
topological order.

>>> maps = frozen.get_topological_reorderings()
>>> maps
((3, 0, 2, 1), (0, 2, 5, 1, 4, 3))
>>> (frozen.toponodes, frozen.topoarcs)
((3, 0, 2, 1), (5, 1, 2, 0, 3, 4))


E.g. node 3 (West) would be the first node in this topological ordering and node
1 (East) would be the last node in this topological ordering.  Similarly, arc 0
(WestToEast) would remain the first arc in this topological ordering and arc 3
(SouthToEast) would be the last arc in this topological ordering.

Let's make such a graph.

>>> topo = FrozenGraph(frozen.get_mapped_tables(maps))
>>> for text in topo.text_iter():
...   print text,
num_nodes 4
0   West
1   North
2   South
3   East
num_arcs 6
0   0  3   WestToEast
1   0  2   WestToSouth
2   0  1   WestToNorth
3   1  2   NorthToSouth
4   1  3   NorthToEast
5   2  3   SouthToEast

In this graph the topological permutations are identities.

>>> topo.get_topological_reorderings()
((0, 1, 2, 3), (0, 1, 2, 3, 4, 5))
>>> topo.toponodes, topo.topoarcs
((0, 1, 2, 3), (2, 1, 3, 0, 4, 5))

So the two terminal nodes have smallest and largest node ids.

>>> (start_node,), (end_node,) = terminals = topo.get_terminals()
>>> terminals
((0,), (3,))

Despite the reordering of ids, the scores of paths and the sequence of arc
labels are the same as for the non-topologically sorted graph.

>>> printpaths(topo.iterpaths(len, start_node, end_node))
score (arc_label, ...)
10 ('WestToEast',)
22 ('WestToSouth', 'SouthToEast')
22 ('WestToNorth', 'NorthToEast')
34 ('WestToNorth', 'NorthToSouth', 'SouthToEast')

A more lexically-involved cost metric can be used to break length ties:

>>> def lexlen(label):
...   accum = 0
...   for item in label:
...     accum = accum * 0x100 + ord(item)
...   return accum

No ties:

>>> printpaths(topo.iterpaths(lexlen, start_node, end_node))
score (arc_label, ...)
412717324528352340570996 ('WestToEast',)
200478143005669390070048732 ('WestToNorth', 'NorthToEast')
206522827443974778547267548 ('WestToSouth', 'SouthToEast')
24481084856605229727991028804 ('WestToNorth', 'NorthToSouth', 'SouthToEast')

Show that randomly shuffling the arcs and nodes doesn't change the paths' scores
or sequences of arc-labels on each path.

>>> random.seed(123)
>>> maps = tuple(map(list, topo.get_topological_reorderings()))

Fiddle the elements from the original frozen graph:

>>> _ = map(random.shuffle, maps)
>>> maps
([1, 2, 3, 0], [2, 3, 1, 5, 4, 0])
>>> rand1 = FrozenGraph(frozen.get_mapped_tables(maps))
>>> (start_node1,), (end_node1,) = rand1.get_terminals()
>>> printpaths(rand1.iterpaths(lexlen, start_node1, end_node1))
score (arc_label, ...)
412717324528352340570996 ('WestToEast',)
200478143005669390070048732 ('WestToNorth', 'NorthToEast')
206522827443974778547267548 ('WestToSouth', 'SouthToEast')
24481084856605229727991028804 ('WestToNorth', 'NorthToSouth', 'SouthToEast')

Now differently fiddle the elements from the topo graph:

>>> _ = map(random.shuffle, maps)
>>> maps
([2, 3, 1, 0], [0, 5, 4, 2, 3, 1])
>>> rand2 = FrozenGraph(topo.get_mapped_tables(maps))
>>> (start_node2,), (end_node2,) = rand2.get_terminals()
>>> printpaths(rand2.iterpaths(lexlen, start_node2, end_node2))
score (arc_label, ...)
412717324528352340570996 ('WestToEast',)
200478143005669390070048732 ('WestToNorth', 'NorthToEast')
206522827443974778547267548 ('WestToSouth', 'SouthToEast')
24481084856605229727991028804 ('WestToNorth', 'NorthToSouth', 'SouthToEast')

But the underlying node and arc ids are different:
>>> printpaths(rand1.iterpaths(lexlen, start_node1, end_node1), True)
score ((arc_id, (arc_start_node, arc_end_node, arc_label)), ...)
412717324528352340570996 ((5, (2, 0, 'WestToEast')),)
200478143005669390070048732 ((3, (2, 3, 'WestToNorth')), (4, (3, 0, 'NorthToEast')))
206522827443974778547267548 ((0, (2, 1, 'WestToSouth')), (1, (1, 0, 'SouthToEast')))
24481084856605229727991028804 ((3, (2, 3, 'WestToNorth')), (2, (3, 1, 'NorthToSouth')), (1, (1, 0, 'SouthToEast')))

>>> printpaths(rand2.iterpaths(lexlen, start_node2, end_node2), True)
score ((arc_id, (arc_start_node, arc_end_node, arc_label)), ...)
412717324528352340570996 ((0, (3, 1, 'WestToEast')),)
200478143005669390070048732 ((3, (3, 2, 'WestToNorth')), (2, (2, 1, 'NorthToEast')))
206522827443974778547267548 ((5, (3, 0, 'WestToSouth')), (1, (0, 1, 'SouthToEast')))
24481084856605229727991028804 ((3, (3, 2, 'WestToNorth')), (4, (2, 0, 'NorthToSouth')), (1, (0, 1, 'SouthToEast')))

"""


def tutorial():
    foo = """
This is a tutorial on using graphtools.  It's also a doctest-based test of the
graphtools module.

This tutorial demonstrates use of the graphtools module.  It shows how to build
and access a graph using GraphBuilder.  It shows how to create a FrozenGraph and
the operations and invariants associtated with a FrozenGraph.

First, create some utility iterators and generators.

>>> three = 3
>>> seven = 7
>>> def iten(): return xrange(10)

Positive numbers for use as arc labels

>>> arclabels = count(1).next

Negative numbers for use as node labels

>>> negidgen = imap(operator.neg, count(1)).next

Create a GraphBuilder object
>>> builder = GraphBuilder()

It starts out empty.
>>> builder.num_nodes == builder.num_arcs == 0
True

Create some nodes.  Node ids are increasing indices.
>>> tuple(builder.new_node(negidgen()) for x in iten())
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

Take a look at the node labels.
>>> tuple(builder.get_node_label(x) for x in iten())
(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10)

Add another node
>>> builder.new_node(negidgen())
10

Now add arcs to make a linear sequence of "backwards" pointing
arcs.  As with nodes, the arc ids are increasing indices.
>>> tuple(builder.new_arc(x+1, x, arclabels()) for x in iten())
(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

Take a look at the linear "backwards" arcs
>>> tuple(builder.get_arc(x) for x in iten())
((1, 0, 1), (2, 1, 2), (3, 2, 3), (4, 3, 4), (5, 4, 5), (6, 5, 6), (7, 6, 7), (8, 7, 8), (9, 8, 9), (10, 9, 10))

Add another arc
>>> builder.new_arc(seven, three, arclabels())
10

>>> builder.num_nodes == builder.num_arcs == 11
True
>>> builder.verify()

Make a FrozenGraph
>>> frozen = FrozenGraph(builder)
>>> frozen.has_cycle()
False

>>> (startnode,), (endnode,) = terminals = frozen.get_terminals()
>>> terminals
((10,), (0,))

>>> for path_score, path in frozen.path_iter(lambda x: x, startnode, endnode):
...   print path_score, tuple((arcid, frozen.get_arc(arcid),) for arcid in path)
44 ((9, (10, 9, 10)), (8, (9, 8, 9)), (7, (8, 7, 8)), (10, (7, 3, 11)), (2, (3, 2, 3)), (1, (2, 1, 2)), (0, (1, 0, 1)))
55 ((9, (10, 9, 10)), (8, (9, 8, 9)), (7, (8, 7, 8)), (6, (7, 6, 7)), (5, (6, 5, 6)), (4, (5, 4, 5)), (3, (4, 3, 4)), (2, (3, 2, 3)), (1, (2, 1, 2)), (0, (1, 0, 1)))


>>> node_map, arc_map = maps = frozen.get_topological_reorderings()
>>> node_map, arc_map
((10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), (9, 8, 7, 6, 10, 5, 4, 3, 2, 1, 0))

>>> frozen2 = FrozenGraph(frozen.get_mapped_tables(maps))
>>> node_map2, arc_map2 = maps = frozen2.get_topological_reorderings()
>>> node_map2, arc_map2
((10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), (1, 2, 3, 0, 4, 5, 10, 6, 7, 8, 9))

>>> (startnode2,), (endnode2,) = terminals2 = frozen2.get_terminals()
>>> terminals2
((10,), (0,))

>>> for path_score, path in frozen2.path_iter(lambda x: x, startnode2, endnode2):
...   print path_score, tuple((arcid, frozen2.get_arc(arcid),) for arcid in path)
44 ((9, (10, 9, 10)), (8, (9, 8, 9)), (7, (8, 7, 8)), (10, (7, 3, 11)), (2, (3, 2, 3)), (1, (2, 1, 2)), (0, (1, 0, 1)))
55 ((9, (10, 9, 10)), (8, (9, 8, 9)), (7, (8, 7, 8)), (6, (7, 6, 7)), (5, (6, 5, 6)), (4, (5, 4, 5)), (3, (4, 3, 4)), (2, (3, 2, 3)), (1, (2, 1, 2)), (0, (1, 0, 1)))


>>> for path_score, path in frozen2.path_iter(lambda x: x, startnode2, endnode2):
...   print tuple(frozen2.get_arc_label(arcid) for arcid in path)
(10, 9, 8, 11, 3, 2, 1)
(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

>>> for path_score, path in frozen.path_iter(lambda x: x, startnode, endnode):
...   print tuple(frozen.get_arc_label(arcid) for arcid in path)
(10, 9, 8, 11, 3, 2, 1)
(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)

>>> frozen.nodelabels
(-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11)
>>> frozen2.nodelabels
(-11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1)
>>> frozen.arclabels
(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
>>> frozen2.arclabels
(10, 9, 8, 7, 11, 6, 5, 4, 3, 2, 1)
"""


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    # _test()
