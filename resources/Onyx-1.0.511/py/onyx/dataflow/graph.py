###########################################################################
#
# File:         graph.py
# Date:         11-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Low-level graph objects for dataflow work
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007, 2009 The Johns Hopkins University
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
Nestable graph-based dataflow support.
"""
import itertools
import functools
import threading
from onyx import builtin
from onyx.util import checkutils
from onyx.graph import graphtools
import onyx.util.dotdisplay
import onyx.containers

class object_base(object):
    r"""
    Uggg!  A baseclass to use when cooperatively-super-using
    multiple-inheritance mix-ins are expecting a \*\*kwargs to __init__, so we
    need to insert a baseclass with an __init__ that makes sure the call to
    object.__init__ has no arguments.
    """
    def __init__(self, **kwargs):
        # users of wkargs must remove them!
        # XXX fix this
        # assert not kwargs, str(kwargs)
        super(object_base, self).__init__()

class Labeled(object_base):
    """
    Mix-in baseclass for objects that get constructed with a read-only label.
    """
    def __init__(self, **kwargs):
        label = kwargs.get('label')
        if label is None:
            self._label = '<unlabeled>'
        else:
            self._label = label
            del kwargs['label']
        super(Labeled, self).__init__(**kwargs)
    @property
    def label(self):
        return self._label
    @property
    def typed_label(self):
        return '%s(%r)' % (type(self).__name__, self.label)

class Managed(object_base):
    """
    Mix-in baseclass exposing a simple API for the concept of the manager of an
    object.
    """
    def __init__(self, **kwargs):
        super(Managed, self).__init__(**kwargs)
        self.set_managed(False)
    def set_managed(self, manager):
        self._managed = manager
    @property
    def is_managed(self):
        return bool(self._managed)
    @property
    def manager(self):
        assert self.is_managed
        return self._managed
    def check_managed(self, status):
        if self.is_managed != status:
            raise ValueError("expected %s labeled %r to be unmanaged, but it is managed by a %s" % (type(self).__name__, self.label, type(self._managed).__name__))

class Primitive(object_base):
    """
    Mix-in baseclass exposing properties for concept of primitive versus composite.

    >>> Primitive().is_primitive
    True
    >>> Primitive().is_composite
    False
    """
    @property
    def is_primitive(self):
        return True
    @property
    def is_composite(self):
        return not self.is_primitive

class Composite(Primitive):
    """
    Mix-in baseclass exposing fixed properties for concept of composite versus primitive.

    >>> Composite().is_composite
    True
    >>> Composite().is_primitive
    False
    """
    @property
    def is_primitive(self):
        return False

class ProcessorNode(Labeled, Managed, Primitive):
    """
    Baseclass for processor nodes.

    Implements functionality that all processor nodes must have.  At present
    this means they can be labeled via a constructor argument, label='some
    string', they can point to a manager, and they are primitive.
    """


class DataflowArc(object):
    """
    An arc in a dataflow network.

    A DataflowArc is a pair of port specifiers, the from_port and the to_port.
    These are keys that identify the particular out_port of the source node and
    the in_port of the target node.  
    """
    default_out_spec = None,
    default_in_spec = None,
    default_arc_spec = default_out_spec, default_in_spec
    def __init__(self, source_node, target_node, arc_spec):
        from_port, to_port = self._arc_spec = self.default_arc_spec if arc_spec == self.default_arc_spec else tuple(arc_spec)
        out_port, in_port = self.arc_spec
        pass_through = self.pass_through
        target_inport = target_node.get_inport(in_port)
        def do_it(item):
            pass_through(item)
            target_inport(item)
        # source_node.set_outtarget(out_port, target_node.get_inport(in_port))
        source_node.set_outtarget(out_port, do_it)

    @property
    def arc_spec(self):
        return self._arc_spec

    def pass_through(self, item):
        # this function is called as each item passes through the arc....
        #print 'pass_through', item
        pass


class MultiOut(object_base):
    """
    Mix-in baseclass for handling the set of output targets of a processor node. 
    """
    def __init__(self, **kwargs):
        super(MultiOut, self).__init__(**kwargs)
        self._outtargets = dict()
    @property
    def num_outtargets(self):
        return len(self._outtargets)
    @property
    def iter_targets(self):
        return self._outtargets.itervalues()
    def set_outtarget(self, outport_label, target):
        #assert outport_label not in self._outtargets
        if not callable(target):
            raise TypeError("%s labeled %r expected a callable target, got a %s" % (type(self).__name__, self.label, type(target).__name__))
        self._outtargets[outport_label] = target
    def get_outtarget(self, outport_label):
        return self._outtargets[outport_label]
        
    
class SingleOut(MultiOut):
    """
    Mix-in baseclass exposing an API for Single-Out processor functionality.
    """
    _only_outtarget_index = DataflowArc.default_out_spec
    @property
    def only_outtarget(self):
        return self.get_outtarget(self._only_outtarget_index)
    def set_only_outtarget(self, target):
        self.set_outtarget(self._only_outtarget_index, target)

    def set_outtarget(self, outport_label, target):
        assert outport_label is self._only_outtarget_index
        super(SingleOut, self).set_outtarget(outport_label, target)
        assert self.num_outtargets == 1
    def get_outtarget(self, outport_label):
        assert outport_label is self._only_outtarget_index
        assert self.num_outtargets == 1
        return super(SingleOut, self).get_outtarget(outport_label)

class MultiIn(object_base):
    """
    Mix-in baseclass for handling the set of inports to a processor node. 
    """
    def __init__(self, **kwargs):
        super(MultiIn, self).__init__(**kwargs)
        self._inports = dict()
    @property
    def num_inports(self):
        return len(self._inports)
    def set_inport(self, inport_label, intarget):
        self._inports[inport_label] = intarget
    def get_inport(self, inport_label):
        return self._inports[inport_label]

class SingleIn(MultiIn):
    """
    Mix-in baseclass exposing an API for Single-In processor functionality.
    """
    _only_inport_index = DataflowArc.default_in_spec
    def __init__(self, **kwargs):
        super(SingleIn, self).__init__(**kwargs)
        self._inport_id = -1
        # subclass constructor must set up
        # self._inports[self._only_inport_index] to be a callable
        
    def set_only_inport(self, only_intarget):
        self.set_inport(self._only_inport_index, only_intarget)

    def get_only_inport(self):
        assert self.num_inports == 1
        intarget = super(SingleIn, self).get_inport(self._only_inport_index)
        # XXX add a check method for this assert
        assert callable(intarget), "expected a callable, got a %s" % (type(intarget).__name__,)

        # wrap intarget in order to check against user losing track of having
        # replumbed connections into this inport
        #
        # XXX is there really a reason to allow get_only_inport to be called
        # more than once!
        # XXX alternatively, could be generalized 
        self._inport_id += 1
        inport_id = self._inport_id
        def inport(item):
            if inport_id != self._inport_id:
                raise ValueError("node labeled %r expected to be called with inport_id %d, but got %d, "
                                 "suggesting that the node was used as the target in more than one "
                                 "call to ProcessorGraphBuilder.connect() and the source from an "
                                 "earlier call to connect() is still being used" % (self._label, self._inport_id, inport_id))
            intarget(item)
        return inport

    def set_inport(self, inport_label, intarget):
        assert inport_label is self._only_inport_index
        super(SingleIn, self).set_inport(inport_label, intarget)
        assert self.num_inports == 1
    def get_inport(self, inport_label):
        assert inport_label is self._only_inport_index
        return self.get_only_inport()

class SisoProcessorNode(ProcessorNode, SingleIn, SingleOut):
    """
    Baseclass for processor nodes having single-input single-output semantics.

    Optional *label* can be provided to give a human-readable label for the
    node; default '<unlabeled>'.

    This baseclass provides the necessary interfaces for a processor node to be
    part of a managed set of processors in a processor graph.

    Subclasses must override <blah> and <blah> in order to implement their
    specific logic.  The baseclass is just an epsilon node that passes its
    inputs to its outputs in an unbuffered, one-to-one fashion.

    >>> sp = SisoProcessorNode('foo')
    >>> sp.is_primitive
    True
    >>> sp.label
    'foo'
    >>> sp.typed_label
    "SisoProcessorNode('foo')"

    >>> sp.set_only_outtarget(-2)
    Traceback (most recent call last):
      ...
    TypeError: SisoProcessorNode labeled 'foo' expected a callable target, got a int
    """
    def __init__(self, label=None):
        super(SisoProcessorNode, self).__init__(label=label)


class SisoMethodProcessorNode(SisoProcessorNode):
    """
    Baseclass for processor nodes that are implemented by overriding the
    process() method to implement processing logic.
    """

    def __init__(self, label=None):
        super(SisoMethodProcessorNode, self).__init__(label=label)

        # set up to call self.process() on an input item and push each of the results into our target
        process = self.process
        def only_intarget(item):
            # get target at usage time in order to dynamically respond to changes to the target
            # XXX rethink the notion of replumbing
            target = self.only_outtarget
            for result in process(item):
                target(result)
        self.set_only_inport(only_intarget)

    def process(self, item):
        """
        Baseclass implements pass-through behavior; it's an epsilon node.
        """
        yield item


class ProcessorGraphBuilder(object):
    """
    Builder for processor graphs.

    A processor graph is a directed acyclic graph of primitive
    SisoProcessorNodes.  Each node implements processing logic, the graph
    implements the connections between the nodes.

    >>> pgb = ProcessorGraphBuilder()
    >>> node1 = SisoMethodProcessorNode('node1')
    >>> n1 = pgb.new_node(node1)
    >>> n2 = pgb.new_node(SisoMethodProcessorNode('node2'))

    >>> pgb.connect(n1, n2)
    >>> p1 = pgb.processor_nodes[n1].get_only_inport()
    >>> res = list()
    >>> pgb.processor_nodes[n2].set_only_outtarget(res.append)
    >>> p1(True); p1(False); p1(None)
    >>> res
    [True, False, None]


    >>> n3 = pgb.new_node(node1)
    Traceback (most recent call last):
      ...
    ValueError: expected SisoMethodProcessorNode labeled 'node1' to be unmanaged, but it is managed by a ProcessorGraphBuilder

    >>> pgb.connect(n1, 1000)
    Traceback (most recent call last):
      ...
    ValueError: expected 0 <= start_node, end_node < 2, but got start_node = 0 and end_node = 1000

    >>> pgb.connect(pgb.new_node(SisoMethodProcessorNode('node3')), n2)
    >>> p1(1)
    Traceback (most recent call last):
      ...
    ValueError: node labeled 'node2' expected to be called with inport_id 1, but got 0, suggesting that the node was used as the target in more than one call to ProcessorGraphBuilder.connect() and the source from an earlier call to connect() is still being used
    """
    def __init__(self, *args):
        super(ProcessorGraphBuilder, self).__init__(*args)

        #self.builder = graphtools.GraphBuilder()
        self.builder = graphtools.GraphStructureBuilder()        
        self.processor_nodes = list()
        self.network_arcs = list()

        self.subgraph_tree_builder = graphtools.GraphStructureBuilder()
        subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels = self.subgraph_data = onyx.containers.tuplenoflist(4)
        self.subgraph_children = list()

        self._frozen = False

    def new_node(self, node):
        """
        Add *node*, a :class:`ProcessorNode`, to the graph.  If *node*\ ``.is_primitive``
        is ``True`` the node will be added to graph.  Otherwise,
        *node*\ ``.is_composite`` must be ``True``, and in this case the network
        structure and contained primitive nodes in *node* will be inserted into
        the graph as a subgraph.  The *node* reqlinquishes control of the
        primitive nodes and becomes useless.

        Returns an identifier for the primitive node or the subgraph.
        """
        self._check_alive()
        #checkutils.check_instance(SisoProcessorNode, node)
        checkutils.check_instance(ProcessorNode, node)        
        node.check_managed(False)

        len_processor_nodes = len(self.processor_nodes)
        assert len_processor_nodes == self.builder.num_nodes
        len_processor_arcs = self.builder.num_arcs
        assert len_processor_arcs == self.builder.num_arcs
        if node.is_primitive:
            # encoding the id type: non-negative for primitive nodes
            id = self.builder.new_node()
            #id = self.builder.new_node_label_is_id()
            assert id == len_processor_nodes
            self.processor_nodes.append(node)
        else:
            assert node.is_composite
            checkutils.check_instance(SisoProcessorGraphNode, node)
            assert node.verify()

            # XXX some of the management logic here sould be in the graph
            # builder; also the renumbering with an offset logic

            # this injects the node's processor network structure, renumbering nodes and arcs and node labels
            starts, ends = self.builder.add_graph(node.graph)
            assert len(starts) == len(ends) == 1
            # standard extension of node list
            self.processor_nodes.extend(node.processor_nodes)
            for nod in node.processor_nodes:
                nod.set_managed(self)
            # XXX need to properly tell node that it no longer owns the nodes....
            node.processor_nodes = None
            # capture the arcs
            self.network_arcs.extend(node.network_arcs)
            node.network_arcs = None

            # pull in the subgraph tree structure
            assert node.subgraph_tree.is_tree_strict
            (substart,), subends = self.subgraph_tree_builder.add_graph(node.subgraph_tree)
            self.subgraph_children.append(substart)
            # subtle: encoding the processor-node-type in the exposed id:
            # negative for composites (subgraphs), non-negative for primitives
            id = ~substart

            subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels = self.subgraph_data
            assert len(subgraph_nodeseqs) == len(subgraph_startseqs) == len(subgraph_endseqs) == len(subgraph_labels)
            # pull in the adjusted tree node data
            for start, end in itertools.izip(subgraph_startseqs, subgraph_endseqs):
                assert len(start) == len(end) == 1

            node_subgraph_nodeseqs, node_subgraph_startseqs, node_subgraph_endseqs, node_subgraph_labels = node.subgraph_data

            # standard extension of node list
            subgraph_labels.extend(node_subgraph_labels)

            # renumber these
            # XXX there could be an interface on the graph builder for managed lists of ids and their (orthogonal) adjustment by offsets....
            # note: at present, the total of subgraph_nodeseqs grows as O(n^2)
            # in the number of nodes; this won't be a problem until graphs are
            # large by our bootstrap standards
            subgraph_nodeseqs.extend(tuple(subnode_id+len_processor_nodes for subnode_id in subnode_seq) for subnode_seq in node_subgraph_nodeseqs)
            subgraph_startseqs.extend(tuple(subnode_id+len_processor_nodes for subnode_id in subnode_seq) for subnode_seq in node_subgraph_startseqs)
            subgraph_endseqs.extend(tuple(subnode_id+len_processor_nodes for subnode_id in subnode_seq) for subnode_seq in node_subgraph_endseqs)
            assert len(subgraph_nodeseqs) == len(subgraph_startseqs) == len(subgraph_endseqs) == len(subgraph_labels)

            for start, end in itertools.izip(subgraph_startseqs, subgraph_endseqs):
                assert len(start) == len(end) == 1

        node.set_managed(self)
        return id

    def _get_node(self, index):
        assert index >= 0
        return self.processor_nodes[index]

    def connect(self, source_id, target_id, arc_ports_spec=DataflowArc.default_arc_spec):
        """
        Connect the output of the node identified by *source_id* to the input of
        the node identified by *target_id*.  Optional arc_ports_spec is a pair
        specifying the output and input ports of the source and target nodes
        respectively.  Update the graph to reflect this connection.
        """
        self._check_alive()

        # resolve the source and/or target ids to the specific nodes 
        if source_id < 0 or target_id < 0:
            # subtle: redirect appropriately if the id is negative, indicating a subgraph
            subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels = self.subgraph_data
            if source_id < 0:
                source_id, = subgraph_endseqs[~source_id]
            if target_id < 0:
                target_id, = subgraph_startseqs[~target_id]
        assert source_id >= 0 and target_id >= 0

        # create the arc in the graph, error checks the ids too
        arc_id = self.builder.new_arc(source_id, target_id)

        # create the arc object that connects the two nodes at their specified ports
        arc = DataflowArc(self.processor_nodes[source_id], self.processor_nodes[target_id], arc_ports_spec)

        assert arc_id == len(self.network_arcs)
        self.network_arcs.append(arc)

    @property
    def frozen(self):
        """
        A ``bool`` indicating whether the builder has been frozen and is thus
        useless.  If ``False`` the builder is still usable.  If ``True`` the
        builder cannot be used and attempts to do so will raise ``ValueError``.
        """
        return self._frozen
    def _check_alive(self):
        if self.frozen:
            raise ValueError("%s has had freeze() called and cannot be used" % (type(self).__name__,))
    def freeze(self, label=None):
        """
        Freezes the builder and returns a pair, (nodes, graph), where nodes is a
        tuple of the :class:`ProcessorNode`\ s that make up the processor network,
        and graph is a :class:`~onyx.graph.graphtools.FrozenGraph` representing
        the structure of the network, where the label on each node in the graph
        is the index of the corresponding ProcessorNode in the nodes tuple.

        The ProcessorNodes in the nodes tuple are (already) internally linked so
        as to implement the network's functionality.  Since these nodes are in
        general mutable objects that cannot be deepcopied, the instance
        relinquishes ownership of the nodes and marks itself as frozen.  As
        such, it cannot be used for any futher work.  See
        :class:`SisoProcessorGraphNode`, which

        Raises ``ValueError`` if the graph does not satisfy the requirements
        that it be a lattice with no self loops.
        """
        self._check_alive()
        nodes, arcs, graph = tuple(self.processor_nodes), tuple(self.network_arcs), graphtools.FrozenGraph(self.builder)
        assert len(nodes) == graph.num_nodes
        assert len(arcs) == graph.num_arcs

        root_id = self.subgraph_tree_builder.new_node()
        for child_id in self.subgraph_children:
            self.subgraph_tree_builder.new_arc(root_id, child_id)
        subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels = self.subgraph_data
        assert len(subgraph_nodeseqs) == len(subgraph_startseqs) == len(subgraph_endseqs) == len(subgraph_labels) == root_id
        subgraph_nodeseqs.append(tuple(xrange(graph.num_nodes)))
        subgraph_startseqs.append(graph.startnodes)
        subgraph_endseqs.append(graph.endnodes)
        subgraph_labels.append(label)
        subgraph_data = tuple(tuple(seq) for seq in (subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels))

        subgraph_tree = graphtools.FrozenGraph(self.subgraph_tree_builder)
        assert subgraph_tree.is_tree_strict

        # XXX we're done owning the nodes, arcs, and subtree data
        self.processor_nodes = self.network_arcs = self.subgraph_data = None

        if graph.has_self_loop():
            raise ValueError("expected processor graph to be a strict DAG, but it has one or more self loops")
        if graph.has_cycle():
            raise ValueError("expected processor graph to be a strict DAG, but it has one or more multinode cycles")
        assert graph.is_strict_dag()
        if not graph.is_lattice():
            raise ValueError("expected processor graph to be a lattice, but it has %d startnodes and %d endnodes"
                             % (len(graph.startnodes), len(graph.endnodes)))

        self._frozen = True
        return nodes, arcs, graph, subgraph_data, subgraph_tree

class SisoProcessorGraphNode(SisoProcessorNode, Composite, onyx.util.dotdisplay.DotDisplay):
    """
    Composite siso processor.

    >>> builder = ProcessorGraphBuilder()
    >>> builder.connect(builder.new_node(SisoMethodProcessorNode('node1')), builder.new_node(SisoMethodProcessorNode('node2')))
    >>> spgn1 = SisoProcessorGraphNode(builder, 'graphnode1')
    >>> spgn1.is_primitive
    False

    >>> p1 = spgn1.get_only_inport()
    >>> res = list()
    >>> spgn1.set_only_outtarget(res.append)
    >>> p1('a');p1(1);p1(None);
    >>> res
    ['a', 1, None]
    """
    def __init__(self, builder, label=None):
        checkutils.check_instance(ProcessorGraphBuilder, builder)
        super(SisoProcessorGraphNode, self).__init__(label=label)
        self.processor_nodes, self.network_arcs, self.graph, self.subgraph_data, self.subgraph_tree = builder.freeze(label=label)
        for node in self.processor_nodes:
            #assert isinstance(node, SisoProcessorNode)
            assert node.is_primitive
            node.set_managed(self)
        startnode, = self.graph.startnodes
        endnode, = self.graph.endnodes
    
        self._inports[self._only_inport_index] = self.processor_nodes[startnode].get_only_inport()
        def proxy(item):
            # runtime use of only_outtarget means we respond to replumbing
            self.only_outtarget(item)
        self.processor_nodes[endnode].set_only_outtarget(proxy)

        self.verify()

    def verify(self):
        # some of these invariants point to a more efficient storage mechanism...

        # graph is of our processor nodes
        assert self.graph.num_nodes == len(self.processor_nodes)
        assert self.graph.num_arcs == len(self.network_arcs)

        # subgraph_tree is a graph of subgraphs
        subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels = self.subgraph_data
        assert self.subgraph_tree.num_nodes == len(subgraph_nodeseqs) == len(subgraph_startseqs) == len(subgraph_endseqs) == len(subgraph_labels)

        # subgraph_tree it's a strict tree of subgraphs
        assert self.subgraph_tree.is_tree_strict
        subgraph_start, = self.subgraph_tree.startnodes
        # top subgraph is all processor nodes
        assert subgraph_nodeseqs[subgraph_start] == tuple(xrange(len(self.processor_nodes)))
        for nodeseq in subgraph_nodeseqs:
            # a consequence of how we build the tree
            assert nodeseq == tuple(xrange(nodeseq[0], nodeseq[-1] + 1))

        return True
    
    # note: we inherit from onyx.util.dotdisplay.DotDisplay, so our dot_display
    # method will call our dot_iter method
    _default_label_callback = lambda label, *_: '' if label is None else str(label)
    def dot_iter(self, graph_label=None,
                 graph_attributes=(),
                 node_attributes=('height=.3', 'width=.4', 'fontsize=8'),
                 depth_limit=None,
                 force_display_labels=(),
                 node_label_callback=_default_label_callback,
                 arc_label_callback=_default_label_callback,
                 subgraph_label_callback=_default_label_callback,
                 node_attributes_callback=None,
                 arc_attributes_callback=None,
                 subgraph_attributes_callback=None):

        subgraph_nodeseqs, subgraph_startseqs, subgraph_endseqs, subgraph_labels = self.subgraph_data
        assert self.subgraph_tree.is_tree_strict
        adjout = self.subgraph_tree.nodeadjout

        def join_callback(obj, start_id, end_id):
            assert False
            return ()
        def cycle_callback(obj, start_id, end_id):
            assert False
            return ()

        # figure out set of tree_node_ids that will display
        force_display_labels = frozenset(force_display_labels)
        print 'force_display_labels'
        print force_display_labels
        force_display_ids = set()
        force_stack = list()
        flags = builtin.attrdict()
        flags.force_depth = 0
        def display_pre_callback(obj, tree_node_id):
            if len(force_stack) == 0:
                assert flags.force_depth == 0
            force_stack.append(tree_node_id)
            my_network_node_ids = set(subgraph_nodeseqs[tree_node_id])
            subgraph_network_node_ids = set(subgraph_network_node_id for subnode_id, arc_id in adjout[tree_node_id] for subgraph_network_node_id in subgraph_nodeseqs[subnode_id])
            assert subgraph_network_node_ids <= my_network_node_ids
            #my_non_nested_network_node_ids = my_network_node_ids - subgraph_network_node_ids
            my_non_nested_network_node_labels = set(self.processor_nodes[my_non_nested_network_node_id].label for my_non_nested_network_node_id in my_network_node_ids - subgraph_network_node_ids)
            if subgraph_labels[tree_node_id] in force_display_labels or my_non_nested_network_node_labels & force_display_labels:
                # this tree node, and all tree nodes above it will display
                flags.force_depth = len(force_stack)
            return ()
        def display_post_callback(obj, tree_node_id):
            if flags.force_depth >= len(force_stack):
                assert flags.force_depth == len(force_stack)
                force_display_ids.add(tree_node_id)
                flags.force_depth -= 1
            force_stack.pop()
            if len(force_stack) == 0:
                assert flags.force_depth == 0
            return ()

        for foo in self.subgraph_tree.depth_first_iter(pre_callback=display_pre_callback, post_callback=display_post_callback, join_callback=join_callback, cycle_callback=cycle_callback):
            print 'yowza3'

        indentation = '  '
        def indented_line(line):
            indent = indentation * (len(subgraph_stack) + 1)
            return '%s%s\n' % (indent, line,)

        node_name_format = 'n%%0%dd' % (len(str(len(self.processor_nodes)-1)),)
        def dot_node_name(network_node_id):
            # name of a network node, for DOT
            return node_name_format % (network_node_id,)

        composite_name_format = 'c%%0%dd' % (len(str(self.subgraph_tree.num_nodes-1)),)
        def dot_composite_name(tree_node_id):
            # name of a composite subgraph, for DOT
            return composite_name_format % (tree_node_id,)

        def get_attrs(label_callback, attributes_callback, node_id):
            attrs = list()
            attrs.append('label="%s"' % (label_callback(node_id),))
            if attributes_callback is not None:
                attrs.extend(attributes_callback(node_id))
            return attrs

        flags = builtin.attrdict()
        flags.cluster_number = 0

        # map from network node id to the id of the composite in which it's hidden
        hidden_node_map = dict()
        subgraph_stack = list()

        def pre_callback(obj, tree_node_id):
            # tree_node_id is a node in the subgraph tree
            assert self.subgraph_tree.get_node_label(tree_node_id) == tree_node_id
            my_network_node_ids = set(subgraph_nodeseqs[tree_node_id])
            len_subgraph_stack = len(subgraph_stack)
            if len_subgraph_stack == 0:
                assert len(self.processor_nodes) == self.graph.num_nodes
                assert my_network_node_ids == set(xrange(self.graph.num_nodes))
            if depth_limit is None or len_subgraph_stack < depth_limit or tree_node_id in force_display_ids:
                # start a new subgraph
                yield indented_line('subgraph cluster_%d {' % (flags.cluster_number,))
                subgraph_stack.append(tree_node_id)
                #attrs = get_attrs(subgraph_label_callback, subgraph_attributes_callback, tree_node_id)
                #yield indented_line('graph [%s];' % (', '.join(attrs),))
                yield indented_line('graph [label="%s", fontname="%s", labelloc=top, labeljust=c];' % (subgraph_labels[tree_node_id], 'Times-Bold' if subgraph_labels[tree_node_id] in force_display_labels else 'Times-Roman',))
                flags.cluster_number += 1
            else:
                if len_subgraph_stack == depth_limit or subgraph_stack[-1] in force_display_ids:
                    # box as a proxy for all enclosed subgraphs
                    assert not (my_network_node_ids & set(hidden_node_map))
                    # put all of the enclosed subgraph nodes into the map
                    hidden_node_map.update((my_network_node_id, tree_node_id) for my_network_node_id in my_network_node_ids)
                    yield indented_line('%s [label="%s", shape=box];' % (dot_composite_name(tree_node_id), subgraph_labels[tree_node_id],))
                else:
                    assert my_network_node_ids & set(hidden_node_map)
                subgraph_stack.append(tree_node_id)
                # note!
                return

            subgraph_network_node_ids = set(subgraph_network_node_id for subnode_id, arc_id in adjout[tree_node_id] for subgraph_network_node_id in subgraph_nodeseqs[subnode_id])
            assert subgraph_network_node_ids <= my_network_node_ids
            for my_non_nested_network_node_id in sorted(my_network_node_ids - subgraph_network_node_ids):
                node_label = self.processor_nodes[my_non_nested_network_node_id].label
                yield indented_line('%s [label="%s", style=%s];' % (dot_node_name(my_non_nested_network_node_id), node_label, 'bold' if node_label in force_display_labels else 'solid',))

        def post_callback(obj, tree_node_id):
            # tree_node_id is a node in the subgraph tree
            len_subgraph_stack = len(subgraph_stack)
            assert len_subgraph_stack >= 1
            assert subgraph_stack[-1] == tree_node_id
            if len_subgraph_stack == 1:
                # time to draw all the arcs
                nodelabels, arcstartnodes, arcendnodes, arclabels = self.graph.graphseqs
                for arc_id in self.graph.topoarcs:
                    start_node, end_node = arcstartnodes[arc_id], arcendnodes[arc_id]
                    if start_node in hidden_node_map and end_node in hidden_node_map and hidden_node_map[start_node] == hidden_node_map[end_node]:
                        # no arc for two hidden nodes inside the same composite
                        continue
                    start_name = dot_composite_name(hidden_node_map[start_node]) if start_node in hidden_node_map else dot_node_name(start_node)
                    end_name = dot_composite_name(hidden_node_map[end_node]) if end_node in hidden_node_map else dot_node_name(end_node)
                    #yield indented_line('%s -> %s [sametail="%s", samehead="%s"];' % (start_name, end_name, start_name, end_name))
                    #yield indented_line('%s -> %s [arrowhead=vee, arrowtail=crow];' % (start_name, end_name))
                    yield indented_line('%s -> %s [arrowhead=vee];' % (start_name, end_name))                    
            if depth_limit is None or len_subgraph_stack <= depth_limit or tree_node_id in force_display_ids:
                subgraph_stack.pop()
                yield indented_line('}')
            else:
                subgraph_stack.pop()

        # start the graph
        yield 'digraph %s {\n' % (('_' if graph_label is None else graph_label),)
        if graph_label is not None or graph_attributes:
            attrs = list()
            if depth_limit is not None:
                attrs.append('label="depth_limit=%d"' % (depth_limit,))
            attrs.extend(graph_attributes)
            yield indented_line('graph [%s];' % (', '.join(attrs),))
            yield indented_line('node [%s];' % (', '.join(node_attributes),))
        # do the guts of the (sub)graph(s)
        for line in self.subgraph_tree.depth_first_iter(pre_callback=pre_callback, post_callback=post_callback, join_callback=join_callback, cycle_callback=cycle_callback):
            yield line
        # finish the graph
        yield '}\n'

def SisoChainProcessor(siso_node_iterable, label=None):
    """
    Make a siso processor that chains together the siso processors making up *siso_node_iterable*.

    >>> chain1 = SisoChainProcessor((SisoMethodProcessorNode('node_%02d' % (index,)) for index in xrange(50)), 'chain1')
    >>> p1 = chain1.get_only_inport()
    >>> res1 = list()
    >>> chain1.set_only_outtarget(res1.append)
    >>> p1(True); p1(False); p1(None); p1('a')
    >>> res1
    [True, False, None, 'a']

    Demonstrate nesting ability of the SisoProcessorGraphNode that gets returned

    >>> numchains, numnodes = 10, 10
    >>> chains = (SisoChainProcessor((SisoMethodProcessorNode('node%02d_%02d' % (chain, node,)) for node in xrange(numnodes)), 'chain%d' % (chain,)) for chain in xrange(numchains))
    >>> nested = SisoChainProcessor(chains, 'nested%02d_%02d' % (numchains, numnodes))
    >>> nested.label
    'nested10_10'
    >>> nested.typed_label
    "SisoProcessorGraphNode('nested10_10')"
    >>> p2 = nested.get_only_inport()
    >>> res2 = list()
    >>> nested.set_only_outtarget(res2.append)
    >>> p2('a');p2('b');p2(123)
    >>> res2
    ['a', 'b', 123]

    >>> # nested.subgraph_tree.dot_display()
    >>> # nested.dot_display(graph_attributes=('label="%s"' % (nested.label,), 'labelloc=top;', 'rankdir=LR;'), node_label_callback=lambda label, x, y: nested.processor_nodes[label].label)
    """
    builder = ProcessorGraphBuilder()
    node_ids = tuple(builder.new_node(node) for node in siso_node_iterable)
    iter1 = iter(node_ids); iter1.next()
    for source, target in itertools.izip(node_ids, iter1):
        builder.connect(source, target)
    return SisoProcessorGraphNode(builder, label=label)

class Splitter(ProcessorNode, SingleIn, MultiOut):
    """
    Undifferentiated splitter.  Each input item is sent to each of the targets
    in an unspecified order.

    Each input item must be immutable.
    """
    def __init__(self, label=None):
        super(Splitter, self).__init__(label=label)
        def only_intarget(item):
            # check immutable
            hash(item)
            for target in self.iter_targets:
                target(item)
        self.set_only_inport(only_intarget)


class Joiner(ProcessorNode, MultiIn, SingleOut):
    """
    Undifferentiated serializing joiner.  Each input item is sent to the single
    target in the order it is received.
    """
    def __init__(self, label=None):
        super(Joiner, self).__init__(label=label)
        inport_lock = threading.Lock()
        def inport(item):
            inport_lock.acquire()
            self.only_outtarget(item)
            inport_lock.release()
        self.inport = inport
    def get_inport(self, inport_label):
        inport = self._inports.get(inport_label)
        if inport is None:
            self.set_inport(inport_label, self.inport)
        return super(Joiner, self).get_inport(inport_label)


def SisoParallelProcessor(siso_node_iterable, label=None):
    """
    Makes a siso processor that implements a parallel graph of the siso
    processors making up *siso_node_iterable*.  The output of the processor is
    serialized from each processor in the parallel graph.  This means the amount
    of data in the network is multiplied.
    
    >>> parallel1 = SisoParallelProcessor((SisoMethodProcessorNode('node_%02d' % (index,)) for index in xrange(3)), 'parallel1')
    >>> p1 = parallel1.get_only_inport()
    >>> res1 = list()
    >>> parallel1.set_only_outtarget(res1.append)
    >>> p1(True); p1(False); p1(None); p1('a')
    >>> res1
    [True, True, True, False, False, False, None, None, None, 'a', 'a', 'a']

    Demonstrate nesting ability of the SisoProcessorGraphNode that gets
    returned, and the data-multiplying effect

    >>> numchains, numnodes = 2, 3
    >>> chains = (SisoParallelProcessor((SisoMethodProcessorNode('node%02d_%02d' % (chain, node,)) for node in xrange(numnodes)), 'chain%d' % (chain,)) for chain in xrange(numchains))
    >>> nested = SisoParallelProcessor(chains, 'nested_%02d_%02d' % (numchains, numnodes))
    >>> nested.label
    'nested_02_03'
    >>> nested.typed_label
    "SisoProcessorGraphNode('nested_02_03')"
    >>> p2 = nested.get_only_inport()
    >>> res2 = list()
    >>> nested.set_only_outtarget(res2.append)
    >>> p2('a');p2('b');p2(123)
    >>> res2
    ['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 123, 123, 123, 123, 123, 123]

    >>> # nested.subgraph_tree.dot_display()
    >>> # nested.dot_display(graph_attributes=('label="%s"' % (nested.label,), 'labelloc=top;', 'rankdir=LR;'), node_label_callback=lambda label, x, y: nested.processor_nodes[label].label)
    """
    builder = ProcessorGraphBuilder()
    start_id = builder.new_node(Splitter(label='Split\\n%s' % (label,)))
    node_ids = tuple(builder.new_node(node) for node in siso_node_iterable)
    end_id = builder.new_node(Joiner(label='Join\\n%s' % (label,)))
    for index, node_id in enumerate(node_ids):
        builder.connect(start_id, node_id, (index, DataflowArc.default_in_spec))
        builder.connect(node_id, end_id, (DataflowArc.default_out_spec, index))
    return SisoProcessorGraphNode(builder, label=label)

def _nested(node_factory, nodes_composers, order, depth, label=None):
    r"""
    Test display features of the nested subgraph structure

    >>> order, depth = 3, 3
    >>> nested = _nested(SisoMethodProcessorNode, (SisoChainProcessor, SisoParallelProcessor,), order, depth, label='nested_1')
    >>> len(nested.processor_nodes), nested.graph.num_nodes, nested.graph.num_arcs
    (64, 64, 90)

    Attributes for the following display work

    >>> graph_attributes = 'labelloc=bottom', 'labeljust=r', 'rankdir=LR', 'size=11,8.5', 'ratio=compress', 'margin=0', 'nodesep=.05', 'ranksep=0', 'mindist=0', 'fontsize=14'

    Note: set visual_display to True in order to actually get visual display of
    whole graph and then the force_display_labels limited versions

    >>> visual_display = None
    >>> visual_display and nested.dot_display(graph_attributes=graph_attributes)
    >>> visual_display and nested.dot_display(depth_limit=1, graph_attributes=graph_attributes, force_display_labels=('composite_7',))
    >>> visual_display and nested.dot_display(depth_limit=2, graph_attributes=graph_attributes, force_display_labels=('primitive_11',))
    >>> visual_display and nested.dot_display(depth_limit=1, graph_attributes=graph_attributes, force_display_labels=('composite_7', 'primitive_11',))


    Look at depth_limited dot output

    >>> for limit in (0, 1, 2):
    ...   print 'limit', limit
    ...   for line in nested.dot_iter(depth_limit=limit, graph_attributes=graph_attributes): print line.rstrip()
    limit 0
    force_display_labels
    frozenset([])
    digraph _ {
      graph [label="depth_limit=0", labelloc=bottom, labeljust=r, rankdir=LR, size=11,8.5, ratio=compress, margin=0, nodesep=.05, ranksep=0, mindist=0, fontsize=14];
      node [height=.3, width=.4, fontsize=8];
      c12 [label="nested_1", shape=box];
    }
    limit 1
    force_display_labels
    frozenset([])
    digraph _ {
      graph [label="depth_limit=1", labelloc=bottom, labeljust=r, rankdir=LR, size=11,8.5, ratio=compress, margin=0, nodesep=.05, ranksep=0, mindist=0, fontsize=14];
      node [height=.3, width=.4, fontsize=8];
      subgraph cluster_0 {
        graph [label="nested_1", fontname="Times-Roman", labelloc=top, labeljust=c];
        n00 [label="primitive_0", style=solid];
        n01 [label="primitive_1", style=solid];
        n02 [label="primitive_2", style=solid];
        n63 [label="primitive_51", style=solid];
        c11 [label="composite_8", shape=box];
        c07 [label="composite_4", shape=box];
        c03 [label="composite_0", shape=box];
        n00 -> n01 [arrowhead=vee];
        n01 -> n02 [arrowhead=vee];
        n02 -> c03 [arrowhead=vee];
        c03 -> c07 [arrowhead=vee];
        c07 -> c11 [arrowhead=vee];
        c11 -> n63 [arrowhead=vee];
      }
    }
    limit 2
    force_display_labels
    frozenset([])
    digraph _ {
      graph [label="depth_limit=2", labelloc=bottom, labeljust=r, rankdir=LR, size=11,8.5, ratio=compress, margin=0, nodesep=.05, ranksep=0, mindist=0, fontsize=14];
      node [height=.3, width=.4, fontsize=8];
      subgraph cluster_0 {
        graph [label="nested_1", fontname="Times-Roman", labelloc=top, labeljust=c];
        n00 [label="primitive_0", style=solid];
        n01 [label="primitive_1", style=solid];
        n02 [label="primitive_2", style=solid];
        n63 [label="primitive_51", style=solid];
        subgraph cluster_1 {
          graph [label="composite_8", fontname="Times-Roman", labelloc=top, labeljust=c];
          n43 [label="Split\ncomposite_8", style=solid];
          n44 [label="primitive_35", style=solid];
          n45 [label="primitive_36", style=solid];
          n46 [label="primitive_37", style=solid];
          n61 [label="primitive_50", style=solid];
          n62 [label="Join\ncomposite_8", style=solid];
          c10 [label="composite_11", shape=box];
          c09 [label="composite_10", shape=box];
          c08 [label="composite_9", shape=box];
        }
        subgraph cluster_2 {
          graph [label="composite_4", fontname="Times-Roman", labelloc=top, labeljust=c];
          n23 [label="Split\ncomposite_4", style=solid];
          n24 [label="primitive_19", style=solid];
          n25 [label="primitive_20", style=solid];
          n26 [label="primitive_21", style=solid];
          n41 [label="primitive_34", style=solid];
          n42 [label="Join\ncomposite_4", style=solid];
          c06 [label="composite_7", shape=box];
          c05 [label="composite_6", shape=box];
          c04 [label="composite_5", shape=box];
        }
        subgraph cluster_3 {
          graph [label="composite_0", fontname="Times-Roman", labelloc=top, labeljust=c];
          n03 [label="Split\ncomposite_0", style=solid];
          n04 [label="primitive_3", style=solid];
          n05 [label="primitive_4", style=solid];
          n06 [label="primitive_5", style=solid];
          n21 [label="primitive_18", style=solid];
          n22 [label="Join\ncomposite_0", style=solid];
          c02 [label="composite_3", shape=box];
          c01 [label="composite_2", shape=box];
          c00 [label="composite_1", shape=box];
        }
        n00 -> n01 [arrowhead=vee];
        n01 -> n02 [arrowhead=vee];
        n02 -> n03 [arrowhead=vee];
        n03 -> n04 [arrowhead=vee];
        n03 -> n05 [arrowhead=vee];
        n03 -> n06 [arrowhead=vee];
        n03 -> c00 [arrowhead=vee];
        n03 -> c01 [arrowhead=vee];
        n03 -> c02 [arrowhead=vee];
        n03 -> n21 [arrowhead=vee];
        n04 -> n22 [arrowhead=vee];
        n05 -> n22 [arrowhead=vee];
        n06 -> n22 [arrowhead=vee];
        c00 -> n22 [arrowhead=vee];
        c01 -> n22 [arrowhead=vee];
        c02 -> n22 [arrowhead=vee];
        n21 -> n22 [arrowhead=vee];
        n22 -> n23 [arrowhead=vee];
        n23 -> n24 [arrowhead=vee];
        n23 -> n25 [arrowhead=vee];
        n23 -> n26 [arrowhead=vee];
        n23 -> c04 [arrowhead=vee];
        n23 -> c05 [arrowhead=vee];
        n23 -> c06 [arrowhead=vee];
        n23 -> n41 [arrowhead=vee];
        n24 -> n42 [arrowhead=vee];
        n25 -> n42 [arrowhead=vee];
        n26 -> n42 [arrowhead=vee];
        c04 -> n42 [arrowhead=vee];
        c05 -> n42 [arrowhead=vee];
        c06 -> n42 [arrowhead=vee];
        n41 -> n42 [arrowhead=vee];
        n42 -> n43 [arrowhead=vee];
        n43 -> n44 [arrowhead=vee];
        n43 -> n45 [arrowhead=vee];
        n43 -> n46 [arrowhead=vee];
        n43 -> c08 [arrowhead=vee];
        n43 -> c09 [arrowhead=vee];
        n43 -> c10 [arrowhead=vee];
        n43 -> n61 [arrowhead=vee];
        n44 -> n62 [arrowhead=vee];
        n45 -> n62 [arrowhead=vee];
        n46 -> n62 [arrowhead=vee];
        c08 -> n62 [arrowhead=vee];
        c09 -> n62 [arrowhead=vee];
        c10 -> n62 [arrowhead=vee];
        n61 -> n62 [arrowhead=vee];
        n62 -> n63 [arrowhead=vee];
      }
    }
    
    Look at the actual dot output from force_display_labels

    >>> # for line in nested.dot_iter(graph_attributes=graph_attributes): print line.rstrip()
    >>> for line in nested.dot_iter(depth_limit=1, graph_attributes=graph_attributes, force_display_labels=('composite_7',)): print line.rstrip()
    force_display_labels
    frozenset(['composite_7'])
    digraph _ {
      graph [label="depth_limit=1", labelloc=bottom, labeljust=r, rankdir=LR, size=11,8.5, ratio=compress, margin=0, nodesep=.05, ranksep=0, mindist=0, fontsize=14];
      node [height=.3, width=.4, fontsize=8];
      subgraph cluster_0 {
        graph [label="nested_1", fontname="Times-Roman", labelloc=top, labeljust=c];
        n00 [label="primitive_0", style=solid];
        n01 [label="primitive_1", style=solid];
        n02 [label="primitive_2", style=solid];
        n63 [label="primitive_51", style=solid];
        c11 [label="composite_8", shape=box];
        subgraph cluster_1 {
          graph [label="composite_4", fontname="Times-Roman", labelloc=top, labeljust=c];
          n23 [label="Split\ncomposite_4", style=solid];
          n24 [label="primitive_19", style=solid];
          n25 [label="primitive_20", style=solid];
          n26 [label="primitive_21", style=solid];
          n41 [label="primitive_34", style=solid];
          n42 [label="Join\ncomposite_4", style=solid];
          subgraph cluster_2 {
            graph [label="composite_7", fontname="Times-Bold", labelloc=top, labeljust=c];
            n37 [label="primitive_30", style=solid];
            n38 [label="primitive_31", style=solid];
            n39 [label="primitive_32", style=solid];
            n40 [label="primitive_33", style=solid];
          }
          c05 [label="composite_6", shape=box];
          c04 [label="composite_5", shape=box];
        }
        c03 [label="composite_0", shape=box];
        n00 -> n01 [arrowhead=vee];
        n01 -> n02 [arrowhead=vee];
        n02 -> c03 [arrowhead=vee];
        c03 -> n23 [arrowhead=vee];
        n23 -> n24 [arrowhead=vee];
        n23 -> n25 [arrowhead=vee];
        n23 -> n26 [arrowhead=vee];
        n23 -> c04 [arrowhead=vee];
        n23 -> c05 [arrowhead=vee];
        n23 -> n37 [arrowhead=vee];
        n37 -> n38 [arrowhead=vee];
        n38 -> n39 [arrowhead=vee];
        n39 -> n40 [arrowhead=vee];
        n23 -> n41 [arrowhead=vee];
        n24 -> n42 [arrowhead=vee];
        n25 -> n42 [arrowhead=vee];
        n26 -> n42 [arrowhead=vee];
        c04 -> n42 [arrowhead=vee];
        c05 -> n42 [arrowhead=vee];
        n40 -> n42 [arrowhead=vee];
        n41 -> n42 [arrowhead=vee];
        n42 -> c11 [arrowhead=vee];
        c11 -> n63 [arrowhead=vee];
      }
    }

    >>> for line in nested.dot_iter(depth_limit=2, graph_attributes=graph_attributes, force_display_labels=('primitive_11',)): print line.rstrip()
    force_display_labels
    frozenset(['primitive_11'])
    digraph _ {
      graph [label="depth_limit=2", labelloc=bottom, labeljust=r, rankdir=LR, size=11,8.5, ratio=compress, margin=0, nodesep=.05, ranksep=0, mindist=0, fontsize=14];
      node [height=.3, width=.4, fontsize=8];
      subgraph cluster_0 {
        graph [label="nested_1", fontname="Times-Roman", labelloc=top, labeljust=c];
        n00 [label="primitive_0", style=solid];
        n01 [label="primitive_1", style=solid];
        n02 [label="primitive_2", style=solid];
        n63 [label="primitive_51", style=solid];
        subgraph cluster_1 {
          graph [label="composite_8", fontname="Times-Roman", labelloc=top, labeljust=c];
          n43 [label="Split\ncomposite_8", style=solid];
          n44 [label="primitive_35", style=solid];
          n45 [label="primitive_36", style=solid];
          n46 [label="primitive_37", style=solid];
          n61 [label="primitive_50", style=solid];
          n62 [label="Join\ncomposite_8", style=solid];
          c10 [label="composite_11", shape=box];
          c09 [label="composite_10", shape=box];
          c08 [label="composite_9", shape=box];
        }
        subgraph cluster_2 {
          graph [label="composite_4", fontname="Times-Roman", labelloc=top, labeljust=c];
          n23 [label="Split\ncomposite_4", style=solid];
          n24 [label="primitive_19", style=solid];
          n25 [label="primitive_20", style=solid];
          n26 [label="primitive_21", style=solid];
          n41 [label="primitive_34", style=solid];
          n42 [label="Join\ncomposite_4", style=solid];
          c06 [label="composite_7", shape=box];
          c05 [label="composite_6", shape=box];
          c04 [label="composite_5", shape=box];
        }
        subgraph cluster_3 {
          graph [label="composite_0", fontname="Times-Roman", labelloc=top, labeljust=c];
          n03 [label="Split\ncomposite_0", style=solid];
          n04 [label="primitive_3", style=solid];
          n05 [label="primitive_4", style=solid];
          n06 [label="primitive_5", style=solid];
          n21 [label="primitive_18", style=solid];
          n22 [label="Join\ncomposite_0", style=solid];
          c02 [label="composite_3", shape=box];
          subgraph cluster_4 {
            graph [label="composite_2", fontname="Times-Roman", labelloc=top, labeljust=c];
            n11 [label="Split\ncomposite_2", style=solid];
            n12 [label="primitive_10", style=solid];
            n13 [label="primitive_11", style=bold];
            n14 [label="primitive_12", style=solid];
            n15 [label="primitive_13", style=solid];
            n16 [label="Join\ncomposite_2", style=solid];
          }
          c00 [label="composite_1", shape=box];
        }
        n00 -> n01 [arrowhead=vee];
        n01 -> n02 [arrowhead=vee];
        n02 -> n03 [arrowhead=vee];
        n03 -> n04 [arrowhead=vee];
        n03 -> n05 [arrowhead=vee];
        n03 -> n06 [arrowhead=vee];
        n03 -> c00 [arrowhead=vee];
        n03 -> n11 [arrowhead=vee];
        n11 -> n12 [arrowhead=vee];
        n11 -> n13 [arrowhead=vee];
        n11 -> n14 [arrowhead=vee];
        n11 -> n15 [arrowhead=vee];
        n12 -> n16 [arrowhead=vee];
        n13 -> n16 [arrowhead=vee];
        n14 -> n16 [arrowhead=vee];
        n15 -> n16 [arrowhead=vee];
        n03 -> c02 [arrowhead=vee];
        n03 -> n21 [arrowhead=vee];
        n04 -> n22 [arrowhead=vee];
        n05 -> n22 [arrowhead=vee];
        n06 -> n22 [arrowhead=vee];
        c00 -> n22 [arrowhead=vee];
        n16 -> n22 [arrowhead=vee];
        c02 -> n22 [arrowhead=vee];
        n21 -> n22 [arrowhead=vee];
        n22 -> n23 [arrowhead=vee];
        n23 -> n24 [arrowhead=vee];
        n23 -> n25 [arrowhead=vee];
        n23 -> n26 [arrowhead=vee];
        n23 -> c04 [arrowhead=vee];
        n23 -> c05 [arrowhead=vee];
        n23 -> c06 [arrowhead=vee];
        n23 -> n41 [arrowhead=vee];
        n24 -> n42 [arrowhead=vee];
        n25 -> n42 [arrowhead=vee];
        n26 -> n42 [arrowhead=vee];
        c04 -> n42 [arrowhead=vee];
        c05 -> n42 [arrowhead=vee];
        c06 -> n42 [arrowhead=vee];
        n41 -> n42 [arrowhead=vee];
        n42 -> n43 [arrowhead=vee];
        n43 -> n44 [arrowhead=vee];
        n43 -> n45 [arrowhead=vee];
        n43 -> n46 [arrowhead=vee];
        n43 -> c08 [arrowhead=vee];
        n43 -> c09 [arrowhead=vee];
        n43 -> c10 [arrowhead=vee];
        n43 -> n61 [arrowhead=vee];
        n44 -> n62 [arrowhead=vee];
        n45 -> n62 [arrowhead=vee];
        n46 -> n62 [arrowhead=vee];
        c08 -> n62 [arrowhead=vee];
        c09 -> n62 [arrowhead=vee];
        c10 -> n62 [arrowhead=vee];
        n61 -> n62 [arrowhead=vee];
        n62 -> n63 [arrowhead=vee];
      }
    }

    >>> for line in nested.dot_iter(depth_limit=2, graph_attributes=graph_attributes, force_display_labels=('primitive_11', 'composite_7')): print line.rstrip()
    force_display_labels
    frozenset(['composite_7', 'primitive_11'])
    digraph _ {
      graph [label="depth_limit=2", labelloc=bottom, labeljust=r, rankdir=LR, size=11,8.5, ratio=compress, margin=0, nodesep=.05, ranksep=0, mindist=0, fontsize=14];
      node [height=.3, width=.4, fontsize=8];
      subgraph cluster_0 {
        graph [label="nested_1", fontname="Times-Roman", labelloc=top, labeljust=c];
        n00 [label="primitive_0", style=solid];
        n01 [label="primitive_1", style=solid];
        n02 [label="primitive_2", style=solid];
        n63 [label="primitive_51", style=solid];
        subgraph cluster_1 {
          graph [label="composite_8", fontname="Times-Roman", labelloc=top, labeljust=c];
          n43 [label="Split\ncomposite_8", style=solid];
          n44 [label="primitive_35", style=solid];
          n45 [label="primitive_36", style=solid];
          n46 [label="primitive_37", style=solid];
          n61 [label="primitive_50", style=solid];
          n62 [label="Join\ncomposite_8", style=solid];
          c10 [label="composite_11", shape=box];
          c09 [label="composite_10", shape=box];
          c08 [label="composite_9", shape=box];
        }
        subgraph cluster_2 {
          graph [label="composite_4", fontname="Times-Roman", labelloc=top, labeljust=c];
          n23 [label="Split\ncomposite_4", style=solid];
          n24 [label="primitive_19", style=solid];
          n25 [label="primitive_20", style=solid];
          n26 [label="primitive_21", style=solid];
          n41 [label="primitive_34", style=solid];
          n42 [label="Join\ncomposite_4", style=solid];
          subgraph cluster_3 {
            graph [label="composite_7", fontname="Times-Bold", labelloc=top, labeljust=c];
            n37 [label="primitive_30", style=solid];
            n38 [label="primitive_31", style=solid];
            n39 [label="primitive_32", style=solid];
            n40 [label="primitive_33", style=solid];
          }
          c05 [label="composite_6", shape=box];
          c04 [label="composite_5", shape=box];
        }
        subgraph cluster_4 {
          graph [label="composite_0", fontname="Times-Roman", labelloc=top, labeljust=c];
          n03 [label="Split\ncomposite_0", style=solid];
          n04 [label="primitive_3", style=solid];
          n05 [label="primitive_4", style=solid];
          n06 [label="primitive_5", style=solid];
          n21 [label="primitive_18", style=solid];
          n22 [label="Join\ncomposite_0", style=solid];
          c02 [label="composite_3", shape=box];
          subgraph cluster_5 {
            graph [label="composite_2", fontname="Times-Roman", labelloc=top, labeljust=c];
            n11 [label="Split\ncomposite_2", style=solid];
            n12 [label="primitive_10", style=solid];
            n13 [label="primitive_11", style=bold];
            n14 [label="primitive_12", style=solid];
            n15 [label="primitive_13", style=solid];
            n16 [label="Join\ncomposite_2", style=solid];
          }
          c00 [label="composite_1", shape=box];
        }
        n00 -> n01 [arrowhead=vee];
        n01 -> n02 [arrowhead=vee];
        n02 -> n03 [arrowhead=vee];
        n03 -> n04 [arrowhead=vee];
        n03 -> n05 [arrowhead=vee];
        n03 -> n06 [arrowhead=vee];
        n03 -> c00 [arrowhead=vee];
        n03 -> n11 [arrowhead=vee];
        n11 -> n12 [arrowhead=vee];
        n11 -> n13 [arrowhead=vee];
        n11 -> n14 [arrowhead=vee];
        n11 -> n15 [arrowhead=vee];
        n12 -> n16 [arrowhead=vee];
        n13 -> n16 [arrowhead=vee];
        n14 -> n16 [arrowhead=vee];
        n15 -> n16 [arrowhead=vee];
        n03 -> c02 [arrowhead=vee];
        n03 -> n21 [arrowhead=vee];
        n04 -> n22 [arrowhead=vee];
        n05 -> n22 [arrowhead=vee];
        n06 -> n22 [arrowhead=vee];
        c00 -> n22 [arrowhead=vee];
        n16 -> n22 [arrowhead=vee];
        c02 -> n22 [arrowhead=vee];
        n21 -> n22 [arrowhead=vee];
        n22 -> n23 [arrowhead=vee];
        n23 -> n24 [arrowhead=vee];
        n23 -> n25 [arrowhead=vee];
        n23 -> n26 [arrowhead=vee];
        n23 -> c04 [arrowhead=vee];
        n23 -> c05 [arrowhead=vee];
        n23 -> n37 [arrowhead=vee];
        n37 -> n38 [arrowhead=vee];
        n38 -> n39 [arrowhead=vee];
        n39 -> n40 [arrowhead=vee];
        n23 -> n41 [arrowhead=vee];
        n24 -> n42 [arrowhead=vee];
        n25 -> n42 [arrowhead=vee];
        n26 -> n42 [arrowhead=vee];
        c04 -> n42 [arrowhead=vee];
        c05 -> n42 [arrowhead=vee];
        n40 -> n42 [arrowhead=vee];
        n41 -> n42 [arrowhead=vee];
        n42 -> n43 [arrowhead=vee];
        n43 -> n44 [arrowhead=vee];
        n43 -> n45 [arrowhead=vee];
        n43 -> n46 [arrowhead=vee];
        n43 -> c08 [arrowhead=vee];
        n43 -> c09 [arrowhead=vee];
        n43 -> c10 [arrowhead=vee];
        n43 -> n61 [arrowhead=vee];
        n44 -> n62 [arrowhead=vee];
        n45 -> n62 [arrowhead=vee];
        n46 -> n62 [arrowhead=vee];
        c08 -> n62 [arrowhead=vee];
        c09 -> n62 [arrowhead=vee];
        c10 -> n62 [arrowhead=vee];
        n61 -> n62 [arrowhead=vee];
        n62 -> n63 [arrowhead=vee];
      }
    }
    """
    xrange_one = xrange(1)
    xrange_order = xrange(order)
    primitive_counter = iter(itertools.count())
    composite_counter = iter(itertools.count())
    composers = iter(itertools.cycle(nodes_composers))
    def recursive_builder(depth):
        # make order nodes and, recursively, a composite of order children, plus one extra
        return itertools.chain((node_factory(label='primitive_' + str(primitive_counter.next())) for i in xrange_order),
                               () if depth == 1 else (composers.next()(recursive_builder(depth-1), label='composite_' + str(composite_counter.next())) for i in xrange_order),
                               (node_factory(label='primitive_' + str(primitive_counter.next())) for i in xrange_one))
    return composers.next()(recursive_builder(depth), label=label)
    

class Join(object):
    """
    Obsolete, empty class, used by objectset for its testing.

    >>> Join(None) #doctest: +ELLIPSIS
    <__main__.Join object at 0x...>
    """
    def __init__(self, _):
        pass

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
