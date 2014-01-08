###########################################################################
#
# File:         decodergraph.py
# Date:         1-Apr-2008
# Author:       Hugh Secker-Walker
# Description:  A graph builder suitable for decoding while building.
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
A graph builder that can decode while it's building.
"""

from itertools import izip
from onyx.graph.graphtools import TopologicalGraphBuilder

class DecoderGraph(TopologicalGraphBuilder):
    """
    A TopologicalGraphBuilder suitable for doing decoding while it's
    being built.

    >>> builder = DecoderGraph()
    >>> num_nodes = 5
    >>> nodes = tuple(builder.new_node_label_is_id() for i in xrange(num_nodes))
    >>> for node in nodes:
    ...   builder.new_arc_label_is_id(node, node) 
    ...   if node + 1 < num_nodes: builder.new_arc_label_is_id(node, node + 1)
    ...   if node + 2 < num_nodes: builder.new_arc_label_is_id(node, node + 2)
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11

    >>> tuple(builder.in_arc_iter(0))
    ((0, 0),)
    >>> tuple(builder.in_arc_iter(num_nodes-1))
    ((2, 8), (3, 10), (4, 11))
    
    A list for each end-node, a tuple of start-node-id, arc-id for
    each arc.
    >>> builder.in_arcs
    [[(0, 0)], [(0, 1), (1, 3)], [(0, 2), (1, 4), (2, 6)], [(1, 5), (2, 7), (3, 9)], [(2, 8), (3, 10), (4, 11)]]
    """
    def __init__(self, *args):
        super(DecoderGraph, self).__init__(allow_self_loops=True, *args)
        assert self.allow_self_loops
        # maintain these during the build
        self.in_arcs = list()

    def new_arc(self, start_node, end_node, arclabel=None):
        ret = super(DecoderGraph, self).new_arc(start_node, end_node, arclabel)
        in_arcs = self.in_arcs
        while len(in_arcs) <= end_node:
            in_arcs.append(list())
        in_arcs[end_node].append((start_node, ret))
        return ret

    def in_arc_iter(self, end_node_id):
        self._check_node_id(end_node_id, "in_arc_iter end_node")
        return iter(self.in_arcs[end_node_id])

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
