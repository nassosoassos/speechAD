###########################################################################
#
# File:         graphsandbox.py
# Date:         08-26-2008
# Author:       Hugh Secker-Walker
# Description:  A place for sandbox and development work for graph tools
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
Sandbox where new graph stuff is developed.
"""
from onyx.containers import frozenbijection, tuplenoflist

def WfstCompose(wfst1, wfst2, epsilon=None):
    """
    >>> # epsilon = '&Epsilon;'
    >>> epsilon = '-'
    >>> wfst1 = FrozenGraph(GraphTables(((0, 1), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (('a', 'A'), ('b', 'B'), ('c', 'C'), ('d', 'D'), ('e', 'E')))))
    >>> wfst2 = FrozenGraph(GraphTables(((0, 1), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (('A', 0), ('B', 1), ('C', 2), ('D', 3), ('E', 4, ))))) #0.5)))))
    >>> wfst3 = wfst1 + wfst2 + WfstCompose(wfst1, wfst2)
    workset: set([(0, 0)])
    a A 0
    b B 1
    c C 2
    d D 3
    e E 4
    >>> #wfst3.dot_display(arc_label_callback=_wfst_arc_label_callback, globals=('rankdir=LR;',)) and None

    >>> wfst10 = FrozenGraph(GraphTables(((0, 1, 2, 3, 4), (0, 1, 1, 2, 3), (1, 2, 3, 1, 4), (('a', 'b'), ('p','a'), ('l','a'), ('p','n'), ('e', epsilon)))))
    >>> wfst11 = FrozenGraph(GraphTables(((0, 1), (0, 0, 0, 1, ), (1, 1, 1, 0, ), (('a', 'A'), ('b', 'B'), ('n', 'N'), (epsilon, epsilon), ))))
    >>> wfst12 = wfst10 + wfst11 + WfstCompose(wfst10, wfst11, epsilon)
    workset: set([(0, 0)])
    a b B
    p a A
    l a A
    p n N
    >>> #wfst12.dot_display(arc_label_callback=_wfst_arc_label_callback, globals=('rankdir=LR;',)) and None
    """

    nodelabels1, arcstartnodes1, arcendnodes1, arclabels1 = wfst1.graphseqs
    nodelabels2, arcstartnodes2, arcendnodes2, arclabels2 = wfst2.graphseqs

    nodeadjout1 = wfst1.nodeadjout
    nodeadjout2 = wfst2.nodeadjout

    starts1 = frozenset((0,))
    starts2 = frozenset((0,))
    ends1 = frozenset((4,))
    ends2 = frozenset((1,))

    # XXX need to get per-node epsilon closures first

    workset = set((s1, s2) for s1 in starts1 for s2 in starts2)
    print 'workset:', workset
    # some may become isolated
    startnodes = frozenset(workset)
    nodes = set(startnodes)
    arcs = set()
    while workset:
        u1, u2 = u = workset.pop()
        outs1 = nodeadjout1[u1]
        outs2 = nodeadjout2[u2]
        for v1, e1 in outs1:
            in1, out1 = arclabels1[e1]
            for v2, e2 in outs2:
                in2, out2 = arclabels2[e2]
                if in2 is out2 is epsilon:
                    v = u1, v2
                    if v not in nodes:
                        nodes.add(v)
                        assert v not in workset
                        workset.add(v)
                elif out1 is epsilon:
                    v = v1, v2
                    if v not in nodes:
                        nodes.add(v)
                        assert v not in workset
                        workset.add(v)
                    arc = u, v, (in1, out1)
                    #assert arc not in arcs
                    arcs.add(arc)
                elif out1 == in2:
                    print in1, out1, out2
                    v = v1, v2
                    if v not in nodes:
                        nodes.add(v)
                        assert v not in workset
                        workset.add(v)
                    arc = u, v, (in1, out2)
                    assert arc not in arcs
                    arcs.add(arc)
    node_by_id, id_by_node = frozenbijection(nodes)
    arcstart, arcend, arclabel = tuplenoflist(3)
    for u, v, label in arcs:
        arcstart.append(id_by_node[u])
        arcend.append(id_by_node[v])
        arclabel.append(label)
    return GraphBase(GraphTables((node_by_id, tuple(arcstart), tuple(arcend), tuple(arclabel))))

def _wfst_arc_label_callback(label):
    format = 'label="%s:%s/%s"' if len(label) == 3 else 'label="%s:%s"'
    return format % label
        
def _confusion_sandbox():
    """
    >>> _confusion_sandbox()
    """
    g3 = FrozenGraph(GraphTables(((0,), (),(),()))) + FrozenGraph(GraphTables(((0,), (0, 0, 0),(0,0,0),(0,1,2))))
    #g3.dot_display(globals=('rankdir=LR;',))

from onyx.graph.graphtools import *

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
