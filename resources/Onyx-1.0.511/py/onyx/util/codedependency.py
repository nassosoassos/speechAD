###########################################################################
#
# File:         codedependency.py
# Date:         12-Nov-2008
# Author:       Hugh Secker-Walker
# Description:  Generate a code dependency (import) diagram
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
Generate a code dependency (import) diagram

"""

from onyx.graph.graphtools import make_initialized_set_graph_builder, FrozenGraph
from onyx.builtin import dict_of

# nodes: name, rank, core
AcousticModels = 'Acoustic Models', 3, 0
Audio = 'Audio', 3, 0
Cfg = 'CFG', 2, 1
Dataflow = 'Dataflow', 2, 1
Decoding = 'Decoding', 4, 0
GraphLattice = 'Graph  /  Lattice', 1, 1
HtkFiles = 'HTK Files', 4, 1
Lexicon = 'Lexicon', 3, 0
Serialization = 'Serialization', 1, 1
SignalProcessing = 'Signal Processing', 3, 0
UtilitiesContainers = '          Utilities  /  Containers          ', 0, 1

dependencies = (
    (AcousticModels, GraphLattice),
    (AcousticModels, Serialization),
    (Audio, Dataflow),
    (Audio, UtilitiesContainers),
    (Cfg, GraphLattice),
    (Cfg, UtilitiesContainers),
    (Dataflow, GraphLattice),
    (Dataflow, UtilitiesContainers),
    (Decoding, Cfg),
    (Decoding, Dataflow),
    (Decoding, Lexicon),
    (GraphLattice, Serialization),
    (GraphLattice, UtilitiesContainers),
    (HtkFiles, AcousticModels),
    (HtkFiles, Dataflow),
    (HtkFiles, Lexicon),
    (Lexicon, Cfg),
    (Serialization, UtilitiesContainers),
    (SignalProcessing, Dataflow),
    )

def go(do_display=False):
    """
    Generate a dependency graph, and display it if optional do_display is True.

    >>> go(do_display=False)
    digraph  { 
      node [shape=box];
      ranksep=0.4;
      {rank=same; "n05";}
      {rank=same; "n01"; "n02";}
      {rank=same; "n04"; "n06";}
      {rank=same; "n00"; "n03"; "n08"; "n10";}
      {rank=same; "n07"; "n09";}
      n00  [label="Acoustic Models", style=bold, shape=box];
      n01  [label="Graph  /  Lattice", style=bold, shape=octagon];
      n02  [label="Serialization", style=bold, shape=octagon];
      n03  [label="Audio", style=bold, shape=box];
      n04  [label="Dataflow", style=bold, shape=octagon];
      n05  [label="          Utilities  /  Containers          ", style=bold, shape=octagon];
      n06  [label="CFG", style=bold, shape=octagon];
      n07  [label="Decoding", style=bold, shape=box];
      n08  [label="Lexicon", style=bold, shape=box];
      n09  [label="HTK Files", style=bold, shape=octagon];
      n10  [label="Signal Processing", style=bold, shape=box];
      n00 -> n01;
      n00 -> n02;
      n03 -> n04;
      n03 -> n05;
      n06 -> n01;
      n06 -> n05;
      n04 -> n01;
      n04 -> n05;
      n07 -> n06;
      n07 -> n04;
      n07 -> n08;
      n01 -> n02;
      n01 -> n05;
      n09 -> n00;
      n09 -> n04;
      n09 -> n08;
      n08 -> n06;
      n02 -> n05;
      n10 -> n04;
    }
    """

    g = FrozenGraph(make_initialized_set_graph_builder(dependencies))

    # make the rank sub graphs
    ranks = dict_of(set)
    for id in xrange(g.num_nodes):
        name, rank, color = g.get_node_label(id)
        ranks[rank].add('n%02d' %(id,))
    rankglobals = list()
    for rank, names in sorted(ranks.iteritems()):
        rankglobals.append('{rank=same; "' + '"; "'.join(sorted(names)) + '";}')

    # log it
    globals=['node [shape=box];', 'ranksep=0.4;'] + rankglobals
    node_label_callback=lambda x, *_: str(x[0])    
    #node_attributes_callback=lambda x, *_: ['color=%s' % (x[2],)]
    node_attributes_callback=lambda x, *_: ['style=bold', 'shape=octagon'] if x[2] else ['style=bold', 'shape=box']
    for line in g.dot_iter(globals=globals, node_label_callback=node_label_callback, node_attributes_callback=node_attributes_callback):
        print line,

    # display it
    do_display and g.dot_display(globals=globals, node_label_callback=node_label_callback, node_attributes_callback=node_attributes_callback)


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
