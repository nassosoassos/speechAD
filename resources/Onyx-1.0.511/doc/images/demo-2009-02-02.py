###########################################################################
#
# File:         demo-2009-02-02.py
# Date:         20-Dec-2008
# Author:       Hugh Secker-Walker
# Description:  Code for generating DOT files for pictures for presentations
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
    >>> True
    True
"""

# note: see range of graphs that actually get created (near the bottom)

import cStringIO as StringIO
from onyx.graph.graphtools import SetGraphBuilder, FrozenGraph

def build_it(linesstring):
    builder = SetGraphBuilder()
    boxen = set()
    for arc in StringIO.StringIO(linesstring):
        parts = arc.split()
        if not parts or parts[0].startswith('#'):
            continue
        if len(parts) == 1:
            builder.add_node(*parts)
        elif len(parts) == 2:
            builder.add_arc(*parts)
        else:
            assert False, str(parts)
    return FrozenGraph(builder)

graphs = list()

mics = """
WaveFile
Mic3 
Mic2 invisible
Mic1
"""
graphs.append(mics)


fftabs = """
ChanSelect Dither
Dither PreEmph
PreEmph Window
Window RemoveDC
RemoveDC Hamming
Hamming FFT
FFT Mag
"""
graphs.append(fftabs)

vuutt = """
dB EndPoint
dB VU-meter
EndPoint VU-meter
"""
graphs.append(vuutt)

melcep = """
MelFilters Log
Log DCT
DCT Lifter
"""
graphs.append(melcep)


train = """
FFT-ABS
FFT-ABS dB
dB EndPoint
dB VU-meter
EndPoint VU-meter
FFT-ABS MelCep1
MelCep1 GaussTrain
EndPoint GaussTrain
GaussTrain logger
"""
graphs.append(train)


ddt = """
FFT-ABS dB
dB EndPoint
dB VU-meter
EndPoint VU-meter
FFT-ABS MelCep1
EndPoint GMM-Classifier1
MelCep1 GMM-Classifier1
FFT-ABS x^2
x^2 MelCep2
EndPoint GMM-Classifier2
MelCep2 GMM-Classifier2
GMM-Classifier1 DDT
GMM-Classifier2 DDT
DDT logger
DDT DDT-display
"""
graphs.append(ddt)


vu = """
FFT-ABS
FFT-ABS dB
dB EndPoint
dB VU-meter
EndPoint VU-meter
"""
graphs.append(vu)

vu2 = """
WaveBuffer\\niterator source1
source1 FFT-ABS
FFT-ABS dB
dB split2
split2 EndPoint
split2 join1
EndPoint join1
join1 VU-meter
"""
graphs.append(vu2)

# have to edit the resulting dot file and add something like this at global
# scope to get the output boxes on the same rank
#   {rank=same; "n08"; "n21"; "n20";}
ddt2 = """
WaveBuffer\\niterator source1
source1 FFT-ABS
FFT-ABS split1
split1 dB
dB split2
split2 EndPoint
split2 join1
join1 VU-meter
EndPoint split3
split3 join1
split3 join2
split1 MelCep1
MelCep1 join2
join2 GMM-Classifier1
split1 x^2
x^2 MelCep2
split3 join3
MelCep2 join3
join3 GMM-Classifier2
GMM-Classifier1 join4
GMM-Classifier2 join4
join4 DDT
DDT split4
split4 logger
split4 DDT-display
"""
graphs.append(ddt2)


# note: legend isn't added to graphs because it crashed Graphviz and eventually
# required a reboot....
# needs something like this to make things line up
#   {rank=same; "n08"; "n21"; "n20";}
legend = """
join2 invisible11
invisible5 join2
invisible6 join2

invisible10 split2
split2 invisible3
split2 invisible4

invisible1 source1
source1 invisible2

invisible7 VU-meter 

invisible8 Processor
Processor invisible9

WaveBuffer\\niterator invisible
"""
#graphs.append(legend)

invisibles = frozenset((
    'invisible',
    'invisible1',
    'invisible2',
    'invisible3',
    'invisible4',
    'invisible5',
    'invisible6',
    'invisible7',
    'invisible8',
    'invisible9',
    'invisible10',
    'invisible11',
    ))
boxes = frozenset((
    'WaveBuffer\\niterator',
    'VU-meter',
    'logger',
    'DDT-display',
    ))
hexagons = frozenset((
    'split1',
    'split2',
    'split3',
    'split4',
    'join1',
    'join2',
    'join3',
    'join4',
    ))
octagons = frozenset((
    'source1',
    ))
def node_attributes_callback(label, is_start, is_end):
    if label in boxes:
        yield 'shape=box'
    elif label in hexagons:
        yield 'shape=hexagon'
        yield 'height=0.125'
        yield 'width=0.25'
        yield 'label=""'
        yield 'style=filled'
    elif label in octagons:
        yield 'shape=octagon'
        #yield 'label="OnDemand"'
        yield 'label=""'
        yield 'fontsize=10'
        yield 'height=0.375'
        yield 'width=0.5'
        yield 'style=filled'
    else:
        yield 'shape=ellipse'

    if label in invisibles:
        yield 'style=invis'

for graph in graphs[:]:
    build_it(graph).dot_display(
        node_attributes_callback=node_attributes_callback,
        globals=['rankdir=LR;',
                 'ranksep=0.2',
                 'edge [sametail=tail, samehead=head, arrowsize=0.75];',
                 ])
    import time
    time.sleep(0.5)

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
