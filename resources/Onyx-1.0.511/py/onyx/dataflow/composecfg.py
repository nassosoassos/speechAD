###########################################################################
#
# File:         composecfg.py
# Date:         10-Nov-2008
# Author:       Hugh Secker-Walker
# Description:  Tools for composing CFGs
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
Proof of concept of composing CFGs.
"""

from collections import deque
from cStringIO import StringIO
from onyx.graph.graphtools import make_initialized_set_graph_builder, FrozenGraph
from onyx.dataflow.simplecfg import CfgBuilder, CfgRecognizer

# top of the comlex lexicon
comlextop = """
<s>     sil
</s>    sil
<sil>   sil
<unk>   unk
<nonspch>       nonspch
<laugh> laugh
a       ei
a's     ei Z
a.      ei
a.'s    ei Z
a.s     ei Z
aachen  a k I n
aalseth a l S E T
aalseth ae l S E T
aames   ei m Z
aancor  ae n k o9wD r
aaron   ei r I n
aarons  ei r I n Z
ababa   ae b @ b @
abaci   ae b @ k aI
aback   @ b ae k

  abaco    @ b ae k oU
abaco   ae b @ k oU
abacus  ae b @ k @ S
# foo
abacuses        ae b @ k @ S I Z
abalkin ae b @ l k I n
abalon  ae b @ l a n
abalone ae b @ l oU n iwD
abandon @ b ae n d I n
abandoned       @ b ae n d I n d
abandoning      @ b ae n d I n I ng
abandonment     @ b ae n d I n m I n t
abandonments    @ b ae n d I n m I n t S
abandons        @ b ae n d I n Z
"""

# fake phones
comlexphones = """
@ h01 h02
@  h31
@  h31 h31a h31b
@  h31c h31c
E  h13
I  h10
S  h14
T  h15
Z  h11
aI h16
ae h08
b h03 b_
b h03 b_ h05
b_ h06 h07
d  h09
e h32
ei h17
iwD h18
k h19
l h20
laugh h21
m h22
n  h12
ng h23
nonspch h24
o9wD h25
oU h26
r h27
sil h28
t h29
unk h30
"""

def make_recognizer(tdstream, comment_prefix='#'):
    builder = CfgBuilder()
    for line in tdstream:
        parts = line.split()
        if not parts or parts[0].startswith(comment_prefix):
            continue
        builder.add_production(parts[0], parts[1:])
    return CfgRecognizer(builder)

def explorer(recognizer):
    # generator only stops on a finite grammar!
    is_sentential, end_id, legal, exception = recognizer(None)
    assert exception is None
    yield end_id, is_sentential
    breadth_first = deque((end_id, is_sentential, symbol) for symbol in reversed(sorted(legal)))
    while breadth_first:
        start_id, was_sentential, symbol = breadth_first.pop()
        is_sentential, end_id, legal, exception = recognizer((start_id, symbol))
        if exception is not None: raise exception
        yield (start_id, was_sentential), (end_id, is_sentential), symbol
        breadth_first.extendleft((end_id, is_sentential, symbol) for symbol in sorted(legal))

def explore_finite(recognizer):
    #assert recognizer.is_finite
    cfg_iter = explorer(recognizer)
    global_start = cfg_iter.next()
    links = set(cfg_iter)
    return links, global_start

def _finite():
    """
    >>> cfg_string = '''
    ... S  A1
    ... S  B1
    ... A1 A11
    ... A1 A12
    ... A11 A111
    ... A11 A112
    ... A12 A121
    ... A12 A122
    ... B1 B11
    ... B1 B12
    ... B11 B111
    ... B11 B112
    ... B12 B121
    ... B12 B122
    ... '''

    >>> cfg_string = '''
    ... S  A1 B
    ... S  A1 C
    ... B  D1 D B1
    ... B  B1 E
    ... C  D1 D C1
    ... C  C1 F
    ... '''

    >>> cfg_string = '''
    ... S  A1 B
    ... S  A1 C
    ... B  B1 D D1
    ... B  B1 E
    ... C  C1 D D1
    ... C  C1 F
    ... '''

    >>> b = CfgBuilder()
    >>> for line in StringIO(cfg_string):
    ...   parts = line.split()
    ...   if parts: b.add_production(parts[0], parts[1:])
    >>> rec1 = CfgRecognizer(b).recognizer(['S'])
    >>> links, (start_id, is_sentential) = explore_finite(rec1)
    >>> len(links)
    9
    >>> None and display(FrozenGraph(make_initialized_set_graph_builder(links)))
    """

def display(g, label=None):
    g.dot_display(node_label_callback=None,
                  node_attributes_callback=lambda (n, s), *_: ['shape=doublecircle' if s else 'shape=circle', 'style=bold' if s else 'style=normal', 'height=0.2', 'width=0.2'],
                  #arc_attributes_callback=lambda a: ['sametail="tailtag"'],
                  globals=['rankdir=LR', 'ranksep=0.25;', 'fontsize=20;', 'labelloc=top;', 'label="%s";' % (label,) if label is not None else '' ])
    
def printgraph(g, label):
    print 'printgraph:', label
    for line in g.text_iter():
        print line,

def go(wordnames, do_display=False):
    """
    A first example of composing finite CFGs.
    """

    comlexcfg = make_recognizer(StringIO(comlextop))
    #for non_terminal in sorted(comlexcfg.non_terminals):
    #    print non_terminal
    #for terminal in sorted(comlexcfg.terminals):
    #    print terminal
    phonecfg = make_recognizer(StringIO(comlexphones))

    wordrecognizer = comlexcfg.recognizer(wordnames)
    links, (start_id, is_sentential) = explore_finite(comlexcfg.recognizer(wordnames))
    g1 = FrozenGraph(make_initialized_set_graph_builder(links))

    printgraph(g1, 'Pronunciation lattice')
    do_display and display(g1, '\\n'.join(wordnames))

    breadth_first = deque()
    global_start = start_id = unstarted = object()
    seen_symbols = set()
    links = set()
    links2 = set()
    send_arg = None
    count = -1
    while True:
        is_sentential, end_id, legal, exception = wordrecognizer(send_arg)
        if global_start is unstarted:
            global_start = end_id
        if exception is not None: raise exception
        if not legal: assert is_sentential

        #print 'legal:', ' '.join(legal)

        if start_id is not unstarted:
            links.add(((start_id, was_sentential), (end_id, is_sentential), symbol))

            count += 1
            count = 0
            substart2 = start_id, substart
            links2.add(((start_id, was_sentential), (substart2, False), '(-'))
            for (sub_start_id, sub_was_sentential), (sub_end_id, sub_is_sentential), subsymbol in sublinks:
                sub_start_id = start_id, sub_start_id
                sub_end_id = start_id, sub_end_id
##                 if sub_start_id == substart:
##                     links2.add(((start_id, was_sentential), (sub_start_id, False), '(-'))
                links2.add(((sub_start_id, False),
                            #((end_id, is_sentential) if sub_is_sentential else (sub_end_id, False)),
                            (sub_end_id, False),
                            subsymbol))
                if sub_is_sentential:
                    links2.add(((sub_end_id, False), (end_id, is_sentential), '(-'))

        for symbol in sorted(legal):
            breadth_first.appendleft(((end_id, is_sentential), symbol, explore_finite(phonecfg.recognizer([symbol]))))
            sublinks, (sub_start_id, sub_is_sentential) = explore_finite(phonecfg.recognizer([symbol]))
            if symbol not in seen_symbols:
                seen_symbols.add(symbol)
                subgraph = FrozenGraph(make_initialized_set_graph_builder(sublinks))
                printgraph(subgraph, 'Phoneme %s' % (symbol,))
                do_display and display(subgraph, symbol)
        if not breadth_first:
            break

        (start_id, was_sentential), symbol, (sublinks, (substart, substart_sentential)) = breadth_first.pop()
        send_arg = start_id, symbol
        
    g = FrozenGraph(make_initialized_set_graph_builder(links))
    None and do_display and display(g)

    g3 = FrozenGraph(make_initialized_set_graph_builder(links2))
    printgraph(g3, 'HMM graph')
    do_display and display(g3)
    #print legal
    
def _test():
    """
    >>> go(['abaco', 'ababa', 'abaci', 'abandon', 'aback'], do_display=False)
    printgraph: Pronunciation lattice
    num_nodes 14
    0   (2, False)
    1   (4, False)
    2   (6, False)
    3   (10, False)
    4   (1, False)
    5   (3, False)
    6   (12, False)
    7   (13, False)
    8   (5, False)
    9   (8, False)
    10  (7, True)
    11  (9, False)
    12  (11, True)
    13  (0, False)
    num_arcs 17
    0   0  1   b
    1   2  3   k
    2   4  5   b
    3   6  7   I
    4   5  8   ae
    5   9  6   d
    6   8  9   n
    7   8  10  k
    8   11 12  @
    9   13 4   @
    10  10 12  oU
    11  3  12  oU
    12  3  12  aI
    13  7  12  n
    14  2  11  b
    15  1  2   @
    16  13 0   ae
    printgraph: Phoneme @
    num_nodes 6
    0   (0, False)
    1   (1, False)
    2   (3, False)
    3   (4, True)
    4   (2, True)
    5   (5, False)
    num_arcs 7
    0   0  1   h01
    1   2  3   h31c
    2   0  2   h31c
    3   4  5   h31a
    4   1  3   h02
    5   5  3   h31b
    6   0  4   h31
    printgraph: Phoneme ae
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h08
    printgraph: Phoneme b
    num_nodes 5
    0   (0, False)
    1   (1, False)
    2   (3, True)
    3   (4, True)
    4   (2, False)
    num_arcs 4
    0   0  1   h03
    1   2  3   h05
    2   4  2   h07
    3   1  4   h06
    printgraph: Phoneme k
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h19
    printgraph: Phoneme n
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h12
    printgraph: Phoneme oU
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h26
    printgraph: Phoneme d
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h09
    printgraph: Phoneme aI
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h16
    printgraph: Phoneme I
    num_nodes 2
    0   (0, False)
    1   (1, True)
    num_arcs 1
    0   0  1   h10
    printgraph: HMM graph
    num_nodes 61
    0   ((1, 3), False)
    1   (3, False)
    2   ((6, 1), False)
    3   ((6, 2), False)
    4   ((0, 0), False)
    5   ((0, 2), False)
    6   ((4, 2), False)
    7   (6, False)
    8   ((9, 0), False)
    9   ((9, 2), False)
    10  ((1, 4), False)
    11  ((13, 0), False)
    12  ((13, 1), False)
    13  (12, False)
    14  ((12, 0), False)
    15  ((9, 5), False)
    16  ((9, 4), False)
    17  (2, False)
    18  ((2, 0), False)
    19  ((6, 3), False)
    20  ((6, 4), False)
    21  (11, True)
    22  ((0, 3), False)
    23  ((0, 4), False)
    24  ((9, 1), False)
    25  ((0, 1), False)
    26  (7, True)
    27  ((7, 0), False)
    28  (9, False)
    29  ((6, 0), False)
    30  (10, False)
    31  (5, False)
    32  ((5, 0), False)
    33  (0, False)
    34  ((5, 1), False)
    35  ((3, 0), False)
    36  ((3, 1), False)
    37  ((0, 5), False)
    38  ((4, 0), False)
    39  ((4, 1), False)
    40  ((4, 4), False)
    41  ((10, 0), False)
    42  ((10, 1), False)
    43  ((8, 1), False)
    44  ((4, 3), False)
    45  (1, False)
    46  ((9, 3), False)
    47  ((2, 2), False)
    48  ((2, 3), False)
    49  ((1, 2), False)
    50  ((4, 5), False)
    51  (8, False)
    52  ((2, 1), False)
    53  ((7, 1), False)
    54  ((8, 0), False)
    55  (4, False)
    56  ((12, 1), False)
    57  (13, False)
    58  ((1, 0), False)
    59  ((1, 1), False)
    60  ((2, 4), False)
    num_arcs 79
    0   0  1   (-
    1   2  3   h06
    2   4  5   h31
    3   6  7   (-
    4   8  9   h31
    5   0  10  h05
    6   11 12  h12
    7   13 14  (-
    8   15 16  h31b
    9   10 1   (-
    10  17 18  (-
    11  3  19  h07
    12  19 20  h05
    13  16 21  (-
    14  22 23  h31c
    15  24 16  h02
    16  4  25  h08
    17  26 27  (-
    18  9  21  (-
    19  20 28  (-
    20  7  29  (-
    21  2  30  (-
    22  31 32  (-
    23  33 4   (-
    24  8  24  h01
    25  32 34  h19
    26  35 36  h08
    27  12 21  (-
    28  5  37  h31a
    29  38 39  h01
    30  39 40  h02
    31  40 7   (-
    32  41 42  h26
    33  43 13  (-
    34  1  35  (-
    35  44 40  h31c
    36  5  45  (-
    37  34 26  (-
    38  38 6   h31
    39  46 16  h31c
    40  23 45  (-
    41  47 48  h07
    42  4  25  h01
    43  4  22  h31c
    44  37 23  h31b
    45  8  46  h31c
    46  49 0   h07
    47  29 2   h19
    48  29 2   h03
    49  30 41  (-
    50  50 40  h31b
    51  34 51  (-
    52  42 21  (-
    53  9  15  h31a
    54  36 31  (-
    55  52 47  h06
    56  25 23  h02
    57  27 53  h26
    58  28 8   (-
    59  51 54  (-
    60  48 55  (-
    61  6  50  h31a
    62  18 52  h03
    63  56 57  (-
    64  58 59  h03
    65  60 55  (-
    66  14 56  h10
    67  45 58  (-
    68  38 44  h31c
    69  19 28  (-
    70  41 42  h16
    71  57 11  (-
    72  53 21  (-
    73  25 17  (-
    74  55 38  (-
    75  32 34  h12
    76  59 49  h06
    77  48 60  h05
    78  54 43  h09
    """

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
