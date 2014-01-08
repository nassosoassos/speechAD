###########################################################################
#
# File:         cfg.py (directory: ./py/onyx/dataflow)
# Date:         3-Apr-2008
# Author:       Hugh Secker-Walker
# Description:  Context Free Grammar support
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008, 2009 The Johns Hopkins University
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
    Work with CFG grammars.
"""

from __future__ import division
import random, cStringIO
from collections import defaultdict, deque
from functools import partial
from onyx.containers import sorteduniquetuple, frozenbijection

class GrammarTables(object):
    """
    Minimal transport and checking for a grammar.  Four components of
    a grammar, each as a sorted tuple of a set of immutable items.

    Constructor takes three or four iterable arguments: the set of
    terminals, the set of non_terminals, the set of productions, and
    the set of start states.  The terminal and non_terminal sets must
    be disjoint.  Each production is a two-item sequence (lhs, rhs)
    where lhs is a non_terminal and rhs is a sequence of terminals and
    non_terminals.  The start_states, if any, must be non_terminals.
    All items must be immutable (hashable) objects.

    >>> g = GrammarTables(('a', 'b'), ('AB',), (('AB', ('a', 'b')), ('AB', ('a', 'AB', 'b'))))
    >>> tuple(len(x) for x in g)
    (2, 1, 2, 0)
    >>> g = GrammarTables(('a', 'b'), ('AB',), (('AB', ('a', 'b')), ('AB', ('a', 'AB', 'b'))), ('AB',))
    >>> tuple(len(x) for x in g)
    (2, 1, 2, 1)
    """
    __slots__ = ('data',)
    def __init__(self, terminals, non_terminals, productions, starts=()):
        terminals = sorteduniquetuple(terminals)
        non_terminals = sorteduniquetuple(non_terminals)
        productions = sorteduniquetuple((lhs, tuple(rhs)) for lhs, rhs in productions)
        starts = sorteduniquetuple(starts)

        self.data = terminals, non_terminals, productions, starts

        self._verify()

    def _verify(self):
        terminals, non_terminals, productions, starts = tuple(set(x) for x in self.data)
        intersection = terminals & non_terminals
        assert not intersection, repr(intersection)
        union = terminals | non_terminals
        seen_non_terminals = set()
        for lhs, rhs in productions:
            assert lhs in non_terminals, repr(lhs) + " isn't in non_terminals"
            seen_non_terminals.add(lhs)
            for item in rhs:
                assert item in union, repr(item) + " isn't in union of terminals and non_terminals"
        unseen_non_terminals = non_terminals - seen_non_terminals
        assert not unseen_non_terminals, "unused non_terminals: %s" % (' '.join(repr(non_terminal) for non_terminal in unseen_non_terminals),)
        for start in starts:
            assert start in non_terminals, repr(start) + " isn't in non_terminals"


    def __iter__(self):
        return iter(self.data)


class ImplicitCfgBuilder(object):
    """
    A builder for CFG grammars.  This supports the simple model of
    specifying a CFG by building it up one production at a time.

    In this simple model terminals and epsilons are implicit, and no
    starting non-terminals are specified.  A terminal is a
    right-hand-side (rhs) items which does not appear on the
    left-hand-side (lhs) of any production.  An epsilon is a
    production with an empty rhs.

    XXX consider implementing that the first production to be added be
    the start state... this would imply some sort of ordering
    semantics....
    
    >>> builder = ImplicitCfgBuilder()

    Names for terminals

    >>> builder.add_production('zero', [0])
    >>> builder.add_production('one', [1])

    Give a name to an epsilon production

    >>> builder.add_production('epsilon')

    Components of binary numbers

    >>> builder.add_production('binary_digit', ['zero'])
    >>> builder.add_production('binary_digit', ['one'])

    Binary number

    >>> builder.add_production('binary_number', ['binary_digit'])
    >>> builder.add_production('binary_number', ['binary_number','binary_digit'])
    
    Possibly empty sequence of binary numbers

    >>> builder.add_production('binary_numbers_list', ['epsilon'])
    >>> builder.add_production('binary_numbers_list', ['binary_numbers_list', 'binary_number'])

    Look inside

    >>> tuple(builder)
    (('binary_digit', ('one',)), ('binary_digit', ('zero',)), ('binary_number', ('binary_digit',)), ('binary_number', ('binary_number', 'binary_digit')), ('binary_numbers_list', ('binary_numbers_list', 'binary_number')), ('binary_numbers_list', ('epsilon',)), ('epsilon', ()), ('one', (1,)), ('zero', (0,)))

    Constructor accepts the iteration to create a copy of the builder

    >>> builder2 = ImplicitCfgBuilder(builder)

    This duplicate rhs won't change the object)

    >>> builder2.add_production('binary_numbers_list', ['binary_numbers_list', 'binary_number'])

    This new rhs will

    >>> builder2.add_production('binary_number_pair', ['binary_number', 'binary_number'])

    >>> tuple(builder2)
    (('binary_digit', ('one',)), ('binary_digit', ('zero',)), ('binary_number', ('binary_digit',)), ('binary_number', ('binary_number', 'binary_digit')), ('binary_number_pair', ('binary_number', 'binary_number')), ('binary_numbers_list', ('binary_numbers_list', 'binary_number')), ('binary_numbers_list', ('epsilon',)), ('epsilon', ()), ('one', (1,)), ('zero', (0,)))

    The grammar_sets() method returns the transportable, frozen view of the grammar

    >>> tuple(builder2.grammar_sets())
    ((0, 1), ('binary_digit', 'binary_number', 'binary_number_pair', 'binary_numbers_list', 'epsilon', 'one', 'zero'), (('binary_digit', ('one',)), ('binary_digit', ('zero',)), ('binary_number', ('binary_digit',)), ('binary_number', ('binary_number', 'binary_digit')), ('binary_number_pair', ('binary_number', 'binary_number')), ('binary_numbers_list', ('binary_numbers_list', 'binary_number')), ('binary_numbers_list', ('epsilon',)), ('epsilon', ()), ('one', (1,)), ('zero', (0,))), ())

    Here's the TypeError if the rhs sequence has a mutable item in it

    >>> builder2.add_production('binary_number_triple', ['binary_number', 'binary_number', range(3)]) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    TypeError: ...unhashable...
    """
    def __init__(self, arg=()):
        self.lhs_sets = defaultdict(set)
        for lhs, rhs in arg:
            self.add_production(lhs, rhs)

    def add_production(self, lhs, rhs=()):
        """
        Add a production to the CFG, where lhs is the non-terminal for the
        production and rhs, if given, is an iterable of non-terminals and
        terminals making up the production.  If rhs is not given or is an empty
        iterable then the production is an epsilon production.

        All non-terminals and terminals must be immutable objects.
        """
        self.lhs_sets[lhs].add(tuple(rhs))

    def __iter__(self):
        for key in sorted(self.lhs_sets.keys()):
            for rhs in sorted(self.lhs_sets[key]):
                # yield (lhs, rhs)
                yield key, rhs

    def grammar_sets(self):
        """
        Return a tuple of the three sets comprising the grammar: (terminals,
        non-terminals, productions) where each production is a two-item tuple
        consisting of the non-terminal lhs and the sequence of non-terminals and
        terminals that is the rhs.

        The return values are suitable for use as the constructor
        arguments for the GrammarTables object.
        """
        non_terminals = set(self.lhs_sets.keys())
        productions = set()
        terminals = set()
        for lhs, rhs_set in self.lhs_sets.iteritems():
            for rhs in rhs_set:
                productions.add((lhs, rhs))
                terminals |= set(rhs) - non_terminals

        return GrammarTables(terminals, non_terminals, productions)


class ExplicitCfgBuilder(object):
    # XXX needs work to be complete: productions, and adopt a model for initialization from iteration(s)

    """
    A builder for CFG grammars.  This supports an explicit model of specifying a
    CFG by naming each terminal and non-terminal before it is used in any
    productions, and by, optionally, specifying the starting non-terminal(s).

    >>> builder = ExplicitCfgBuilder()
    >>> builder.has_terminal(None)
    False
    >>> builder.has_non_terminal(None)
    False

    Put in some terminal items with checking that they're not already
    there

    >>> builder.add_new_terminal(-1)
    >>> builder.add_new_terminal(-2)

    Can't use add_new_terminal for an existing item

    >>> builder.add_new_terminal(-2)
    Traceback (most recent call last):
      ...
    ValueError: item is already a terminal: -2

    But you can use the less careful add_terminal

    >>> builder.add_terminal(-2)

    Similarly, put in some non_terminal items with checking that
    they're not already there

    >>> builder.add_new_non_terminal(1)
    >>> builder.add_new_non_terminal(2)

    Can't use add_new_non_terminal for an existing item

    >>> builder.add_new_non_terminal(1)
    Traceback (most recent call last):
      ...
    ValueError: item is already a non-terminal: 1

    But you can use the less careful add_non_terminal

    >>> builder.add_non_terminal(2)


    Careful or not, the methods keep the terminal and non-terminal
    sets disjoint

    >>> builder.add_terminal(2)
    Traceback (most recent call last):
      ...
    ValueError: item is already a non-terminal: 2
    >>> builder.add_non_terminal(-2)
    Traceback (most recent call last):
      ...
    ValueError: item is already a terminal: -2
    
    """

    def __init__(self):
        self.terminals = set()
        self.non_terminals = set()
        self.starts = set()
        self.lhs_sets = defaultdict(set)

    def _check_not_terminal(self, item):
        if self.has_terminal(item):
            raise ValueError("item is already a terminal: %r" % (item,))
    def _check_not_non_terminal(self, item):
        if self.has_non_terminal(item):
            raise ValueError("item is already a non-terminal: %r" % (item,))
        
    def has_terminal(self, terminal):
        return terminal in self.terminals
    def add_new_terminal(self, terminal):
        """
        Introduce a terminal to the CFG.  A ValueError is raised if the terminal
        token is already in the CFG as either a terminal or a non-terminal.
        """
        self._check_not_terminal(terminal)
        self.add_terminal(terminal)
    def add_terminal(self, terminal):
        """
        Ensure that the terminal is in the CFG.  A ValueError is raised if the
        terminal is already in the CFG as a non-terminal.
        """
        self._check_not_non_terminal(terminal)
        self.terminals.add(terminal)
        
    def has_non_terminal(self, non_terminal):
        return non_terminal in self.non_terminals
    def add_new_non_terminal(self, non_terminal):
        """
        Introduce a non_terminal to the CFG.  A ValueError is raised if the
        non_terminal token is already in the CFG as either a terminal or a
        non-terminal.
        """
        self._check_not_non_terminal(non_terminal)
        self.add_non_terminal(non_terminal)
    def add_non_terminal(self, non_terminal):
        """
        Ensure that the non_terminal is in the CFG.  A ValueError is raised if
        the non_terminal is already in the CFG as a terminal.
        """
        self._check_not_terminal(non_terminal)
        self.non_terminals.add(non_terminal)
        
        

# section delimited, line-based tokenization helper
def until(stream, value, flatten=False):
    stream = iter(stream)
    result = list()
    update = result.extend if flatten else result.append
    while True:
        try:
            parts = stream.next().split()
        except StopIteration:
            return result
        # empty or comment
        if not parts or parts[0].startswith('#'):
            continue
        if len(parts) == 1 and parts[0] == value:
            return result
        update(parts)

def parse_grammar(stream):
    """
    Simple line-based parser for grammars, returns a GrammarTables instance.

    >>> g = parse_grammar(cStringIO.StringIO(_binary_cfg))

    Counts of terminals, non-terminals, productions,starts

    >>> tuple(len(x) for x in g)
    (2, 2, 4, 1)

    >>> g = parse_grammar(cStringIO.StringIO(_hello_world_cfg))
    >>> tuple(len(x) for x in g)
    (41, 19, 64, 1)
    """

    parse_until = partial(until, stream)

    empty_prolog = parse_until('__terminals__')
    assert not empty_prolog, repr(empty_prolog)

    terminals = frozenset(parse_until('__productions__', flatten=True))

    productions = list()
    for parts in parse_until('__starts__'):
        productions.append((parts[0], tuple(parts[1:])))
    productions = frozenset(productions)
    non_terminals = frozenset(lhs for lhs, rhs in productions)

    starts = frozenset(parse_until(None, flatten=True))
        
    return GrammarTables(terminals, non_terminals, productions, starts)

class Grammar(object):
    """
    A CFG grammar object, initialized from a GrammarTables instance.  Can be
    used to generate random sentences from the language.

    A simple grammar

    >>> g0 = Grammar(parse_grammar(cStringIO.StringIO(_tiny_cfg)), debug=False)
    >>> for line in g0.str_iter: print line
    __terminals__
    A
    B
    __productions__
    abab A B A B
    __starts__
    abab
    >>> result = ' '.join(item for item in g0.gen(-1, debug=False))
    >>> print result
    A B A B

    A binary number grammar

    >>> g1 = Grammar(parse_grammar(cStringIO.StringIO(_binary_cfg)))

    Canonical text-based representation

    >>> for line in g1.str_iter: print line
    __terminals__
    0
    1
    __productions__
    binary_digit 1
    binary_digit 0
    binary_number binary_digit
    binary_number binary_number binary_digit
    __starts__
    binary_number

    A BNF-style representation

    >>> for line in g1.bnf_iter: print line
    binary_digit ::= "1"
                   | "0"
    binary_number ::= binary_digit
                    | binary_number binary_digit

    Run the simple guy until five sentential forms have been seen

    >>> result = ' '.join(item for item in g1.gen(5, seed=1))
    >>> print result
    1 0 0 1 1

    Run a more complicate grammar until 50 sentential forms have been seen

    >>> g2 = Grammar(parse_grammar(cStringIO.StringIO(_hello_world_cfg)))
    >>> result = ' '.join(item for item in g2.gen(25, seed=81))
    >>> print result
    H J F M 4 U J 7 D 1 P N P V Y G P E + ( 7 5 5 * R 4 R J E * 8 + N N 2 D 8 X Y J O Q 4 0 S W T T 8 L ) + G V 3 M R E

    Ambigous grammar

    >>> g3 = Grammar(parse_grammar(cStringIO.StringIO(_ambiguous_cfg)))
    >>> #g3 = Grammar(parse_grammar(cStringIO.StringIO(_ambiguous_cfg)), debug=True)
    >>> # for line in g3.bnf_iter: print line
    >>> result = ' '.join(item for item in g3.gen(2, seed=4))
    >>> #result = ' '.join(item for item in g3.gen(2, seed=4, debug=True))
    >>> print result
    IF FOO THEN IF FOO THEN EXPR ELSE EXPR

    Copy construction of the simple guy

    >>> Grammar(g1).grammar_sets == g1.grammar_sets
    True

    Augment the productions so as to have a shared lhs and a shared rhs, also
    use an epsilon, and have multiple starts, and turn on debugging spew

    >>> terminals, non_terminals, productions, starts = g1.grammar_sets
    >>> g3 = Grammar(GrammarTables(terminals, non_terminals + ('zero', 'zero1', 'list_of_zero'), productions + (('zero', ('0',)), ('zero1', ('0',)), ('list_of_zero', ()), ('list_of_zero', ('list_of_zero', 'zero'))), starts + ('list_of_zero',)), debug=True)    
    productions: ((0, ('binary_digit', ('0',))), (1, ('binary_digit', ('1',))), (2, ('binary_number', ('binary_digit',))), (3, ('binary_number', ('binary_number', 'binary_digit'))), (4, ('list_of_zero', ())), (5, ('list_of_zero', ('list_of_zero', 'zero'))), (6, ('zero', ('0',))), (7, ('zero1', ('0',))))
    starts: ('binary_number', 'list_of_zero')
    terminal_by_id: ((0, '0'), (1, '1'))
    terminal by integer: ((-2, '0'), (-3, '1'))
    non_terminal_by_id: ((0, 'binary_digit'), (1, 'binary_number'), (2, 'list_of_zero'), (3, 'zero'), (4, 'zero1'))
    prod1: ((0, (-3, -1)), (0, (-2, -1)), (1, (0, -1)), (1, (1, 0, -1)), (2, (-1,)), (2, (2, 3, -1)), (3, (-2, -1)), (4, (-2, -1)))
    rhs_by_id: ((0, (-3, -1)), (1, (-2, -1)), (2, (-1,)), (3, (0, -1)), (4, (1, 0, -1)), (5, (2, 3, -1)))
    rhs uniqueness: 6 of 8 : 0.75
    prod2: ((0, 0), (0, 1), (1, 3), (1, 4), (2, 2), (2, 5), (3, 1), (4, 1))
    ruleset_by_id: ((0, (0, 1)), (1, (1,)), (2, (2, 5)), (3, (3, 4)))
    ruleset uniqueness: 4 of 5 : 0.8
    self.ruleset_id_by_lhs_id: ((0, 0), (1, 3), (2, 2), (3, 1), (4, 1))
    starts: (1, 2)
    
    Watch the internals of this new guy

    >>> result = ' '.join(item for item in g3.gen(4, seed=1, debug=True))
    start_states: ('binary_number', 'list_of_zero')
    start_ids: (1, 2)
    start_rhs_ids: (6, 7)
    rhs_by_id: ((0, (-3, -1)), (1, (-2, -1)), (2, (-1,)), (3, (0, -1)), (4, (1, 0, -1)), (5, (2, 3, -1)), (6, (1, -1)), (7, (2, -1)))
    start_ruleset_id: 4
    ruleset_by_id: ((0, (0, 1)), (1, (1,)), (2, (2, 5)), (3, (3, 4)), (4, (6, 7)))
    start_lhs_id: 5
    ruleset_id_by_lhs_id: ((0, 0), (1, 3), (2, 2), (3, 1), (4, 1), (5, 4))
    non_terminal_by_id: ((0, 'binary_digit'), (1, 'binary_number'), (2, 'list_of_zero'), (3, 'zero'), (4, 'zero1'), (5, '<start_state>'))
    <BLANKLINE>
    predicting:   <start_state>
    prediction:     (5, 6, 0, 0)  <start_state> ( . binary_number ) 0
    prediction:     (5, 7, 0, 0)  <start_state> ( . list_of_zero ) 0
    <BLANKLINE>
    k: 0
    predicting:   list_of_zero
    prediction:     (2, 2, 0, 0)  list_of_zero ( . ) 0    from: (5, 7, 0, 0)  <start_state> ( . list_of_zero ) 0
    prediction:     (2, 5, 0, 0)  list_of_zero ( . list_of_zero zero ) 0    from: (5, 7, 0, 0)  <start_state> ( . list_of_zero ) 0
    predicting:   list_of_zero
    prediction: +   (2, 2, 0, 0)  list_of_zero ( . ) 0    from: (2, 5, 0, 0)  list_of_zero ( . list_of_zero zero ) 0
    prediction: +   (2, 5, 0, 0)  list_of_zero ( . list_of_zero zero ) 0    from: (2, 5, 0, 0)  list_of_zero ( . list_of_zero zero ) 0
    predicting:   binary_number
    prediction:     (1, 3, 0, 0)  binary_number ( . binary_digit ) 0    from: (5, 6, 0, 0)  <start_state> ( . binary_number ) 0
    prediction:     (1, 4, 0, 0)  binary_number ( . binary_number binary_digit ) 0    from: (5, 6, 0, 0)  <start_state> ( . binary_number ) 0
    predicting:   binary_number
    prediction: +   (1, 3, 0, 0)  binary_number ( . binary_digit ) 0    from: (1, 4, 0, 0)  binary_number ( . binary_number binary_digit ) 0
    prediction: +   (1, 4, 0, 0)  binary_number ( . binary_number binary_digit ) 0    from: (1, 4, 0, 0)  binary_number ( . binary_number binary_digit ) 0
    predicting:   binary_digit
    prediction:     (0, 0, 0, 0)  binary_digit ( . "1" ) 0    from: (1, 3, 0, 0)  binary_number ( . binary_digit ) 0
    prediction:     (0, 1, 0, 0)  binary_digit ( . "0" ) 0    from: (1, 3, 0, 0)  binary_number ( . binary_digit ) 0
    completing:   list_of_zero  (2, 2, 0, 0)  list_of_zero ( . ) 0
    completion:     (2, 5, 1, 0)  list_of_zero ( list_of_zero . zero ) 0    from: (2, 2, 0, 0)  list_of_zero ( . ) 0
    completion:     (5, 7, 1, 0)  <start_state> ( list_of_zero . ) 0    from: (2, 2, 0, 0)  list_of_zero ( . ) 0
    completing: * <start_state>  (5, 7, 1, 0)  <start_state> ( list_of_zero . ) 0
    predicting:   zero
    prediction:     (3, 1, 0, 0)  zero ( . "0" ) 0    from: (2, 5, 1, 0)  list_of_zero ( list_of_zero . zero ) 0
    scanset: 0 1
    yielding: 1
    <BLANKLINE>
    k: 1
    scanning:    "1"
    scanned:        (0, 0, 1, 0)  binary_digit ( "1" . ) 0
    completing:   binary_digit  (0, 0, 1, 0)  binary_digit ( "1" . ) 0
    completion:     (1, 3, 1, 0)  binary_number ( binary_digit . ) 0    from: (0, 0, 1, 0)  binary_digit ( "1" . ) 0
    completing:   binary_number  (1, 3, 1, 0)  binary_number ( binary_digit . ) 0
    completion:     (5, 6, 1, 0)  <start_state> ( binary_number . ) 0    from: (1, 3, 1, 0)  binary_number ( binary_digit . ) 0
    completion:     (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0    from: (1, 3, 1, 0)  binary_number ( binary_digit . ) 0
    predicting:   binary_digit
    prediction:     (0, 0, 0, 1)  binary_digit ( . "1" ) 1    from: (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0
    prediction:     (0, 1, 0, 1)  binary_digit ( . "0" ) 1    from: (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0
    completing: * <start_state>  (5, 6, 1, 0)  <start_state> ( binary_number . ) 0
    scanset: 0 1
    yielding: 0
    <BLANKLINE>
    k: 2
    scanning:    "0"
    scanned:        (0, 1, 1, 1)  binary_digit ( "0" . ) 1
    completing:   binary_digit  (0, 1, 1, 1)  binary_digit ( "0" . ) 1
    completion:     (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0    from: (0, 1, 1, 1)  binary_digit ( "0" . ) 1
    completing:   binary_number  (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0
    completion:     (5, 6, 1, 0)  <start_state> ( binary_number . ) 0    from: (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0
    completion:     (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0    from: (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0
    completing: * <start_state>  (5, 6, 1, 0)  <start_state> ( binary_number . ) 0
    predicting:   binary_digit
    prediction:     (0, 0, 0, 2)  binary_digit ( . "1" ) 2    from: (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0
    prediction:     (0, 1, 0, 2)  binary_digit ( . "0" ) 2    from: (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0
    scanset: 0 1
    yielding: 0
    <BLANKLINE>
    k: 3
    scanning:    "0"
    scanned:        (0, 1, 1, 2)  binary_digit ( "0" . ) 2
    completing:   binary_digit  (0, 1, 1, 2)  binary_digit ( "0" . ) 2
    completion:     (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0    from: (0, 1, 1, 2)  binary_digit ( "0" . ) 2
    completing:   binary_number  (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0
    completion:     (5, 6, 1, 0)  <start_state> ( binary_number . ) 0    from: (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0
    completion:     (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0    from: (1, 4, 2, 0)  binary_number ( binary_number binary_digit . ) 0
    completing: * <start_state>  (5, 6, 1, 0)  <start_state> ( binary_number . ) 0
    predicting:   binary_digit
    prediction:     (0, 0, 0, 3)  binary_digit ( . "1" ) 3    from: (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0
    prediction:     (0, 1, 0, 3)  binary_digit ( . "0" ) 3    from: (1, 4, 1, 0)  binary_number ( binary_number . binary_digit ) 0
    <BLANKLINE>
    completed: countdown

    >>> print result
    1 0 0
    """

    def __init__(self, grammar_sets, debug=False):
        # allow copy construction
        if isinstance(grammar_sets, Grammar):
            grammar_sets = grammar_sets.grammar_sets

        assert isinstance(grammar_sets, GrammarTables), repr(type(grammar_sets).__name__)
        grammar_sets._verify()
        terminals, non_terminals, productions, starts = self._grammar_sets = grammar_sets

        if debug:
            print 'productions:', tuple(enumerate(productions))
            print 'starts:', tuple(starts)

        # bijective maps with non-negative ids
        self.terminal_by_id, self.id_by_terminal = frozenbijection(terminals)
        self.non_terminal_by_id, self.id_by_non_terminal = frozenbijection(non_terminals)

        if debug:
            print 'terminal_by_id:', tuple(enumerate(self.terminal_by_id))
            print 'terminal by integer:', tuple((self.symbol_to_integer(terminal), terminal) for terminal in self.terminal_by_id)
            print 'non_terminal_by_id:', tuple(enumerate(self.non_terminal_by_id))

        # create the index-based/integer-based left-hand sides and integer-based right-hand sides
        # note: we put completion_id at end of each rhs as a cursor-indexible sentinel
        for lhs, rhs in productions:
            assert self.id_by_non_terminal[lhs] == self.symbol_to_integer(lhs)
        prod1 = sorteduniquetuple((self.id_by_non_terminal[lhs], tuple(self.symbol_to_integer(symbol) for symbol in rhs) + (self.completion_id,)) for lhs, rhs in productions)
        if debug:
            print 'prod1:', prod1
        
        # introduce two further levels of indirection: sharing of rhs sequences, sharing of rule sets

        self.rhs_by_id, id_by_rhs = frozenbijection(rhs for lhs, rhs in prod1)
        if debug:
            print 'rhs_by_id:', tuple(enumerate(self.rhs_by_id))
            print 'rhs uniqueness:', len(self.rhs_by_id), 'of', len(prod1), ':', len(self.rhs_by_id) / len(prod1)

        # each production as a pair: this uses an index for each rhs sequence
        prod2 = sorteduniquetuple((lhs, id_by_rhs[rhs]) for lhs, rhs in prod1)
        if debug:
            print 'prod2:', prod2

        proddict = defaultdict(set)
        for lhs, rhs in prod2:
            proddict[lhs].add(rhs)
        # here, for each production, a tuple of the set of rhs indices is used for rhs
        prod3 = sorteduniquetuple((lhs, sorteduniquetuple(proddict[lhs])) for lhs in xrange(len(proddict)))

        self.ruleset_by_id, id_by_ruleset = frozenbijection(rhs for lhs, rhs in prod3)
        if debug:
            print 'ruleset_by_id:', tuple(enumerate(self.ruleset_by_id))
            print 'ruleset uniqueness:', len(self.ruleset_by_id), 'of', len(prod3), ':', len(self.ruleset_by_id) / len(prod3)

        # note: lhs are now a (useless) enumeration
        self.ruleset_id_by_lhs_id = tuple(id_by_ruleset[rhs] for lhs, rhs in prod3)
        if debug:
            print 'self.ruleset_id_by_lhs_id:', tuple(enumerate(self.ruleset_id_by_lhs_id))

        self.starts = sorteduniquetuple(self.id_by_non_terminal[start] for start in starts)
        if debug:
            print 'starts:', self.starts

        self._verify()

    def _verify(self):
        self.grammar_sets._verify()
        assert tuple(parse_grammar(self.str_iter)) == tuple(self.grammar_sets)

        # XXX needs many more

        # for the stringifying bailouts
        for rhs in self.rhs_by_id:
            assert list(rhs).index(self.completion_id) == len(rhs) - 1

    #  ways to stringify ourself!

    @property
    def grammar_sets(self):
        return self._grammar_sets

    @property
    def str_iter(self):
        # XXX would like to verify, currently can't due to recursion
        # self._verify()
        yield '__terminals__'
        for terminal in self.terminal_by_id:
            yield terminal

        integer_to_symbol = self.integer_to_symbol

        yield '__productions__'
        for lhs, ruleset_id in enumerate(self.ruleset_id_by_lhs_id):
            for rhs in self.ruleset_by_id[ruleset_id]:
                parts = list()
                parts.append(self.non_terminal_by_id[lhs])
                for item in self.rhs_by_id[rhs]:
                    if item is self.completion_id:
                        # in _verify() we asserted it's the last guy
                        break
                    parts.append(integer_to_symbol(item))
                yield ' '.join(parts)
                
        yield '__starts__'
        for id in self.starts:
            yield self.non_terminal_by_id[id]

    @property
    def bnf_iter(self):
        self._verify()
        for lhs, ruleset_id in enumerate(self.ruleset_id_by_lhs_id):        
            indent = None
            for rhs in self.ruleset_by_id[ruleset_id]:
                parts = list()
                if indent is None:
                    non_terminal = self.non_terminal_by_id[lhs]
                    indent = ' ' * len(non_terminal)
                    parts.append(non_terminal)
                    parts.append('::=')
                else:
                    parts.append(indent)
                    parts.append('  |')
                for item in self.rhs_by_id[rhs]:
                    if item is self.completion_id:
                        # in _verify() we asserted it's the last guy
                        break
                    parts.append(self.integer_to_symbol(item, '"'))
                yield ' '.join(parts)

    # items in rhs are integers: if non-negative, it's a non_terminal
    # id, if negative, and not completion_id (-1), it's offset by one
    # from the ones-complement of the terminal id; completion_id is a
    # sentinel used at the end of each rhs
    completion_symbol = '<complete>'
    completion_id = -1
    @staticmethod
    def symetrical_coder(id):
        # map from non-negatives to negatives (and back), leaving small negatives aside (e.g. -1) as sentinel(s)
        assert id != Grammar.completion_id
        result = ~id + Grammar.completion_id
        assert result != Grammar.completion_id
        return result
    def symbol_to_integer(self, symbol):
        assert symbol != self.completion_symbol
        id = self.id_by_non_terminal[symbol] if symbol in self.id_by_non_terminal else self.symetrical_coder(self.id_by_terminal[symbol])
        assert id != self.completion_id
        return id
    def integer_to_symbol(self, id, terminal_quote='', completion_symbol=''):
        assert id is not None
        if id == self.completion_id:
            return completion_symbol
        return self.non_terminal_by_id[id] if id >= 0 else terminal_quote + self.terminal_by_id[self.symetrical_coder(id)] + terminal_quote        

    def gen(self, countdown, seed=None, debug=False):
        """
        Returns a generator yielding terminal tokens giving a random sentence in
        the language.

        If countdown is positive the iteration will stop after the generated
        sentence contains countdown sub-sentences.

        countdown completions have occured (where a completion is an opaque
        internal occurence in the parser that indicates a sentential prefix has
        been observed).  If countdown is negative the iterator will stop when
        the end of the grammar is reached, which can easily be never for
        non-trivial grammars.

        If seed is given and is not None it is used to seed the randomness,
        giving reproducible results.

        If debug is not False, the verbose activity of the parser will be
        printed.
        """
        start_states = tuple(self.non_terminal_by_id[start] for start in self.starts)

        if seed is None:
            choice = random.choice
        else:
            rand = random.Random()
            rand.seed(seed)
            choice = rand.choice


        completion_id = self.completion_id

        # augment sequences with the start states info by creating a
        # new non-terminal called '<start_state>'

        # ids of the starting non_terminals
        start_ids = tuple(self.id_by_non_terminal[start_state] for start_state in start_states)
        if debug:
            print 'start_states:', start_states
            print 'start_ids:', start_ids
        assert start_ids == self.starts

        # new rhs tuples
        start_rhss = tuple((start_id, completion_id) for start_id in start_ids)

        # new rhs ids for the 
        start_rhs_ids = tuple(xrange(len(self.rhs_by_id), len(self.rhs_by_id) + len(start_rhss)))
        # augmented rhs collection
        rhs_by_id = self.rhs_by_id + start_rhss
        if debug:
            print 'start_rhs_ids:', start_rhs_ids
            print 'rhs_by_id:', tuple(enumerate(rhs_by_id))

        # new ruleset id
        start_ruleset_id = len(self.ruleset_by_id)
        # new ruleset
        ruleset_by_id = self.ruleset_by_id + (start_rhs_ids,)
        if debug:
            print 'start_ruleset_id:', start_ruleset_id
            print 'ruleset_by_id:', tuple(enumerate(ruleset_by_id))

        # the new lhs id
        start_lhs_id = len(self.ruleset_id_by_lhs_id)
        ruleset_id_by_lhs_id = self.ruleset_id_by_lhs_id + (start_ruleset_id,)
        if debug:
            print 'start_lhs_id:', start_lhs_id
            print 'ruleset_id_by_lhs_id:', tuple(enumerate(ruleset_id_by_lhs_id))

        # also, the new non_terminal
        assert start_lhs_id == len(self.non_terminal_by_id)
        non_terminal_by_id = self.non_terminal_by_id  + ('<start_state>',)
        if debug:
            print 'non_terminal_by_id:', tuple(enumerate(non_terminal_by_id))

        class EarleyState(tuple):
            parser = self
            def __new__(cls, *args):
                return args[0] if len(args) == 1 and type(args[0]) is EarleyState else super(EarleyState, cls).__new__(cls, args)
            def __init__(self, *_):
                super(EarleyState, self).__init__()
                assert len(self) == 4
                assert min(self) >= 0

            @property
            def lhs_id(self): return self[0]
            @property
            def rhs_id(self): return self[1]
            @property
            def cursor(self): return self[2]
            @property
            def origin(self): return self[3]

            @property
            def rhs_token(self):
                return rhs_by_id[self.rhs_id][self.cursor]                

            @property
            def is_sentential(self):
                return (self.lhs_id is start_lhs_id) and (self.rhs_token is completion_id)

            @property
            def shifted(self):
                return EarleyState(self.lhs_id, self.rhs_id, self.cursor + 1, self.origin)

            def __str__(self):
                parser = self.parser
                parts = list()
                parts.append(super(EarleyState, self).__repr__())
                parts.append('')
                parts.append(non_terminal_by_id[self.lhs_id])                                
                items = tuple(parser.integer_to_symbol(item, '"') for item in rhs_by_id[self.rhs_id]) + ('',)
                dotted = items[:self.cursor] + ('.',) + items[self.cursor:]
                # note: conditional in the generator skips empty items, e.g. the default-empty completion_symbol
                parts.append("( %s )" % (' '.join(item for item in dotted if item),))
                parts.append(repr(self.origin))
                return ' '.join(parts)


        statesets = list()
        workset = set()

        def do_prediction(stateset_id, lhs_id, from_state=None):
            stateset = statesets[stateset_id]
            assert lhs_id >= 0
            if debug:
                print 'predicting:', ' ', non_terminal_by_id[lhs_id]
            zero_cursor = 0
            ruleset_id = ruleset_id_by_lhs_id[lhs_id]            
            for rhs_id in ruleset_by_id[ruleset_id]:
                prediction = EarleyState(lhs_id, rhs_id, zero_cursor, stateset_id)
                if prediction not in stateset and prediction not in workset:
                    workset.add(prediction)
                    if debug:
                        print 'prediction:', ' ', ' ', prediction,
                        if from_state is not None:
                            print '  ', 'from:', from_state,
                        print
                else:
                    if debug:
                        print 'prediction:', '+', ' ', prediction,
                        if from_state is not None:
                            print '  ', 'from:', from_state,
                        print

        def do_completion(stateset_id, pred):
            stateset = statesets[stateset_id]
            my_lhs = pred.lhs_id
            if debug:
                print 'completing:', ('*' if pred.is_sentential else ' '), non_terminal_by_id[my_lhs], '', pred
            for parent in statesets[pred.origin]:
                if parent.rhs_token == my_lhs:
                    completion = parent.shifted
                    assert completion not in stateset
                    if completion not in workset:                    
                        workset.add(completion)
                        if debug:
                            print 'completion:', ' ', ' ', completion, '  ', 'from:', pred
                    else:
                        if debug:
                            print 'completion:', '@', ' ', completion, '  ', 'from:', pred

        if debug:
            print

        # bootstrap
        k = len(statesets)
        statesets.append(set())
        do_prediction(k, start_lhs_id)

        if debug:
            print
            print 'k:', k

        while True:
            completed = False
            stateset = statesets[k]
            assert workset

            scansets = defaultdict(list)
            while workset:
                pred = workset.pop()
                assert pred not in stateset
                stateset.add(pred)

                if pred.is_sentential:
                    completed = True
                    if countdown > 0:
                        countdown -= 1

                rhs_token = pred.rhs_token
                if rhs_token >= 0:
                    # prediction
                    do_prediction(k, rhs_token, pred)
                elif rhs_token is completion_id:
                    # completion
                    do_completion(k, pred)
                else:
                    # scanning
                    scansets[rhs_token].append(pred)

            if not scansets:
                assert completed
                if debug:
                    print
                    print 'completed: end-of-grammar'
                return

            if completed and countdown == 0:
                if debug:
                    print
                    print 'completed: countdown'
                return

            scan_token_id = choice(tuple(scansets))
            scan_token = self.integer_to_symbol(scan_token_id)
            if debug:
                print 'scanset:', ' '.join(sorted(self.integer_to_symbol(scanitem) for scanitem in scansets))                        
                print 'yielding:', scan_token
            yield scan_token

            # do the piled-up scanning work to seed the next state
            assert len(workset) == 0
            k = len(statesets)
            statesets.append(set())
            if debug:
                print
                print 'k:', k

            if debug:
                print 'scanning:', ' ', '', self.integer_to_symbol(scan_token_id, '"')
            # workset.update(state.shifted for state in scansets[scan_token_id])
            for state in scansets[scan_token_id]:
                assert state.rhs_token == scan_token_id
                scan_state = state.shifted
                assert scan_state not in workset
                workset.add(scan_state)
                if debug:
                    print 'scanned:', '    ', ' ', scan_state

if __name__ == '__main__':
    # text-based grammars for doctest
    
    _tiny_cfg = """
    __terminals__
    A B
    __productions__
    abab A B A B
    __starts__
    abab
    """

    _binary_cfg = """
    __terminals__
    0 1
    __productions__
    binary_digit 0
    binary_digit 1
    binary_number binary_digit
    binary_number binary_number binary_digit
    __starts__
    binary_number
    """

    _hello_world_cfg = """
    # algebraic expressions: the "hello world" of context-free grammars

    # terminal tokens
    __terminals__
    0 1 2 3 4 5 6 7 8 9 0
    +
    *
    ( )

    A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
    _

    # the production rules
    __productions__ 

    # first, abstract the terminals
    # digits
    digit 0
    digit 1
    digit 2
    digit 3
    digit 4
    digit 5
    digit 6
    digit 7
    digit 8
    digit 9

    # operations
    plus +
    times *

    # grouping
    left (
    right )

    # now the stuff for building expressions

    # expressions
    expr  term
    expr  expr plus term
    term  primary
    term  term times primary
    primary  number
    primary  symbol
    primary  left expr right
    number  digit
    number  number digit

    # symbols
    alpha A
    alpha B
    alpha C
    alpha D
    alpha E
    alpha F
    alpha G
    alpha H
    alpha I
    alpha J
    alpha K
    alpha L
    alpha M
    alpha N
    alpha O
    alpha P
    alpha Q
    alpha R
    alpha S
    alpha T
    alpha U
    alpha V
    alpha W
    alpha X
    alpha Y
    alpha Z

    under _

    initial  alpha
    trailing  alpha
    trailing  under
    trailing  digit

    symbol initial
    symbol symbol trailing

    # stuff to exercise the sharing of rhs and rule sets:
    # this rhs is same as one of those for expr
    foo  expr plus term
    # bar has same rhs set as expr
    bar  expr plus term
    bar  term

    # an epsilon expression
    baz

    # epsilon and something
    baz2 
    baz2 digit digit digit alpha

    # epsilon rule and something
    baz3 baz
    baz3 digit alpha digit alpha

    # the start state(s)
    __starts__
    expr
    #symbol
    #baz2
    #baz3
    """

    _ambiguous_cfg = """
    __terminals__
    IF
    THEN
    ELSE
    EXPR
    FOO
    (
    )
    __productions__
    expr EXPR
    # using FOO instead of expr as conditional to simplify generating an ambiguous sentence
    expr IF  FOO  THEN  expr
    expr IF  FOO  THEN  expr  ELSE  expr 
    __starts__
    expr
    """

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
