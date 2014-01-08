###########################################################################
#
# File:         simplecfg.py (directory: ./py/onyx/dataflow)
# Date:         5-May-2008
# Author:       Hugh Secker-Walker
# Description:  A "simpler" Earley parsing framework.
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
Tools for working with CFG grammars.
"""
from __future__ import with_statement
import operator, random
from collections import defaultdict, deque
from itertools import count, chain
from onyx.builtin import frozendict, dict_of_set, frozendict_of_set, dict_of
from onyx.containers import sorteduniquetuple, frozenbijection
from onyx.graph.graphtools import NodeLabelIdGraphBuilder, FrozenGraph, make_initialized_set_graph_builder
from onyx.util.checkutils import check_instance
from onyx.util.debugprint import dprint, DebugPrint

class CfgBuilder(object):
    """
    A simple builder for CFG grammars.

    The CfgBuilder provides two set-style methods for adding productions:
    add_production() adds a single production for a non-terminal to the grammar;
    update_production() adds a sequence of productions for a non-terminal to the
    grammar.

    CfgBuilder supports containment queries via Python's in syntax, returning
    True if the given symbol is either a non-terminal or a terminal that appears
    in the grammar.  This allows a client to ensure that non-terminals or
    terminals that are planned to be added are either new to the grammar or
    already exist in the grammar.
    """
    def __init__(self):
        self.productions = production_dict()
        self.symbols = set()

    def __contains__(self, symbol):
        """
        Returns True if symbol appears anywhere in the grammar.
        """
        return symbol in self.symbols

    @property
    def size(self):
        """
        The size of the grammar.  We follow Robert Moore in calculating the size
        of the grammar: the size is the number of non-terminal symbols plus the
        sum of the lengths of the right-hand-side sequences over all the
        productions in the grammar.  By counting each non-terminal just once,
        instead of once for each of its productions, this size statistic more
        closely tracks storage requirements of actual implementations of grammar
        structures.  An empty right-hand-side sequence (epsilon) is counted has
        having length one.
        """
        return self.productions.size

    def add_production(self, lhs, rhs):
        """
        Add a production to the CFG.  The lhs argument is an immutable object
        that is the left-hand-side (non-terminal) for the production.  The rhs
        argument is a possibly-empty iterable of immutable objects (symbols)
        that are the sequence of non-terminals and terminals that make up the
        right-hand-side of the production.  An empty rhs is used to add an
        epsilon production.  The productions for a given non-terminal are
        treated as a set; this means that duplicate right-hand sides are
        ignored.  See also update_production().

        >>> builder = CfgBuilder()
        >>> builder.size
        0
        >>> builder.add_production('A', ('x', 'y', 'zoo'))
        >>> builder.size
        4
        >>> builder.add_production('B', ())
        >>> builder.size
        6
        >>> builder.add_production('B', ('b',))
        >>> builder.size
        7
        >>> builder.add_production('B', ('B', 'b'))
        >>> builder.add_production('Cows', ('A',))
        >>> builder.add_production('Cows', ('B',))
        >>> builder.size
        12
        >>> builder.add_production('Cows', ('B',))
        >>> builder.size
        12
        >>> 'A' in builder
        True
        >>> 'zoo' in builder
        True
        >>> 'Moo' in builder
        False
        """
        rhs = tuple(rhs)
        self.productions[lhs].add(rhs)
        self.symbols.add(lhs)
        self.symbols.update(rhs)

    def update_production(self, lhs, rhs_set):
        """
        Add a set of productions to the CFG.  The lhs argument is an immutable
        object that is the left-hand-side (non-terminal) for each of the
        productions.  The rhs_set argument is a possibly-empty iterable of rhs
        sequences.  Each rhs sequence is an iterable of immutable objects
        (symbols) that are the sequence of non-terminals and terminals that make
        up the right-hand-side of the given production.  An empty rhs is used to
        add an epsilon production.  The productions for a given non-terminal are
        treated as a set; this means that duplicate right-hand sides are
        ignored.  See also add_production().

        >>> builder = CfgBuilder()
        >>> builder.add_production('A', ('x', 'y', 'zoo'))
        >>> builder.update_production('B', ((), ('b',), ('B', 'b')))
        >>> builder.size
        9
        >>> builder.update_production('Cows', (('A',), ('B',)))
        >>> builder.size
        12
        >>> builder.add_production('Cows', ('A',))
        >>> builder.size
        12
        >>> builder.update_production('Cows', (('B',), ('A',), ))
        >>> builder.size
        12
        >>> 'A' in builder, 'zoo' in builder, 'Moo' in builder
        (True, True, False)
        """
        rhs_set = tuple(tuple(rhs) for rhs in rhs_set)
        self.productions[lhs].update(rhs_set)
        self.symbols.add(lhs)
        self.symbols.update(rhs_symbol for rhs in rhs_set for rhs_symbol in rhs)

    
class production_dict(dict_of_set):
    """
    A specialized dictionary for working with CFG productions.  The keys are
    non-terminal symbols.  Each value is a set of the right-hand-side sequences
    for the productions corresponding to the left-hand-side key.  An empty set
    is automatically added when a new key is used to access the dictionary.
    """
    @property
    def size(self):
        """
        The size of the grammar.  We follow Robert Moore in calculating the size
        of the grammar: the size is the number of non-terminal symbols plus the
        sum of the lengths of the right-hand-side sequences over all the
        productions in the grammar.  By counting each non-terminal just once,
        instead of once for each of its productions, this size statistic more
        closely tracks storage requirements of actual implementations of grammar
        structures.  An empty right-hand-side sequence (epsilon) is counted as
        having length one.
        """
        return reduce(operator.add, (max(len(rhs), 1) for rhs_set in self.itervalues() for rhs in rhs_set), len(self))

    @property
    def iter_flat(self):
        """
        Returns a generator that yields a pair corresponding to each production
        in the set of productions.  The first item in each pair is the
        non-terminal left-hand-side for the production, the second item is
        sequence of non-terminals and terminals making up the right-hand-side of
        the production.
        """
        # XXX this is not robust against changing the dict while we're iterating
        return ((lhs, rhs) for lhs in sorted(self.keys()) for rhs in sorted(self[lhs]))

    def get_non_terminals(self):
        """
        Return a set of the non_terminal symbols in the collection of productions.
        """
        return set(self.iterkeys())

    def get_terminals(self):
        """
        Return a set of the terminals symbols in the productions.
        """
        return set(symbol for rhs_set in self.itervalues() for rhs in rhs_set for symbol in rhs if symbol not in self)


def iter_nullable(productions):
    """
    Yield the nullable non-terminals in the productions.  Productions is a
    mapping with non-terminal key and an iterable of sequences as value, where
    each (possibly empty) sequence consists of non-terminals and terminals.

    This function returns the list of non-terminals which, through transitive
    closure, can derive an empty sentence.

    >>> tuple(iter_nullable({}))
    ()
    >>> tuple(iter_nullable({'x':(('x', 'y'),)}))
    ()
    >>> tuple(iter_nullable({'x':((), ('x', 'y'))}))
    ('x',)
    >>> tuple(iter_nullable({'x':((), ('x', 'y')), 'z':(('a', 'b'), ('x',)), 'a':(('a',),)}))
    ('x', 'z')
    """

    def iter_epsilon(productions):
        # generator of lhs of productions with direct epsilons on the rhs
        return (lhs for lhs, rhs_set in productions.iteritems() for rhs in rhs_set if not rhs)

    # see if there's any work to be done due to an epsilon production
    if not any(iter_epsilon(productions)):
        return

    non_terminals = frozenset(productions)
    # print 'non_terminals:', ' '.join(non_terminals)

    # setify the rhss of each non-terminal
    work_prod = dict((lhs, set(frozenset(rhs) for rhs in rhs_set)) for lhs, rhs_set in productions.iteritems())
    # eliminate productions with terminals by keeping only those rhs_set which
    # are subsets of non_terminals
    work_prod = dict((lhs, tuple(set(rhs_set) for rhs_set in rhs_sets if rhs_set <= non_terminals)) for lhs, rhs_sets in work_prod.iteritems())
    # keep only those productions with non-empty rhs_sets, i.e. eliminate
    # non-terminals whose productions were composed only of terminals and thus
    # gave empty rhs_sets
    work_prod = dict((lhs, rhs_sets) for lhs, rhs_sets in work_prod.iteritems() if rhs_sets)    
    # print 'work_prod:', work_prod

    # now iteratively find empty rhss and eliminate new nullables from rhss
    nullables = set()
    while True:
        # find the lhs that have a direct empty rhs
        new_nullables = set(iter_epsilon(work_prod))
        if not new_nullables:
            return
        # print 'new_nullables:', new_nullables
        # eliminate those productions, yield the lhs
        for lhs in new_nullables:
            del work_prod[lhs]
            yield lhs

        # remove new_nullables from rhss
        for lhs, rhs_sets in work_prod.iteritems():
            assert lhs not in nullables
            for rhs_set in rhs_sets:
                assert not (rhs_set & nullables)
                rhs_set -= new_nullables

        # update nullables
        nullables.update(new_nullables)


class FsaItem(tuple):
    """
    Item in an Fsa decomposition of a grammar.  An FsaItem is a production and a
    cursor position within the production.  Specifically it's a triple: the
    non-terminal for the production, the right-hand side sequence that the
    production expands to, and the zero-based cursor position within the rhs
    sequence.

    >>> lhs = 'A'
    >>> rhs = 'a1', 'a2', 'a3'
    >>> a = FsaItem.inital_item(lhs, rhs)
    >>> a.is_initial, a.is_penultimate, a.is_final
    (True, False, False)
    >>> a.is_production, a.is_terminal
    (True, False)
    >>> a.is_epsilon
    False

    >>> a.lhs
    'A'
    >>> a.rhs_symbol
    'a1'

    >>> a.cursor
    0
    >>> a.rhs_seq
    ('a1', 'a2', 'a3')
    >>> while not a.is_final: a.lhs; a.rhs_symbol; print a; a = a.shifted
    'A'
    'a1'
    A ( . a1 a2 a3 )
    'A'
    'a2'
    A ( a1 . a2 a3 )
    'A'
    'a3'
    A ( a1 a2 . a3 )

    >>> print a
    A ( a1 a2 a3 . )
    >>> a.is_final
    True
    >>> a.is_penultimate
    False
    >>> print a.penultimate
    A ( a1 a2 . a3 )
    >>> a.penultimate.is_penultimate
    True

    >>> print a.rhs_symbol
    Traceback (most recent call last):
      ...
    ValueError: cannot access rhs_symbol property of an FsaItem that is_final
    >>> a.shifted
    Traceback (most recent call last):
      ...
    ValueError: cannot shift an FsaItem that is_final


    >>> e = FsaItem.inital_item('E',())
    >>> e.is_initial, e.is_penultimate, e.is_final
    (True, False, True)
    >>> e.is_production, e.is_terminal
    (True, False)
    >>> e.is_epsilon
    True
    >>> e.lhs
    'E'

    >>> b = FsaItem.terminal_item('x')
    >>> b.is_initial, b.is_final
    (True, False)
    >>> b.is_initial, b.is_penultimate, b.is_final
    (True, True, False)
    >>> b.is_epsilon
    False
    >>> b.as_terminal
    'x'
    >>> b.lhs
    'x'
    >>> b.rhs_symbol
    'x'

    >>> print b
    x ( . x )
    >>> b = b.shifted
    >>> b.is_initial, b.is_penultimate, b.is_final
    (False, False, True)
    >>> b.is_production, b.is_terminal
    (False, True)
    """

    @staticmethod
    def inital_item(lhs, rhs_seq):
        item = FsaItem(lhs, tuple(rhs_seq), 0)
        assert item.is_production
        return item

    @staticmethod
    def terminal_item(symbol):
        item = FsaItem(symbol, (symbol,), 0)
        assert item.is_terminal
        return item
    
    def __new__(cls, *args):
        if len(args) == 1:
            args, = args
            if type(args) is FsaItem:
                return args
        return super(FsaItem, cls).__new__(cls, args)
    def __init__(self, *_):
        super(FsaItem, self).__init__()
        assert len(self) == 3, str(len(self))
        assert self.is_production ^ self.is_terminal

    @property
    def lhs(self): return self[0]
    @property
    def rhs_seq(self): return self[1]
    @property
    def cursor(self): return self[2]

    @property
    def shifted(self):
        if self.is_final:
            raise ValueError("cannot shift an FsaItem that is_final")
        rhs_seq = self.rhs_seq
        new_cursor = self.cursor + 1
        assert 0 < new_cursor <= len(rhs_seq)
        return FsaItem(self.lhs, rhs_seq, new_cursor)

    @property
    def unshifted(self):
        if self.is_initial:
            raise ValueError("cannot unshift an FsaItem that is_initial")
        rhs_seq = self.rhs_seq
        new_cursor = self.cursor - 1
        assert 0 <= new_cursor < len(rhs_seq)
        return FsaItem(self.lhs, rhs_seq, new_cursor)

    @property
    def initial(self):
        assert self.is_penultimate
        return FsaItem(self.lhs, self.rhs_seq, 0)

    @property
    def penultimate(self):
        assert not self.is_epsilon, str(self)
        rhs_seq = self.rhs_seq
        new_cursor = len(rhs_seq) - 1
        assert new_cursor >= 0
        return FsaItem(self.lhs, rhs_seq, new_cursor)

    @property
    def final(self):
        assert self.is_initial
        rhs_seq = self.rhs_seq
        new_cursor = len(rhs_seq)
        return FsaItem(self.lhs, rhs_seq, new_cursor)

    @property
    def is_initial(self):
        return self.cursor == 0
    @property
    def is_penultimate(self):
        return self.cursor + 1 == len(self.rhs_seq)
    @property
    def is_final(self):
        return self.cursor == len(self.rhs_seq)
    @property
    def is_epsilon(self):
        return self.is_initial and self.is_final

    @property
    def rhs_symbol(self):
        if self.is_final:
            raise ValueError("cannot access rhs_symbol property of an FsaItem that is_final")
        rhs_seq = self.rhs_seq
        len_rhs_seq = len(rhs_seq)
        cursor = self.cursor
        assert 0 <= cursor <= len_rhs_seq
        return rhs_seq[cursor] if cursor < len_rhs_seq else self.complete_symbol

    @property
    def is_production(self):
        # not a strict assessment
        #return len(self.rhs_seq) != 1 or self.rhs_seq[0] != self.lhs
        return not self.is_terminal

    @property
    def is_terminal(self):
        # not a strict assessment, but any such cyclic production would be useless
        # XXX so it should be forbidden
        return len(self.rhs_seq) == 1 and self.rhs_seq[0] == self.lhs

    @property
    def as_terminal(self):
        assert self.is_terminal
        return self.lhs

    def __str__(self):
        parts = list()
        # note: use of str here is to support the root item
        parts.append(str(self.lhs))
        items = tuple(self.rhs_seq)
        dotted = items[:self.cursor] + ('.',) + items[self.cursor:]
        parts.append("( %s )" % (' '.join(dotted)))
        return ' '.join(parts)
    

def fsa_dot_lines(successor_links, prediction_links, completion_links,
                  graph_label='', graph_type='digraph', globals=()):

    all_nodes = set(node for link in chain(successor_links, prediction_links, completion_links) for node in link)
    node_by_id, id_by_node = frozenbijection(all_nodes)
    label_by_id = tuple(str(node) for node in node_by_id)
    node_name_num_digits = len(str(len(label_by_id)))
    name_by_node = dict((node, 'n%0*d' % (node_name_num_digits, id_by_node[node])) for node in node_by_id)

    prod_starts = set(node for node in all_nodes if node.is_initial)
    ranks = list()
    for item in prod_starts:
        rank = list()
        rank.append(item)
        while not item.is_final:
            item = item.shifted
            rank.append(item)
        ranks.append(rank)

    group_by_node = dict()
    for group, rank in enumerate(ranks):
        for item in rank:
            group_by_node[item] = '"g%d"' % (group,)

    node_set_by_lhs = dict_of_set()
    for node in all_nodes:
        node_set_by_lhs[node.lhs].add(node)

    # opening
    yield "%s %s { \n" % (graph_type, graph_label)

    # globals
    yield '  rankdir=TB;\n'
    yield '  nodesep=0.3625;\n'
    # yield '  ranksep=1.0;\n'
    for line in globals:
        yield '  %s\n' % (line,)

    # nodes
    for node in node_by_id:
        yield '  %s [label="%s", group=%s];\n' % (name_by_node[node], node, group_by_node[node])

    # node ranks
    for rank in ranks:
        yield '  { rank=same; %s; }\n' % ('; '.join('%s' % (name_by_node[node],) for node in rank),)

    # production links
    for from_node, to_node in successor_links:
        yield '  %s -> %s [style=dotted, weight=8];\n' % (name_by_node[from_node], name_by_node[to_node])

    # prediction links
    for from_node, to_node in prediction_links:
        yield '  %s -> %s [style=solid];\n' % (name_by_node[from_node], name_by_node[to_node])

    for from_node, to_node in completion_links:
        yield '  %s -> %s [style=dashed, dir=back];\n' % (name_by_node[to_node], name_by_node[from_node])

    # closing
    yield "}\n"

def render(dot_lines_iterable,
           temp_file_prefix='simplecfg_',
           display_tool_format="open -a /Applications/Graphviz.app %s"):

    import os
    from onyx.util import opentemp
    with opentemp('wb', suffix='.dot', prefix=temp_file_prefix) as (name, outfile):
        for line in dot_lines_iterable:
            outfile.write(line)
    os.system(display_tool_format % (name,))
    return name

def plot_fsa(successor_links, prediction_links, completion_links):
    return render(fsa_dot_lines(successor_links, prediction_links, completion_links))


class CfgDecoderBase(object):
    """
    A baseclass for objects that use a Cfg.  This baseclass creates the
    structures necessary to decode against the Cfg represented by productions.

    >>> b = CfgBuilder()
    >>> b.add_production('A', ('x', 'y', 'z'))
    >>> b.add_production('B', ())
    >>> b.add_production('B', ('b',))
    >>> b.add_production('B', ('B', 'b'))
    >>> b.add_production('C', ('A',))
    >>> b.add_production('C', ('B',))

    >>> x = CfgDecoderBase(b)
    >>> x.dump()
    has_epsilon: True
    prediction_items:
      B ( . B b )
        B ( . B b )
        B ( . b )
      C ( . A )
        A ( . x y z )
      C ( . B )
        B ( . B b )
        B ( . b )
    prediction_terminals:
      A ( . x y z )
        x
      A ( x . y z )
        y
      A ( x y . z )
        z
      B ( . B b )
        b
      B ( B . b )
        b
      B ( . b )
        b
      C ( . A )
        x
      C ( . B )
        b
    item_completions_unshifted:
      A ( x y . z )
        C ( . A )
      B ( B . b )
        B ( . B b )
        C ( . B )
      B ( . b )
        B ( . B b )
        C ( . B )
    item_completions_unshifted5:
      A
        C ( . A )
      B
        B ( . B b )
        C ( . B )
      C
      b
        B ( . B b )
        B ( B . b )
        B ( . b )
        C ( . B )
      x
        A ( . x y z )
      y
        A ( x . y z )
      z
        A ( x y . z )
        C ( . A )
    terminal_completions_unshifted:
      b
        B ( . B b )
        B ( B . b )
        B ( . b )
        C ( . B )
      x
        A ( . x y z )
      y
        A ( x . y z )
      z
        A ( x y . z )
        C ( . A )
    """
    def __init__(self, arg):
        productions = arg.productions if isinstance(arg, CfgBuilder) else arg
        check_instance(production_dict, productions)
        
        # XXX need to deal with epsilons

        # sets of symbols
        non_terminals = self.non_terminals = frozenset(productions.get_non_terminals())
        terminals = self.terminals = frozenset(productions.get_terminals())
        nullable_non_terminals = self.nullable_non_terminals = frozenset(iter_nullable(productions))
        assert nullable_non_terminals <= non_terminals
        assert not (nullable_non_terminals & terminals)

        # the set of fsa items for productions
        fsa_set = set()
        # set of initial FsaItems for each lhs
        initial_items_by_non_terminal = dict_of_set()
        # map from non-terminal to set of fsa items with that non-terminal as rhs_symbol
        actives = dict_of_set()
        for lhs, rhs in productions.iter_flat:
            assert lhs in non_terminals
            fsa_item = FsaItem.inital_item(lhs, rhs)
            assert not fsa_item.is_terminal
            assert fsa_item not in fsa_set
            fsa_set.add(fsa_item)
            initial_items_by_non_terminal[lhs].add(fsa_item)
            while not fsa_item.is_final:
                actives[fsa_item.rhs_symbol].add(fsa_item)
                fsa_item = fsa_item.shifted
                assert fsa_item not in fsa_set
                fsa_set.add(fsa_item)
        self.actives = frozendict_of_set(actives)

        # augment both fsa_set and initial_items_by_non_terminal with terminal items
        #
        # note: these are non-standard FsaItems because they do not correspond
        # to productions, but they are useful for giving uniform node entities
        # in the graphical representation
        for terminal in terminals:
            assert terminal not in fsa_set
            assert terminal not in initial_items_by_non_terminal
            terminal_item = FsaItem.terminal_item(terminal)
            fsa_set.add(terminal_item)
            fsa_set.add(terminal_item.shifted)
            initial_items_by_non_terminal[terminal].add(terminal_item)
        fsa_set = frozenset(fsa_set)
        initial_items_by_non_terminal = self.initial_items_by_non_terminal = frozendict_of_set(initial_items_by_non_terminal)

        # ultimately we build transitive closures of the items for prediction
        # and for completion; we do this by generating the set of links in the
        # graphs, then build the graphs, then use the graph to get the
        # transitive closure of each node
        #
        # we will include terminal items in the graphs, and we will check
        # item.is_production when we are using the items and we care
        has_epsilon = False
        prediction_links = list()
        completion_links = list()
        completion_links2 = list()
        for item in fsa_set:
            if item.is_epsilon:
                has_epsilon = True
            if item.is_final or item.is_terminal:
                # skip endings (including epsilons), and skip terminals
                continue
            for down_item in initial_items_by_non_terminal[item.rhs_symbol]:
                if down_item.is_final:
                    # skip epsilons
                    assert down_item.cursor == 0
                    continue
                # note: down_item can be a terminal item
                prediction_links.append((item, down_item))
                completion_links.append((down_item.final, item.shifted))
                completion_links2.append((down_item.penultimate, item))
                #completion_links2.append((down_item.final.unshifted, item))
        self.has_epsilon = has_epsilon
        assert len(set(prediction_links)) == len(prediction_links)
        assert len(set(completion_links)) == len(completion_links)
        assert len(set(completion_links2)) == len(completion_links2)

        # may want to expose these at some point
        prediction_graph = FrozenGraph(make_initialized_set_graph_builder(prediction_links))
        completion_graph = FrozenGraph(make_initialized_set_graph_builder(completion_links))
        completion_graph2 = FrozenGraph(make_initialized_set_graph_builder(completion_links2))

        display = False
        if display:
            def node_label_callback(item, *_):
                check_instance(FsaItem, item)
                # an instance of caring about is_production: terminal items display just as the terminal
                return str(item if item.is_production else item.as_terminal)
            def node_attributes_callback(item, *_):
                check_instance(FsaItem, item)
                if item.is_terminal:
                    yield 'style=bold'
            prediction_graph.dot_display(node_label_callback=node_label_callback, node_attributes_callback=node_attributes_callback, globals=('label="Predictions";', 'labelloc=top;',))
            completion_graph.dot_display(node_label_callback=node_label_callback, node_attributes_callback=node_attributes_callback, globals=('label="Completions";', 'labelloc=top;', 'rankdir=BT;'))
            completion_graph2.dot_display(node_label_callback=node_label_callback, node_attributes_callback=node_attributes_callback, globals=('label="Completions 2";', 'labelloc=top;', 'rankdir=BT;'))

        prediction_sets = dict_of_set((item, set(prediction_graph.get_forward_transitive_closures([item])[0])) for item in fsa_set if item.is_production and not item.is_final)
        completion_sets = dict_of_set((item, set(completion_graph.get_forward_transitive_closures([item])[0])) for item in fsa_set if not item.is_initial)
        completion_sets2 = dict_of_set((item, set(completion_graph2.get_forward_transitive_closures([item])[0])) for item in fsa_set if not item.is_final)

        debug = False
        if debug:
            print 'prediction_sets:'
            for item, items in sorted(prediction_sets.iteritems()):
                print item
                for x in sorted(items):
                    print ' ', x
            print 'completion_sets:'
            for item, items in sorted(completion_sets.iteritems()):
                print item
                for x in sorted(items):
                    print ' ', x
            print 'completion_sets2:'
            for item, items in sorted(completion_sets2.iteritems()):
                print item
                for x in sorted(items):
                    print ' ', x
        
        # XXX needs work, the completeables work in senderator could be moved here

        # these are the structures that our CfgDecoderBase-subclassing objects
        # will use

        # these two structures are used for prediction work
        self.prediction_items = frozendict_of_set((item, frozenset(prediction for prediction in predictions if prediction.is_production)) for item, predictions in prediction_sets.iteritems())
        self.prediction_terminals = frozendict_of_set((item, frozenset(prediction.as_terminal for prediction in predictions if prediction.is_terminal)) for item, predictions in prediction_sets.iteritems())

        # helper
        item_completions = frozendict_of_set((item, frozenset(completions)) for item, completions in completion_sets.iteritems() if item.is_production)
        # what gets used
        self.item_completions_unshifted = frozendict_of_set((item.unshifted, frozenset(completion.unshifted for completion in completions)) for item, completions in item_completions.iteritems())
        self.item_completions_unshifted2 = frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions)) for item, completions in item_completions.iteritems() if item.unshifted.rhs_symbol in non_terminals)

        #item_completions3 = frozendict_of_set((item, frozenset(completions)) for item, completions in completion_sets.iteritems())
        #item_completions3 = frozendict_of_set((item, frozenset(completions)) for item, completions in completion_sets.iteritems())
        #assert item_completions3 == frozendict_of_set(completion_sets)
        #item_completions3 = frozendict_of_set(completion_sets)
        #self.item_completions_unshifted3 = frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions if not completion.unshifted.is_penultimate)) for item, completions in item_completions3.iteritems())
        self.item_completions_unshifted3 = frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions if not completion.unshifted.is_penultimate)) for item, completions in completion_sets.iteritems())
#        assert self.item_completions_unshifted3 == frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions if not completion.unshifted.is_penultimate)) for item, completions in item_completions3.iteritems())
#        assert set(self.item_completions_unshifted3) == set(frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions if not completion.unshifted.is_penultimate)) for item, completions in frozendict_of_set(completion_sets).iteritems()))
#        assert self.item_completions_unshifted3 == frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions if not completion.unshifted.is_penultimate)) for item, completions in frozendict_of_set(completion_sets).iteritems())
#        assert self.item_completions_unshifted3 == frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions if not completion.unshifted.is_penultimate)) for item, completions in completion_sets.iteritems())
        #self.item_completions_unshifted3 = frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions)) for item, completions in item_completions3.iteritems())

        self.item_completions_unshifted4 = frozendict_of_set((item.unshifted.rhs_symbol, frozenset(completion.unshifted for completion in completions)) for item, completions in completion_sets.iteritems())

        #self.item_completions_unshifted5 = frozendict_of_set((item.rhs_symbol, frozenset(completions)) for item, completions in completion_sets2.iteritems())
        # map from symbol to all items that can advance when that symbol is seen
        item_completions_unshifted5 = dict_of(set)
        for item, completions in completion_sets2.iteritems():        
            item_completions_unshifted5[item.lhs].update(completions)
        #self.item_completions_unshifted5 = frozendict_of_set(item_completions_unshifted5)
        self.item_completions_unshifted5 = item_completions_unshifted5

##         assert set(self.item_completions_unshifted4) == set(self.item_completions_unshifted5)
##         for foo in set(self.item_completions_unshifted5):
##             #assert self.item_completions_unshifted4[foo] == self.item_completions_unshifted5[foo], str(foo)
##             if self.item_completions_unshifted4[foo] != self.item_completions_unshifted5[foo]:
##                 print 'foo:', foo
##                 print ' ',  self.item_completions_unshifted4[foo]
##                 print ' ',  self.item_completions_unshifted5[foo]

        # help
        terminal_completions = frozendict_of_set((item.as_terminal, frozenset(completions)) for item, completions in completion_sets.iteritems() if item.is_terminal)
        # this is what's used in practice for completion work
        self.terminal_completions_unshifted = frozendict_of_set((terminal, frozenset(completion.unshifted for completion in completions)) for terminal, completions in terminal_completions.iteritems())
                
    def dump(self):
        print 'has_epsilon:', self.has_epsilon
        print 'prediction_items:'
        for lhs, item_set in sorted(self.prediction_items.iteritems()):
          if item_set:
            print ' ', lhs
            for item in sorted(item_set):
              print ' ', ' ', item
        print 'prediction_terminals:'
        for lhs, terminal_set in sorted(self.prediction_terminals.iteritems()):
          if terminal_set:
            print ' ', lhs
            for item in sorted(terminal_set):
                print ' ', ' ', item
        print 'item_completions_unshifted:'
        for lhs, item_set in sorted(self.item_completions_unshifted.iteritems()):
          if item_set:
            print ' ', lhs
            for item in sorted(item_set):
              print ' ', ' ', item
##         print 'item_completions_unshifted2:'
##         for symbol, item_set in sorted(self.item_completions_unshifted2.iteritems()):
##           if item_set:
##             print ' ', symbol
##             for item in sorted(item_set):
##               print ' ', ' ', item
        print 'item_completions_unshifted5:'
        for symbol, item_set in sorted(self.item_completions_unshifted5.iteritems()):
            print ' ', symbol
            for item in sorted(item_set):
                print ' ', ' ', item
        print 'terminal_completions_unshifted:'
        for lhs, terminal_set in sorted(self.terminal_completions_unshifted.iteritems()):
          if terminal_set:
            print ' ', lhs
            for item in sorted(terminal_set):
                print ' ', ' ', item
    
class RecognizerState(tuple):
    """
    Helper for Cfg recognition.  It's a container for the immutablesignature of
    a recognizer state in the decoder, (is_sentential, earley_seeders), and a
    mutable member, extenders.  When copy constructed, the signature is
    preserved, but the extenders is copied.

    Make an empty one
    >>> RecognizerState()
    (False, frozendict_of_set({}), dict_of(set, {}))

    Make a non-empty one
    >>> r1 = RecognizerState(True, dict_of(set, (('a', set((0, 1))), ('b', set((10, 20, 30))))))

    Make a copy of it
    >>> r2 = RecognizerState(r1)

    Their extenders are distinct
    >>> r1.extenders['c'].add(100)
    >>> r2.extenders['d'].add(200)
    >>> r1
    (True, frozendict_of_set({'a': frozenset([0, 1]), 'b': frozenset([10, 20, 30])}), dict_of(set, {'c': set([100])}))
    >>> r2
    (True, frozendict_of_set({'a': frozenset([0, 1]), 'b': frozenset([10, 20, 30])}), dict_of(set, {'d': set([200])}))
    """
    required_len = 3
    def __new__(cls, *args):
        if len(args) == 0:
            # empty case, not very useful
            args = False, frozendict_of_set(), dict_of(set)
        else:
            if len(args) == 1:
                # copy construction
                args, = args
                is_sentential, earley_seeders, extenders = args
            else:
                # standard construction with is_sentential and earley_seeders arguments
                is_sentential, earley_seeders = args
                extenders = ()
            # make sure earley_seeders is immutable; always copy the extenders
            args = True if is_sentential else False, frozendict_of_set(earley_seeders), dict_of(set, extenders)
        return super(RecognizerState, cls).__new__(cls, args)
    def __init__(self, *_):
        if len(self) != self.required_len:
            raise ValueError("expected to initialize from %d items, but got %d in %r" % (self.required_len, len(self), self))
        # verify immutability
        hash(self.is_sentential)
        hash(self.seeders)
    @property
    def is_sentential(self):
        return self[0]
    @property
    def seeders(self):
        return self[1]
    @property
    def extenders(self):
        return self[2]

    

class CfgRecognizer(CfgDecoderBase):
    """
    A subclass of CfgDecoderBase that can be used to recognize sequences of
    terminals against languages defined by sets of non-terminals of a Cfg
    grammar.
    """
    def recognizer(self, start_non_terminals, force_tree=False):
        """
        Given a set of starting non-terminals in the Cfg productions from which
        CfgRecognizer was constructed, returns a (generator) function that can
        be used to recognize whether a sequence of terminals forms a sentence in
        the language defined by the starting non_terminals.

        Optional force_tree defaults to False, and the recognizer optimizes its
        internal structures to merge decodes that have identical futures
        (potentially resulting in space-saving DAGs or even cyclic graphs in
        these data structures.  If force_tree is True, this optimization is
        disabled and the recognizer will generate a strict tree of internal
        states.  This can use exponentially more space.

        The returned function, func, should be called with each terminal symbol
        in a sequence of terminals in order to recognize whether the stream of
        symbols is a sentence in the language of the grammar.

        Each func(terminal) call returns a tuple, (is_sentential,
        legal_terminals), where is_sentential is True if the sentential prefix
        ending in terminal is a sentential form, and where legal_terminals is
        the frozenset of legal terminals from which to choose the terminal for
        the next call to send.  If the next putative terminal in the client's
        token sequence is not in the set of legal_terminals then the recognition
        of the client's sequence has failed.  It is up to the client to notice
        this.

        As with all generators, the first call to func must be func(None).  From
        this initial call, is_sentential is True if the empty string is a
        sentence in the language, and legal_terminals is the set of legal
        initial terminals in (non-empty) sentences of the language.

        If an empty legal_terminals is ever returned then the recognition is
        finished.  In this case is_sentential will be True and the recognition
        will have found a sentence that is a proper prefix of no other sentence
        in the language.

        It is an error to call func with a symbol that is not in the most
        recently returned legal_terminals set.

        Implementation note: This recognizer is implemented with an
        FSA-item-centric version of Earley's algorithm.  As such it can handle
        any CFG.  It makes extensive use of mappings and sets.  It is prelimary
        work for a transducer-based CFG (Context Free Transducer?) that will
        transduce lattices into lattices.
        """
        if self.has_epsilon:
            raise ValueError("cannot handle CFGs with epsilons, yet")
        def senderator(start_non_terminals, force_tree):
            sentential_state_id = None
            start_state_id = 0

            recognizer_states = list()
            recognizer_id = dict()
            def get_recognizer_state(is_sentential, earley_seeders):
                is_sentential =  bool(is_sentential)
                earley_seeders = frozendict_of_set(earley_seeders)
                recog_spec = is_sentential, earley_seeders
                recog_id = recognizer_id.get(recog_spec)
                if recog_id is None or force_tree:
                    recog_id = recognizer_id[recog_spec] = len(recognizer_states)
                    recognizer_states.append(RecognizerState(*recog_spec))
                return recog_id, recognizer_states[recog_id]

            # for the first yield, is_sentential means the empty sequence is a
            # valid sentence
            is_sentential = bool(set(start_non_terminals) & self.nullable_non_terminals)
            
            # seeding items given the starting non_terminals
            start_items = set(start_item for non_terminal in start_non_terminals for start_item in self.initial_items_by_non_terminal[non_terminal])
            assert all(start_item.is_initial for start_item in start_items)

            earley_seeders = dict_of(set)
            for start_item in start_items:
                earley_seeders[start_item].add(sentential_state_id)

            # this loop is entered with the is_sentential and earley_seeders for
            # the state we're going to yield to the user; we see if this is a
            # new state, and if so, we flesh it out before yielding and building
            # the next earley_seeders at the bottom of the loop
            while True:
                state_id, recognizer_state = get_recognizer_state(is_sentential, earley_seeders)
                earley_seeders = recognizer_state.seeders
                #print 'get_recognizer_state:', r

                # keys of earley_parents, items we were seeded with
                #seed_items = frozenset(earley_parents)
                seed_items = frozenset(earley_seeders)                

                # only start_state_id has is_initial items, the sententials
                assert all(item.is_initial == (state_id == start_state_id) for item in seed_items)

                # set of terminals that are legal from this recog_state; we use
                # the baseclass's prediction_terminals to build the set from the
                # seed_items in earley_parents
                legal_terminals = frozenset(legal_terminal for item in seed_items for legal_terminal in self.prediction_terminals[item])

                debug = False

                if debug:
                    print '============================================'
                    print 'state_id:', state_id
                    print 'seed_items:'
                    for item in sorted(seed_items):
                        if item.rhs_symbol not in self.terminals:
                            #print ' ', item, ' ', sorted(earley_parents[item])
                            print ' ', item, ' ', sorted(earley_seeders[item])
                    print ' ', '---'
                    for item in sorted(seed_items):
                        if item.rhs_symbol in self.terminals:
                            #print ' ', item, ' ', sorted(earley_parents[item])
                            print ' ', item, ' ', sorted(earley_seeders[item])

                extenders = recognizer_state.extenders
                if not extenders:
                    # lazy, first time making these for this state

                    # mutable copy used to build extenders
                    earley_parents = dict_of(set, earley_seeders)
                    # here we implement the prediction step using the baseclass's
                    # prediction_items; this adds in all the intermediate items,
                    # with derivations starting at this state_id
                    for item in set(item for early_item in earley_parents
                                    for item in self.prediction_items[early_item]):
                        earley_parents[item].add(state_id)

                    assert all(item.is_production and not item.is_final for item in earley_parents)
                    assert set(item.rhs_symbol for item in earley_parents if item.rhs_symbol in self.terminals) == legal_terminals

                    if debug:
                        print 'earley_parents:'
                        for item in sorted(earley_parents):
                            if not item.is_initial and item.rhs_symbol not in self.terminals:
                                print ' ', item, ' ', sorted(earley_parents[item])
                        print ' ', '---'
                        for item in sorted(earley_parents):
                            if not item.is_initial and item.rhs_symbol in self.terminals:
                                print ' ', item, ' ', sorted(earley_parents[item])
                        print ' ', '---', '---'
                        for item in sorted(earley_parents):
                            if item.is_initial and item.rhs_symbol not in self.terminals:
                                print ' ', item, ' ', sorted(earley_parents[item])
                        print ' ', '---'
                        for item in sorted(earley_parents):
                            if item.is_initial and item.rhs_symbol in self.terminals:
                                print ' ', item, ' ', sorted(earley_parents[item])

                    # hmmm, we do a lot of work here... some should be done once at
                    # construction time, other work should only be done that could
                    # be used, e.g. only the scanned terminal, only shift-reachable
                    # states....

                    completeables = dict_of(set)
                    for rhs_symbol in set(item.rhs_symbol for item in earley_parents):
                        completeables[rhs_symbol].update(item for item in (self.item_completions_unshifted5[rhs_symbol] & set(earley_parents))
                                                         if not item.is_penultimate or item in seed_items)
                    if debug:
                        print 'completeables:'
                        for completeable in sorted(completeables):
                            print ' ', completeable
                            for item in sorted(completeables[completeable]):
                                print ' ', ' ', item, ' ', sorted(set(earley_parents[item]) - set([state_id] if item.is_penultimate else []))

                    for symbol, items in completeables.iteritems():
                        for item in items:
                            parents = frozenset(earley_parents[item])
                            #if not parents:
                            #    continue
                            assert parents, str(symbol) + '  ' + str(item)
                            # emprical assertion
                            assert len(parents) == 1 or (len(parents) == 2 and sentential_state_id in parents), str(parents)
                            # XXX need to deal with sententials
                            if not item.is_penultimate:
                                # simple case
                                extenders[symbol].add((item, parents))
                            else:
                                # set up to handle completions
                                assert item in seed_items
                                for parent in parents:
                                    if parent == sentential_state_id:
                                        # sentinel for sentential forms
                                        extenders[symbol].add((item, frozenset([sentential_state_id])))
                                    else:
                                        # note: reach back to an earlier state's extenders
                                        #extenders[symbol].update(recog_states[parent].extenders[item.lhs])
                                        extenders[symbol].update(recognizer_states[parent].extenders[item.lhs])

                assert set(symbol for symbol in extenders if symbol in self.terminals) == legal_terminals

                if debug:
                    print 'extenders:'
                    for symbol in sorted(extenders):
                        print ' ', symbol
                        for item, parents in sorted(extenders[symbol]):
                            print ' ', ' ', item, ' ', sorted(parents)

                    print 'legal_terminals:', ' ', ' '.join(sorted(legal_terminals))
                    print 'is_sentential:', is_sentential

                # note: raising an exception from inside a generator destroys
                # the generator due to Python's termination model of error
                # handling; so, we yield an exception when one has occured, and
                # keep doing so until we get valid arguments; this lets trivial
                # clients just raise the exception themselves, while a
                # sophisticated client can deal with it....
                exception = None
                while True:
                    #  yield info for the state we've worked on and its
                    #  legal_terminals, and get the scanned_state_id and scanned
                    #  terminal for parsing the next token
                    scanned_state_id, scanned_terminal = yield is_sentential, state_id, legal_terminals, exception

                    # validate the state_id
                    if not (start_state_id <= scanned_state_id <= len(recognizer_states) - 1):
                        exception = ValueError("expected a state_id in range (%d, %d) inclusive, got %r" % (start_state_id, len(recognizer_states) - 1, scanned_state_id,))
                        continue

                    # validate the terminal
                    check_legal_terminals = set(symbol for symbol in recognizer_states[scanned_state_id].extenders if symbol in self.terminals)
                    if not scanned_terminal in check_legal_terminals:
                        exception = ValueError("expected a terminal item in %r, got %r" % (tuple(sorted(check_legal_terminals)), scanned_terminal,))
                        continue

                    # we're good to go
                    break

                state_id = scanned_state_id
                recognizer_state = recognizer_states[scanned_state_id]
                extenders = recognizer_state.extenders
                legal_terminals = set(symbol for symbol in extenders if symbol in self.terminals)
                assert scanned_terminal in legal_terminals

##                 # global set of items that could be advanced by the scanned_terminal
##                 scanned_items = self.terminal_completions_unshifted[scanned_terminal]
                
                # scanning-type step: find set of this state's items that are
                # advanced by the scanned_terminal; intersect items in
                # terminal_completions_unshifted (for the scanned_terminal) with
                # items in earley_parents; subtle, set(earley_parents) makes a
                # set of the keys (the items), corresponding to earley_parents
##                 shifters = scanned_items & set(earley_parents)

                if debug:
                    print 'scanned_terminal:', ' ', scanned_terminal

##                     if False:
##                         print 'scanned_items:'
##                         for item in sorted(scanned_items):
##                             if not item.is_penultimate:
##                                 print ' ', item
##                         print ' ', '---'
##                         for item in sorted(scanned_items):
##                             if item.is_penultimate:
##                                 print ' ', item

##                     print 'shifters:'
##                     for item in sorted(shifters):
##                         if not item.is_penultimate:
##                             print ' ', item, ' ', sorted(earley_parents[item])
##                     print ' ', '---'
##                     for item in sorted(shifters):
##                         if item.is_penultimate:
##                             print ' ', item, ' ', sorted(earley_parents[item])


                # create spec for following recognizer state
                is_sentential = False
                earley_seeders = dict_of(set)                

                # do the token-passing work
                for item, parents in extenders[scanned_terminal]:
                    if item.is_penultimate:
                        assert len(parents) == 1 and sentential_state_id in parents
                        is_sentential = True
                    else:
                        earley_seeders[item.shifted].update(parents)

                if debug:
                    print 'earley_seeders:'
                    for item in sorted(earley_seeders.keys()):
                        if not item.is_penultimate:
                            print ' ', item, ' ', sorted(earley_seeders[item])
                    print ' ', '---'
                    for item in sorted(earley_seeders.keys()):
                        if item.is_penultimate:
                            print ' ', item, ' ', sorted(earley_seeders[item])

        # recognizer() returns the send attribute of the generator
        return senderator(start_non_terminals, force_tree).send

def _testing_CfgDecoder_1():
    """
    >>> b = CfgBuilder()
    >>> b.add_production('expr', ('term',))
    >>> b.add_production('expr', ('expr', '+', 'term',))
    >>> b.add_production('term', ('factor',))
    >>> b.add_production('term', ('term', '*', 'factor',))
    >>> b.add_production('factor', ('primary',))
    >>> b.add_production('factor', ( 'factor', '^', 'primary',))
    >>> b.add_production('primary', ('(', 'expr', ')',))
    >>> #b.add_production('primary', ('n',))
    >>> b.add_production('primary', ('number',))
    >>> b.add_production('number', ('digit',))
    >>> b.add_production('number', ('digit', 'number', ))
    >>> #b.add_production('number', ('number', 'digit',))
    >>> # b.add_production('number', ('d',))
    >>> # b.add_production('number', ('number', 'd',))
    >>> b.add_production('digit', ('0',))
    >>> b.add_production('digit', ('1',))
    >>> #b.add_production('expr', ('SS',))
    >>> #b.add_production('SS', ('a', 'b', 'B', 'c', 'C', 'd', 'E', 'e'))
    >>> #b.add_production('SS', ('1', '2', '3'))
    >>> #b.add_production('B', ('D', 'b1',))
    >>> #b.add_production('B', ('x3',))
    >>> #b.add_production('B', ('b', 'B',))
    >>> #b.add_production('D', ('d1', 'd2',))
    >>> #b.add_production('C', ('c1',))
    >>> #b.add_production('C', ('c2',))
    >>> #b.add_production('E', ('D', 'e1',))
    >>> #b.add_production('Z', ('C',))

    >>> d = CfgRecognizer(b)
    >>> rand = random.Random()
    >>> rand.seed(8)
    >>> choice = rand.choice

    >>> recognizer = d.recognizer(['expr'], force_tree=False)
    >>> is_sentential, state_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> rand.seed(8)
    >>> choice = rand.choice
    >>> sentence = list()
    >>> countdown = 25
    >>> while legal and countdown > 0:
    ...   sentence.append(choice(sorted(legal)))
    ...   is_sentential, state_id, legal, exception = recognizer((state_id, sentence[-1]))
    ...   assert exception is None
    ...   if is_sentential: print 'sentential:', ' ', ' '.join(sentence)
    ...   countdown -= 1
    sentential:   ( 1 )
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) )
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0 ^ 0
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0 ^ 0 * ( 1 )
    >>> print 'ending prefix:', ' '.join(sentence)
    ending prefix: ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0 ^ 0 * ( 1 )
    >>> print 'is_sentential:', is_sentential
    is_sentential: True
    >>> print state_id
    22

    >>> recognizer = d.recognizer(['expr'], force_tree=True)
    >>> is_sentential, state_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> rand.seed(8)
    >>> choice = rand.choice
    >>> sentence = list()
    >>> countdown = 25
    >>> while legal and countdown > 0:
    ...   sentence.append(choice(sorted(legal)))
    ...   is_sentential, state_id, legal, exception = recognizer((state_id, sentence[-1]))
    ...   assert exception is None
    ...   if is_sentential: print 'sentential:', ' ', ' '.join(sentence)
    ...   countdown -= 1
    sentential:   ( 1 )
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) )
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0 ^ 0
    sentential:   ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0 ^ 0 * ( 1 )
    >>> print 'ending prefix:', ' '.join(sentence)
    ending prefix: ( 1 ) ^ ( ( 1 * 0 + 0 + ( 1 ) ) ) * 0 ^ 0 * ( 1 )
    >>> print 'is_sentential:', is_sentential
    is_sentential: True
    >>> print state_id
    25


    >>> recognizer = d.recognizer(['number'])
    >>> is_sentential, state_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> rand.seed(8)
    >>> choice = rand.choice
    >>> sentence = list()
    >>> countdown = 12
    >>> while legal and countdown > 0:
    ...   sentence.append(choice(tuple(legal)))
    ...   is_sentential, state_id, legal, exception = recognizer((state_id, sentence[-1]))
    ...   assert exception is None
    ...   if is_sentential: print 'sentential:', ' ', ' '.join(sentence)
    ...   countdown -= 1
    sentential:   1
    sentential:   1 0
    sentential:   1 0 1
    sentential:   1 0 1 0
    sentential:   1 0 1 0 1
    sentential:   1 0 1 0 1 1
    sentential:   1 0 1 0 1 1 0
    sentential:   1 0 1 0 1 1 0 1
    sentential:   1 0 1 0 1 1 0 1 0
    sentential:   1 0 1 0 1 1 0 1 0 1
    sentential:   1 0 1 0 1 1 0 1 0 1 1
    sentential:   1 0 1 0 1 1 0 1 0 1 1 1
    >>> print 'ending prefix:', ' '.join(sentence)
    ending prefix: 1 0 1 0 1 1 0 1 0 1 1 1
    >>> print 'is_sentential:', is_sentential
    is_sentential: True

    Show how you get errors from the CFG recognizer.  Here we raise them in our
    usual doctest fashion and doctest lets us keep going, but a different
    error-savy client would have to deal as it saw fit.

    >>> assert legal
    >>> is_sentential, state_id, legal, exception = recognizer((-1, sentence[-1]))
    >>> assert exception is not None
    >>> raise exception
    Traceback (most recent call last):
      ...
    ValueError: expected a state_id in range (0, 12) inclusive, got -1

    >>> assert legal
    >>> is_sentential, state_id, legal, exception = recognizer((state_id, 'a'))
    >>> assert exception is not None
    >>> raise exception
    Traceback (most recent call last):
      ...
    ValueError: expected a terminal item in ('0', '1'), got 'a'

    It's still good to go
    >>> assert legal
    >>> is_sentential, state_id, legal, exception = recognizer((state_id, sentence[-1]))
    >>> assert exception is None
    >>> assert is_sentential

    >>> recognizer = d.recognizer(['digit'])
    >>> is_sentential, state_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> rand.seed(8)
    >>> choice = rand.choice
    >>> sentence = list()
    >>> countdown = 25
    >>> while legal and countdown > 0:
    ...   sentence.append(choice(tuple(legal)))
    ...   is_sentential, state_id, legal, exception = recognizer((state_id, sentence[-1]))
    ...   assert exception is None
    ...   if is_sentential: print 'sentential:', ' ', ' '.join(sentence)
    ...   countdown -= 1
    sentential:   1
    >>> print 'ending prefix:', ' '.join(sentence)
    ending prefix: 1
    >>> print 'is_sentential:', is_sentential
    is_sentential: True

    >>> recognizer = d.recognizer(['expr'])
    >>> is_sentential, state_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> trial = list(reversed('10+(10^1+(1*1))'))
    >>> sentence = list()
    >>> countdown = 25
    >>> while legal and trial and countdown > 0:
    ...   sentence.append(trial.pop())
    ...   is_sentential, state_id, legal, exception = recognizer((state_id, sentence[-1]))
    ...   if exception is not None: raise exception
    ...   if is_sentential: print 'sentential:', ' ', ' '.join(sentence)
    ...   countdown -= 1
    sentential:   1
    sentential:   1 0
    sentential:   1 0 + ( 1 0 ^ 1 + ( 1 * 1 ) )
    >>> print 'ending prefix:', ' '.join(sentence)
    ending prefix: 1 0 + ( 1 0 ^ 1 + ( 1 * 1 ) )
    >>> print 'is_sentential:', is_sentential
    is_sentential: True

    Pattern for a breadth-first enumeration of all strings in the language
    >>> recognizer = d.recognizer(['expr'])
    >>> is_sentential, to_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> breadth_first = deque()
    >>> for symbol in sorted(legal):
    ...   breadth_first.appendleft(((to_id, is_sentential), symbol))
    >>> links = set()
    >>> countdown = 50
    >>> while breadth_first and countdown > 0:
    ...   (from_id, was_sentential), symbol = breadth_first.pop()
    ...   is_sentential, to_id, legal, exception = recognizer((from_id, symbol))
    ...   assert exception is None
    ...   links.add(((from_id, was_sentential), (to_id, is_sentential), symbol))
    ...   for symbol in sorted(legal):
    ...     breadth_first.appendleft(((to_id, is_sentential), symbol))
    ...   countdown -= 1
    >>> for (from_id, was_sentential), (to_id, is_sentential), symbol in sorted(links):
    ...   print from_id, 'o' if was_sentential else ' ', ' ', to_id, 'o' if is_sentential else ' ', '  ', symbol
    0     1      (
    0     2 o    0
    0     2 o    1
    1     3      (
    1     4      0
    1     4      1
    2 o   5      *
    2 o   6      +
    2 o   7 o    0
    2 o   7 o    1
    2 o   8      ^
    3     9      (
    3     10      0
    3     10      1
    4     11 o    )
    4     12      *
    4     13      +
    4     14      0
    4     14      1
    4     15      ^
    5     16      (
    5     17 o    0
    5     17 o    1
    6     18      (
    6     19 o    0
    6     19 o    1
    7 o   5      *
    7 o   6      +
    7 o   8      ^
    7 o   20 o    0
    7 o   20 o    1
    8     21      (
    8     22 o    0
    8     22 o    1

    >>> g = FrozenGraph(make_initialized_set_graph_builder(links))
    >>> None and g.dot_display(node_label_callback=None,
    ...               node_attributes_callback=lambda (n, s), *_: ['shape=doublecircle' if s else 'shape=circle', 'style=bold' if s else 'style=normal', 'height=0.2', 'width=0.2'],
    ...               arc_attributes_callback=lambda a: ['sametail="tailtag"'],
    ...               globals=['rankdir=LR', 'ranksep=0.25;', 'fontsize=20;'])

    Pattern for a breadth-first enumeration of all strings in the language
    >>> b2 = CfgBuilder()
    >>> b2.add_production('expr', ('term',))
    >>> b2.add_production('expr', ('expr', '+', 'term',))
    >>> b2.add_production('term', ('factor',))
    >>> b2.add_production('term', ('term', '*', 'factor',))
    >>> b2.add_production('factor', ('primary',))
    >>> b2.add_production('factor', ( 'factor', '^', 'primary',))
    >>> b2.add_production('primary', ('(', 'expr', ')',))
    >>> #b2.add_production('primary', ('n',))
    >>> b2.add_production('primary', ('number',))
    >>> b2.add_production('number', ('digit',))
    >>> b2.add_production('number', ('digit', 'number', ))
    >>> #b2.add_production('number', ('number', 'digit',))
    >>> # b2.add_production('number', ('d',))
    >>> # b2.add_production('number', ('number', 'd',))
    >>> b2.add_production('digit', ('0',))
    >>> b2.add_production('digit', ('1',))
    >>> for lhs, rhs in b2.productions.iter_flat:
    ...   print lhs, ':', rhs
    digit : ('0',)
    digit : ('1',)
    expr : ('expr', '+', 'term')
    expr : ('term',)
    factor : ('factor', '^', 'primary')
    factor : ('primary',)
    number : ('digit',)
    number : ('digit', 'number')
    primary : ('(', 'expr', ')')
    primary : ('number',)
    term : ('factor',)
    term : ('term', '*', 'factor')

    >>> d2 = CfgRecognizer(b2)
    >>> rand.seed(8)
    >>> recognizer = d2.recognizer(['expr'])
    >>> seen = set()
    >>> is_sentential, to_id, legal, exception = recognizer(None)
    >>> assert exception is None
    >>> breadth_first = deque()
    >>> if to_id not in seen:
    ...   seen.add(to_id)
    ...   for symbol in sorted(legal):
    ...     breadth_first.appendleft(((to_id, is_sentential), symbol))
    >>> links = set()
    >>> countdown = 35
    >>> countdown
    35
    >>> while breadth_first and countdown > 0:
    ...   (from_id, was_sentential), symbol = breadth_first.pop()
    ...   is_sentential, to_id, legal, exception = recognizer((from_id, symbol))
    ...   assert exception is None
    ...   links.add(((from_id, was_sentential), (to_id, is_sentential), symbol))
    ...   if to_id not in seen:
    ...     seen.add(to_id)
    ...     for symbol in sorted(legal):
    ...       breadth_first.appendleft(((to_id, is_sentential), symbol))
    ...   countdown -= 1
    >>> for (from_id, was_sentential), (to_id, is_sentential), symbol in sorted(links):
    ...   print from_id, 'o' if was_sentential else ' ', ' ', to_id, 'o' if is_sentential else ' ', '  ', symbol
    0     1      (
    0     2 o    0
    0     2 o    1
    1     3      (
    1     4      0
    1     4      1
    2 o   5      *
    2 o   6      +
    2 o   7 o    0
    2 o   7 o    1
    2 o   8      ^
    3     9      (
    3     10      0
    3     10      1
    4     11 o    )
    4     12      *
    4     13      +
    4     14      0
    4     14      1
    4     15      ^
    5     16      (
    5     17 o    0
    5     17 o    1
    6     18      (
    6     19 o    0
    6     19 o    1
    7 o   5      *
    7 o   6      +
    7 o   8      ^
    7 o   20 o    0
    7 o   20 o    1
    8     21      (
    8     22 o    0
    8     22 o    1
    9     23      (
    >>> g = FrozenGraph(make_initialized_set_graph_builder(links))
    >>> None and g.dot_display(node_label_callback=None,
    ...               node_attributes_callback=lambda (n, s), *_: ['shape=doublecircle' if s else 'shape=circle', 'style=bold' if s else 'style=normal', 'height=0.2', 'width=0.2'],
    ...               #arc_attributes_callback=lambda a: ['sametail="tailtag"'],
    ...               globals=['rankdir=LR', 'ranksep=0.25;', 'fontsize=20;'])

    >>> d.dump()
    has_epsilon: False
    prediction_items:
      expr ( . expr + term )
        digit ( . 0 )
        digit ( . 1 )
        expr ( . expr + term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
      expr ( expr + . term )
        digit ( . 0 )
        digit ( . 1 )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
      expr ( . term )
        digit ( . 0 )
        digit ( . 1 )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
      factor ( . factor ^ primary )
        digit ( . 0 )
        digit ( . 1 )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
      factor ( factor ^ . primary )
        digit ( . 0 )
        digit ( . 1 )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
      factor ( . primary )
        digit ( . 0 )
        digit ( . 1 )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
      number ( . digit )
        digit ( . 0 )
        digit ( . 1 )
      number ( . digit number )
        digit ( . 0 )
        digit ( . 1 )
      number ( digit . number )
        digit ( . 0 )
        digit ( . 1 )
        number ( . digit )
        number ( . digit number )
      primary ( ( . expr ) )
        digit ( . 0 )
        digit ( . 1 )
        expr ( . expr + term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
      primary ( . number )
        digit ( . 0 )
        digit ( . 1 )
        number ( . digit )
        number ( . digit number )
      term ( . factor )
        digit ( . 0 )
        digit ( . 1 )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
      term ( . term * factor )
        digit ( . 0 )
        digit ( . 1 )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
      term ( term * . factor )
        digit ( . 0 )
        digit ( . 1 )
        factor ( . factor ^ primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        primary ( . ( expr ) )
        primary ( . number )
    prediction_terminals:
      digit ( . 0 )
        0
      digit ( . 1 )
        1
      expr ( . expr + term )
        (
        0
        1
      expr ( expr . + term )
        +
      expr ( expr + . term )
        (
        0
        1
      expr ( . term )
        (
        0
        1
      factor ( . factor ^ primary )
        (
        0
        1
      factor ( factor . ^ primary )
        ^
      factor ( factor ^ . primary )
        (
        0
        1
      factor ( . primary )
        (
        0
        1
      number ( . digit )
        0
        1
      number ( . digit number )
        0
        1
      number ( digit . number )
        0
        1
      primary ( . ( expr ) )
        (
      primary ( ( . expr ) )
        (
        0
        1
      primary ( ( expr . ) )
        )
      primary ( . number )
        0
        1
      term ( . factor )
        (
        0
        1
      term ( . term * factor )
        (
        0
        1
      term ( term . * factor )
        *
      term ( term * . factor )
        (
        0
        1
    item_completions_unshifted:
      digit ( . 0 )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      digit ( . 1 )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      expr ( expr + . term )
        expr ( . expr + term )
        primary ( ( . expr ) )
      expr ( . term )
        expr ( . expr + term )
        primary ( ( . expr ) )
      factor ( factor ^ . primary )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        primary ( ( . expr ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      factor ( . primary )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        primary ( ( . expr ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      number ( . digit )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      number ( digit . number )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      primary ( ( expr . ) )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        primary ( ( . expr ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      primary ( . number )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        primary ( ( . expr ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      term ( . factor )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        primary ( ( . expr ) )
        term ( . term * factor )
      term ( term * . factor )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        primary ( ( . expr ) )
        term ( . term * factor )
    item_completions_unshifted5:
      (
        primary ( . ( expr ) )
      )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        primary ( ( . expr ) )
        primary ( ( expr . ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      *
        term ( term . * factor )
      +
        expr ( expr . + term )
      0
        digit ( . 0 )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      1
        digit ( . 1 )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      ^
        factor ( factor . ^ primary )
      digit
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      expr
        expr ( . expr + term )
        primary ( ( . expr ) )
      factor
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        primary ( ( . expr ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      number
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      primary
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        primary ( ( . expr ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      term
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        primary ( ( . expr ) )
        term ( . term * factor )
    terminal_completions_unshifted:
      (
        primary ( . ( expr ) )
      )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        primary ( ( . expr ) )
        primary ( ( expr . ) )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      *
        term ( term . * factor )
      +
        expr ( expr . + term )
      0
        digit ( . 0 )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      1
        digit ( . 1 )
        expr ( . expr + term )
        expr ( expr + . term )
        expr ( . term )
        factor ( . factor ^ primary )
        factor ( factor ^ . primary )
        factor ( . primary )
        number ( . digit )
        number ( . digit number )
        number ( digit . number )
        primary ( ( . expr ) )
        primary ( . number )
        term ( . factor )
        term ( . term * factor )
        term ( term * . factor )
      ^
        factor ( factor . ^ primary )
    """


class FrozenCfg(object):
    """
    >>> b = CfgBuilder()
    >>> b.add_production('A', ('x', 'y', 'z'))
    >>> b.add_production('B', ())
    >>> b.add_production('B', ('b',))
    >>> b.add_production('B', ('B', 'b'))
    >>> b.add_production('C', ('A',))
    >>> b.add_production('C', ('B',))
    >>> cfg = FrozenCfg(b)
    >>> cfg = FrozenCfg(b, 'x')
    Traceback (most recent call last):
        ...
    ValueError: expected (optional) start to be in the set of non_terminals, but got 'x'
    >>> cfg = FrozenCfg(b, 'B')
    >>> for lhs, rhs in cfg: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
    'A' := 'x' 'y' 'z'
    'B' := 
    'B' := 'B' 'b'
    'B' := 'b'
    'C' := 'A'
    'C' := 'B'
    """
    def __init__(self, builder, start=None):
        """
        >>> cfg = FrozenCfg(CfgBuilder())
        >>> cfg.size
        0
        """
        check_instance(CfgBuilder, builder)
        self.start = start
        self._size = builder.productions.size
        productions = self.productions = frozendict((lhs, frozenset(rhs_set)) for lhs, rhs_set in builder.productions.iteritems())
        non_terminals = self._non_terminals = frozenset(productions)
        if start is not None and start not in non_terminals:
            raise ValueError("expected (optional) start to be in the set of non_terminals, but got %r" %(start,))
        symbols = self._symbols = non_terminals | frozenset(rhs_token for rhs_set in productions.itervalues() for rhs in rhs_set for rhs_token in rhs)
        terminals = self._terminals = symbols - non_terminals

        self._num_productions = reduce(operator.add, (len(rhs_set) for rhs_set in productions.itervalues()), 0)
        #print 'non_terminals:', non_terminals
        #print 'terminals:', terminals
        #print 'symbols:', symbols
        nullables = self.nullables = frozenset(iter_nullable(productions))
        #print 'nullables:', nullables
        
        self._verify()
        
    def _verify(self):
        assert self.size >= len(self.non_terminals) + len(self.terminals)

    def __iter__(self):
        """
        Returns a generator that yields a pair corresponding to each production
        in the CFG.  The first item in each pair is the non-terminal
        left-hand-side for the production, the second item is sequence of
        non-terminals and terminals making up the right-hand-side of the
        production.
        """
        productions = self.productions
        return ((lhs, rhs) for lhs in sorted(productions.keys()) for rhs in sorted(productions[lhs]))

    @property
    def non_terminals(self):
        """
        A frozenset of the non-terminals in the grammar.
        """
        return self._non_terminals
        
    @property
    def terminals(self):
        """
        A frozenset of the terminals in the grammar.
        """
        return self._terminals

    @property
    def num_productions(self):
        """
        The number of productions in the grammar.
        """
        return self._num_productions
    
    @property
    def size(self):
        """
        The size of the grammar.

        We follow Robert Moore in calculating the size of the grammar: the size
        is the number of non-terminal symbols plus the sum of the lengths of the
        right-hand-side sequences over all the productions in the grammar.  By
        counting each non-terminal just once, instead of once for each of its
        productions, this size statistic more closely tracks storage
        requirements of actual implementations of grammar structures.  An empty
        right-hand-side sequence (epsilon) is counted has having length one.
        """
        return self._size

    def make_no_epsilon_cfg(self):
        """
        Return a FrozenCfg that is equivalent to self but which
        contains no epsilon productions.
        
        >>> b = CfgBuilder()
        >>> b.add_production('A', ('x', 'y', 'z'))
        >>> b.add_production('B', ())
        >>> b.add_production('B', ('b',))
        >>> b.add_production('B', ('B', 'b'))
        >>> b.add_production('C', ('A',))
        >>> b.add_production('C', ('B',))
        >>> b.add_production('D', ('(', 'C', ')'))
        >>> b.add_production('E', ('B', '(', 'C', ')'))
        >>> b.add_production('F', ('B', '(', 'C', ')', 'C'))
        >>> b.add_production('F', ('B', '(', 'C', ')', 'C', 'C'))
        >>> b.add_production('G', ('B',))
        >>> b.add_production('G', ('B', 'B', 'B'))
        >>> b.add_production('S', ('A',))
        >>> b.add_production('S', ('A', 'F'))
        >>> b.add_production('S', ('A', 'C'))
        >>> cfg = FrozenCfg(b)
        >>> cfg = FrozenCfg(b, 'S')
        >>> for lhs, rhs in cfg: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'x' 'y' 'z'
        'B' := 
        'B' := 'B' 'b'
        'B' := 'b'
        'C' := 'A'
        'C' := 'B'
        'D' := '(' 'C' ')'
        'E' := 'B' '(' 'C' ')'
        'F' := 'B' '(' 'C' ')' 'C'
        'F' := 'B' '(' 'C' ')' 'C' 'C'
        'G' := 'B'
        'G' := 'B' 'B' 'B'
        'S' := 'A'
        'S' := 'A' 'C'
        'S' := 'A' 'F'
        >>> cfg1 = cfg.make_no_epsilon_cfg()
        >>> for lhs, rhs in cfg1: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'x' 'y' 'z'
        'B_er1' := 'B_er1' 'b'
        'B_er1' := 'b'
        'C_er0' := 'A'
        'C_er0' := 'B_er1'
        'D' := '(' ')'
        'D' := '(' 'C_er0' ')'
        'E' := '(' ')'
        'E' := '(' 'C_er0' ')'
        'E' := 'B_er1' '(' ')'
        'E' := 'B_er1' '(' 'C_er0' ')'
        'F' := '(' ')'
        'F' := '(' ')' 'C_er0'
        'F' := '(' ')' 'C_er0' 'C_er0'
        'F' := '(' 'C_er0' ')' 'C_er0'
        'F' := '(' 'C_er0' ')' 'C_er0' 'C_er0'
        'F' := 'B_er1' '(' ')'
        'F' := 'B_er1' '(' ')' 'C_er0'
        'F' := 'B_er1' '(' ')' 'C_er0' 'C_er0'
        'F' := 'B_er1' '(' 'C_er0' ')'
        'F' := 'B_er1' '(' 'C_er0' ')' 'C_er0'
        'F' := 'B_er1' '(' 'C_er0' ')' 'C_er0' 'C_er0'
        'G_er2' := 'B_er1'
        'G_er2' := 'B_er1' 'B_er1'
        'G_er2' := 'B_er1' 'B_er1' 'B_er1'
        'S' := 'A'
        'S' := 'A' 'C_er0'
        'S' := 'A' 'F'
        """

        # XXX in the new CFG, none of the set of nullable non_terminals from the
        # old CFG can be used as a start state!  We will need a mechanism to
        # enforce this, e.g. by having hidden productions....  Or we could adopt
        # Chomsky's normalization rules that allow epsilons only on the start
        # state which would allow nullable non_terminals to remain; this would
        # be more general, but would be complicated due to our not requiring a
        # start state to be specified....  Or we could let the client maintain a
        # set of "special" non-terminals that would have method-specific
        # semantics, e.g. start states, don't elimiminate states when reducing,
        # automaticaly introduced states, etc

        if self.start in self.nullables:
            raise ValueError("start appears in the set of nullable non_terminals: start %r" % (self.start,))

        if False:

            def make_non_terminal_generator():
                count = 0
                prefix = yield
                while True:
                    prefix = yield str(prefix) + '_$E' + str(count)
                    count += 1
            non_terminal_generator = make_non_terminal_generator().send
            non_terminal_generator(None)
            #print tuple(non_terminal_generator('expr'+str(x*2+1)) for x in xrange(10))

            class new_symbol_dict(dict):
                def __missing__(self, key):
                    result = self[key] = non_terminal_generator(str(key))
                    return result
            nullables_map = new_symbol_dict()
            #print nullables_map['a']
            #print nullables_map['a']
            #print nullables_map['b']
            nullables_map.clear()

            # XXX way too much cleverness
            # use tuple to run the generator for the side effect of populating the dictionary
            tuple(nullables_map[nullable] for nullable in self.nullables)

        # XXX not taking care to avoid duplicate symbols; we need to resolve
        # some deeper issues about CFGs to solve this problem, e.g. a nullable
        # start state, multiple passes of symbol-synthesizing logic, etc
        nullables_map = frozendict((lhs, str(lhs) + '_er' + str(index)) for index, lhs in enumerate(self.nullables))
        assert not (frozenset(nullables_map.itervalues()) & self._symbols), 'synthetic symbol(s) collide with non_terminals: ' + ' '.join(repr(x) for x in (frozenset(nullables_map.itervalues()) & self.symbols))
        builder = CfgBuilder()
        for lhs, rhs in self:
            if not rhs:
                # drop the explicit epsilon productions
                assert lhs in nullables_map
                continue

            lhs = nullables_map.get(lhs, lhs)
            nullable_indices = tuple(i for i, val in enumerate(rhs) if val in nullables_map)
            rhs = tuple(nullables_map.get(rhs_token, rhs_token) for rhs_token in rhs)
            builder.add_production(lhs, rhs)

            if not nullable_indices:
                continue
            #print 'nullable_indices:', lhs, ':=', rhs, '', nullable_indices

            # XXX we could be more efficient by introducing new non-terminals to
            # eliminate some of the combinatorics; this would then leave only
            # the case of adjacent nullable non-terminals which would require a
            # combinatoric generation of alternative productions
            #
            # deal with combinatorics of making the rhss which exclude
            # one or more of the nullable non-terminals
            subsets = (nullable_indices[start:stop]
                       for start in xrange(len(nullable_indices))
                       for stop in xrange(start+1, len(nullable_indices)+1))
            #print 'subsets:', tuple(sorted(subsets))
            for subset in subsets:
                # note: we're using "in" syntax on subset, which is a tuple not
                # a set; but we expect subset to be small and so not worth the
                # overhead of constructing a set for the membership checks
                new_rhs = tuple(rhs_token for index, rhs_token in enumerate(rhs) if index not in subset)
                # new_rhs will be empty (once) for a rhs that consists only of
                # nullable non-terminals
                if new_rhs:
                    builder.add_production(lhs, new_rhs)

        return FrozenCfg(builder, self.start)

    def make_left_factored_cfg(self):
        """
        Return a FrozenCfg that is equivalent to self but for which the
        productions for a non_terminal are all left factored.  This means that
        there is no prefix sharing across the productions for a given
        non-terminal.  Another way to state this is that there will be only one
        production for each direct left corner of each non-terminal.

        >>> b = CfgBuilder()
        >>> b.add_production('A', ('x', 'y', 'z'))
        >>> b.add_production('A', ('x', 'y', 'q'))
        >>> b.add_production('A', ('l', 'n', 'z'))
        >>> b.add_production('A', ('l', 'n', 'q'))
        >>> b.add_production('A', ('l',))
        >>> b.add_production('A', ('x', 'z', 'z'))
        >>> b.add_production('A', ('x', 'q', 'q'))
        >>> b.add_production('B', ())
        >>> b.add_production('B', ('b',))
        >>> b.add_production('B', ('C', 'b'))
        >>> b.add_production('C', ('A',))
        >>> b.add_production('C', ('D',))
        >>> b.add_production('C', ('E', 'e'))
        >>> b.add_production('D', ('B',))
        >>> b.add_production('E', ('B', 'f'))
        >>> b.add_production('S', ('A', 'C', 'x'))
        >>> b.add_production('S', ('B', 'C', 'x'))
        >>> cfg = FrozenCfg(b, 'S')
        >>> for lhs, rhs in cfg: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'l'
        'A' := 'l' 'n' 'q'
        'A' := 'l' 'n' 'z'
        'A' := 'x' 'q' 'q'
        'A' := 'x' 'y' 'q'
        'A' := 'x' 'y' 'z'
        'A' := 'x' 'z' 'z'
        'B' := 
        'B' := 'C' 'b'
        'B' := 'b'
        'C' := 'A'
        'C' := 'D'
        'C' := 'E' 'e'
        'D' := 'B'
        'E' := 'B' 'f'
        'S' := 'A' 'C' 'x'
        'S' := 'B' 'C' 'x'
        >>> cfg1 = cfg.make_left_factored_cfg()
        >>> for lhs, rhs in cfg1: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'l' 'A_lf3'
        'A' := 'x' 'A_lf2'
        'A_lf0' := 'q'
        'A_lf0' := 'z'
        'A_lf1' := 'q'
        'A_lf1' := 'z'
        'A_lf2' := 'q' 'q'
        'A_lf2' := 'y' 'A_lf0'
        'A_lf2' := 'z' 'z'
        'A_lf3' := 
        'A_lf3' := 'n' 'A_lf1'
        'B' := 
        'B' := 'C' 'b'
        'B' := 'b'
        'C' := 'A'
        'C' := 'D'
        'C' := 'E' 'e'
        'D' := 'B'
        'E' := 'B' 'f'
        'S' := 'A' 'C' 'x'
        'S' := 'B' 'C' 'x'
        >>> cfg2 = cfg1.make_no_epsilon_cfg()
        >>> for lhs, rhs in cfg2: print str(lhs), ': ', '  '.join(str(rhs_token) for rhs_token in rhs)
        A :  l
        A :  l  A_lf3_er2
        A :  x  A_lf2
        A_lf0 :  q
        A_lf0 :  z
        A_lf1 :  q
        A_lf1 :  z
        A_lf2 :  q  q
        A_lf2 :  y  A_lf0
        A_lf2 :  z  z
        A_lf3_er2 :  n  A_lf1
        B_er1 :  C_er0  b
        B_er1 :  b
        C_er0 :  A
        C_er0 :  D_er3
        C_er0 :  E  e
        D_er3 :  B_er1
        E :  B_er1  f
        E :  f
        S :  A  C_er0  x
        S :  A  x
        S :  B_er1  C_er0  x
        S :  B_er1  x
        S :  C_er0  x
        S :  x
        >>> cfg3 = cfg2.make_left_factored_cfg()
        >>> for lhs, rhs in cfg3: print str(lhs), ': ', '  '.join(str(rhs_token) for rhs_token in rhs)
        A :  l  A_lf0_lf0
        A :  x  A_lf2
        A_lf0 :  q
        A_lf0 :  z
        A_lf0_lf0 :  
        A_lf0_lf0 :  A_lf3_er2
        A_lf1 :  q
        A_lf1 :  z
        A_lf2 :  q  q
        A_lf2 :  y  A_lf0
        A_lf2 :  z  z
        A_lf3_er2 :  n  A_lf1
        B_er1 :  C_er0  b
        B_er1 :  b
        C_er0 :  A
        C_er0 :  D_er3
        C_er0 :  E  e
        D_er3 :  B_er1
        E :  B_er1  f
        E :  f
        S :  A  S_lf0
        S :  B_er1  S_lf1
        S :  C_er0  x
        S :  x
        S_lf0 :  C_er0  x
        S_lf0 :  x
        S_lf1 :  C_er0  x
        S_lf1 :  x
        >>> cfg4 = cfg3.make_no_epsilon_cfg()
        >>> for lhs, rhs in cfg4: print str(lhs), ': ', '  '.join(str(rhs_token) for rhs_token in rhs)
        A :  l
        A :  l  A_lf0_lf0_er0
        A :  x  A_lf2
        A_lf0 :  q
        A_lf0 :  z
        A_lf0_lf0_er0 :  A_lf3_er2
        A_lf1 :  q
        A_lf1 :  z
        A_lf2 :  q  q
        A_lf2 :  y  A_lf0
        A_lf2 :  z  z
        A_lf3_er2 :  n  A_lf1
        B_er1 :  C_er0  b
        B_er1 :  b
        C_er0 :  A
        C_er0 :  D_er3
        C_er0 :  E  e
        D_er3 :  B_er1
        E :  B_er1  f
        E :  f
        S :  A  S_lf0
        S :  B_er1  S_lf1
        S :  C_er0  x
        S :  x
        S_lf0 :  C_er0  x
        S_lf0 :  x
        S_lf1 :  C_er0  x
        S_lf1 :  x
        >>> cfg5 = cfg4.make_left_factored_cfg()
        >>> for lhs, rhs in cfg5: print str(lhs), ': ', '  '.join(str(rhs_token) for rhs_token in rhs)
        A :  l  A_lf0_lf0
        A :  x  A_lf2
        A_lf0 :  q
        A_lf0 :  z
        A_lf0_lf0 :  
        A_lf0_lf0 :  A_lf0_lf0_er0
        A_lf0_lf0_er0 :  A_lf3_er2
        A_lf1 :  q
        A_lf1 :  z
        A_lf2 :  q  q
        A_lf2 :  y  A_lf0
        A_lf2 :  z  z
        A_lf3_er2 :  n  A_lf1
        B_er1 :  C_er0  b
        B_er1 :  b
        C_er0 :  A
        C_er0 :  D_er3
        C_er0 :  E  e
        D_er3 :  B_er1
        E :  B_er1  f
        E :  f
        S :  A  S_lf0
        S :  B_er1  S_lf1
        S :  C_er0  x
        S :  x
        S_lf0 :  C_er0  x
        S_lf0 :  x
        S_lf1 :  C_er0  x
        S_lf1 :  x
        >>> cfg.size, cfg1.size, cfg2.size, cfg3.size, cfg4.size, cfg5.size
        (42, 44, 51, 55, 55, 57)
        >>> cfg.num_productions, cfg1.num_productions, cfg2.num_productions, cfg3.num_productions, cfg4.num_productions, cfg5.num_productions
        (17, 21, 25, 28, 28, 29)
        """

        # XXX we should have an option to not introduce new epsilon productions

        builder = CfgBuilder()        
        for lhs, rhs_set in self.productions.iteritems():
            assert len(rhs_set) >= 1
            if len(rhs_set) == 1:
                # nothing to do for a single-symbol production
                builder.update_production(lhs, rhs_set)
                continue

            rhs_set = set(rhs_set)
            new_non_terminal_index = 0

            # Note: this is a simple n^2 algorithm where n is the sum of the
            # lengths of the production's rhss.  A tree-based algorithm could be
            # more efficient for a non-terminal with numerous long productions,
            # but it would rarely be of actual benefit and it would be
            # considerably more complicated.

            # start from maximum possible length of a shared left-prefix for
            # these productions
            prefix_length = max(len(x) for x in rhs_set)
            while prefix_length >= 1:
                # create a dictionary of sets of productions based on their
                # shared prefix
                prefix_seqs = dict_of_set()
                for rhs in rhs_set:
                    # Note: using >= means we will generate epsilon productions;
                    # using > would prevent creating epsilon productions, but
                    # would leave some duplicate prefixes....  This should
                    # perhaps be controlled by an argument to this method.
                    if len(rhs) >= prefix_length:
                        prefix_seqs[rhs[:prefix_length]].add(rhs)
                # find a largest set
                max_seq_count = 1
                for prefix, seqs in prefix_seqs.iteritems():
                    if len(seqs) > max_seq_count:
                        max_seq_count = len(seqs)
                        best_seqs = seqs
                        best_prefix = prefix
                if max_seq_count == 1:
                    # no (more) prefix sharing for this prefix_length
                    prefix_length -= 1
                    continue

                #print 'max_seq_count', max_seq_count, ' best_prefix', best_prefix, ' best_seqs', best_seqs

                # rewrite grammar by adding a new non_terminal and adjusting the
                # current rhs_set
                
                # XXX we are not taking good care to avoid duplicate symbols; we
                # need to resolve some deeper issues about CFGs to solve this
                # recuring problem, e.g. a nullable start state, or
                # distinguished states, multiple passes of symbol-synthesizing
                # logic, etc
                new_non_terminal = str(lhs)
                while new_non_terminal in self.productions:
                    new_non_terminal += '_lf' + str(new_non_terminal_index)
                new_non_terminal_index += 1
                rhs_set.add(best_prefix + (new_non_terminal,))
                for seq in best_seqs:
                    rhs_set.remove(seq)
                    assert len(seq) >= prefix_length
                    builder.add_production(new_non_terminal, seq[prefix_length:])
                # note: we don't decrease prefix_length here because there could
                # have been more than one set of productions sharing prefixes of
                # the given prefix_length, so we need to try again with the
                # current prefix_length


            # inclue the (rewritten) productions for this lhs
            builder.update_production(lhs, rhs_set)

        return FrozenCfg(builder, self.start)


    def make_nlrg_cfg(self):
        """
        Return a FrozenCfg that is equivalent to self but for which the
        non-left-recursive productions for each non-terminal are grouped into a
        new non-terminal; this follows Robert Moore's
        non-left-recursion-grouping (NLRG) algorithm

        >>> b = CfgBuilder()
        >>> b.add_production('A', ('x', 'y', 'z'))
        >>> b.add_production('A', ('x', 'y', 'q'))
        >>> b.add_production('A', ('l', 'n', 'z'))
        >>> b.add_production('A', ('l', 'n', 'q'))
        >>> b.add_production('A', ('l',))
        >>> b.add_production('A', ('x', 'z', 'z'))
        >>> b.add_production('A', ('x', 'q', 'q'))
        >>> b.add_production('B', ())
        >>> b.add_production('B', ('b',))
        >>> b.add_production('B', ('C', 'b'))
        >>> b.add_production('C', ('A',))
        >>> b.add_production('C', ('D',))
        >>> b.add_production('C', ('E', 'e'))
        >>> b.add_production('D', ('B',))
        >>> b.add_production('E', ('B', 'f'))
        >>> b.add_production('S', ('A', 'C', 'x'))
        >>> b.add_production('S', ('B', 'C', 'x'))
        >>> cfg = FrozenCfg(b, 'S')
        >>> for lhs, rhs in cfg: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'l'
        'A' := 'l' 'n' 'q'
        'A' := 'l' 'n' 'z'
        'A' := 'x' 'q' 'q'
        'A' := 'x' 'y' 'q'
        'A' := 'x' 'y' 'z'
        'A' := 'x' 'z' 'z'
        'B' := 
        'B' := 'C' 'b'
        'B' := 'b'
        'C' := 'A'
        'C' := 'D'
        'C' := 'E' 'e'
        'D' := 'B'
        'E' := 'B' 'f'
        'S' := 'A' 'C' 'x'
        'S' := 'B' 'C' 'x'
        >>> cfg1 = cfg.make_nlrg_cfg()
        >>> for lhs, rhs in cfg1: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'l'
        'A' := 'l' 'n' 'q'
        'A' := 'l' 'n' 'z'
        'A' := 'x' 'q' 'q'
        'A' := 'x' 'y' 'q'
        'A' := 'x' 'y' 'z'
        'A' := 'x' 'z' 'z'
        'B' := 
        'B' := 'C' 'b'
        'B' := 'b'
        'C' := 'C_nlg'
        'C' := 'D'
        'C_nlg' := 'A'
        'C_nlg' := 'E' 'e'
        'D' := 'B'
        'E' := 'B' 'f'
        'S' := 'A' 'C' 'x'
        'S' := 'B' 'C' 'x'
        """

        nullables = self.nullables
        builder = CfgBuilder()        
        for lhs, rhs_set in self.productions.iteritems():
            if lhs not in nullables:
                builder.update_production(lhs, rhs_set)
                continue
            # note: explicit exclusion of empty rhs (epsilons)
            nlrg = set(rhs for rhs in rhs_set if rhs and rhs[0] not in nullables)
            if len(nlrg) < 2:
                builder.update_production(lhs, rhs_set)
                continue

            rhs_set = set(rhs_set)
            new_non_terminal = str(lhs)
            while new_non_terminal in self.productions:
                new_non_terminal += '_nlg'
            rhs_set.add((new_non_terminal,))
            for rhs in nlrg:
                rhs_set.remove(rhs)
                builder.add_production(new_non_terminal, rhs)
            builder.update_production(lhs, rhs_set)

        return FrozenCfg(builder, self.start)


    def make_no_left_recursion_cfg(self):
        """
        Return a FrozenCfg that is equivalent to self but which contains no
        left-recursive production chains.

        >>> b = CfgBuilder()
        >>> b.add_production('A', ('x', 'y', 'z'))
        >>> b.add_production('B', ('b',))
        >>> b.add_production('B', ('C', 'b'))
        >>> b.add_production('C', ('A',))
        >>> b.add_production('C', ('D',))
        >>> b.add_production('C', ('E', 'e'))
        >>> b.add_production('D', ('B',))
        >>> b.add_production('E', ('B', 'f'))
        >>> b.add_production('S', ('A', 'C'))
        >>> b.add_production('S', ('B', 'C'))
        >>> cfg = FrozenCfg(b, 'S')
        >>> for lhs, rhs in cfg: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'A' := 'x' 'y' 'z'
        'B' := 'C' 'b'
        'B' := 'b'
        'C' := 'A'
        'C' := 'D'
        'C' := 'E' 'e'
        'D' := 'B'
        'E' := 'B' 'f'
        'S' := 'A' 'C'
        'S' := 'B' 'C'
        >>> cfg1 = cfg.make_no_left_recursion_cfg()
        >>> for lhs, rhs in cfg1: print repr(lhs), ':=', ' '.join(repr(rhs_token) for rhs_token in rhs)
        'S' := 
        """

        # XXX this is a start of a naive implementation which suffers from
        # exponential growth of some grammars; a (much) better algorithm exists,
        # "Removing Left Recursion from Context-Free Grammars" by Robert C Moore
        # of Microsoft; it just needs to be implemented

        # XXX this restriction could be relaxed
        if self.nullables:
            raise ValueError("expected a CFG with no nullable non_terminals, but these are nullable: %s:" %(' '.join(repr(x) for x in self.nullables),))

        for lhs, rhsset in self.productions.iteritems():
            assert rhsset
            lhset = set()
            lhchain = list()
            lhstack = list(rhsset)
            while lhstack:
                rhs = lhstack.pop()
                assert rhs
                rhs0 = rhs[0]
                if rhs0 in self.terminals:
                    continue

                if rhs0 == lhs:
                    #print lhs, lhchain, rhs
                    pass
                if rhs0 in lhset:
                    continue
                else:
                    lhchain.append(rhs0)
                    lhset.add(rhs0)
                    lhstack.extend(self.productions[rhs0])

        builder = CfgBuilder()        
        builder.add_production(self.start, ())
        return FrozenCfg(builder, self.start)


def _msg(*args):
    return ' '.join(str(arg) for arg in args)

class Cfg(object):
    """
    The is obsolete but still contains some useful ideas looking towards
    CFG-based lattice transduction.
    """
    def __init__(self):
        self.productions = dict_of_set()

    def add_production(self, lhs, rhs):
        self.productions[lhs].add(tuple(rhs))

    def fsa(self, starts):
        assert set(starts) <= set(self.productions), ' '.join(repr(x) for x in set(starts) - set(self.productions))
        gendict = dict(self.productions)

        # we allow the client to give us a set of non_terminals for which we
        # create a new lhs and a new set of productions
        start_non_terminal = '_starts'
        while start_non_terminal in gendict:
            start_non_terminal += '_'
        gendict[start_non_terminal] = set((start,) for start in starts)
        nullable = frozenset(iter_nullable(gendict))

        class Root(object):
            def __repr__(self): return 'Root()'
            def __str__(self): return '<root>'
        root = Root()

        class Completion(object):
            def __repr__(self): return 'Completion()'
            def __str__(self): return '<completion>'
        completion_token = Completion()

        # note FsaItem has a Python closure on start_non_terminal and
        # completion_token
        class FsaItem(tuple):
            _seed_cursor = 0

            def __new__(cls, *args):
                len_args = len(args)
                if len_args == 1:
                    # we can create ourself from a sequence
                    args, = args
                    if type(arg) is FsaItem:
                        return args
                elif len_args == 2:
                    # we can create ourself from two args, by adding the third
                    args = args + (FsaItem._seed_cursor,)
                # we create a three-item tuple
                assert len(args) == 3
                return super(FsaItem, cls).__new__(cls, args)
            def __init__(self, *_):
                super(FsaItem, self).__init__()
                assert self._seed_cursor <= self.cursor <= self._complete_cursor

            @property
            def lhs_id(self): return self[0]
            @property
            def rhs_seq(self): return self[1]
            @property
            def cursor(self): return self[2]

            @property
            def _complete_cursor(self):
                return len(self.rhs_seq)

            @property
            def rhs_token(self):
                cursor = self.cursor
                return self.rhs_seq[cursor] if cursor < self._complete_cursor else completion_token

            @property
            def is_complete(self):
                return self.cursor == self._complete_cursor
                
            @property
            def is_sentential(self):
                return (self.lhs_id is start_non_terminal) and self.is_complete

            @property
            def shifted(self):
                return FsaItem(self.lhs_id, self.rhs_seq, self.cursor + 1)

            @property
            def seeder(self):
                seed_cursor = self._seed_cursor
                return self if self.cursor == seed_cursor else FsaItem(self.lhs_id, self.rhs_seq, seed_cursor)

            @property
            def completer(self):
                complete_cursor = self._complete_cursor - 1
                return self if self.cursor == complete_cursor else FsaItem(self.lhs_id, rhs_seq, complete_cursor)

            def __str__(self):
                parts = list()
                # note: use of str here is to support the use of
                # singletons, e.g. the root item
                parts.append(str(self.lhs_id))
                items = tuple(self.rhs_seq) + ('',)
                dotted = items[:self.cursor] + ('.',) + items[self.cursor:]
                parts.append("( %s)" % (' '.join(dotted)))
                return ' '.join(parts)


        # create the set of FSA items, and the dictionaries
        fsa_item_set = set()
        predict_dict = dict_of_set()
        complete_dict = dict_of_set()
        for lhs, rhs_seq_set in gendict.iteritems():
            for rhs_seq in rhs_seq_set:
                item = FsaItem(lhs, rhs_seq)
                assert item not in fsa_item_set
                fsa_item_set.add(item)
                predict_dict[lhs].add(item)
                complete_dict[lhs].add(item.completer)
                while not item.is_complete:
                    # note: first time through is a duplicate add; structured
                    # this way so as to handle epsilons
                    fsa_item_set.add(item)
                    rhs_token = item.rhs_token
                    if rhs_token not in gendict:
                        terminal = FsaItem(rhs_token, (rhs_token,))
                        if terminal not in fsa_item_set:
                            fsa_item_set.add(terminal)
                            predict_dict[rhs_token].add(terminal)
                            complete_dict[rhs_token].add(terminal.completer)
                    item = item.shifted

        print
        print 'fsa_item_set:'
        for item in sorted(fsa_item_set):
            print ' ', str(item)
        print

        print
        print 'predict_dict:'
        for lhs in sorted(predict_dict.keys()):
            print ' ', lhs
            for item in sorted(predict_dict[lhs]):
                print '   ', str(item)
        print

        print
        print 'complete_dict:'
        for lhs in sorted(complete_dict.keys()):
            print ' ', lhs
            for item in sorted(complete_dict[lhs]):
                print '   ', str(item)
        print

        # prediction graph
        builder = NodeLabelIdGraphBuilder()
        node_by_item = dict((item, builder.new_node(item)) for item in fsa_item_set)
        for item in node_by_item:
            if item.lhs_id in gendict:
                for prediction in predict_dict[item.rhs_token]:
                    builder.new_arc(node_by_item[item], node_by_item[prediction])
        fsa3 = FrozenGraph(builder)

        # completion graph
        builder = NodeLabelIdGraphBuilder()
        node_by_item = dict((item, builder.new_node(item.shifted)) for item in fsa_item_set)
        for item in node_by_item:
            if item.lhs_id in gendict:
                for completion in complete_dict[item.rhs_token]:
                    builder.new_arc(node_by_item[item], node_by_item[completion])
        fsa4 = FrozenGraph(builder).get_reversed()

        earleyitems = set()
        null_rhs = ((),)
        def predictions(parent_item):
            # return a generator yielding predictions
            dprint('cfg', _msg('  predicting from: ', parent_item))
            assert parent_item in earleyitems, str(parent_item)
            parent_rhs_token = parent_item.rhs_token
            rhs_set = gendict.get(parent_rhs_token)
            if rhs_set is None:
                # terminal
                prediction = FsaItem(parent_rhs_token, (parent_rhs_token,))
                yield prediction
            else:
                # non-terminal productions
                for rhs in rhs_set:
                    prediction = FsaItem(parent_rhs_token, rhs)
                    yield prediction

        builder = NodeLabelIdGraphBuilder()
        item_to_node_id = dict()
        
        # the root_item is "virtual" in that its lhs (root) isn't
        # entered into gendict
        root_item = FsaItem(root, (start_non_terminal,), 0)
        earleyitems.add(root_item)

        item_stack = list(predictions(root_item))
        earleyitems.update(item_stack)
        for item in item_stack:
            item_to_node_id[item] = builder.new_node(item)            

        while item_stack:
            parent = item_stack.pop()
            assert parent in item_to_node_id, str(parent)

            if not (parent.lhs_id is root or parent.lhs_id in gendict):
                continue
            
            for item in predictions(parent):
                # if item not in earleyitems and item.lhs_id in gendict:
                if item not in earleyitems:
                    assert not item.is_complete
                    earleyitems.add(item)
                    assert item not in item_to_node_id
                    item_to_node_id[item] = builder.new_node(item)
                    item_stack.append(item)
                builder.new_arc(item_to_node_id[parent], item_to_node_id[item], 'prediction')

            if not parent.is_complete:
                item = parent.shifted
                assert item not in earleyitems, str(item)
                if not item.is_complete:
                    earleyitems.add(item)
                    assert item not in item_to_node_id
                    item_to_node_id[item] = builder.new_node(item)
                    #builder.new_arc(item_to_node_id[parent], item_to_node_id[item], 'completion')
                    item_stack.append(item)
                    
        fsa_predictions = FrozenGraph(builder)
        fsa_completions = fsa_predictions.get_nocycles().get_reversed()

        # generate completion graph
        builder = NodeLabelIdGraphBuilder()
        nodeadjout = fsa_completions.nodeadjout
        label = fsa_completions.get_node_label
        queue = deque()
        queue.extend(fsa_completions.startnodes)
        seen = set(label(node) for node in queue)
        node_id_by_item = dict((item.shifted, builder.new_node(item.shifted)) for item in seen)
        while queue:
            node = queue.popleft()
            item = label(node)
            assert item in seen
            shifted = item.shifted
            assert shifted in node_id_by_item
            if not shifted.is_complete:
                continue
            for child_node, child_arc in nodeadjout[node]:
                child_item = label(child_node)
                child_shifted = child_item.shifted
                if child_item not in seen:
                    seen.add(child_item)
                    node_id_by_item[child_shifted] = builder.new_node(child_shifted)
                    queue.append(child_node)
                builder.new_arc(node_id_by_item[shifted], node_id_by_item[child_shifted])

        fsa2 = FrozenGraph(builder)
            #print 'node_id_by_item:', node_id_by_item
            
        def start_end_closures(graph):
            # returns a frozendict of the frozenset of labels of graph-end nodes
            # indexed by the labels of the graph-start nodes, i.e. the
            # frozendict indexes from each start-node-label to the set of
            # leaf-labels reachable from that start node; assumes labels are
            # unique
            get_node_label = graph.get_node_label
            starts, ends = graph.startnodes, graph.endnodes
            closures = dict()
            for start_node_id in starts:
                node_ids, arc_ids = graph.get_forward_transitive_closures((start_node_id,), True)
                closures[get_node_label(start_node_id)] = frozenset(get_node_label(node_id) for node_id in node_ids if node_id in ends)
            return frozendict(closures)
            
        print
        prediction_closures = start_end_closures(fsa_predictions)
        print 'prediction_closures:'
        for item in sorted(prediction_closures.keys()):
            closure = prediction_closures[item]
            print str(item)
            for item in sorted(closure):
                print ' ', str(item)
            
        completion_closures = start_end_closures(fsa_completions)
        print 'completion_closures:'
        for item in sorted(completion_closures.keys()):
            closure = completion_closures[item]
            print str(item)
            for item in sorted(closure):
                print ' ', str(item)
        print

        def node_label_callback(item, is_start, is_end):
            is_start_or_end = is_start or is_end
            is_terminal = item.lhs_id not in gendict
            is_start_state = item.lhs_id == start_non_terminal
            parts = list()
            parts.append('label="%s"' % (str(item),))
            if is_start_or_end or is_terminal:
                parts.append('style=bold')
            if is_terminal:
                parts.append('fontsize=16')
            if is_start_state:
                parts.append('shape=octagon')
            return ', '.join(parts)
            # return 'label="%s", shape=%s, style=%s ' % (str(item), 'ellipse' if is_non_terminal else 'tripleoctagon', 'normal' if is_non_terminal else 'bold')
            # return 'label="%s", shape=%s ' % (str(item), 'ellipse' if is_non_terminal else 'tripleoctagon, fontsize=16')
            #return 'label="%s", shape=%s ' % (str(item), 'ellipse' if is_non_terminal else 'ellipse, style=bold, fontsize=16')               
        def arc_label_callback(label):
            return ''
            # return 'style=dotted, arrowtail=dot' if label == 'completion' else ''
            # return '' if label == 'prediction' else 'style=bold, weight=10'
            # fsa_predictions.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('ratio=auto;', 'size="6.5,8";', 'page="8.5,11";'))

        #fsa_predictions.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('size="4.7,6.125";', 'label="predictions"'))
        #fsa_completions.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('size="4.7,6.125";', 'label="completions"'))
        # fsa_completions = fsa_completions.get_nocycles()
        # fsa_completions.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('size="4.7,6.125";', 'label="completions"'))
        #fsa2.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('size="4.7,6.125";', 'label="completions 2"'))
        
        #fsa3.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('size="4.7,6.125";', 'label="predictions 10"'))
        #fsa4.dot_display(node_label_callback=node_label_callback, arc_label_callback=arc_label_callback,  globals=('size="4.7,6.125";', 'label="completions 10"'))

        print 'earleyitems:'
        for item in sorted(earleyitems):
            print ' ', item

        print
        print 'root_item:'
        return root_item

    def gen(self):
        # generator protocol is that the first send from the client is the
        # starting non-terminals and the graph builder; subsequent yields give
        # the client a tuple (sentential, scan-set) where sentential is True if
        # we're at a sentential point and scan-set is the set of terminals that
        # are currently valid; if it wants the parse to continue the client
        # sends a terminal from scan-set as the next non-terminal to parse
        starts, builder = yield
        dprint('cfg', _msg('starts:', starts))
        #print 'starts:', starts
        assert set(starts) <= set(self.productions), ' '.join(repr(x) for x in set(starts) - set(self.productions))
        gendict = dict(self.productions)

        # we allow the client to give us a set of non_terminals for
        # which we create a new lhs and a new set of productions
        start_non_terminal = '_starts'
        while start_non_terminal in gendict:
            start_non_terminal += '_'
        gendict[start_non_terminal] = set((start,) for start in starts)

        # productions dictionary for this gen/parse
        gendict = frozendict((lhs, sorteduniquetuple(rhs)) for lhs, rhs in gendict.iteritems())
        dprint('cfg', _msg('gendict:', gendict))
        # print 'gendict:', gendict

        nullable = frozenset(iter_nullable(gendict))
        dprint('cfg', _msg('nullable:', ' '.join(nullable)))
        # print 'nullable:', ' '.join(nullable)

        # there must be a way to abstract this named-singleton pattern using __metaclass__ ...
        class Root(object):
            def __repr__(self): return 'Root()'
            def __str__(self): return '<root>'
        class Completion(object):
            def __repr__(self): return 'Completion()'
            def __str__(self): return '<completion>'
        class Child(object):
            def __repr__(self): return 'Child()'
            def __str__(self): return '<child>'
        class Sibling(object):
            def __repr__(self): return 'Sibling()'
            def __str__(self): return '<sibling>'

        root = Root()
        completion_token = Completion()
        child = Child()
        sibling = Sibling()

        # note EarleyState has a Python closure on start_non_terminal and
        # completion_token
        class EarleyState(tuple):
            def __new__(cls, *args):
                if len(args) == 1:
                    args, = args
                    if type(arg) is EarleyState:
                        return args
                return super(EarleyState, cls).__new__(cls, args)
            def __init__(self, *_):
                super(EarleyState, self).__init__()
                assert len(self) == 4

            @property
            def lhs_id(self): return self[0]
            @property
            def rhs_seq(self): return self[1]
            @property
            def cursor(self): return self[2]
            @property
            def origin(self): return self[3]

            @property
            def rhs_token(self):
                rhs_seq = self.rhs_seq
                cursor = self.cursor
                assert 0 <= cursor <= len(rhs_seq)
                return rhs_seq[cursor] if cursor < len(rhs_seq) else completion_token

            @property
            def shifted(self):
                rhs_seq = self.rhs_seq
                new_cursor = self.cursor + 1
                assert new_cursor <= len(rhs_seq)
                return EarleyState(self.lhs_id, rhs_seq, new_cursor, self.origin)

            @property
            def is_complete(self):
                #return self.cursor == len(self.rhs_seq)
                return self.rhs_token is completion_token

            @property
            def is_sentential(self):
                return (self.lhs_id is start_non_terminal) and self.is_complete

            def __str__(self):
                # return super(EarleyState, self).__repr__()
                parts = list()
                #parts.append(super(EarleyState, self).__repr__())
                #parts.append('')
                # note: use of str here is to support the root item
                parts.append(str(self.lhs_id))
                items = tuple(self.rhs_seq) + ('',)
                dotted = items[:self.cursor] + ('.',) + items[self.cursor:]
                parts.append("( %s )" % (' '.join(dotted)))
                parts.append("%s" %(self.origin,))
                return ' '.join(parts)

        class NodeDict(dict):
            def __missing__(self, item):
                node = self[item] = builder.new_node(item)
                #print 'missing:', node, '', item
                return node

        # the root_item is "virtual" in that its lhs (root) isn't entered into
        # gendict
        root_item = EarleyState(root, (start_non_terminal,), 0, 0)

        node_by_item = NodeDict()
        # root item is first node in the parses graph
        node_by_item[root_item]
        earleystates = set(node_by_item)
        assert len(earleystates) == 1

        null_rhs = ((),)
        zero_cursor = 0
        def predictions(parent_item, stateset_id):
            # return a generator yielding predictions
            dprint('cfg', _msg('  predicting from: ', parent_item))
            assert parent_item in earleystates, repr(parent_item)
            #assert non_terminal is parent_item.rhs_token
            parent_rhs_token = parent_item.rhs_token
            # note: this is subtle, we're dealing with three cases of the rhs_token:
            # - not empty -> non-terminal
            # - empty but parent_rhs_token not nullable -> terminal
            # - empty, parent_rhs_token is nullable -> epsilon
            # first two lead to a prediction; epsilons are handled in the workset loop
            #for rhs in gendict[parent_rhs_token] if parent_rhs_token in gendict else ((),):
            for rhs in gendict.get(parent_rhs_token, null_rhs):
                if rhs or parent_rhs_token not in nullable:
                    # terminals have the empty rhs field, so they are always is_complete
                    prediction = EarleyState(parent_rhs_token, rhs, zero_cursor, stateset_id)
##                     if prediction not in earleystates:
                    if True:
                        # note: deal with duplicate issues

                        # XXX give builder a non-duplicate arcs mode so no duplicate arcs
                        # builder.new_arc(node_by_item[parent_item], node_by_item[prediction], child)
                        builder.new_arc(node_by_item[parent_item], node_by_item[prediction], child)
##                     builder.new_arc(node_by_item[(parent_item, stateset_id)], node_by_item[(prediction, stateset_id)], child)
                        earleystates.add(prediction)

                    dprint('cfg', _msg('    prediction:' if rhs else '  terminal:  ', prediction))
#                    print 'prediction:' if rhs else 'terminal:  ', prediction
                    yield prediction

        def make_completion(parent, stateset_id):
            assert parent in earleystates, repr(parent)
            # completion is just a shift, no duplicate target issues
            completion = parent.shifted
##             if completion not in earleystates:
            if True:
                earleystates.add(completion)
                if False and not completion.is_complete:
##                 if not parent.is_complete:
                    dprint('cfg', _msg('    completion: ', sibling))
                    #print 'completion:', ' ', sibling

                    builder.new_arc(node_by_item[parent], node_by_item[completion], sibling)
##                     builder.new_arc(node_by_item[(parent, stateset_id)], node_by_item[(completion, stateset_id)], sibling)
            return completion
            
        statesets = list()
        def completions(item, stateset_id):
            # return a generator yielding completions
            assert item in earleystates, repr(item)
            dprint('cfg', _msg('  completing: ', item))
#            print 'completing:', item
            my_lhs = item.lhs_id
            for parent in statesets[item.origin]:
                if parent.rhs_token == my_lhs:
                    completion = make_completion(parent, stateset_id)
                    # note: no duplicate issues for completion...
                    dprint('cfg', _msg('    completion: ', completion, '  from: ', parent))
                    #print 'completion:', ' ', completion, ' ', 'from:', parent
                    yield completion


        # bootstrap the Early parser
        k = len(statesets)
        assert k == 0
##         seed_items = (root_item,)
        dprint('cfg', 'seeding:')
        seed_items = set(predictions(root_item, k))
        while seed_items:        

            # note: seed the workset with predictions from the cursor-facing
            # items (usually from scansets); in this iteration, these items will
            # pull in completions, leading to predictions and/or new terminals
            # in the iteration; as a special case, on bootstrapping, the
            # root_item will pull in the _starts predictions
##             workset = set(prediction for item in seed_items for prediction in predictions(item, k))
            workset = seed_items

            k = len(statesets)
            stateset = set()
            statesets.append(stateset)
            if DebugPrint.active('cfg'):
                dprint('cfg', '')
                dprint('cfg', _msg('k:', k))
                dprint('cfg', _msg('workset:'))
                for item in workset:
                    dprint('cfg', _msg('  ', item))
##             print
##             print 'k:', k
##             print 'workset:'
##             for item in workset:
##                 print ' ', item

            scansets = dict_of_set()
            sentential = False
            while workset:
                item = workset.pop()
                assert item not in stateset
                stateset.add(item)

                rhs_token = item.rhs_token
                dprint('cfg', _msg('rhs_token: ', rhs_token, ' ', 'in: ', item))
#                print 'rhs_token: ', rhs_token, ' ', 'in: ', item
                if rhs_token in gendict:
                    # deal with non-terminal

                    # XXX have to decide how to handle epsilons; e.g. do all the
                    # combinatorics at the production level or the parses
                    # graph/traceback level...
                    #
                    # XXX have still not proven that the completion work on
                    # nullables is OK in the face of the unordered handling of
                    # the workset items...
                    if rhs_token in nullable:
                        dprint('cfg', _msg('  nullable:', item))
#                        print 'nullable:', item
                        workset.add(make_completion(item, k))
                    workset.update(prediction for prediction in predictions(item, k) if prediction not in stateset)
                elif item.is_sentential:
                    # deal with end of grammar, no completion possible (assert this?)
                    dprint('cfg', _msg('  sentential:', item))
#                    print 'sentential:', item
                    sentential = True
                elif rhs_token is completion_token:
                    # deal with end of a production
                    workset.update(completion for completion in completions(item, k) if completion not in stateset and completion not in workset)
                else:
                    # store up work on this terminal; held until client selects
                    # (a) "scanned" terminal(s)
                    #scansets[rhs_token].append(item)
##                     scansets[rhs_token].update(predictions(item, k))
                    scansets[rhs_token].add(item)
                    dprint('cfg', _msg('  terminal:', rhs_token))
#                    print 'terminal:  ', rhs_token

            dprint('cfg', '')
            scanset = set(scansets)
            dprint('cfg', _msg('scans:', tuple(scanset)))
#            print 'scans:', tuple(scanset)
            # yield point
            scanned = yield sentential, scanset
            dprint('cfg', _msg('scanned:', scanned))
#            print 'scanned:', scanned
            assert scanned in scanset
            dprint('cfg', 'seeding:')
            seed_items = set(prediction for item in scansets[scanned] for prediction in predictions(item, k))

# some private data that's shared across tests    
if __name__ == '__main__':

    _cfg0 = """
    expr x
    expr expr + expr
    """
    
    _cfg1 = """
    expr primary
    expr expr + expr
    primary alpha alpha_num_list
    alpha a
    alpha b
    alpha c
    alpha d
    num 0 
    num 1
    num 2 
    alpha_num_list
    alpha_num_list alpha_num_list alpha
    alpha_num_list alpha_num_list num
    spudge alpha
    spudge num
    spudge alpha_num_list
    foo xxx expr
    foo d
    """

    _cfg2 = """
    chain chain1
    chain1 chain2
    chain1 chain4
    chain2 chain3 baz
    chain3 chain4
    chain4 foobar
    """

    _cfg3 = """
    s a
    a b
    b c
    c W X Y Z
    """

    _cfg4 = """
    expr term
    expr expr + term
    term factor
    term term * factor
    factor primary
    factor primary ^ factor
    primary ( expr )
    primary number
    number digit
    number number digit
    digit 0
    digit 1
    """

    _cfg5 = """
    expr term
    expr expr + term
    term primary
    term term * primary
    primary A B C
    primary number
    primary ( expr )
    number digit
    number number digit
    digit 0
    digit 1
    """

    _cfg6 = """
    expr number
    expr expr + number
    number digit
    number number digit
    digit 0
    digit 1
    """

    def _cfg_from_lines(iterable, comment_prefix='#', reverse=False):
        cfg = Cfg()
        for line in iterable:
            parts = line.split()
            if parts and not parts[0].startswith(comment_prefix):
                cfg.add_production(parts[0], parts[1:] if not reverse else reversed(parts[1:]))
        return cfg

    def _cfg_from_string(string, comment_prefix='#', reverse=False):
        from cStringIO import StringIO
        return _cfg_from_lines(StringIO(string), comment_prefix, reverse)

    def _gen(cfg, starts, count_down, seed=None):
        assert count_down >= 1
        
        rand = random.Random()
        rand.seed(seed)
        choice = rand.choice

        gen = cfg.gen()
        gen.next()
        
        #builder = TopologicalGraphBuilder()
        builder = NodeLabelIdGraphBuilder()
        sentential, scans = gen.send((starts, builder))
        if sentential:
            count_down -= 1
            if count_down == 0:
                print 'done:', 'count_down'
                return (), FrozenGraph(builder)
##         print 'scans 2:', scans

        seq = list()
        while True:
##         for i in xrange(count_down):
##             print 'scans 3:', scans
            if not scans:
                dprint('cfg', _msg('done:', 'end of grammar'))
#                print 'done:', 'end of grammar'
                return tuple(seq), FrozenGraph(builder)
            scan = choice(tuple(scans))
            seq.append(scan)
            sentential, scans = gen.send(scan)
            if sentential:
                count_down -= 1
                if count_down == 0:
                    dprint('cfg', _msg('done:', 'count_down'))
#                    print 'done:', 'count_down'
                    return tuple(seq), FrozenGraph(builder)
        
    def _gen2(cfg, starts, seq):
        gen = cfg.gen()
        gen.next()
        
        builder = NodeLabelIdGraphBuilder()
        sentential, scans = gen.send((starts, builder))
        for token in seq:
            assert token in scans
            sentential, scans = gen.send(token)
        assert sentential
        return tuple(seq), FrozenGraph(builder)


    binary_math_cfg = """
    expr     term
    expr     expr + term
    term     factor
    term     term * factor
    factor   primary
    factor   primary ^ factor
    primary  ( expr )
    primary  number
    number   digit
    number   number digit
    digit    0
    digit    1
    """

    """
    Simple binary-number expression recognizer:
    >>> recognizer = CfgRecognizer(FrozenCfg(make_cfg_from_string(binary_math_cfg)))
    >>> ' '.join(recognizer.terminals)
    ( ) * + 0 1 ^

    Get a deterministic sentence generator for the grammar:
    >>> sentencer = recognizer.iter_sentences.next

    Get the first 100 sentences and look at the first and the last of them:
    >>> sentences = list(sentencer() for i in xrange(100))
    >>> print sentence[0]
    0
    >>> print sentence[-1]
    01+(0)*01    

    See if sentences are in the language:
    >>> recognizer.recognize('001+100*(10+11^(1+1))')
    True
    >>> recognizer.recognize('001+^100'):
    False
    """
    

def _test_a_bunch_o_stuff():
    """
    >>> cfg0 = _cfg_from_string(_cfg2)
    >>> seq0, graph0 = _gen2(cfg0, ('chain',), ('foobar', 'baz'))
    >>> #graph0.dot_display(node_label_callback=lambda item, x, y: 'label="%s", shape=%s, style=%s ' % (str(item), 'box' if item.is_complete else 'ellipse', 'bold' if item.is_complete else 'normal'), globals=('ordering=out;',))
    >>> print ' '.join(seq0)
    foobar baz

    >>> #cfg0.fsa(('chain',))

    >>> cfg1 = _cfg_from_string(_cfg0)
    >>> seq1, graph1 = _gen2(cfg1, ('expr',), 'x+x')
    >>> #graph1.dot_display(node_label_callback=lambda item, x, y: 'label="%s", shape=%s, style=%s ' % (str(item), 'box' if item.is_complete else 'ellipse', 'bold' if item.is_complete else 'normal'))
    >>> print ' '.join(seq1)
    x + x

    >>> cfg3 = _cfg_from_string(_cfg3)
    >>> with DebugPrint(None): seq3, graph3 = _gen(cfg3, ('s',), 25, 0)
    >>> print ' '.join(seq3)
    W X Y Z


    >>> # cfg4 = _cfg_from_string(_cfg4)
    >>> # with DebugPrint('cfg'): seq4, graph4 = _gen(cfg4, ('expr',), 3, 0)
    >>> # print ' '.join(seq4)


    >>> cfg5 = _cfg_from_string(_cfg5)
    >>> with DebugPrint(None): seq, graph = _gen2(cfg5, ('expr',), '(1+0)*1+0')
    >>> print ' '.join(seq)
    ( 1 + 0 ) * 1 + 0
    >>> print cfg5.fsa(('expr', 'primary'))
    <BLANKLINE>
    fsa_item_set:
      ( ( . ( )
      ) ( . ) )
      * ( . * )
      + ( . + )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
      B ( . B )
      C ( . C )
      _starts ( . expr )
      _starts ( . primary )
      digit ( . 0 )
      digit ( . 1 )
      expr ( . expr + term )
      expr ( expr . + term )
      expr ( expr + . term )
      expr ( . term )
      number ( . digit )
      number ( . number digit )
      number ( number . digit )
      primary ( . ( expr ) )
      primary ( ( . expr ) )
      primary ( ( expr . ) )
      primary ( . A B C )
      primary ( A . B C )
      primary ( A B . C )
      primary ( . number )
      term ( . primary )
      term ( . term * primary )
      term ( term . * primary )
      term ( term * . primary )
    <BLANKLINE>
    <BLANKLINE>
    predict_dict:
      (
        ( ( . ( )
      )
        ) ( . ) )
      *
        * ( . * )
      +
        + ( . + )
      0
        0 ( . 0 )
      1
        1 ( . 1 )
      A
        A ( . A )
      B
        B ( . B )
      C
        C ( . C )
      _starts
        _starts ( . expr )
        _starts ( . primary )
      digit
        digit ( . 0 )
        digit ( . 1 )
      expr
        expr ( . expr + term )
        expr ( . term )
      number
        number ( . digit )
        number ( . number digit )
      primary
        primary ( . ( expr ) )
        primary ( . A B C )
        primary ( . number )
      term
        term ( . primary )
        term ( . term * primary )
    <BLANKLINE>
    <BLANKLINE>
    complete_dict:
      (
        ( ( . ( )
      )
        ) ( . ) )
      *
        * ( . * )
      +
        + ( . + )
      0
        0 ( . 0 )
      1
        1 ( . 1 )
      A
        A ( . A )
      B
        B ( . B )
      C
        C ( . C )
      _starts
        _starts ( . expr )
        _starts ( . primary )
      digit
        digit ( . 0 )
        digit ( . 1 )
      expr
        expr ( expr + . term )
        expr ( . term )
      number
        number ( . digit )
        number ( number . digit )
      primary
        primary ( ( expr . ) )
        primary ( A B . C )
        primary ( . number )
      term
        term ( . primary )
        term ( term * . primary )
    <BLANKLINE>
    <BLANKLINE>
    prediction_closures:
    _starts ( . expr )
      ( ( . ( )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
    _starts ( . primary )
      ( ( . ( )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
    expr ( expr . + term )
      + ( . + )
    expr ( expr + . term )
      ( ( . ( )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
    number ( number . digit )
      0 ( . 0 )
      1 ( . 1 )
    primary ( ( . expr ) )
      ( ( . ( )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
    primary ( ( expr . ) )
      ) ( . ) )
    primary ( A . B C )
      B ( . B )
    primary ( A B . C )
      C ( . C )
    term ( term . * primary )
      * ( . * )
    term ( term * . primary )
      ( ( . ( )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
    completion_closures:
    ( ( . ( )
      _starts ( . expr )
      _starts ( . primary )
      expr ( expr + . term )
      primary ( ( . expr ) )
      term ( term * . primary )
    ) ( . ) )
      primary ( ( expr . ) )
    * ( . * )
      term ( term . * primary )
    + ( . + )
      expr ( expr . + term )
    0 ( . 0 )
      _starts ( . expr )
      _starts ( . primary )
      expr ( expr + . term )
      number ( number . digit )
      primary ( ( . expr ) )
      term ( term * . primary )
    1 ( . 1 )
      _starts ( . expr )
      _starts ( . primary )
      expr ( expr + . term )
      number ( number . digit )
      primary ( ( . expr ) )
      term ( term * . primary )
    A ( . A )
      _starts ( . expr )
      _starts ( . primary )
      expr ( expr + . term )
      primary ( ( . expr ) )
      term ( term * . primary )
    B ( . B )
      primary ( A . B C )
    C ( . C )
      primary ( A B . C )
    <BLANKLINE>
    earleyitems:
      <root> ( . _starts )
      ( ( . ( )
      ) ( . ) )
      * ( . * )
      + ( . + )
      0 ( . 0 )
      1 ( . 1 )
      A ( . A )
      B ( . B )
      C ( . C )
      _starts ( . expr )
      _starts ( . primary )
      digit ( . 0 )
      digit ( . 1 )
      expr ( . expr + term )
      expr ( expr . + term )
      expr ( expr + . term )
      expr ( . term )
      number ( . digit )
      number ( . number digit )
      number ( number . digit )
      primary ( . ( expr ) )
      primary ( ( . expr ) )
      primary ( ( expr . ) )
      primary ( . A B C )
      primary ( A . B C )
      primary ( A B . C )
      primary ( . number )
      term ( . primary )
      term ( . term * primary )
      term ( term . * primary )
      term ( term * . primary )
    <BLANKLINE>
    root_item:
    <root> ( . _starts )

    >>> cfg6 = _cfg_from_string(_cfg6)
    >>> with DebugPrint(None): seq, graph = _gen2(cfg6, ('expr',), '10+1+1')
    >>> with DebugPrint(None): seq, graph = _gen2(cfg6, ('expr',), '1+10+1')
    >>> cfg6.fsa(('expr',))
    <BLANKLINE>
    fsa_item_set:
      + ( . + )
      0 ( . 0 )
      1 ( . 1 )
      _starts ( . expr )
      digit ( . 0 )
      digit ( . 1 )
      expr ( . expr + number )
      expr ( expr . + number )
      expr ( expr + . number )
      expr ( . number )
      number ( . digit )
      number ( . number digit )
      number ( number . digit )
    <BLANKLINE>
    <BLANKLINE>
    predict_dict:
      +
        + ( . + )
      0
        0 ( . 0 )
      1
        1 ( . 1 )
      _starts
        _starts ( . expr )
      digit
        digit ( . 0 )
        digit ( . 1 )
      expr
        expr ( . expr + number )
        expr ( . number )
      number
        number ( . digit )
        number ( . number digit )
    <BLANKLINE>
    <BLANKLINE>
    complete_dict:
      +
        + ( . + )
      0
        0 ( . 0 )
      1
        1 ( . 1 )
      _starts
        _starts ( . expr )
      digit
        digit ( . 0 )
        digit ( . 1 )
      expr
        expr ( expr + . number )
        expr ( . number )
      number
        number ( . digit )
        number ( number . digit )
    <BLANKLINE>
    <BLANKLINE>
    prediction_closures:
    _starts ( . expr )
      0 ( . 0 )
      1 ( . 1 )
    expr ( expr . + number )
      + ( . + )
    expr ( expr + . number )
      0 ( . 0 )
      1 ( . 1 )
    number ( number . digit )
      0 ( . 0 )
      1 ( . 1 )
    completion_closures:
    + ( . + )
      expr ( expr . + number )
    0 ( . 0 )
      _starts ( . expr )
      expr ( expr + . number )
      number ( number . digit )
    1 ( . 1 )
      _starts ( . expr )
      expr ( expr + . number )
      number ( number . digit )
    <BLANKLINE>
    earleyitems:
      <root> ( . _starts )
      + ( . + )
      0 ( . 0 )
      1 ( . 1 )
      _starts ( . expr )
      digit ( . 0 )
      digit ( . 1 )
      expr ( . expr + number )
      expr ( expr . + number )
      expr ( expr + . number )
      expr ( . number )
      number ( . digit )
      number ( . number digit )
      number ( number . digit )
    <BLANKLINE>
    root_item:
    (Root(), ('_starts',), 0)

    >>> #cfg6a = _cfg_from_string(_cfg6, reverse=True)
    >>> #cfg6a.fsa(('expr',))


    >>> cfg = _cfg_from_string(_cfg1)
    >>> seq, graph = _gen(cfg, ('expr', 'foo'), 25, 0)
    >>> print ' '.join(seq)
    d 0 d b + c 0 b d + d + c 0 + c 2 2 0 2 b 1 2 1 d a d + d 2

    >>> for text in graph.text_iter():
    ... #  print text,
    ...   pass
    >>> print ' '.join(seq)
    d 0 d b + c 0 b d + d + c 0 + c 2 2 0 2 b 1 2 1 d a d + d 2

    >>> seq, graph = _gen(cfg, ('alpha_num_list',), 30, 1)
    >>> print ' '.join(seq)
    a 0 0 c d d 1 0 a a 0 d 0 a d 0 c 2 2 a a d 2 b c b a c d

    >>> for text in graph.text_iter():
    ...   #print text,
    ...   pass

    XXX move this to a  --logreftest 
    >>> with DebugPrint('cfg'): seq1, graph1 = _gen2(cfg1, ('expr',), 'x+x')
    cfg: starts: ('expr',)
    cfg: gendict: frozendict({'expr': (('expr', '+', 'expr'), ('x',)), '_starts': (('expr',),)})
    cfg: nullable: 
    cfg: seeding:
    cfg:   predicting from:  <root> ( . _starts  ) 0
    cfg:     prediction: _starts ( . expr  ) 0
    cfg: 
    cfg: k: 0
    cfg: workset:
    cfg:    _starts ( . expr  ) 0
    cfg: rhs_token:  expr   in:  _starts ( . expr  ) 0
    cfg:   predicting from:  _starts ( . expr  ) 0
    cfg:     prediction: expr ( . expr + expr  ) 0
    cfg:     prediction: expr ( . x  ) 0
    cfg: rhs_token:  expr   in:  expr ( . expr + expr  ) 0
    cfg:   predicting from:  expr ( . expr + expr  ) 0
    cfg:     prediction: expr ( . expr + expr  ) 0
    cfg:     prediction: expr ( . x  ) 0
    cfg: rhs_token:  x   in:  expr ( . x  ) 0
    cfg:   terminal: x
    cfg: 
    cfg: scans: ('x',)
    cfg: scanned: x
    cfg: seeding:
    cfg:   predicting from:  expr ( . x  ) 0
    cfg:   terminal:   x ( .  ) 0
    cfg: 
    cfg: k: 1
    cfg: workset:
    cfg:    x ( .  ) 0
    cfg: rhs_token:  <completion>   in:  x ( .  ) 0
    cfg:   completing:  x ( .  ) 0
    cfg:     completion:  expr ( x .  ) 0   from:  expr ( . x  ) 0
    cfg: rhs_token:  <completion>   in:  expr ( x .  ) 0
    cfg:   completing:  expr ( x .  ) 0
    cfg:     completion:  _starts ( expr .  ) 0   from:  _starts ( . expr  ) 0
    cfg:     completion:  expr ( expr . + expr  ) 0   from:  expr ( . expr + expr  ) 0
    cfg: rhs_token:  +   in:  expr ( expr . + expr  ) 0
    cfg:   terminal: +
    cfg: rhs_token:  <completion>   in:  _starts ( expr .  ) 0
    cfg:   sentential: _starts ( expr .  ) 0
    cfg: 
    cfg: scans: ('+',)
    cfg: scanned: +
    cfg: seeding:
    cfg:   predicting from:  expr ( expr . + expr  ) 0
    cfg:   terminal:   + ( .  ) 1
    cfg: 
    cfg: k: 2
    cfg: workset:
    cfg:    + ( .  ) 1
    cfg: rhs_token:  <completion>   in:  + ( .  ) 1
    cfg:   completing:  + ( .  ) 1
    cfg:     completion:  expr ( expr + . expr  ) 0   from:  expr ( expr . + expr  ) 0
    cfg: rhs_token:  expr   in:  expr ( expr + . expr  ) 0
    cfg:   predicting from:  expr ( expr + . expr  ) 0
    cfg:     prediction: expr ( . expr + expr  ) 2
    cfg:     prediction: expr ( . x  ) 2
    cfg: rhs_token:  x   in:  expr ( . x  ) 2
    cfg:   terminal: x
    cfg: rhs_token:  expr   in:  expr ( . expr + expr  ) 2
    cfg:   predicting from:  expr ( . expr + expr  ) 2
    cfg:     prediction: expr ( . expr + expr  ) 2
    cfg:     prediction: expr ( . x  ) 2
    cfg: 
    cfg: scans: ('x',)
    cfg: scanned: x
    cfg: seeding:
    cfg:   predicting from:  expr ( . x  ) 2
    cfg:   terminal:   x ( .  ) 2
    cfg: 
    cfg: k: 3
    cfg: workset:
    cfg:    x ( .  ) 2
    cfg: rhs_token:  <completion>   in:  x ( .  ) 2
    cfg:   completing:  x ( .  ) 2
    cfg:     completion:  expr ( x .  ) 2   from:  expr ( . x  ) 2
    cfg: rhs_token:  <completion>   in:  expr ( x .  ) 2
    cfg:   completing:  expr ( x .  ) 2
    cfg:     completion:  expr ( expr + expr .  ) 0   from:  expr ( expr + . expr  ) 0
    cfg:     completion:  expr ( expr . + expr  ) 2   from:  expr ( . expr + expr  ) 2
    cfg: rhs_token:  <completion>   in:  expr ( expr + expr .  ) 0
    cfg:   completing:  expr ( expr + expr .  ) 0
    cfg:     completion:  _starts ( expr .  ) 0   from:  _starts ( . expr  ) 0
    cfg:     completion:  expr ( expr . + expr  ) 0   from:  expr ( . expr + expr  ) 0
    cfg: rhs_token:  <completion>   in:  _starts ( expr .  ) 0
    cfg:   sentential: _starts ( expr .  ) 0
    cfg: rhs_token:  +   in:  expr ( expr . + expr  ) 0
    cfg:   terminal: +
    cfg: rhs_token:  +   in:  expr ( expr . + expr  ) 2
    cfg:   terminal: +
    cfg: 
    cfg: scans: ('+',)
    >>> print ' '.join(seq1)
    x + x
    """

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
