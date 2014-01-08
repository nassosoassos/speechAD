###########################################################################
#
# File:         bwtrainer.py
# Date:         Mon 12 May 2008 16:17
# Author:       Ken Basye
# Description:  Baum-Welch trainer
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
    Baum-Welch trainer for HMMs - see class TrainingGraph
"""
from __future__ import division
from __future__ import with_statement
from itertools import imap, repeat, chain, izip, count
from onyx.am.hmm import Hmm
from onyx.am.modelmgr import GmmMgr
from onyx.am.hmm_mgr import HmmMgr
from onyx.graph.graphtools import FrozenGraph, GraphBuilder
from onyx.util.debugprint import dcheck, DebugPrint
from onyx.util.floatutils import float_to_readable_string
from onyx.util.safediv import safely_divide_float_array
from collections import deque, defaultdict
from random import seed, random
import numpy
import sys

class TrainingGraph(object):
    """
    A graph that supports BW training (and perhaps other types later).  The
    graph is made up of Hmm units, connected in terms of their inputs and
    outputs, so that the entire graph is also a hidden Markov model.

    See the __init__function for more complete documentation on building
    TrainingGraphs.
    """

    # XXX Maybe do something more with split_prob_dict, like make them
    # into elements in the graph itself?  Need a lot of changes to do
    # that, though.
    def __init__(self, graph, hmm_mgr, split_prob_dict):
        """
        graph is a FrozenGraph object with non-negative integer label pairs,
        (node_label, model_label), on the nodes.  No node_label may appear more
        than once.  hmm_mgr is a HmmMgr indexed by model_label.  Arc labels are
        not used.  graph.is_lattice() must return True.  graph.has_self_loop()
        must return False.  All hmms must be using the same GmmMgr to store
        their models.  split_prob_dict is a mapping from pairs of labels (t1,
        t2) to a Numpy array of floats between 0 and 1.  For a given pair, the
        array must have a length equal to the number of outputs in the model
        indicated by t1 (which must also be equal to the number of inputs in the
        model indicated by t2).  Each number gives the probability of making a
        transition from t1 through an output state to the corresponding input
        state of t2.  The dict must obey the following restrictions: All nodes
        t1 with more than one successor must have entries for every successor.
        For a given t1 with multiple sucessors, summing the values element-wise
        for all successors must give an array with all values equal to 1.0 (OK,
        very close to 1.0).

        >>> gmm_mgr = make_gmm_mgr(20)

        Here's a trivial case of a graph with only one node,
        containing one model:
        >>> gb = GraphBuilder()
        >>> node_id = gb.new_node((0,0))
        >>> gr0 = FrozenGraph(gb)

        Make an Hmm with 8 states and order 3 (self loop, forward 1,
        forward 2)
        >>> hmm = make_forward_hmm(gmm_mgr, 8, 3) 
        >>> hmm_mgr = HmmMgr((hmm,))
        >>> tg0 = TrainingGraph(gr0, hmm_mgr, dict())
        >>> gmm_mgr.set_adaptation_state("INITIALIZING")
        >>> gmm_mgr.clear_all_accumulators()
        >>> tg0.begin_training()
        >>> gmm_mgr.set_adaptation_state("ACCUMULATING")

        Perturb things a bit to avoid numerical problems
        >>> seed(0)
        >>> obs_seq = list(imap(array, repeat((3 + 0.1*random(), 2 + 0.1*random()), 20) ))
        >>> tg0.train_one_sequence(obs_seq)

        >>> obs_seq = list(imap(array, repeat((3 + 0.1*random(), 2 + 0.1*random()), 20) ))
        >>> tg0.train_one_sequence(obs_seq)

        >>> obs_seq = list(imap(array, repeat((3 + 0.1*random(), 2 + 0.1*random()), 20) ))
        >>> tg0.train_one_sequence(obs_seq)

        >>> with DebugPrint("gaussian_numeric_error"):
        ...    gmm_mgr.set_adaptation_state("APPLYING")
        ...    gmm_mgr.apply_all_accumulators()
        ...    tg0.end_training()
        ...    gmm_mgr.set_adaptation_state("NOT_ADAPTING")

        >>> gb = GraphBuilder()
        >>> node_ids = []
        >>> for label in xrange(7):
        ...    node_ids.append(gb.new_node((label,label)))
        >>> for s, e in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 5), (5, 6)):
        ...    arc_id = gb.new_arc(node_ids[s], node_ids[e])
        >>> gr0 = FrozenGraph(gb)
        >>> gr0.is_lattice()
        True
        >>> gr0.has_self_loop()
        False
        >>> gmm_mgr = make_gmm_mgr(20)
        >>> hmms = tuple(make_forward_hmm(gmm_mgr, 8, 3, exact=True) for i in xrange(7))
        >>> hmm_mgr = HmmMgr(hmms)

        Split-prob dictionary
        >>> spd = {}

        Keys are pairs of (pred_node_label, succ_node_label).  NB that
        the summing-to-one condition is across tuple positions from a
        given predeccessor.
        >>> spd[(1,2)] = (0.5, 0.5, 0.5)
        >>> spd[(1,3)] = (0.5, 0.5, 0.5)

        >>> tg = TrainingGraph(gr0, hmm_mgr, spd)
        
        >>> gmm_mgr.set_adaptation_state("INITIALIZING")
        >>> hmm_mgr.set_adaptation_state("INITIALIZING")
        >>> gmm_mgr.clear_all_accumulators()
        >>> tg.begin_training()
        >>> gmm_mgr.set_adaptation_state("ACCUMULATING")
        >>> hmm_mgr.set_adaptation_state("ACCUMULATING")
        >>> obs_seq1 = list(chain(imap(array, repeat((1.0 + 0.1*random(), 0.5 + 0.1*random()), 20)) ,
        ...                       imap(array, repeat((3.0 + 0.1*random(), 4.0 + 0.1*random()), 10)) ))

        >>> tg.train_one_sequence(obs_seq1)

        >>> obs_seq1 = list(chain(imap(array, repeat((1.0 + 0.1*random(), 0.5 + 0.1*random()), 20)) ,
        ...                       imap(array, repeat((3.0 + 0.1*random(), 4.0 + 0.1*random()), 10)) ))

        >>> tg.train_one_sequence(obs_seq1)

        >>> obs_seq1 = list(chain(imap(array, repeat((1.0 + 0.1*random(), 0.5 + 0.1*random()), 20)) ,
        ...                       imap(array, repeat((3.0 + 0.1*random(), 4.0 + 0.1*random()), 10)) ))

        >>> tg.train_one_sequence(obs_seq1)

        >>> with DebugPrint("gaussian_numeric_error"):
        ...    gmm_mgr.set_adaptation_state("APPLYING")
        ...    hmm_mgr.set_adaptation_state("APPLYING")
        ...    gmm_mgr.apply_all_accumulators()
        ...    hmm_mgr.apply_all_accumulators()
        ...    tg.end_training()
        ...    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
        ...    hmm_mgr.set_adaptation_state("NOT_ADAPTING")

        >>> print tg
        TrainingGraph with 7 nodes:

        >>> hmm_conv = tg.convert_to_standalone_hmm()
        >>> hmm_conv.to_string()
        'Hmm: num_states = 56, model dim = 2'
        """
        self._hmm_mgr = hmm_mgr
        self._graph = graph
        if not graph.is_lattice() or graph.has_self_loop():
            raise ValueError("graph is not a lattice or has a self loop")

        starts, ends  = graph.get_terminals()
        assert len(starts) == len(ends) == 1
        self._start_node_id, self._end_node_id = starts[0], ends[0]
        self._start_node_label, self._start_model_label = self._graph.get_node_label(self._start_node_id)
        self._end_node_label, self._end_model_label = self._graph.get_node_label(self._end_node_id)

        self._supported_dimensions = None
        self._label_pairs = set()
        # map from node labels to model labels
        self._nt_to_mt = dict()
        for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes():
            self._label_pairs.add(label_pair)
            node_label, model_label = label_pair
            assert model_label >= 0
            self._nt_to_mt[node_label] = model_label
            m = self._hmm_mgr[model_label]

            # XXX what we'd really like here is a set class with a "universal"
            # singleton that handled intersection and union operations
            # correctly.  Instead, we're going to use None as the universal set
            # and handle intersection here explicitly.  See also the uses of
            # this attribute elsewhere
            if m.dimension is not None:
                if self._supported_dimensions is not None:
                    self._supported_dimensions = self._supported_dimensions.intersection(set((m.dimension,)))
                else:
                    self._supported_dimensions = set((m.dimension,))
                    
        # Also keep a set of just the model labels, which may be smaller
        self._model_labels = set([pair[1] for pair in self._label_pairs])
        self._gmm_mgr = self._hmm_mgr[self._start_model_label].gmm_manager

        self._split_prob_dict = dict()
        self._split_accum_dict = dict()
        # Validate split_prob_dict in simple ways.  Much more thorough checking is done in _verify()
        for (pair,v) in split_prob_dict.items():
            if len(pair) != 2:
                raise ValueError("split_prob_dict has a bad key: %s; expected a pair of node labels" % (pair,))
            (pred, succ) = pair
            pred_model_idx = self._nt_to_mt[pred]
            no = self._hmm_mgr[pred_model_idx].num_outputs
            if len(v) != no:
                raise ValueError("split_prob_dict has a bad value for key %s; expected length of %d" % (pair, no))
            self._split_prob_dict[pair] = numpy.array(v, dtype=float)
            self._split_accum_dict[pair] = numpy.zeros((no), dtype=float)
            
        self._split_pairs = frozenset(self._split_prob_dict.keys())
        self._verify()
        self._adapting = False
        


######################## PRINTING AND DIAGNOSTICS ###########################

    def __str__(self):
        return self.to_string()

    def to_string(self, full = False):
        ret = "TrainingGraph with %d nodes:" % self._graph.get_num_nodes()
        return ret

    def _default_node_callback(self, node_label_pair, is_start, is_end):
        (node_index, model_index) = node_label_pair
        attribs = ('label="Node %d (HMM %d)"' % (node_index, model_index), 'style=bold')
        return attribs
    
    def dot_iter(self, expand_hmms=False, bwt_label='', graph_type='digraph', globals=(),
                 node_callback=_default_node_callback):
        """
        Returns a generator that yields lines of text, including newlines.  The
        text represents the TrainingGraph in the DOT language for which there
        are many displayers.  See also dot_display().

        Optional argument bwt_label is a string label for the TrainingGraph.
        Optional argument graph_type defaults to 'digraph', can also be 'graph'
        and 'subgraph'

        Optional globals is an iterable that yields strings to put in the
        globals section of the DOT file.
        """
        # By default we use a left-to-right layout, but we prepend this so that if the caller's
        # globals includes a different rankdir attribute it will be effective.
        globals = ('rankdir=LR',) + globals

        # If we're not expanding the HMMs, we just generate the graph
        if not expand_hmms:
            for line in self._graph.dot_iter(graph_label=bwt_label, graph_type=graph_type, globals=globals,
                                             node_attributes_callback=self._default_node_callback):
                yield line

        else:
            yield "digraph %s {\n" % (bwt_label,)

            # Change fonts on nodes and labels globally; you could probably
            # override these with globals, see below
            yield 'node [fontname=ArialRoundedMTBold, label="\N"];'
            yield 'edge [fontname=ArialRoundedMTBold];'
            
            for g in globals:
                yield "%s\n" % (g,)
                
            current_state_num = 0
            out_names_dict = {}
            for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes():
                node_label, model_label = label_pair
                m = self._hmm_mgr[model_label]
                label = 'cluster_hmm_%d_%d' % label_pair
                line_gen, in_names, out_names = m.dot_iter(hmm_label=label, graph_type='subgraph',
                                                           start_state_num=current_state_num)
                out_names_dict[node_label] = out_names
                for line in line_gen:
                    yield line
                current_state_num += m.num_inputs + m.num_states + m.num_outputs

                for (arc_label, start_node_pair) in in_arc_iter:
                    start_node_label, start_model_label = start_node_pair
                    assert out_names_dict.has_key(start_node_label)
                    start_out_names = out_names_dict[start_node_label]
                    assert len(start_out_names) == len(in_names)
                    for start, end, i in izip(start_out_names, in_names, count()):
                        node_pair_key = (start_node_label, node_label)
                        if self._split_prob_dict.has_key(node_pair_key):
                            prob = self._split_prob_dict[node_pair_key][i]
                            prob_label = '[label="%s", style=dashed, fontcolor=red]' % (prob,)
                        else:
                            prob_label = '[style=dashed]'
                        yield "%s -> %s %s\n" % (start, end, prob_label)

            yield "} /* digraph %s */\n" % (bwt_label,)
                



    def dot_display(self, temp_file_prefix='bwt_',
                    display_tool_format="open -a /Applications/Graphviz.app %s", **kwargs):
        """
        Display a dot-generated representation of the TrainingGraph.  Returns
        the name of the temporary file that the display tool is working from.
        The caller is responsible for removing this file.

        Optional temp_file_prefix is the prefix used for the filename.

        Optional display_tool_format is formatting string, with %s where the filename goes, used to
        generate the command that will display the file.  By default it assumes you're on a Mac and
        have Graphviz.app installed in the /Applications directory.

        Remaining keyword arguments are handed to the dot_iter function.
        """
        import os
        from onyx.util import opentemp
        with opentemp('wb', suffix='.dot', prefix=temp_file_prefix) as (name, outfile):
            for line in self.dot_iter(**kwargs):
                outfile.write(line)
        os.system(display_tool_format % (name,))
        return name


    def convert_to_standalone_hmm(self, alt_gmm_mgr=None):
        """
        Convert a TrainingGraph instance into a standalone Hmm.  Returns the new
        Hmm.  If alt_gmm_mgr is not None, it will be used as the GmmMgr of the
        resulting Hmm.  This allows the result to avoid sharing models with the
        original TrainingGraph.
        """

        # This function converts a lattice of Hmms into a single Hmm.
        # The resulting Hmm will have inputs corresponding to the
        # inputs of the starting Hmm in the lattice and outputs
        # corresponding to the outputs of the end Hmm in the lattice.
        # In between, the new Hmm will have as many real states as the
        # sum of the real states in the lattice, with transition
        # probabilities matching those between corresponding states in
        # the lattice Hmms.  That is, for any original (real) state R1
        # in some Hmm in the lattice, there will be a new state N1 in
        # the resulting Hmm, and for any pair of states R1,R2, the
        # transition probability for N1->N2 will be the same as that
        # from R1->R2.  The case in which the R1->R2 transition is
        # really across nodes in the lattice is the tricky part; we
        # need to consider all ways of getting from R1 through an
        # output node, into an input node, and then to R2, and
        # generate the new transition probability by taking the sum of
        # all products of probabilities on such paths.
        
        # Count numbers of virtual input, virtual output, and real
        # states.  Virtual inputs are just those of the Hmm in the
        # start node
        dc = dcheck("bwt_ctsh")
        start_hmm = self._hmm_mgr[self._start_model_label]
        num_virtual_inputs = start_hmm.num_inputs
        # Virtual outputs are just those of the Hmm in the end node
        end_hmm = self._hmm_mgr[self._end_model_label]
        num_virtual_outputs = end_hmm.num_outputs
        # Real states are all real states of all Hmms

        num_real_states = 0
        models = []
        # Walk the graph in topological order, collecting states in
        # each model

        # This maps from node labels to their starting position in the
        # new transition matrix
        offsets = dict()
        current_offset = start_hmm.num_inputs
        for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes():
            node_label, model_label = label_pair
            m = self._hmm_mgr[model_label]
            num_real_states += m.num_states
            models += m.models
            offsets[node_label] = current_offset
            current_offset += m.num_states
                    
        dc and dc("offsets = %s" % (offsets,))
        assert len(models) == num_real_states
        trans_size = num_virtual_inputs + num_real_states + num_virtual_outputs
        trans = numpy.zeros((trans_size, trans_size), dtype = float)
        dc and dc("trans.shape = %s" % (trans.shape,))

        # This keeps track of transitions through HMMs which have non-zero
        # transitions directly from inputs to outputs
        skip_info = defaultdict(list)

        # Extract transition probs from starting model.  ctp is the
        # current trans position, where we are in the big transition
        # array we're building.
        ctp = 0
        hmm_trans = start_hmm.transition_matrix
        dc and dc("start node hmm_trans = \n%s" % (hmm_trans,))
        ncols, nrows = hmm_trans.shape
        ni = start_hmm.num_inputs
        ns = start_hmm.num_states
        assert ncols == nrows
        trans[:ni,:ncols] = hmm_trans[:ni,:ncols]
        ctp += ni
        
        # Walk the graph in topological order again, extracting
        # remaining transition probs
        for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes():
            # First do work needed for just this node
            node_label, model_label = label_pair
            m = self._hmm_mgr[model_label]
            ni = m.num_inputs
            ns = m.num_states
            no = m.num_outputs
            hmm_trans = m.transition_matrix

            # This section extracts the probabilities for transitions that stay
            # within the Hmm.  These are put into the new, larger matrix at the
            # correct location.
            dc and dc("ctp = %d, ni = %d, ns = %d, no = %d" % (ctp, ni, ns, no))
            # We have to index differently for the last node in the network
            if node_label == self._end_node_label:
                trans[ctp:ctp+ns+no,ctp:ctp+ns+no] = hmm_trans[ni:,ni:]
            else:
                trans[ctp:ctp+ns,ctp:ctp+ns] = hmm_trans[ni:ni+ns,ni:ni+ns]

            ctp += ns

            # Now do work that must be done for each successor node.
            for (arc_label, end_node_pair) in out_arc_iter:
                end_node_label, end_model_label = end_node_pair
                m2 = self._hmm_mgr[end_model_label]

                # This section extracts the probabilities for transitions that
                # cross from one Hmm to a successor.  The new transition
                # probability for S1 -> S2 is a product of the prob.  going to
                # the output state and the prob. going from the input state.
                hmm2_trans = m2.transition_matrix
                ni2 = m2.num_inputs
                ns2 = m2.num_states
                assert no == ni2
                dc and dc("hmm_trans = \n%s" % (hmm_trans,))
                dc and dc("hmm_trans[ni:ni+ns,ni+ns:] = \n%s" % (hmm_trans[ni:ni+ns,ni+ns:],))
                out1 = hmm_trans[ni:ni+ns,ni+ns:]
                dc and dc("hmm2_trans = \n%s" % (hmm2_trans,))
                dc and dc("hmm2_trans[0:ni2,ni2:ni2+ns2:] = \n%s" % (hmm2_trans[0:ni2,ni2:ni2+ns2:],))
                in2 = hmm2_trans[0:ni2,ni2:ni2+ns2:]

                dc and dc("hmm2_trans[0:ni2,ni2+ns2:] = \n%s" % (hmm2_trans[0:ni2,ni2+ns2:],))
                skip2 = hmm2_trans[0:ni2,ni2+ns2:]

                # Here is where we apply the split probabilities for
                # nodes with more than one successor
                if self._split_prob_dict.has_key((node_label, end_node_label),):
                    split_probs = self._split_prob_dict[(node_label, end_node_label)]
                    temp = out1 * split_probs
                    dc and dc("split_probs = \n%s" % (split_probs,))
                    dc and dc("temp = \n%s" % (temp,))
                    dc and dc("skip2 = \n%s" % (skip2,))
                    prod = numpy.dot(temp, in2)
                    skip_prod = numpy.dot(temp, skip2)
                else:
                    prod = numpy.dot(out1, in2)
                    skip_prod = numpy.dot(out1, skip2)

                # Write cross-HMM transition probs for this pair into the new
                # table
                o1 = offsets[node_label]
                o2 = offsets[end_node_label]
                trans[o1:o1+ns, o2:o2+ns2] = prod

                # Deal with HMMs which can skip directly from their inputs to
                # their outputs.  We do this in two steps.  First, if the
                # second of the current pair can skip, we store some information
                # in a dictionary, unless the second of the current pair is the
                # end node of the lattice.
                dc and dc("prod = \n%s" % (prod,))
                if (skip_prod != 0.0).any():
                    if end_node_label == self._end_node_label:
                        dc and dc("Adding output trans prob %s at trans location %d:%d, %d:" % (skip_prod, o1, o1+ns,
                                                                                                  o2+ns2))
                        assert skip_prod.shape == (ns, num_virtual_outputs)
                        trans[o1:o1+ns, o2+ns2:] += skip_prod
                    else:
                        dc and dc("Storing skip info for node %s (from node %s), skip_prod = \n%s" %
                                  (end_node_label, node_label, skip_prod))
                        skip_info[end_node_label].append((node_label, ns, skip_prod))

                            
                # Second, if there's skip information stored for
                # the first of the current pair, we use it to add additional
                # entries in the new table and/or to store more skip
                # information for HMMs downstream.
                if skip_info.has_key(node_label):
                    for (pred_node_label, pred_ns, pred_skip_prod) in skip_info[node_label]:
                        if self._split_prob_dict.has_key((node_label, end_node_label),):
                            split_probs = self._split_prob_dict[(node_label, end_node_label)]
                            temp = pred_skip_prod * split_probs
                            dc and dc("split_probs = \n%s" % (split_probs,))
                            dc and dc("temp = \n%s" % (temp,))
                            dc and dc("skip2 = \n%s" % (skip2,))
                            new_prod = numpy.dot(temp, in2)
                            new_skip_prod = numpy.dot(temp, skip2)
                        else:
                            new_prod = numpy.dot(pred_skip_prod, in2)
                            new_skip_prod = numpy.dot(pred_skip_prod, skip2)

                        o1 = offsets[pred_node_label]
                        o2 = offsets[end_node_label]
                        # Note that in this case, there may be several
                        # contributions to the transition probabilities from
                        # some (possibly distant) predecessor to a given state.
                        # This will happen if there are two distinct paths
                        # through skippable HMMs from one state to another.  So
                        # we "accumulate" transition probabilities here with '+='
                        dc and dc("Adding trans prob %s at trans location %d:%d, %d:%d" % (new_prod, o1, o1+pred_ns,
                                                                                           o2, o2+ns2))
                        assert new_prod.shape == (pred_ns, ns2)
                        trans[o1:o1+pred_ns, o2:o2+ns2] += new_prod
                        if (new_skip_prod != 0.0).any():
                            if end_node_label == self._end_node_label:

                                dc and dc("Adding secondary output trans prob %s at trans location %d:%d, %d:" % (new_skip_prod, o1, o1+pred_ns,
                                                                                                  o2+ns2))
                                assert new_skip_prod.shape == (pred_ns, num_virtual_outputs)
                                trans[o1:o1+pred_ns, o2+ns2:] += new_skip_prod

                            else:
                                dc and dc("Storing secondary skip info for node %s (from node %s), skip_prod = \n%s" %
                                          (end_node_label, pred_node_label, new_skip_prod))
                                skip_info[end_node_label].append((pred_node_label, pred_ns, new_skip_prod))


                    
        ret = Hmm(num_real_states, log_domain=self._hmm_mgr.log_domain)
        dc and dc("trans = \n%s" % (trans,))
        mm = self._gmm_mgr if alt_gmm_mgr is None else alt_gmm_mgr
        ret.build_model(mm, models, num_virtual_inputs, num_virtual_outputs, trans)
        return ret
    

######################## PUBLIC  TRAINING INTERFACE ###################################


    def begin_training(self):
        """
        Set up this TrainingGraph for training.
        """
        if self._adapting:
            raise RuntimeError("begin_training called while already training")

        # Note that we do this just once per model, not once per node
        for label in self._model_labels:
            self._hmm_mgr[label].begin_adapt("NETWORK")

        self._adapting = True
        self._clear_split_prob_accums()

    def train_one_sequence(self, obs_iter):
        """
        Train the graph on one iterable of observations.  This call only accumulates; changes to
        models are only made by end_training.
        """
        dc = dcheck("bwt_tos")
        if not self._adapting:
            raise RuntimeError("train_one_sequence called without a prior call to begin_training")
        obs_seq = tuple(obs_iter)
        seq_len = len(obs_seq)

        if len(obs_seq) == 0:
            raise ValueError("empty training sequence")
            
        obs_dim = len(obs_seq[0])
        if self._supported_dimensions is not None and obs_dim not in self._supported_dimensions:
            raise ValueError("expected observations with dimension in %s, but got %d" % (self._supported_dimensions,
                                                                                         obs_dim))
        
        contexts = dict()
        # Get Hmms ready for forward pass
        for nt,mt in self._label_pairs:
            assert not contexts.has_key(nt)
            contexts[nt] = self._hmm_mgr[mt].init_for_forward_pass(obs_seq, nt is self._end_node_label)

        start_hmm = self._hmm_mgr[self._start_model_label]
        start_context = contexts[self._start_node_label]
        # Add initial mass to inputs of start Hmm.  By specifying alphas=None,
        # the Hmm will divide a mass of 1.0 evenly across all inputs.
        start_hmm.accum_input_alphas(start_context, alphas=None)
        
        # Do forward pass
        for i in xrange(seq_len+1):
            dc and dc(DebugPrint.NEWLINE_PREFIX, "Processing frame %d forward" % (i,))
            self._train_one_frame_forward(contexts)
            
        # Get Hmms ready for backward pass
        for nt,mt in self._label_pairs:
            self._hmm_mgr[mt].init_for_backward_pass(contexts[nt])

        end_hmm = self._hmm_mgr[self._start_model_label]
        end_context = contexts[self._end_node_label]
        # Add initial mass to inputs of start Hmm.  By specifying alphas=None,
        # the Hmm will use a mass of 1.0 on every output.
        end_hmm.accum_input_betas(end_context, betas=None)

        # Do backward pass
        for i in xrange(seq_len+1):
            dc and dc(DebugPrint.NEWLINE_PREFIX, "Processing frame %d backward" % (i,))
            self._train_one_frame_backward(contexts)

        # Collect gamma sums for normalization
        total_gamma_sum = start_hmm.get_initial_gamma_sum()
        for nt,mt in self._label_pairs:
            self._hmm_mgr[mt].add_to_gamma_sum(total_gamma_sum, contexts[nt])

        # Do accumulation
        for nt,mt in self._label_pairs:
            self._hmm_mgr[mt].do_accumulation(contexts[nt], total_gamma_sum)
        
        # Accumulate split probability sums
        self._accum_split_prob_sums(contexts, total_gamma_sum)

    def end_training(self):
        """
        End training on this TrainingGraph, which does the actual adaptation of split-probabilities,
        but not of the Hmms in this graph.  To adapt Hmms, call apply_all_accumulators on the
        relevant HmmMgr.
        """
        dc = dcheck("bwt_et")
        if not self._adapting:
            raise RuntimeError("end_training called without a prior call to begin_training")
        dc and dc("end_training called")

        # Apply split probability accumulators
        self._apply_split_prob_accums()
        self._adapting = False

######################## INTERNAL FUNCTIONS ###################################

    def _train_one_frame_forward(self, contexts):
        # Walk the graph in topological order, generating alphas in
        # each model and transferring outputs to inputs.
        dc = dcheck("bwt_toff")
        for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes():
            # First do work needed for just this node
            start_nt, start_mt = label_pair
            m = self._hmm_mgr[start_mt]
            output_alphas = m.process_one_frame_forward(contexts[start_nt])
            if output_alphas is None:
                continue

            # Now do work needed for each successor
            for (arc_label, end_node_pair) in out_arc_iter:
                end_nt, end_mt = end_node_pair
                m2 = self._hmm_mgr[end_mt]
                if self._split_prob_dict.has_key((start_nt, end_nt)):
                    m2.accum_input_alphas(contexts[end_nt], output_alphas, self._split_prob_dict[(start_nt, end_nt)])
                    dc and dc("transferring output alphas %s with split_probs %s from node %d to node %d" %
                              (output_alphas, self._split_prob_dict[(start_nt, end_nt)], start_nt, end_nt))
                else:
                    m2.accum_input_alphas(contexts[end_nt], output_alphas)
                    dc and dc("transferring output alphas %s from node %d to node %d" %
                              (output_alphas, start_nt, end_nt))


    def _train_one_frame_backward(self, contexts):
        # Walk the graph in *reverse* topological order, generating
        # betas in each model and transferring outputs to inputs.
        dc = dcheck("bwt_tofb")
        for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes(reverse_order=True):
            # First do work needed for just this node
            end_nt, end_mt = label_pair
            m = self._hmm_mgr[end_mt]
            output_betas = m.process_one_frame_backward(contexts[end_nt])
            if output_betas is None:
                continue

            # Now do work needed for each successor
            for (arc_label, start_node_pair) in in_arc_iter:
                start_nt, start_mt = start_node_pair
                m2 = self._hmm_mgr[start_mt]
                if self._split_prob_dict.has_key((start_nt, end_nt)):
                    m2.accum_input_betas(contexts[start_nt], output_betas, self._split_prob_dict[(start_nt, end_nt)])
                    dc and dc("transferring output betas %s with split_probs %s from node %d to node %d" %
                              (output_betas, self._split_prob_dict[(start_nt, end_nt)], end_nt, start_nt))
                else:
                    dc and dc("transferring output betas %s from node %d to node %d" %
                              (output_betas, end_nt, start_nt))
                    m2.accum_input_betas(contexts[start_nt], output_betas)
            

    def _clear_split_prob_accums(self):
        for pair,accum in self._split_accum_dict.items():
            accum[:] *= 0.0
        
    def _accum_split_prob_sums(self, contexts, total_gamma_sum):
        dc = dcheck("bwt_asps")
        for pred_label, succ_label in self._split_pairs:
            dc and dc("Accumulating for pair %d --> %d" % (pred_label, succ_label))
            pred_mt = self._nt_to_mt[pred_label]
            succ_mt = self._nt_to_mt[succ_label]
            mp = self._hmm_mgr[pred_mt]
            ms = self._hmm_mgr[succ_mt]
            
            pcxt = contexts[pred_label]
            scxt = contexts[succ_label]

            accum_vals = mp.get_accum_values(total_gamma_sum, pcxt, ms, scxt)
            num_frames = len(accum_vals)
            accum = self._split_accum_dict[pred_label, succ_label]
            for t in xrange(num_frames-1):
                accum += accum_vals[t]
            dc and dc("accum = %s" % (accum))

    def _apply_split_prob_accums(self):
        dc = dcheck("bwt_aspa")
        norm_dict = dict()
        for pred_label, succ_label in self._split_pairs:
            val = self._split_prob_dict[pred_label, succ_label] * self._split_accum_dict[pred_label, succ_label]
            if norm_dict.has_key(pred_label):
                norm_dict[pred_label] += val
            else:
                norm_dict[pred_label] = val
                
        for pred_label, succ_label in self._split_pairs:
            dc and dc("Applying for pair %d --> %d" % (pred_label, succ_label))
            dc and dc("norm_dict[pred_label] = %s" % (norm_dict[pred_label],))
            accum = self._split_accum_dict[pred_label, succ_label]
            split = self._split_prob_dict[pred_label, succ_label]
            val = split * accum
            assert(norm_dict.has_key(pred_label))
            val /= norm_dict[pred_label]
            self._split_prob_dict[pred_label, succ_label] = numpy.nan_to_num(val)
            dc and dc("New split for pair %d --> %d is %s" % (pred_label, succ_label, val))
            

    def _verify(self):
        # Walk the graph and make sure that the models, their inputs and
        # outputs, and the connections between units, are all sensible.
        dc = dcheck("bwt_vrfy")

        node_labels_seen = set()
        max_label = self._hmm_mgr.num_models - 1
        spd_sums = dict()
        preds_seen_but_not_in_spd = set()
        
        for (label_pair, in_arc_iter, out_arc_iter) in self._graph.iter_nodes():
            dc and dc("label_pair is %s"%(label_pair,))
            start_node_label, model_label = label_pair
            m = self._hmm_mgr[model_label]

            if start_node_label in node_labels_seen:
                raise ValueError("graph has two nodes with the same node label")
            node_labels_seen.add(start_node_label)
            if m.gmm_manager_id != id(self._gmm_mgr):
                raise ValueError("graph has a model with the wrong GmmMgr instance")
            if model_label > max_label:
                raise ValueError("graph has a node with label %d, max is %d" % (start_model_label, max_label))

            for (arc_label, end_node_pair) in out_arc_iter:
                end_node_label, end_model_label = end_node_pair
                dc and dc("Node %s feeds into node %s\n" % (start_node_label, end_node_label))
                m2 = self._hmm_mgr[end_model_label]
                if m.num_outputs != m2.num_inputs:
                    raise ValueError("graph has a node pair with mismatched models - num_outputs (%d) != num_inputs (%d)" %
                                     (m.num_outputs, m2.num_inputs))
                        
                # Verify correct split_prob_dict entries.  We accumulate, for
                # each node in the graph, a Numpy array of values from the dict
                # for each successor.  Separately, we check to make sure any
                # pred node not present in the SPD has only one successor.  At
                # the end, every node with multiple successors should have an
                # array with values all 1.
                spd = self._split_prob_dict
                pair_key = (start_node_label, end_node_label)
                if spd.has_key(pair_key):
                    probs = spd[pair_key]
                    assert probs.shape == (m.num_outputs,)
                        
                    if spd_sums.has_key(start_node_label):
                        spd_sums[start_node_label] += probs
                    else:
                        spd_sums[start_node_label] = array(probs) # Make copy to avoid aliasing!!
                else:
                    if start_node_label in preds_seen_but_not_in_spd:
                        raise ValueError("graph has a start node (label %d) with missing split_prob_dict values" % start_node_label)
                    preds_seen_but_not_in_spd.add(start_node_label)
                
            dc and dc(m.to_string(full=True))

        # Finish verifying split_prob_dict:
        for k,v in spd_sums.items():
            dc and dc("start_node_label = %d; sum = %s" % (k,v))
            correct = numpy.ones((m.num_outputs,), dtype=float)
            if not numpy.allclose(correct, v):
                raise ValueError("graph has a start node (label %d) with bad split_prob_dict values" % k)


######################## TEST HELPER FUNCTIONS ###################################

from onyx.am.hmm import toy_probs
from random import randint, seed
from onyx.am.gaussian import GaussianMixtureModel, DummyModel
from numpy import array, eye

def make_gmm_mgr(num_models):
    dimension = 2
    num_mixtures = 2

    def make_gmm(dimension, num_mixtures):
        gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.FULL_COVARIANCE, num_mixtures)
        w = [1.0 / num_mixtures for n in xrange(num_mixtures)]
        gmm.set_weights(array(w))
        mu = array(((1.5,1.5), (3,3)))
        v = array((eye(2), eye(2)))
        gmm.set_model(mu, v)
        return gmm

    gmm_mgr = GmmMgr( (make_gmm(dimension, num_mixtures) for i in xrange(num_models)) )
    return gmm_mgr

def make_forward_hmm(mm, num_states, order, models=None, exact=False):
    hmm0 = Hmm(num_states)
    # generate a set of random indices from the GmmMgr
    models = tuple(randint(0, mm.num_models-1) for i in xrange(num_states)) if models is None else models
    trans = tuple([p] * num_states for p in toy_probs(order))
    if exact:
        hmm0.build_forward_model_exact(mm, models, order, trans)
    else:
        hmm0.build_forward_model_compact(mm, models, order, trans)

    return hmm0

def _write_dot_output(tg):
    # Write two versions of this graph in dot form
    ret = "\n========= DOT OUTPUT ============\n"
    for line in tg.dot_iter():
        ret += line
    for line in tg.dot_iter(expand_hmms=True):
        ret += line
    ret += "\n========= END DOT OUTPUT ============\n"
    return ret


def _test0():
    ret1 = ""
    num_states = 3
    dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4) )
    gmm_mgr = GmmMgr(dummies)
    models = range(3)
    
    hmm1 = Hmm(num_states)
    hmm1.build_forward_model_compact(gmm_mgr, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    gmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    hmm1.begin_adapt("STANDALONE")
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    obs = [array((0,0))]*12
    hmm1.adapt_one_sequence(obs)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm1.end_adapt()
    gmm_mgr.apply_all_accumulators()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    ret1 += hmm1.to_string(full=True)
    
    # Hmm: num_states = 3
    # Models (dim = 2):
    # GmmMgr index list: [0, 1, 2]
    # DummyModel (Dim = 2) always returning a score of 0.1
    # DummyModel (Dim = 2) always returning a score of 0.2
    # DummyModel (Dim = 2) always returning a score of 0.4
    # Transition probabilities:
    # [[ 0.          1.          0.          0.          0.        ]
    # [ 0.          0.2494527   0.7505473   0.          0.        ]
    # [ 0.          0.          0.49667697  0.50332303  0.        ]
    # [ 0.          0.          0.          0.88480382  0.11519618]
    # [ 0.          0.          0.          0.          0.        ]]
    durs = hmm1.find_expected_durations(12)
    ret1 += "\nExpected durations = %s\n" % (durs,)
    # Expected durations = [ 1.33236099  1.98543653  5.62090219  3.06130028]
    

    # Now do the same thing in a BWTrainingGraph
    ret2 = ""
    hmm1 = Hmm(num_states)
    hmm1.build_forward_model_compact(gmm_mgr, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    hmm2 = Hmm(num_states)
    hmm2.build_forward_model_compact(gmm_mgr, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    hmm_mgr = HmmMgr((hmm1,hmm2))
    
    # Here's a trivial case of a graph with only one node, containing
    # one model:
    gb = GraphBuilder()
    node_id = gb.new_node((0,0))
    gr0 = FrozenGraph(gb)

    # with DebugPrint("bwt_vrfy"):
    tg0 = TrainingGraph(gr0, hmm_mgr, dict())
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    tg0.begin_training()
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    hmm_mgr.set_adaptation_state("ACCUMULATING")
    tg0.train_one_sequence(obs)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    tg0.end_training()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")

    ret2 +=  hmm1.to_string(full=True)
    durs = hmm1.find_expected_durations(12)
    ret2 += "\nExpected durations = %s\n" % (durs,)
    return (ret1, ret2)
    

def _test1():
    gmm_mgr = make_gmm_mgr(20)

    # Here's a trivial case of a graph with only one node, containing
    # one model:
    gb = GraphBuilder()
    node_id = gb.new_node((0,0))
    gr0 = FrozenGraph(gb)
    
    obs_seq = list(chain(imap(array, repeat((1.0, 0.5), 5)) , imap(array, repeat((3.0, 4.0), 4)) ))

    # Perturb things a bit to avoid numerical problems
    for i in xrange(len(obs_seq)):
        obs_seq[i] *= (1 + i / 1000)

    # Make an Hmm with 8 states and order 3 (self loop, forward 1, forward 2)
    seed(0)
    hmm = make_forward_hmm(gmm_mgr, 8, 3) 
    hmm_mgr = HmmMgr((hmm,))
    # print hmm.to_string(full=True)
    tg0 = TrainingGraph(gr0, hmm_mgr, dict())
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    tg0.begin_training()
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    hmm_mgr.set_adaptation_state("ACCUMULATING")
    tg0.train_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    tg0.end_training()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")
    
    return hmm.to_string(full=True)

def _test1s():
    # Standalone version of _test1
    gmm_mgr = make_gmm_mgr(20)

    # Make an Hmm with 8 states and order 3 (self loop, forward 1, forward 2)
    seed(0)
    hmm = make_forward_hmm(gmm_mgr, 8, 3) 
    # print hmm.to_string(full=True)

    gmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    hmm.begin_adapt("STANDALONE")
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    
    obs_seq = list(chain(imap(array, repeat((1.0, 0.5), 5)) , imap(array, repeat((3.0, 4.0), 4)) ))

    # Perturb things a bit to avoid numerical problems
    for i in xrange(len(obs_seq)):
        obs_seq[i] *= (1 + i / 1000)
    # print obs_seq

    hmm.adapt_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm.end_adapt()
    gmm_mgr.apply_all_accumulators()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    return hmm.to_string(full=True)


def _test2(write_dot_output=False):
    # Here's a less trivial case of a graph with only two nodes
    # chained together.  Each node contains a 4-node order-3 Hmm.  The
    # results here should match those of _test1s(), which does the same
    # thing with one 8-state model.
    ret = ""
    gmm_mgr = make_gmm_mgr(20)
    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    arc_id = gb.new_arc(node_id0, node_id1)
    gr0 = FrozenGraph(gb)
    
    obs_seq = list(chain(imap(array, repeat((1.0, 0.5), 5)) , imap(array, repeat((3.0, 4.0), 4)) ))

    # Perturb things a bit to avoid numerical problems
    for i in xrange(len(obs_seq)):
        obs_seq[i] *= (1 + i / 1000)

    # Make two Hmms with 4 states and order 3 (self loop, forward 1, forward 2)
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, 4, 3) 
    hmm1 = make_forward_hmm(gmm_mgr, 4, 3) 
    hmm_mgr = HmmMgr((hmm0,hmm1))

    tg0 = TrainingGraph(gr0, hmm_mgr, dict())
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    tg0.begin_training()
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    hmm_mgr.set_adaptation_state("ACCUMULATING")
    with DebugPrint(DebugPrint.TIMESTAMP_ON, "bwt_tos") if True else DebugPrint():
        tg0.train_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    tg0.end_training()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")
    
    if write_dot_output:
        ret += _write_dot_output(tg0)

    ret += "\nModel 0:\n" + hmm0.to_string(full=True)
    ret += "\nModel 1:\n" + hmm1.to_string(full=True)
    return ret

def _test3(write_dot_output=False):
    # Here's a version of _test2 with DummyModels used to give scores
    # of 1 in every state on every frame.  Each node contains a 4-node
    # order-3 Hmm.  The results here should match those of _test3s(),
    # which does the same thing with one 8-state model.
    ret = ""
    # GmmMgr setup

    num_states = 4
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)
    models = range(num_states)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    arc_id = gb.new_arc(node_id0, node_id1)
    gr0 = FrozenGraph(gb)
    
    obs_seq = list(chain(imap(array, repeat((1.0, 0.5), 5)) , imap(array, repeat((3.0, 4.0), 4)) ))

    # Perturb things a bit to avoid numerical problems
    for i in xrange(len(obs_seq)):
        obs_seq[i] *= (1 + i / 1000)

    # Make two Hmms with 4 states and order 3 (self loop, forward 1, forward 2)
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 3) 
    hmm1 = make_forward_hmm(gmm_mgr, num_states, 3) 
    hmm_mgr = HmmMgr((hmm0,hmm1))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, dict())
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    tg0.begin_training()
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    hmm_mgr.set_adaptation_state("ACCUMULATING")
    with DebugPrint("hmm_atgs", "bwt_tofb", "hmm_pofb", "hmm_da") if False else DebugPrint():
        tg0.train_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    tg0.end_training()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")
    
    if write_dot_output:
        ret += _write_dot_output(tg0)

    ret += "\nModel 0:\n" + hmm0.to_string(full=True)
    ret += "\nModel 1:\n" + hmm1.to_string(full=True)
    return ret


def _test3s(write_dot_output=False):
    # Standalone version of _test3
    dimension = 2
    num_states = 8
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)
    models = range(num_states)

    # Make an Hmm with 8 states and order 3 (self loop, forward 1, forward 2)
    seed(0)
    hmm = make_forward_hmm(gmm_mgr, 8, 3) 
    # print hmm.to_string(full=True)

    gmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    hmm.begin_adapt("STANDALONE")
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    
    obs_seq = list(chain(imap(array, repeat((1.0, 0.5), 5)) , imap(array, repeat((3.0, 4.0), 4)) ))

    # Perturb things a bit to avoid numerical problems
    for i in xrange(len(obs_seq)):
        obs_seq[i] *= (1 + i / 1000)

    hmm.adapt_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm.end_adapt()
    gmm_mgr.apply_all_accumulators()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    return hmm.to_string(full=True)



def _test4(write_dot_output=False, do_display=False):
    # Here's a test of converting a TrainingGraph to a standalong Hmm.
    # Each of the 4 nodes contains a 4 (or 6)-node order-3 Hmm; the
    # nodes are connected in a diamond pattern
    ret = ""
    # GmmMgr setup

    num_states = 4
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,2))
    node_id3 = gb.new_node((3,3))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id1, node_id3)
    arc_id = gb.new_arc(node_id2, node_id3)
    gr0 = FrozenGraph(gb)
    
    # Make four Hmms with 4 (or 6) states and order 3 (self loop,
    # forward 1, forward 2)
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    # NB: the asymetry between the two successors is a key part of
    # this test; otherwise, there are no differences between the
    # transition probs going to these successors, which is the tricky
    # case
    hmm1 = make_forward_hmm(gmm_mgr, num_states + 2, 3, exact=True) 
    hmm2 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    hmm3 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2, hmm3))

    spd = {}
    spd[(0,1)] = (0.4, 0.3, 0.8)
    spd[(0,2)] = (0.6, 0.7, 0.2)

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret




def _test4a(write_dot_output=False, do_display=False):
    # Here's a test of converting a TrainingGraph to a standalong Hmm.
    # Each of the 4 nodes contains a 4 (or 6)-node order-3 Hmm; the
    # nodes are connected in a diamond pattern
    ret = ""
    # GmmMgr setup

    num_states = 4
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,2))
    node_id3 = gb.new_node((3,3))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id1, node_id3)
    arc_id = gb.new_arc(node_id2, node_id3)
    gr0 = FrozenGraph(gb)
    
    # Make four Hmms with 4 (or 6) states and order 3 (self loop,
    # forward 1, forward 2)
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 3, exact=False) 
    # NB: the asymetry between the two successors is a key part of
    # this test; otherwise, there are no differences between the
    # transition probs going to these successors, which is the tricky
    # case
    hmm1 = make_forward_hmm(gmm_mgr, num_states + 2, 3, exact=False) 
    hmm2 = make_forward_hmm(gmm_mgr, num_states, 3, exact=False) 
    hmm3 = make_forward_hmm(gmm_mgr, num_states, 3, exact=False) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2, hmm3))

    spd = {}
    spd[(0,1)] = (0.4, 0.3)
    spd[(0,2)] = (0.6, 0.7)

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret



def _test5(write_dot_output=False):
    # A reduced version of _test4
    ret = ""
    # GmmMgr setup

    num_states = 1
    dimension = 2
    models = []

    dm = DummyModel(dimension, 1.0)
    models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,2))
    node_id3 = gb.new_node((3,3))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id1, node_id3)
    arc_id = gb.new_arc(node_id2, node_id3)
    gr0 = FrozenGraph(gb)
    
    spd = {}
    spd[(0,1)] = (0.5,)
    spd[(0,2)] = (0.5,)

    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, 1, 2, exact=True) 
    # NB: the asymetry between the two successors is a key part of
    # this test; otherwise, there are now differences between the
    # transition probs going to these successors, which is the tricky
    # case
    hmm1 = make_forward_hmm(gmm_mgr, 2, 2, exact=True) 
    hmm2 = make_forward_hmm(gmm_mgr, 1, 2, exact=True) 
    hmm3 = make_forward_hmm(gmm_mgr, 1, 2, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2, hmm3))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, spd)

    result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    # Try adaptation on both models.  Since we're using dummy models,
    # the data values won't matter
    obs_seq = list(imap(array, repeat((1.0, 1.0), 6)))

    # Adapt standalone Hmm
    gmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    result_hmm.begin_adapt("STANDALONE")
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    result_hmm.adapt_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    result_hmm.end_adapt()
    gmm_mgr.apply_all_accumulators()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    print "\n=========================== DONE ADAPTING STANDALONE ========================\n"

    # Now adapt original TrainingGraph
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    tg0.begin_training()
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    hmm_mgr.set_adaptation_state("ACCUMULATING")
    tg0.train_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    tg0.end_training()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")

    # Now convert TG to Hmm again and see how they line up
    result_hmm2 = tg0.convert_to_standalone_hmm()

    if write_dot_output:
        ret += _write_dot_output(tg0)

    ret += "\n\n========= CONVERTED THEN ADAPTED AS Hmm =========\n\n" + result_hmm.to_string(full=True)
    ret += "\n\n========= ADAPTED AS TG THEN CONVERTED =========\n\n" + result_hmm2.to_string(full=True)
    return ret


def _test6(write_dot_output=False):
    # Same as _test5, but now with the expanded models of _test4 Each of
    # the 4 nodes contains a 4 (or 6)-node order-3 Hmm; the nodes are
    # connected in a diamond pattern
    ret = ""
    # GmmMgr setup

    num_states = 4
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)
    models = range(num_states)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,2))
    node_id3 = gb.new_node((3,3))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id1, node_id3)
    arc_id = gb.new_arc(node_id2, node_id3)
    gr0 = FrozenGraph(gb)
    
    # Make four Hmms with 4 (or 6) states and order 3 (self loop,
    # forward 1, forward 2)
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    # NB: the asymetry between the two successors is a key part of
    # this test; otherwise, there are no differences between the
    # transition probs going to these successors, which is the tricky
    # case
    hmm1 = make_forward_hmm(gmm_mgr, num_states + 2, 3, exact=True) 
    hmm2 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    hmm3 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2, hmm3))

    spd = {}
    spd[(0,1)] = (0.4, 0.3, 0.8)
    spd[(0,2)] = (0.6, 0.7, 0.2)

    tg0 = TrainingGraph(gr0, hmm_mgr, spd)

    result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    # Try adaptation on both models.  Since we're using dummy models,
    # the data values won't matter.
    obs_seq = list(imap(array, repeat((1.0, 1.0), 20)))

    # Adapt standalone Hmm
    gmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    result_hmm.begin_adapt("STANDALONE")
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    result_hmm.adapt_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    result_hmm.end_adapt()
    gmm_mgr.apply_all_accumulators()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    

    # Now adapt original TrainingGraph
    gmm_mgr.set_adaptation_state("INITIALIZING")
    hmm_mgr.set_adaptation_state("INITIALIZING")
    gmm_mgr.clear_all_accumulators()
    tg0.begin_training()
    gmm_mgr.set_adaptation_state("ACCUMULATING")
    hmm_mgr.set_adaptation_state("ACCUMULATING")
    tg0.train_one_sequence(obs_seq)
    gmm_mgr.set_adaptation_state("APPLYING")
    hmm_mgr.set_adaptation_state("APPLYING")
    gmm_mgr.apply_all_accumulators()
    hmm_mgr.apply_all_accumulators()
    tg0.end_training()
    gmm_mgr.set_adaptation_state("NOT_ADAPTING")
    hmm_mgr.set_adaptation_state("NOT_ADAPTING")

    # Now convert TG to Hmm again and see how they line up
    result_hmm2 = tg0.convert_to_standalone_hmm()

    if write_dot_output:
        ret += _write_dot_output(tg0)

    model1str = result_hmm.to_string(full=True)
    model2str = result_hmm2.to_string(full=True)
    ret += "\n\n========= CONVERTED THEN ADAPTED AS Hmm =========\n\n" + model1str
    ret += "\n\n========= ADAPTED AS TG THEN CONVERTED =========\n\n" + model2str
    ret += "\n model1str == model2str is: %s" % (model1str == model2str,)
    
    return ret


def _test7(write_dot_output=False, do_display=False):
    # Here's a test of converting a TrainingGraph to a standalong Hmm in which
    # one of the HMMs has a transition from an input directly to an output, so
    # it can behave as an epsilon.  This node is between two other nodes in a
    # linear arrangement.
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,2))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id2)
    gr0 = FrozenGraph(gb)
    
    # Make two Hmms with 3 states and order 2 (self loop, forward 1) The model
    # in the middle is special in that it can skip directly from the input state
    # to the output state.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 2, exact=False) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 1, 1, trans)
    hmm2 = make_forward_hmm(gmm_mgr, num_states, 2, exact=False) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def _test8(write_dot_output=False, do_display=False):
    # Like test7, but now with multiple skippable HMMs in a row.
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,1))
    node_id3 = gb.new_node((3,1))
    node_id4 = gb.new_node((4,1))
    node_id5 = gb.new_node((5,2))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id2)
    arc_id = gb.new_arc(node_id2, node_id3)
    arc_id = gb.new_arc(node_id3, node_id4)
    arc_id = gb.new_arc(node_id4, node_id5)
    gr0 = FrozenGraph(gb)
    
    # Make two Hmms with 3 states and order 2 (self loop, forward 1)
    # The models in the middle are special and can skip
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 2, exact=False) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 1, 1, trans)
    hmm2 = make_forward_hmm(gmm_mgr, num_states, 2, exact=False) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    # Write two versions of this graph in dot form
    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def _test9(write_dot_output=False, do_display=False):
    # Like test8, but now HMMs have multiple inputs and outputs.
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,1))
    node_id3 = gb.new_node((3,1))
    node_id4 = gb.new_node((4,1))
    node_id5 = gb.new_node((5,2))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id2)
    arc_id = gb.new_arc(node_id2, node_id3)
    arc_id = gb.new_arc(node_id3, node_id4)
    arc_id = gb.new_arc(node_id4, node_id5)
    gr0 = FrozenGraph(gb)
    
    # Make two Hmms with 3 states and order 3 (self loop, forward 1, forward 2)
    # The models in the middle are special and can skip directly
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, order=3, exact=True) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),
                   (0.0, 0.0, 0.0, 0.5, 0.35, 0.1, 0.05),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 3, 3, trans)
    hmm2 = make_forward_hmm(gmm_mgr, num_states, order=3, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def _test10(write_dot_output=False, do_display=False):
    # Like test9, but now HMMs are arranged in a diamond pattern so inter-HMM
    # probabilities come into play
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,1))
    node_id3 = gb.new_node((3,1))
    node_id4 = gb.new_node((4,1))
    node_id5 = gb.new_node((5,2))

    # The topology here is more complex than previous examples
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id5)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id2, node_id3)
    arc_id = gb.new_arc(node_id3, node_id4)
    arc_id = gb.new_arc(node_id3, node_id5)
    arc_id = gb.new_arc(node_id4, node_id5)
    gr0 = FrozenGraph(gb)

    # Make two Hmms with 3 states and order 3 (self loop, forward 1, forward 2)
    # The models in the middle are special and can skip.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, order=3, exact=True) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),
                   (0.0, 0.0, 0.0, 0.5, 0.35, 0.1, 0.05),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 3, 3, trans)
    hmm2 = make_forward_hmm(gmm_mgr, num_states, order=3, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    spd = {}
    spd[(0,1)] = (0.4, 0.3, 0.8)
    spd[(0,2)] = (0.6, 0.7, 0.2)

    spd[(3,4)] = (0.4, 0.3, 0.8)
    spd[(3,5)] = (0.6, 0.7, 0.2)


    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret

def _test11(write_dot_output=False, do_display=False):
    # A reduced version of test10
    ret = ""
    # GmmMgr setup

    num_states = 2
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,1))
    node_id3 = gb.new_node((3,1))
    node_id4 = gb.new_node((4,2))

    # The topology here is more complex than previous examples
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id4)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id2, node_id3)
    arc_id = gb.new_arc(node_id3, node_id4)
    arc_id = gb.new_arc(node_id2, node_id4)
    gr0 = FrozenGraph(gb)

    # Make two Hmms with 3 states and order 2 (self loop, forward 1)
    # The models in the middle are special and can skip.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, order=2, exact=False) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 1, 1, trans)
    hmm2 = make_forward_hmm(gmm_mgr, num_states, order=2, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    spd = {}
    spd[(0,1)] = (0.4,)
    spd[(0,2)] = (0.6,)

    spd[(2,3)] = (0.4,)
    spd[(2,4)] = (0.6,)


    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def _test12(write_dot_output=False, do_display=False):
    # Like test7, but now the skipping guy is the end node in the lattice.
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    arc_id = gb.new_arc(node_id0, node_id1)
    gr0 = FrozenGraph(gb)
    
    # Make one Hmm with 3 states and order 2 (self loop, forward 1) The model
    # at the end is special in that it can skip directly from the input state
    # to the output state.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 2, exact=False) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 1, 1, trans)
    hmm_mgr = HmmMgr((hmm0, hmm1))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret

def _test13(write_dot_output=False, do_display=False):
    # Like test12, but now the skipping guy is the *only* node in the lattice.
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    gr0 = FrozenGraph(gb)
    
    # Make one Hmm with 3 states and order 2 (self loop, forward 1) The model
    # at the end is special in that it can skip directly from the input state
    # to the output state.
    seed(0)
    hmm0 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm0.build_model(gmm_mgr, (0,), 1, 1, trans)
    hmm_mgr = HmmMgr((hmm0,))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def _test13(write_dot_output=False, do_display=False):
    # Like test10, but now the final HMM can skip
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,1))
    node_id3 = gb.new_node((3,1))
    node_id4 = gb.new_node((4,1))
    node_id5 = gb.new_node((5,1))

    # The topology here is more complex than previous examples
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id5)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id2, node_id3)
    arc_id = gb.new_arc(node_id3, node_id4)
    arc_id = gb.new_arc(node_id3, node_id5)
    arc_id = gb.new_arc(node_id4, node_id5)
    gr0 = FrozenGraph(gb)

    # Make two Hmms with 3 states and order 3 (self loop, forward 1, forward 2)
    # The models in the middle are special and can skip.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, order=3, exact=True) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),
                   (0.0, 0.0, 0.0, 0.5, 0.35, 0.1, 0.05),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 3, 3, trans)
    hmm_mgr = HmmMgr((hmm0, hmm1))

    spd = {}
    spd[(0,1)] = (0.4, 0.3, 0.8)
    spd[(0,2)] = (0.6, 0.7, 0.2)

    spd[(3,4)] = (0.4, 0.3, 0.8)
    spd[(3,5)] = (0.6, 0.7, 0.2)


    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret

def _test14(write_dot_output=False, do_display=False):
    # Like test13, but with very simple topology
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))

    # The topology here is less complex than the previous example
    arc_id = gb.new_arc(node_id0, node_id1)
    gr0 = FrozenGraph(gb)

    # Make one Hmm with 3 states and order 3 (self loop, forward 1, forward 2)
    # The model at the end is special and can skip.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, order=3, exact=True) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),
                   (0.0, 0.0, 0.0, 0.5, 0.35, 0.1, 0.05),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 3, 3, trans)
    hmm_mgr = HmmMgr((hmm0, hmm1))

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def _test15(write_dot_output=False, do_display=False):
    # Like test10, but now the final HMM can skip
    ret = ""
    # GmmMgr setup

    num_states = 3
    dimension = 2
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    gmm_mgr = GmmMgr(models)

    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,1))
    node_id3 = gb.new_node((3,1))

    # The topology here is medium complex 
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id1, node_id3)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id2, node_id3)
    gr0 = FrozenGraph(gb)

    # Make one Hmm with 2 states and order 2 (self loop, forward 1)
    # The models in the middle and end are special and can skip.
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 2, exact=False) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr, (0,), 1, 1, trans)
    hmm_mgr = HmmMgr((hmm0, hmm1))

    spd = {}
    spd[(0,1)] = (0.4,)
    spd[(0,2)] = (0.6,)

    with DebugPrint("bwt_vrfy") if False else DebugPrint():
        tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    if write_dot_output:
        ret += _write_dot_output(tg0)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    if do_display:
        result_hmm.dot_display(globals=("ratio=0.4","edge [fontsize=36]"))

    return ret


def logreftest():
    numpy.set_printoptions(linewidth = 200)
    print "================ TEST 0 ================"
    s0, s1 = _test0()
    print s0 == s1
    
    print "================ TEST 1S ================"
    s1s = _test1s()
    print s1s
    print "================ TEST 1 ================"
    s1 = _test1()
    print s1 == s1s
        
    print "================ TEST 2 ================"
    with DebugPrint("hmm_ea", "hmm_atgs", "bwt_tofb", "hmm_pofb", "hmm_poff", "hmm_da", "mm_as"):
        s2 = _test2()
    print s2
    
    remaining_test_fns = (_test3s, _test3, _test4, _test5, _test6, _test7, _test8, _test9,
                          _test10, _test11, _test12, _test13, _test14, _test15,   )
    dot_output_set = set((_test4,))

    for test_fn in remaining_test_fns:
        print "================ TEST %s ================" % (test_fn.__name__,)
        dot_output = (test_fn in dot_output_set)
        result = test_fn(write_dot_output=dot_output)
        print result



if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    args = argv[1:]
    if '--logreftest' in args:
        logreftest()
    else:
        # ADD YOUR TEMPORARY STUFF HERE AND COMMENT OUT THE EXIT
        exit()
    
##         print "================ TEST 0 ================"
##         s0, s1 = _test0()
##         print s0 == s1

##         print "================ TEST 1S ================"
##         with DebugPrint("hmm_aos", "hmm_gbff") if False else DebugPrint():
##             s1s = _test1s()
##         print s1s

##         print "================ TEST 1 ================"
##         s1 = _test1()
##         print s1 == s1s

##         print "================ TEST 2 ================"
##         with DebugPrint("hmm_ea") if False else DebugPrint():
##             s2 = _test2()
##         print s2

##         print "================ TEST 3S ================"
##         with DebugPrint("hmm_aos", "hmm_ea") if False else DebugPrint():
##             s3s = _test3s()
##         print s3s

##         print "================ TEST 3 ================"
##         with DebugPrint("hmm_ea") if False else DebugPrint():
##             s3 = _test3()
##         print s3

##         print "================ TEST 4 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if False else DebugPrint():
##             s4 = _test4(do_display=True)
##         print s4

##         print "================ TEST 4a ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if False else DebugPrint():
##             s4 = _test4a(do_display=True)
##         print s4

##         print "================ TEST 5 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("hmm_ea", "hmm_da", "hmm_aos", "hmm_atgs", "hmm_gxfs", "bwt_asps", "bwt_aspa", "bwt_vrfy", "bwt_ctsh") if True else DebugPrint():
##             s5 = _test5()
##         print s5         

##         print "================ TEST 6 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("hmm_ea", "hmm_da", "hmm_aos", "hmm_atgs", "hmm_gxfs", "bwt_asps", "bwt_aspa", "bwt_vrfy", "bwt_ctsh", "bwt_tos") if True else DebugPrint():
##             s6 = _test6()
##         print s6

##         print "================ TEST 7 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if False else DebugPrint():
##             s7 = _test7(do_display=False)
##         print s7

##         print "================ TEST 8 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if False else DebugPrint():
##             s8 = _test8(do_display=True)
##         print s8

##         print "================ TEST 9 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if True else DebugPrint():
##             s9 = _test9(do_display=False)
##         print s9

##         print "================ TEST 10 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if True else DebugPrint():
##             s10 = _test10(do_display=False)
##         print s10

##         print "================ TEST 11 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if True else DebugPrint():
##             s11 = _test11(do_display=False)
##         print s11

##         print "================ TEST 12 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if True else DebugPrint():
##             s12 = _test12(do_display=True)
##         print s12

        print "================ TEST 13 ================"
        numpy.set_printoptions(linewidth = 300)
        with DebugPrint("bwt_ctsh") if True else DebugPrint():
            s13 = _test13(do_display=True)
        print s13

##         print "================ TEST 14 ================"
##         numpy.set_printoptions(linewidth = 300)
##         with DebugPrint("bwt_ctsh") if True else DebugPrint():
##             s14 = _test14(do_display=True)
##         print s14        

        print "================ TEST 15 ================"
        numpy.set_printoptions(linewidth = 300)
        with DebugPrint("bwt_ctsh") if True else DebugPrint():
            s15 = _test15(do_display=True)
        print s15        
