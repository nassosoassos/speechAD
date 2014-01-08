###########################################################################
#
# File:         framesynchronous.py
# Date:         25-March-2008
# Author:       Hugh Secker-Walker
# Description:  Pedagogical outline of frame synchronous decoding
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
Pedagogical outline of frame synchronous decoding

"""

from __future__ import with_statement
from __future__ import division
from os import path
from itertools import izip, chain, repeat, tee, count
from collections import defaultdict
import time
from math import log10
import numpy as N

import onyx
from onyx.am.gaussian import DummyModel
from onyx.am.hmm import Hmm
from onyx.am.hmm_mgr import HmmMgr
from onyx.am.modelmgr import GmmMgr
from onyx.builtin import frozendict, frozentuple
from onyx.containers import tuplen
from onyx.graph.decodergraph import DecoderGraph
from onyx.graph.graphtools import TopologicalGraphBuilder, FrozenGraph
from onyx.htkfiles.htkaudio import read_htk_audio_file
from onyx.htkfiles.htkmmf import read_htk_mmf_file
from onyx.util.floatutils import float_to_readable_string
from onyx.util.streamprocess import ProcessorBase, ChainProcessor
from onyx.util.debugprint import dcheck, DebugPrint
from onyx.dataflow.source import IteratorSource

def frame_synchronous_decode(model, parameter, observation_stream, callback=None):

    """
    A somewhat abstract pedagogical overview of frame-synchronous processing
    steps.

    *The Arguments*

    The model contains everything that's static from the point-of-view of doing
    decoding.  This includes the acoustic and language models, the lexicon, the
    grammar, etc.

    The parameter argument includes all the parametric settings for this
    particular decode.  These would include things like algorithm selection,
    pruning thresholds, resource limits, etc.  You might have several different
    parameter values to select from based on e.g. application state, speed
    requirements, etc.  You'd hand one of them into this function.  The
    parameter needs to be compatible with the model.

    The observation_stream argument is an iterable of the (single) stream of
    observations to be decoded.

    The callback argument is an optional callable that will be called with the
    observation, the map of active elements, and the intermediate results at the
    end of each frame.


    *Internal to the Function*

    There is a decoding context that the model produces.  It has the dynamic
    state for the decode.  It has methods for the few top-level operations that
    are performed during a decode.  Unrealistically, the pedagogical version of
    contexts used here gives the client full control over the list of active
    elements.

    There is a map of active_elements keyed off of some unique identifier.
    These active_elements are abstract.  Each one is active in that it holds one
    or more active hypotheses and their histories.  Each one is an element in
    that it represents some part of the decomposition of an entire decode into
    smaller pieces.  An active_element could be as small as a single Hmm state,
    or as large as the network for decoding an entire utterance, or larger....

    The use of an identifier to key the map of active elements supports
    possibility of having joins during the decode, i.e. so that histories with
    provably shared futures can be merged and only one such future actually
    activated for decoding.  This is such an important optimization that it's
    visible in this abstract overview.  Again, unrealistically, the context
    manages the ids without explicitly owning the active list....

    Each of the active elements has some basic features:
    - it has one or more inputs, each of which accepts a history
    - it has one or more outputs, each of which can generate a history
    - it knows how to update its hypotheses and histories given an observation

    The function loops over the sequence of observations.  In the loop it does a
    few things:
    - setup the context for the observation
    - activate successors to active elements with active outputs
    - apply observation likelihoods to the scores
    - prune inactive elements
    - update the result based on active elements with active outputs
    - clean up the observation-specific context
    - yield the intermediate results

    See _toy_example_1() for a toy example that uses this function rather
    trvially.  See _toy_example_2() for a toy example that uses this function
    more realistically (as far as building a complex network is concerned, and
    normalization).
    """

    # Context holds dynamic state for this decode.  Among other things, it
    # manages the uniqueness of active-elements.  However, for pedagogical
    # reasons, it does not manage the list of active_elements.
    context = model.make_decode_context(parameter)

    # context must have the following methods:
    #   get_initial_elements
    #   set_observation
    #   propagate_hypotheses
    #   apply_likelihoods
    #   prune_inactive
    #   update_result
    #   finish_observation
    #   get_result    


    # XXX make a context manager for doing timing and its algebra
    start_time = time.time()

    # Set up the initial elements corresponding to the start states of any
    # grammar
    active_elements = dict(context.get_initial_elements())

    for observation in observation_stream:
        # Do work to prepare the decoder to use this observation.  This includes
        # things like getting codebook entries, scoring lookahead models,
        # setting up normalization, setting thresholds, getting fast-match
        # results from some other decoder, etc. etc.
        context.set_observation(active_elements, observation)

        # Decoding is broken up into two separate steps

        # This step propagates scores and tokens according to the network's
        # transition probabilities.  It is during this step that the grammar is
        # querired and new elements are created and activated.
        context.propagate_hypotheses(active_elements)

        # Here is where the model likelihoods for the observation are folded
        # into the hypothesis scores.
        context.apply_likelihoods(active_elements)

        # Pruning; this modifies the set of active_elements
        context.prune_inactive(active_elements)

        # Add what's been learned to the result
        context.update_result(active_elements)

        # Gather stats, tear down state associated with the observation
        context.finish_observation()

        # Could offer up intermediate results
        if callback is not None:
            callback(observation, active_elements, context.get_result())

    return context, context.get_result(), time.time() - start_time


class FrameSynchronousProcessor(ProcessorBase):
    """
    Pedagogical example of a frame-synchronous decoder, written as a dataflow
    Processor.
    """
    def __init__(self, model, thresholds, sendee=None, sending=True):
        print 'FrameSynchronousProcessor.__init__():'
        super(FrameSynchronousProcessor, self).__init__(sendee=sendee, sending=sending)
        self.model = model
        self.thresholds = thresholds

        self.context = model.make_decode_context(thresholds)

        # Set up the initial elements corresponding to the start states of any
        # grammar
        self.active_elements = dict(self.context.get_initial_elements())

    def process(self, observation):
        context = self.context
        active_elements = self.active_elements

        # Do work to prepare the decoder to use this observation.  This includes
        # things like getting codebook entries, scoring lookahead models,
        # setting up normalization, setting thresholds, getting fast-match
        # results from some other decoder, etc. etc.
        context.set_observation(active_elements, observation)

        # Decoding is broken up into two separate steps

        # This step propagates scores and tokens according to the network's
        # transition probabilities.  It is during this step that the grammar is
        # querired and new elements are created and activated.
        context.propagate_hypotheses(active_elements)

        # Here is where the model likelihoods for the observation are folded
        # into the hypothesis scores.
        context.apply_likelihoods(active_elements)

        # Pruning; this modifies the set of active_elements
        context.prune_inactive(active_elements)

        # Add what's been learned to the result
        result = context.update_result(active_elements)

        # Gather stats, tear down state associated with the observation
        context.finish_observation()

        self.send(result)


def _msg(*args):
    return ' '.join(repr(x) for x in args)


# These elements are crying out for a baseclass

# For these lightweight HMMs we have the following usage semantics loop:
# - any number of calls to activate, e.g. from predecessor sequences
# - one call to pass_tokens
# - only at this point, any use of .has_output and .output is legal, e.g. to activate successors
# - one call to apply_likelihoods
# - only at this point, .scores, if any, are conditional likelihoods

class Epsilon(object):
    """
    A network epsilon.  Just passes its activations to its outputs.  Never has
    any scores.
    """
    scores = ()
    def __init__(self, num_virtual_states, user_data=None):
        self._num_virtual_states = num_virtual_states
        self.user_data = user_data

        # score values for probabilities zero and one, abstracted for when we
        # start using log probs....
        prob_0 = 0

        self.output_scores = self.zeroes = tuple(prob_0 for i in xrange(num_virtual_states))

    @property
    def num_virtual_states(self):
        return self._num_virtual_states

    # XXX rename this to account for the accumulation semantics
    def activate(self, scores):
        # just accumulate activations into the outputs.
        activation = tuple(scores)
        if len(activation) != self.num_virtual_states:
            raise ValueError("expected %d activation scores, got %d" % (self.num_virtual_states, len(activation)))
        # schur sum
        self.output_scores = tuple(existing + new for existing, new in izip(self.output_scores, activation))

    def pass_tokens(self):
        pass

    @property
    def has_output(self):
        """
        True if the output is active, that is if it has any probability mass.
        """
        return self.output_scores is not self.zeroes

    @property
    def output(self):
        """
        The output scores; suitable for activation of successors.
        """
        assert self.has_output
        return self.output_scores

    def apply_likelihoods(self, observation, normalization):
        self.output_scores = self.zeroes

class Sink(object):
    """
    A network sink for probability mass.  Accumulates (and correctly normalizes)
    all the mass with which it gets activated.  Handles any dimensionality of
    activation.  Never has any output.  Always has scores of length one.
    """
    def __init__(self):
        self._scores = [0]
        self.activation = 0

    # XXX rename this to account for the accumulation semantics
    def activate(self, activation):
        self.activation += sum(activation)

    def pass_tokens(self):
        self._scores[0] += self.activation
        self.activation = 0

    def apply_likelihoods(self, observation, normalization):
        self._scores[0] *= normalization

    @property
    def scores(self):
        return tuple(self._scores)

    @property
    def has_output(self):
        return False

class ForwardSequence(object):
    """
    Simple implementation of a linear sequence of states with self loops and
    forward arcs.  With no skipping this implements the Bakis structure.  This
    structure also allows regularized forward skipping of states.  It works
    correctly when the number of states is zero; i.e. it implements epsilon
    sequence behavior correctly.

    Create a Hmm with single-state skipping and observation likelihoods of one
    so that no probability leaks out
    >>> fs = ForwardSequence(lambda x,y: 1, xrange(10), (repeat(0.0625, 10), repeat(0.75, 10), repeat(0.1875, 10)))
    >>> fs.num_states, fs.transition_order
    (10, 3)
    >>> fs.in_trans
    ((0.0625, 0, 1), (0.0625, 0.75, 1), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0.0625, 0.75, 0.1875), (0, 0.75, 0.1875), (0, 0, 0.1875))

    Inject some mass, twice, e.g. as at a join, total injected mass sums to one
    >>> fs.activate((0.0, 0.75))
    >>> fs.activate((0.25, 0))
    >>> sum(fs.scores) == 0
    True

    We can decode six times total without any mass leaking out of the right-hand
    side because that's how long it takes for the mass to get to the (virtual)
    output states.
    >>> fs.pass_tokens()
    >>> fs.has_output and fs.output
    False
    >>> fs.apply_likelihoods(())
    >>> sum(fs.scores)
    1.0
    >>> for i in xrange(4): fs.pass_tokens(); fs.apply_likelihoods(())
    >>> sum(fs.scores)
    1.0

    And here's the non-zero output
    >>> fs.pass_tokens()
    >>> fs.has_output and fs.output #doctest: +ELLIPSIS
    (0.00353407859802246..., 0.000173807144165039...)

    One more set of likelihoods and we're dropping the output mass on the floor
    >>> fs.apply_likelihoods(())
    >>> sum(fs.scores) < 1
    True

    >>> num_states = 1
    >>> out_order = 2
    >>> fs = ForwardSequence(lambda x,y: 1, xrange(num_states), tuple(repeat(p, num_states) for p in toy_probs(out_order)))
    >>> fs.activate(toy_probs(out_order - 1))
    >>> sum(fs.scores)
    0
    >>> fs.pass_tokens()
    >>> fs.has_output and fs.output
    False
    >>> fs.apply_likelihoods(())
    >>> sum(fs.scores)
    1.0
    >>> fs.pass_tokens()
    >>> fs.has_output and fs.output
    (0.75,)
    >>> fs.apply_likelihoods(())

    See that we've lost the output mass
    >>> sum(fs.scores)
    0.25

    Note that a zero-state Hmm makes sense, even with state skipping.  It's just
    an epsilon.
    >>> num_states = 0
    >>> out_order = 4
    >>> fs = ForwardSequence(lambda x, y: 1, xrange(0), tuple(repeat(p, num_states) for p in toy_probs(out_order)))
    >>> fs.activate(toy_probs(out_order - 1))
    >>> len(fs.scores)
    0
    >>> sum(fs.scores)
    0
    >>> fs.pass_tokens()
    >>> fs.has_output and fs.output
    (0.25, 0.625, 0.125)
    >>> sum(fs.output)
    1.0
    >>> fs.apply_likelihoods(())
    >>> sum(fs.scores)
    0

    >>> ForwardSequence(lambda x, y: 1, xrange(10), (tuple(xrange(10)), ))
    Traceback (most recent call last):
      ...
    ValueError: expected at least two transition_iters sequences, got 1
    """

    def __init__(self, scorer, model_iter, transition_iters, user_data=None):
        """
        Create a sequence of states that can decode observations.  The scorer
        argument is a callable that returns the likelihood that a model produced
        an observation.  The sequence of states corresponds to the models in
        model_iter.  These are used to score observations via scorer(model,
        observation).

        The transition_iters argument is a sequence of sequences.  Each
        transition-iter sequence must be the same length as the model_iter
        sequence.  The first two sequences are required; they are the self-loop
        transition probabilities for the states and the forward-one transition
        probabilities for the states.

        The third and subsequent sequences are optional.  They are the forward
        skipping transition probabilities for skipping, respectively, the next
        state, the next-plus-one state, etc.
        """

        # XXX need an argument with the identities and operators for the
        # "addition" and "multiplication" operations on the "probabilities"

        self.scorer = scorer
        assert callable(self.scorer)
        self.model = tuple(model_iter)
        transition_iters = tuple(transition_iters)
        self._transition_order = len(transition_iters)
        if self.transition_order < 2:
            raise ValueError("expected at least two transition_iters sequences, got %d" % (self.transition_order,))
        self.user_data = user_data

        # Note: client gives us states, models, transitions, etc in the
        # "natural" left-to-right forward order. That's how we use them.  This
        # rules out the DP optimization of overwritting scores in place....

        def in_trans_iters(transition_iters, trans_prob_0, trans_prob_1):
            """
            Make the transition probs in the structures we need for decoding.
            Yield tuple of the self transition probability, prior state
            transition probability, etc for the skipped states.
            """

            # This is all a bit hairy!
            #
            # Using transition_iters, a sequence of sequences of outgoing
            # forward transition scores, return a tuple of iterators for the
            # incoming transition scores for the real states and for the
            # trailing virtual states.  The trans_prob_0 and trans_prob_1
            # arguments are the scores to use for synthesized transitions with
            # probability of zero and one respectively.
            num_virtual = len(transition_iters) - 1
            def front(offset):
                # the virtual front nodes have zero prob on their outgoing
                # transitions, except for their furthest "skip ahead" which is
                # one; this ensures that the activation scores are put into
                # their correct states
                return repeat(trans_prob_0 if offset < num_virtual else trans_prob_1, offset)
            def back(offset):
                # the virtual end nodes have zero probability on all their
                # exiting arcs
                return repeat(trans_prob_0, num_virtual - offset)
            return (chain(front(offset), trans, back(offset)) for offset, trans in enumerate(transition_iters))

        # score values for probabilities zero and one, abstracted for when we
        # start using log probs....
        self.prob_0 = 0
        self.prob_1 = 1

        # note: here is where transpose the shifted transitions
        in_trans = tuple(in_tran for in_tran in izip(*in_trans_iters(transition_iters, self.prob_0, self.prob_1)))
        assert len(in_trans) == self.num_states + self.num_virtual_states
        for trans in in_trans:
            assert len(trans) == self.transition_order
        self.in_trans = in_trans

        # some zeroes for virtual states
        self.virtual_zeroes = tuplen(self.num_virtual_states, self.prob_0)

        self._mass_one = tuplen(1, self.prob_1) + tuplen(self.num_virtual_states - 1, self.prob_0)

        # starting likelihoods
        self.scores = tuplen(self.num_states, self.prob_0)

        # activation
        self.activation = self.virtual_zeroes
        self.seeds = None
        self.output_scores = None

    def activate(self, scores):
        activation = tuple(scores)
        if len(activation) != self.num_virtual_states:
            raise ValueError("expected %d activation scores, got %d" % (self.num_virtual_states, len(activation)))
        # schur sum
        self.activation = tuple(existing + new for existing, new in izip(self.activation, activation))

    def pass_tokens(self):
        assert self.seeds is self.output_scores is None
        assert len(self.scores) == self.num_states
        # create a tuple of iterators over the activation and previous scores
        prev_score_iters = tee(chain(self.activation, self.scores, self.virtual_zeroes), 1 + self.num_virtual_states)
        # shift the iterators
        for index, itr in enumerate(prev_score_iters):
            for i in xrange(self.num_virtual_states - index):
                itr.next()
        # seed scores for each state: sum over previous_state_score times
        # transition_probability; this is the dot product inner loop for the
        # mass coming into each state
        self.seeds = tuple(sum(prev_score * trans_prob for prev_score, trans_prob in izip(scores, trans))
                           for scores, trans in izip(izip(*prev_score_iters), self.in_trans))
        assert len(self.seeds) == len(self.scores) + self.num_virtual_states

        self.output_scores = self.seeds[self.num_states:]
        assert len(self.output_scores) == self.num_virtual_states
        self.seeds = self.seeds[:self.num_states]

        self.activation = None

    def apply_likelihoods(self, observation, normalization=1):
        assert self.activation is None
        assert type(self.seeds) is tuple
        assert len(self.seeds) == self.num_states

        # score the observations for each state
        observation_scores = (self.scorer(model, observation) for model in self.model)
        # apply normalization
        if normalization != 1:
            assert normalization >= 2
            observation_scores = (score * normalization for score in observation_scores)
        # schur product: multiply seeding mass by observation likelihood
        new_scores = tuple(seed * observation_score for seed, observation_score in izip(self.seeds, observation_scores))
        assert len(new_scores) == self.num_states

        self.scores = new_scores
        self.seeds = None
        self.output_scores = None

        self.activation = self.virtual_zeroes

    @property
    def mass_one(self):
        """
        An activation suitable for initial sequences in a decode.
        E.g. seq.activate(seq.mass_one).
        """
        return self._mass_one

    @property
    def has_output(self):
        """
        True if the output is active, that is if it has any probability mass.
        """
        assert self.activation is None, 'strict check on usage semantics'
        assert self.output_scores is not None
        return self.output_scores != self.virtual_zeroes

    @property
    def output(self):
        """
        The output scores; suitable for activation of successors.
        """
        assert self.has_output
        return self.output_scores

    @property
    def num_states(self):
        """
        The number of model states in this ForwardSequence.
        """
        return len(self.model)

    @property
    def transition_order(self):
        """
        The degree of transitions out of each state.
        """
        return self._transition_order

    @property
    def num_virtual_states(self):
        """
        The number of virtual states at each end of the sequence
        """
        # don't count the self loop
        return self.transition_order - 1

    def get_decode_context(self, user_data=None):
        # XXX hack hack hack - if we continue to support this Pythonic
        # ForwardSequence guy, it needs to return a bona fide separate object as
        # its decode contex; better to just use real Hmm objects...
        self.user_data = user_data
        return self


def forward_sequence_builder(scorer, state_iter, probs, user_data=None):
    """
    This function serves as a plug-in replacement for the ForwardSequence c'tor
    under the assumption that the scoring function will always return 1.0.  It
    uses Hmms (from onyx.am.hmm) as the underlying model.
    """
    order = len(probs)
    assert order > 0
    probs = tuple(tuple(p) for p in probs)
    models = tuple(DummyModel(1, 1.0) for s in state_iter)
    num_states = len(models)
    model_indices = range(num_states)
    hmm = Hmm(num_states, user_data)
    hmm.build_forward_model_compact(GmmMgr(models), model_indices, order, probs)
    return hmm


def uniform_probs(count):
    """
    Create a uniform distribution of count elements.
    """
    assert count >= 0
    return (1/count,) * count if count > 0 else ()

def toy_probs(count):
    """
    Create a toy distribution of out-arc weights given a count of out arcs.
    Make forward-one arc get the most weight.
    - 1/4 to self loop
    - 1/2 + eps to forward one
    - 1/8 to forward two
    - 1/16 to forward three
    - etc
    - 1/(2^n) to forward n-1
    - eps = 1/(2^n)

    >>> toy_probs(5)
    (0.25, 0.53125, 0.125, 0.0625, 0.03125)

    >>> toy_probs(1)
    (1.0,)
    """
    if count == 0:
        return tuple()
    assert count >= 1
    weights = list()
    mass = 1
    for i in xrange(count):
        mass /= 2
        weights.append(mass)
    weights[0] += mass
    if count > 1:
        # swap first two, giving a more realistic speech model....
        weights[0], weights[1] = weights[1], weights[0]
    assert sum(weights) == 1
    return tuple(weights)


def lozenge_ids(growth, steady, id, verbose=False):
    """
    Generate successor id sets to support a lozenge shape network.

    >>> for id in xrange(12):
    ...   ids = lozenge_ids(2, 2, id, True)
    ...   print id, ids
    growth width steady: 2 2 2
    grow_upper width_upper shrink_upper: 2 6 9
    0 (1,)
    1 (2, 3)
    2 (4,)
    3 (5,)
    4 (6,)
    5 (7,)
    6 (8,)
    7 (8,)
    8 (9,)
    9 (10,)
    10 (11,)
    11 (12,)

    >>> for id in xrange(17):
    ...   ids = lozenge_ids(3, 1, id, True)
    ...   print id, ids
    growth width steady: 3 4 1
    grow_upper width_upper shrink_upper: 4 8 15
    0 (1,)
    1 (2, 3)
    2 (4, 5)
    3 (6, 7)
    4 (8,)
    5 (9,)
    6 (10,)
    7 (11,)
    8 (12,)
    9 (12,)
    10 (13,)
    11 (13,)
    12 (14,)
    13 (14,)
    14 (15,)
    15 (16,)
    16 (17,)
    """

    assert growth >= 1
    assert steady >= 1
    assert id >= 0

    width = 1 << (growth - 1)
    grow_upper = width
    width_upper = grow_upper + steady * width
    shrink_upper = width_upper + width + grow_upper - 1

    if verbose and id == 0:
        print 'growth width steady:', growth, width, steady
        print 'grow_upper width_upper shrink_upper:', grow_upper, width_upper, shrink_upper

    if id == 0:
        # special case
        ids = 1,
    elif id < grow_upper:
        # heap-type growth
        ids = 2 * id, 2 * id + 1
    elif id < width_upper:
        # constant width
        ids = id + width,
    elif id < shrink_upper:
        # heap-type shrink
        ids = shrink_upper - (shrink_upper - id) // 2,
    else:
        # single chain
        ids = id + 1,

    return ids


class decode_context(object):
    """
    A toy implementation of a context that activates a sequence of Hmms,
    dynamically creating new Hmm(s) each time the current end of a sequence
    first activates its output.  After a certain number of Hmms have been
    created it merges everything into a sink that just accumulates mass.
    Never prunes.

    Used in _toy_example() to show that probability mass is preserved.
    Does splitting and merging to explore a network with a lozenge shape
    with a tail.  Also does normalization so as to keep the best score near
    1.  And it uses some epsilon sequences; one at the start of the lattice.

    XXX there's not a clean split between the decode_context and the
    toy_model... most functionality belongs in decode_context
    """
    def __init__(self, model, parameter):
        self.model = model
        self.max_states, self.max_hmms = parameter
        self.history = list()

        self.total_normalization = 1
        self.hmms = list()
        self.num_states = 0
        self.max_user_data = None
        self._sink_element = None

        # number of layers of lozenge at its widest
        self.steady = 2

        self.finish_observation()


    @property
    def sink(self):
        # there's only one sink, but it can be a target many times, and is used
        # to accumulate all inputs
        if self._sink_element is None:
            self._sink_element = Sink()
            self.hmms.append(self._sink_element)
        return self._sink_element

    def get_initial_elements(self):
        depth_and_states = 0, 0

        id = 0
        element = self.model.new_epsilon_element(depth_and_states)
        self.hmms.append(element)
        # note: in general the seeding mass should be distributed across the
        # states in a non-toy_probs way....  note: this should be the only
        # time we activate this epsilon node; XXX could subclass epsilon to
        # give start-enforcing and sink-enforcing semantics....
        element.activate(toy_probs(element.num_virtual_states))
        assert self.hmms[id] is element
        yield id, element

    def set_observation(self, elements, observation):
        assert self.observation is self.normalization is self.threshold is self.seen_ids is None

        # set up observation-specific state
        self.observation = observation

        # normalization work: use previous frames scores to come up with a
        # normalization to use for the current state

        # note: conditional in the generator protects against epsilon
        # elements, which have no scores
        maxen = tuple(max(element.scores) for element in elements.itervalues() if element.scores)
        # XXX note: we assume that there's some prob mass floating around,
        # but maxen will be empty the first time through since only the
        # start_element is active and, being an epsilon, it doesn't have
        # scores, only output
        max_likelihood = max(maxen) if maxen else 1
        if max_likelihood <= 0:
            assert max_likelihood == 0
            raise ValueError("no likelihood in the hypotheses")
        recip_max_likelihood = 1 / max_likelihood
        normalization = 1
        while recip_max_likelihood > normalization:
            normalization *= 2
        # print 'max_likelihood, normalization:', max_likelihood, normalization
        self.normalization = normalization
        self.total_normalization *= normalization
        #self.model.total_normalization *= normalization

        self.threshold = len(elements)
        self.seen_ids = set()

    def finish_observation(self):
        # clear observation-specific state
        self.observation = None
        self.normalization = None
        self.threshold = None
        self.seen_ids = None

    def successor_ids(self, id):
        # implements the grammar
        #
        # in this case max_hmms is special, used for the sink
        return tuple(min(succ_id, self.max_hmms) for succ_id in lozenge_ids(self.max_states, self.steady, id))

    def propagate_hypotheses(self, elements):
        # Handles seeding successors, including branching with probabilistic
        # weights.

        # XXX this should be factored; this is where we would integrate with
        # a grammar-type element, e.g. for FSG or CFG constrained behavior

        # stack based successor activation, supports topological constraints
        ids = elements.keys()
        # reversed ids: backwards topological ordering, so pop-based iteration
        # (i.e. from the end) is topologically forwards; also, new ids are
        # appended to the end, so they are handled topologically too
        ids.sort(None, None, reverse=True)
        while ids:
            id = ids.pop()
            element = elements[id]
            assert element is self.hmms[id]
            # assert that we only deal with an element once per observation
            assert id not in self.seen_ids
            self.seen_ids.add(id)

            # deal with token passing
            element.pass_tokens()

            if element.has_output:
                # do work to find or create new elements, then activate them
                successor_ids = self.successor_ids(id)
                weights = toy_probs(len(successor_ids))
                for successor_id, weight in izip(successor_ids, weights):
                    successor_model = elements.get(successor_id)
                    if successor_model is None:
                        # get a successor
                        if successor_id < self.max_hmms:
                            # create a new Hmm
                            depth, depth_states = element.user_data
                            new_depth = depth + 1
                            # note: num_states can be zero, meaning the element will be an epsilon
                            num_states = new_depth % self.max_states
                            new_depth_and_states = new_depth, depth_states + num_states
                            successor_model = self.model.new_element(num_states, new_depth_and_states)
                            self.hmms.append(successor_model)
                            self.num_states += num_states
                            if new_depth_and_states > self.max_user_data:
                                self.max_user_data = new_depth_and_states
                        else:
                            # one sink model, just accumulates (and normalizes) the activation probs
                            assert successor_id == self.max_hmms
                            successor_model = self.sink

                        elements[successor_id] = successor_model
                        ids.append(successor_id)

                    # here's where the output(s) from the element seed the
                    # input(s) to the successor
                    activation = tuple(weight * output for output in element.output)
                    successor_model.activate(activation)

    def apply_likelihoods(self, elements):
        # incorporate likelihoods for this observation into the models' scores
        for id, element in elements.iteritems():
            assert element is self.hmms[id]
            element.apply_likelihoods(self.observation, self.normalization)

    def prune_inactive(self, elements):
        # never prune for this mass-preserving example
        return

    def update_result(self, elements):
        self.history.append(list())

    def get_result(self):
        # here's where we return the result of having done something
        # interesting in propagate_hypotheses(), apply_likelihoods(), and
        # update_result()
        #
        # in this case, we get the summed mass and the histories
        return self.mass, tuple(tuple(items) for items in self.history)

    @property
    def mass(self):
        return sum(sum(hmm.scores) for hmm in self.hmms if hmm.scores) / self.total_normalization


class toy_model(object):

    def __init__(self, hmm_factory, out_transitions):
        # factory/constructor for Hmms
        self._hmm_factory = hmm_factory

        # strict probability model
        self._out_transitions = tuple(out_transitions)
        assert sum(self.out_transitions) == 1

    @property
    def hmm_factory(self):
        return self._hmm_factory

    @property
    def out_transitions(self):
        return self._out_transitions

    def scorer(self, model_id, observation):
        # trivial models with unity observation likelihoods
        return 1

    def new_element(self, num_states, user_data=None):
        # this is called dynamically during the decode when an existing
        # element has an active output and no successor(s)
        return self.hmm_factory(self.scorer,
                                xrange(num_states),
                                tuple(repeat(prob, num_states) for prob in self.out_transitions)).get_decode_context(user_data)

    def new_epsilon_element(self, user_data=None):
        return Epsilon(len(self.out_transitions)-1, user_data)

    def make_decode_context(self, parameter):
        return decode_context(self, parameter)


def _toy_example(parameter):
    """
    A toy example that actually does Hmm decoding using ForwardSequence.
    Trivially, all observations have likelihood of one; this lets us assert that
    the network does not lose probability mass.

    This example shows that epsilon sequences work.  This toy is still
    constrained that (out_order - 1) scores are passed between elements.

    >>> out_order = 4
    >>> max_states = 10
    >>> max_hmms = 300
    >>> num_observations = 50
    >>> use_hmms = False
    >>> parameter = out_order, max_states, max_hmms, num_observations, use_hmms
    >>> parameter
    (4, 10, 300, 50, False)

    >>> _toy_example(parameter)
    len(result): 50
    len(context.hmms): 301
    context.max_user_data: (9, 45)
    context.num_states: 2189
    context.total_normalization: 64
    mass: 1.0
    sink: 0.993148407154

    >>> use_hmms = True
    >>> parameter = out_order, max_states, max_hmms, num_observations, use_hmms
    >>> parameter
    (4, 10, 300, 50, True)

    >>> _toy_example(parameter)
    len(result): 50
    len(context.hmms): 301
    context.max_user_data: (9, 45)
    context.num_states: 2189
    context.total_normalization: 64
    mass: 1.0
    sink: 0.993148407154
    """

    out_order, max_states, max_hmms, num_observations, use_hmm = parameter
    hmm_factory = forward_sequence_builder if use_hmm else ForwardSequence
    model = toy_model(hmm_factory, toy_probs(out_order))
    # Note that Hmm expects observations to be tuples of the correct dimension,
    # even if using DummyModels
    observation_stream = repeat((0,), num_observations)
    context, (mass, result), elapsed_time = frame_synchronous_decode(model, (max_states, max_hmms), observation_stream)

    print 'len(result):', len(result)
    print 'len(context.hmms):', len(context.hmms)    
    print 'context.max_user_data:', context.max_user_data
    print 'context.num_states:', context.num_states
    print 'context.total_normalization:', context.total_normalization
    assert mass == context.mass
    print 'mass:', mass
    print 'sink:', sum(context.sink.scores) / context.total_normalization    

def hmm_is_skipable(hmm):
    # return True if there is a way through the Hmm that doesn't require an observation
    return hmm._transition_matrix[:hmm.num_inputs, -hmm.num_outputs:].sum() > 0.0

class real_model(object):

    def __init__(self, model_filename):
        print 'real_model():'
        print '  filename', repr(model_filename.replace(onyx.home, '#').replace('\\', '/'))
        self.model_filename = model_filename
        with open(self.model_filename, 'rb') as model_file:
            self.model_dict, self.hmm_mgr, self.gmm_mgr = read_htk_mmf_file(model_file)
        hmm_mgr = self.hmm_mgr
        model_dict = self.model_dict

        # from id to class
        self.inv_dict = dict((value, item) for item, value in model_dict.iteritems())
        assert len(self.inv_dict) == len(model_dict)

        print '  num_models', hmm_mgr.num_models
        print '  arity sets:', ' input', tuple(sorted(hmm_mgr.input_arity_set)), ' output', tuple(sorted(hmm_mgr.output_arity_set))
        print '  model classes:', ' '.join(sorted(self.model_dict))

        assert len(hmm_mgr.input_arity_set) == len(hmm_mgr.output_arity_set) == 1
        assert hmm_mgr.input_arity_set == hmm_mgr.output_arity_set
        arity, = hmm_mgr.input_arity_set
        assert not hmm_mgr.has_epsilon_model(arity)
        self.epsilon_id = hmm_mgr.add_epsilon_model(self.gmm_mgr, arity)
        assert hmm_is_skipable(hmm_mgr[self.epsilon_id])

        self.skipable = set()
        # make sure we can get all decode contexts
        for hmm_class, hmm_id in model_dict.iteritems():
            hmm = hmm_mgr[hmm_id]
            hmm.get_decode_context(hmm_id)
            if hmm_is_skipable(hmm):
                self.skipable.add(hmm_id)
        print '  skipable classes:', ' '.join(sorted(self.inv_dict[skipable] for skipable in self.skipable))

    @property
    def num_classes(self):
        return len(self.model_dict)

    def scorer(self, model_id, observation):
        # trivial models with unity observation likelihoods
        return 1

    def new_element(self, num_states, user_data=None):
        # this is called dynamically during the decode when an existing
        # element has an active output and no successor(s)
        return self.hmm_factory(self.scorer,
                                xrange(num_states),
                                tuple(repeat(prob, num_states) for prob in self.out_transitions)).get_decode_context(user_data)

    def new_epsilon_element(self, user_data=None):
        epsilon = self.hmm_mgr[self.epsilon_id].get_decode_context(user_data)
        assert [epsilon.hmm.num_inputs] == list(self.hmm_mgr.input_arity_set)
        return epsilon

    def make_decode_context(self, parameter):
        grammar_spec = set(self.inv_dict), self.skipable
        thresholds = parameter
        return real_decode_context(self, (grammar_spec, thresholds))

class real_decode_context(object):
    """
    A real decoder!

    A toy implementation of a context that activates a sequence of Hmms,
    dynamically creating new Hmm(s) each time the current end of a sequence
    first activates its output.  After a certain number of Hmms have been
    created it merges everything into a sink that just accumulates mass.
    Never prunes.

    Used in _toy_example() to show that probability mass is preserved.
    Does splitting and merging to explore a network with a lozenge shape
    with a tail.  Also does normalization so as to keep the best score near
    1.  And it uses some epsilon sequences; one at the start of the lattice.
    """
    def __init__(self, model, (grammar_spec, thresholds)):
        self.model = model
        self.star_grammar_ids, self.skipable_ids = grammar_spec
        self.prune_rank, self.seed_threshold_rank, self.seed_threshold_scale = thresholds

        def debug(*args):
            print ' '.join(str(arg) for arg in args)
        self.debug = False

        self.epsilon_user_data = -1, -1, [], [False], -1

        # XXX need to deal with skipable hmms
        self.usable_star_grammar_ids = self.star_grammar_ids - self.skipable_ids

        self.hmms = list()
        self.history = list()

        self._obs_id = count().next
        self.starts_by_observation_id = dict()

        self.total_log2_normalization = 0

        self.bestscores = None

        self._sink_element = None

        self.finish_observation()

    @property
    def sink(self):
        # there's only one sink, but it can be a target many times, and is used
        # to accumulate all inputs
        # XXX yeah right: it violates the topological constraint!
        if self._sink_element is None:
            self._sink_element = Sink()
            self.hmms.append(self._sink_element)
        return self._sink_element

    def get_initial_elements(self):
        id = len(self.hmms)
        # XXX user_data should be something distinctive for an initial epsilon
        element = self.model.new_epsilon_element(self.epsilon_user_data)
        self.hmms.append(element)
        # note: in general the seeding mass should be distributed across the
        # states in a principled way; note: this should be the only time we
        # activate this epsilon node; XXX could subclass epsilon to give
        # start-enforcing and sink-enforcing semantics....
        element.activate(uniform_probs(element.hmm.num_inputs))
        element.user_data[3][:] = [True]
        self.bestscores = N.array([1/element.hmm.num_inputs], dtype=float)
        assert id == 0
        assert self.hmms[id] is element
        yield id, element

    def finish_observation(self):
        # clear observation-specific state
        self.observation_id = None
        self.observation = None
        self.activate_threshold = None
        self.normalization = None
        self.log2_normalization = None
        self.threshold = None
        self.seen_ids = None
        self.num_active = None
        self.temp1 = None

    def set_observation(self, elements, observation):
        # XXX give this a pred_id and have it return an id, and then
        # frame_synchronous_decode can tell us about the observation lattice

        assert self.observation is self.normalization is self.threshold is self.seen_ids is self.temp1 is None

        self.temp1 = N.empty_like(observation)

        # set up observation-specific state
        self.observation_id = self._obs_id()
        self.observation = observation

        # normalization work: use previous frames scores to come up with a
        # normalization to use for the current state

        # note: conditional in the generator protects against epsilon
        # elements, which have no scores

        # duplicate logic
        #bestscores = list(max(element.scores) for element in elements.itervalues() if element.scores)
        #bestscores.sort()
        bestscores = self.bestscores
        self.activate_threshold = bestscores[-self.seed_threshold_rank] / self.seed_threshold_scale if len(bestscores) > self.seed_threshold_rank else 0

        # XXX note: we assume that there's some prob mass floating around,
        # but bestscores will be empty the first time through since only the
        # start_element is active and, being an epsilon, it doesn't have
        # scores, only output
        #max_likelihood = max(bestscores) if bestscores else 1
        max_likelihood = bestscores[-1]
        if max_likelihood <= 0:
            assert max_likelihood == 0
            raise ValueError("no likelihood in the hypotheses")
        recip_max_likelihood = 1 / max_likelihood
        normalization = 1
        log2_normalization = 0
        while recip_max_likelihood > normalization:
            normalization *= 2
            log2_normalization += 1
        # print 'max_likelihood, normalization:', max_likelihood, normalization
        self.normalization = normalization
##         self.total_normalization *= normalization
        self.log2_normalization = log2_normalization
        self.total_log2_normalization += log2_normalization
##         #self.model.total_normalization *= normalization
        self.debug and self.debug('set_observation:', ' log2_normalization', log2_normalization, ' total_log2_normalization', self.total_log2_normalization)

        self.threshold = len(elements)
        self.seen_ids = set()

    def successor_ids(self, id):
        # hmmm, not really a win on the phoneme decoder
        use_epsilon = False

        # star grammar: everyone can go everywhere
        #print 'successor_ids:', ' observation_id', self.observation_id, ' from id', id
        succ_ids = self.starts_by_observation_id.get(self.observation_id)
        if succ_ids is None:
            epsilon_user_data = -1, -1, [], [False], -1

            if use_epsilon:
                # epsilon for the merge to avoid cross-bar combinatorics
                epsilon_id = len(self.hmms)
                succ_ids2 = []
                epsilon = self.model.new_epsilon_element((self.observation_id, epsilon_id, succ_ids2, [False], -1))
                self.hmms.append(epsilon)

            #print ' ', 'observation_id:', self.observation_id
            # first time we're called for observation_id, so create the new succ_ids and put hmms at those slots
            succ_ids = self.starts_by_observation_id[self.observation_id] = tuple(xrange(len(self.hmms), len(self.hmms)+len(self.usable_star_grammar_ids)))
            self.hmms.extend(self.model.hmm_mgr[hmm_id].get_decode_context((self.observation_id, succ_ids[index], [], [False], hmm_id))
                             for index, hmm_id in enumerate(self.usable_star_grammar_ids))
            assert self.hmms[succ_ids[-1]] is self.hmms[-1]

            # use the epsilon
            if use_epsilon:
                succ_ids2[:] = succ_ids
                succ_ids = epsilon_id,

        else:
            # we've created the successor_ids for elements with outputs on this
            # frame; this allows joins
            pass
        #print ' ', 'return:', succ_ids
        return succ_ids

    def propagate_hypotheses(self, elements):
        self.debug and self.debug('propagate_hypotheses:', ' observation_id', self.observation_id, ' len(elements)', len(elements))

        # Handles seeding successors, including branching with probabilistic
        # weights.

        # XXX this should be factored; this is where we would integrate with
        # a grammar-type element, e.g. for FSG or CFG constrained behavior

        temp1 = self.temp1

        # stack based successor activation, supports topological constraints
        ids = elements.keys()

        # reversed ids: backwards topological ordering, so pop-based iteration
        # (i.e. from the end) is topologically forwards; also, new ids are
        # appended to the end, so they are handled topologically too
        activate_threshold = self.activate_threshold
        while ids:
            # expensive (in top 10% on profile): instead arrange to build a new list topologically
            ids.sort(None, None, reverse=True)
            id = ids.pop()
            element = elements[id]
            assert element is self.hmms[id]

            # assert that we only deal with an element once per observation
            assert id not in self.seen_ids
            __debug__ and self.seen_ids.add(id)

            user_data = element.user_data
            assert user_data[3] == [True]
            assert element._current_state in ('activate', 'apply_likelihoods'), _msg(element._current_state)

            #class_name = 'epsilon' if user_data[:2] == self.epsilon_user_data[:2] else self.model.inv_dict[user_data[-1]]
            #print ' ', 'propagate_hypotheses:', ' id', id, ' state', repr(element._current_state), ' user_data', user_data, ' class_name', class_name

            # deal with token passing
            element.pass_tokens()

##             if element.has_output:
##                 #print ' ', 'has_output', ' good', user_data[0] < self.observation_id
##                 assert user_data[0] < self.observation_id

##             if element.has_output and max(element.output) >= activate_threshold:
##                 assert max(element.output) == element.max_output, str(max(element.output)) + '  ' + str(element.max_score)
            #if element.has_output and element.max_score >= activate_threshold:
            if element.max_output > activate_threshold:
                # XXX has to have started earlier since our epsilon support is lame
                assert user_data[0] <= self.observation_id

                # do work to find or create new elements, then activate them
                successor_ids = user_data[2]
                if len(successor_ids) == 0:
                    successor_ids[:] = self.successor_ids(id)

##                 weights = N.empty((len(successor_ids),), dtype=float)
##                 weights.fill(1 / len(successor_ids))
##                 # in general, make a cache of these....
##                 assert (weights == N.array(uniform_probs(len(successor_ids)), dtype=float)).all()

                # XXX here is where you'd use inter-hmm transition probs,
                # e.g. from a transition structure manager or from an LM, etc.
                #weights = uniform_probs(len(successor_ids))

                weight = N.float64(1/len(successor_ids))
                
                #for successor_id, weight in izip(successor_ids, weights):
                for successor_id in successor_ids:
                    assert successor_id > id, _msg(id, successor_id)
                    successor_element = elements.get(successor_id)
                    if successor_element is None:
                        #print ' ', ' successor_id', successor_id, ' new'
                        successor_element = elements[successor_id] = self.hmms[successor_id]
                        assert successor_element._current_state in ('start', 'apply_likelihoods'), successor_element._current_state
                        # expensive
                        assert successor_id not in ids
                        ids.append(successor_id)
                        assert successor_element.user_data[3] == [False]
                        successor_element.user_data[3][:] = [True]

                    # here's where the output(s) from the element seed the
                    # input(s) to the successor
                    assert element._current_state in ('pass_tokens',), _msg(self.observation_id, element._current_state)
                    assert successor_element.user_data[3] == [True]
                    assert successor_element._current_state in ('start', 'activate', 'pass_tokens', 'apply_likelihoods'), _msg(self.observation_id, successor_element._current_state)
                    #activation = tuple(weight * output for output in element.output)
                    element_outputX = element.outputX
                    # have a cache of sized temps?....
                    temp1.resize(element_outputX.shape, refcheck=0)
                    N.multiply(element_outputX, weight, temp1)
                    successor_element._update_state('activate')
                    successor_element._input += temp1
##                     N.add(temp1, successor_element._input, successor_element._input)
##                     activation = element_outputX * weight
##                     successor_element.activate(activation)

        if __debug__:
            for element in elements.itervalues():
                if element.user_data[-1] >= 0:
                    assert element._current_state == 'pass_tokens', str(element._current_state)

    def apply_likelihoods(self, elements):
        self.debug and self.debug('apply_likelihoods:', ' len(elements)', len(elements))
        if __debug__:
            for element in elements.itervalues():
                if element.user_data[-1] >= 0:
                    assert element._current_state == 'pass_tokens', str(element._current_state)

        # incorporate likelihoods for this observation into the models' scores
        for id, element in elements.iteritems():
            assert element is self.hmms[id]
            assert element.user_data[3] == [True]
            #print ' ', 'user_data', element.user_data
            # would be nice to be able to skip epsilons
            if element.user_data[-1] >= 0:
                element.apply_likelihoods(self.observation, self.normalization)

        if __debug__:
            for element in elements.itervalues():
                if element.user_data[-1] >= 0:
                    assert element._current_state == 'apply_likelihoods', str(element._current_state)

    def prune_inactive(self, elements):
        assert self.num_active is None
        self.num_active = len(elements)        
        # XXX use a threshold, use a hard limit, use a ranked percentile, etc. etc.
        #bestscores = list(max(element.scores) for element in elements.itervalues() if element.scores)
        #bestscores = list(element.max_score for element in elements.itervalues())
        #bestscores = self.bestscores = N.fromiter(((element.max_score, id) for id, element in elements.iteritems()), dtype=(float, int))
        bestscores = self.bestscores = self.bestscores = N.fromiter((element.max_score for element in elements.itervalues()), dtype=float)
        bestscores.sort()
        # remove elements who's best score is below the threshold
        threshold = bestscores[-self.prune_rank] if len(bestscores) > self.prune_rank else N.float64(0.0)
        delem = set()
        for key, element in elements.iteritems():
            #if not element.scores or max(element.scores) < threshold:
            if element.max_score < threshold:
                #print ' ', ' id', key, ' user_data', element.user_data
                delem.add(key)
                assert element.user_data[3] == [True]
                element.user_data[3][:] = [False]
        for key in delem:
            del elements[key]
        self.debug and self.debug('prune_inactive:', ' self.num_active', self.num_active, ' removed', self.num_active - len(elements), ' len(elements)', len(elements))

    def update_result(self, elements):
        dc = dcheck("update_result")
        #maxscores = list((max(element.scores), id) for id, element in elements.iteritems() if element.scores)
        maxscores = list((element.max_score, id) for id, element in elements.iteritems())        
        maxscores.sort()
        #maxscores = self.bestscores
        bestscore, bestid = maxscores[-1]
        bestelement = elements[bestid]
        user_data = bestelement.user_data
        class_name = 'eps' if user_data[:2] == self.epsilon_user_data[:2] else self.model.inv_dict[user_data[-1]]
        self.history.append((class_name, bestscore))
        neglog10likelihood = int(-log10(bestscore))
        dc and dc(' obs_id', '%3d' % self.observation_id, ' class', '%4s' % (str(class_name),), ' ', neglog10likelihood, ' ', '-' * neglog10likelihood, ' ', self.num_active)

        return self.observation_id, class_name, neglog10likelihood, self.num_active

    def get_result(self):
        # here's where we return the result of having done something
        # interesting in propagate_hypotheses(), apply_likelihoods(), and
        # update_result()
        #
        # in this case, we copy the history
        return tuple(tuple(items) for items in self.history)


def get_acoustic_observations(audio_filename):
    print 'get_acoustic_observations():'
    print '  filename', repr(audio_filename.replace(onyx.home, '#').replace('\\', '/'))
    with open(audio_filename, 'rb') as audio_file:
        audio_data, (kind, qualifiers), samp_period = read_htk_audio_file(audio_file)
    print '  samp_period', samp_period
    print '  kind:', kind
    print '  qualifiers:', qualifiers
    print '  shape:', audio_data.shape
    return audio_data, samp_period, kind, qualifiers


def _real_example2(parameter):
    """
    A decoding example that uses real models from HTK and a real utterance file from HTK.

    XXX need to make this use build dir when SCons is running things
    >>> model_dirname = path.normpath(path.join(onyx.home, 'py/onyx/htkfiles'))
    >>> model_filename = 'monophones4.mmf'

    XXX need to make this use build dir when SCons is running things
    >>> audio_dirname = path.normpath(path.join(onyx.home, 'data/htk_r1_adg0_4'))
    >>> audio_filename = "adg0_4_sr089.mfc"

    >>> prune_rank = 2
    >>> seed_threshold_rank = 1
    >>> seed_threshold_scale = 1

    >>> parameter = (model_dirname, model_filename), (audio_dirname, audio_filename), (prune_rank, seed_threshold_rank, seed_threshold_scale)

    Run with debugging.  The output here comes from the DebugPrint, we examine the result object later.

    >>> with DebugPrint("update_result"):
    ...     result, elapsed_time = _real_example2(parameter)
    real_model():
      filename '#/py/onyx/htkfiles/monophones4.mmf'
      num_models 49
      arity sets:  input (1,)  output (1,)
      model classes: aa ae ah ao aw ax ay b ch d dd dh dx eh el en er ey f g hh ih iy jh k kd l m n ng ow oy p pd r s sh sil sp t td th ts uh uw v w y z
      skipable classes: sp
    get_acoustic_observations():
      filename '#/data/htk_r1_adg0_4/adg0_4_sr089.mfc'
      samp_period 100000
      kind: (6, 'MFCC', 'mel-frequency cepstral coefficients')
      qualifiers: (('A', 'has acceleration coefficients'), ('E', 'has energy'), ('D', 'has delta coefficients'))
      shape: (246, 39)
    FrameSynchronousProcessor.__init__():
    update_result:  obs_id   0  class  sil   28   ----------------------------   49
    update_result:  obs_id   1  class  sil   27   ---------------------------   2
    update_result:  obs_id   2  class  sil   27   ---------------------------   2
    update_result:  obs_id   3  class  sil   26   --------------------------   2
    update_result:  obs_id   4  class  sil   22   ----------------------   2
    update_result:  obs_id   5  class  sil   25   -------------------------   2
    update_result:  obs_id   6  class  sil   23   -----------------------   2
    update_result:  obs_id   7  class  sil   26   --------------------------   2
    update_result:  obs_id   8  class  sil   26   --------------------------   2
    update_result:  obs_id   9  class  sil   24   ------------------------   2
    update_result:  obs_id  10  class  sil   27   ---------------------------   2
    update_result:  obs_id  11  class  sil   29   -----------------------------   2
    update_result:  obs_id  12  class  sil   31   -------------------------------   2
    update_result:  obs_id  13  class  sil   35   -----------------------------------   2
    update_result:  obs_id  14  class  sil   42   ------------------------------------------   2
    update_result:  obs_id  15  class  sil   46   ----------------------------------------------   2
    update_result:  obs_id  16  class  sil   45   ---------------------------------------------   2
    update_result:  obs_id  17  class  sil   46   ----------------------------------------------   2
    update_result:  obs_id  18  class  sil   42   ------------------------------------------   2
    update_result:  obs_id  19  class  sil   33   ---------------------------------   2
    update_result:  obs_id  20  class  sil   33   ---------------------------------   2
    update_result:  obs_id  21  class  sil   36   ------------------------------------   2
    update_result:  obs_id  22  class   hh   35   -----------------------------------   2
    update_result:  obs_id  23  class   hh   35   -----------------------------------   2
    update_result:  obs_id  24  class   hh   37   -------------------------------------   2
    update_result:  obs_id  25  class   hh   41   -----------------------------------------   2
    update_result:  obs_id  26  class   hh   42   ------------------------------------------   2
    update_result:  obs_id  27  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  28  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  29  class   hh   40   ----------------------------------------   2
    update_result:  obs_id  30  class   hh   38   --------------------------------------   2
    update_result:  obs_id  31  class   hh   34   ----------------------------------   2
    update_result:  obs_id  32  class   hh   32   --------------------------------   2
    update_result:  obs_id  33  class   hh   34   ----------------------------------   2
    update_result:  obs_id  34  class   hh   32   --------------------------------   2
    update_result:  obs_id  35  class   hh   32   --------------------------------   2
    update_result:  obs_id  36  class   hh   32   --------------------------------   2
    update_result:  obs_id  37  class   hh   34   ----------------------------------   2
    update_result:  obs_id  38  class   hh   39   ---------------------------------------   2
    update_result:  obs_id  39  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  40  class   hh   55   -------------------------------------------------------   2
    update_result:  obs_id  41  class   hh   55   -------------------------------------------------------   2
    update_result:  obs_id  42  class   hh   56   --------------------------------------------------------   2
    update_result:  obs_id  43  class   hh   57   ---------------------------------------------------------   2
    update_result:  obs_id  44  class   hh   42   ------------------------------------------   2
    update_result:  obs_id  45  class   hh   32   --------------------------------   2
    update_result:  obs_id  46  class   hh   29   -----------------------------   2
    update_result:  obs_id  47  class   hh   29   -----------------------------   2
    update_result:  obs_id  48  class   hh   28   ----------------------------   2
    update_result:  obs_id  49  class   hh   30   ------------------------------   2
    update_result:  obs_id  50  class   hh   29   -----------------------------   2
    update_result:  obs_id  51  class   hh   29   -----------------------------   2
    update_result:  obs_id  52  class   hh   28   ----------------------------   2
    update_result:  obs_id  53  class   hh   30   ------------------------------   2
    update_result:  obs_id  54  class   hh   37   -------------------------------------   2
    update_result:  obs_id  55  class   hh   41   -----------------------------------------   2
    update_result:  obs_id  56  class   hh   44   --------------------------------------------   2
    update_result:  obs_id  57  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  58  class   hh   56   --------------------------------------------------------   2
    update_result:  obs_id  59  class   hh   57   ---------------------------------------------------------   2
    update_result:  obs_id  60  class   hh   50   --------------------------------------------------   2
    update_result:  obs_id  61  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  62  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  63  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  64  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id  65  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id  66  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  67  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  68  class   hh   44   --------------------------------------------   2
    update_result:  obs_id  69  class   hh   37   -------------------------------------   2
    update_result:  obs_id  70  class   hh   36   ------------------------------------   2
    update_result:  obs_id  71  class   hh   36   ------------------------------------   2
    update_result:  obs_id  72  class   hh   38   --------------------------------------   2
    update_result:  obs_id  73  class   hh   36   ------------------------------------   2
    update_result:  obs_id  74  class   hh   36   ------------------------------------   2
    update_result:  obs_id  75  class   hh   37   -------------------------------------   2
    update_result:  obs_id  76  class   hh   37   -------------------------------------   2
    update_result:  obs_id  77  class   hh   38   --------------------------------------   2
    update_result:  obs_id  78  class   hh   43   -------------------------------------------   2
    update_result:  obs_id  79  class   hh   51   ---------------------------------------------------   2
    update_result:  obs_id  80  class   hh   58   ----------------------------------------------------------   2
    update_result:  obs_id  81  class   hh   60   ------------------------------------------------------------   2
    update_result:  obs_id  82  class   hh   81   ---------------------------------------------------------------------------------   2
    update_result:  obs_id  83  class   hh   80   --------------------------------------------------------------------------------   2
    update_result:  obs_id  84  class   hh   44   --------------------------------------------   2
    update_result:  obs_id  85  class   hh   33   ---------------------------------   2
    update_result:  obs_id  86  class   hh   33   ---------------------------------   2
    update_result:  obs_id  87  class   hh   36   ------------------------------------   2
    update_result:  obs_id  88  class   hh   38   --------------------------------------   2
    update_result:  obs_id  89  class   hh   40   ----------------------------------------   2
    update_result:  obs_id  90  class   hh   38   --------------------------------------   2
    update_result:  obs_id  91  class   hh   37   -------------------------------------   2
    update_result:  obs_id  92  class   hh   33   ---------------------------------   2
    update_result:  obs_id  93  class   hh   33   ---------------------------------   2
    update_result:  obs_id  94  class   hh   33   ---------------------------------   2
    update_result:  obs_id  95  class   hh   34   ----------------------------------   2
    update_result:  obs_id  96  class   hh   36   ------------------------------------   2
    update_result:  obs_id  97  class   hh   42   ------------------------------------------   2
    update_result:  obs_id  98  class   hh   40   ----------------------------------------   2
    update_result:  obs_id  99  class   hh   37   -------------------------------------   2
    update_result:  obs_id 100  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 101  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 102  class   hh   38   --------------------------------------   2
    update_result:  obs_id 103  class   hh   36   ------------------------------------   2
    update_result:  obs_id 104  class   hh   33   ---------------------------------   2
    update_result:  obs_id 105  class   hh   32   --------------------------------   2
    update_result:  obs_id 106  class   hh   32   --------------------------------   2
    update_result:  obs_id 107  class   hh   34   ----------------------------------   2
    update_result:  obs_id 108  class   hh   38   --------------------------------------   2
    update_result:  obs_id 109  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 110  class   hh   38   --------------------------------------   2
    update_result:  obs_id 111  class   hh   38   --------------------------------------   2
    update_result:  obs_id 112  class   hh   37   -------------------------------------   2
    update_result:  obs_id 113  class   hh   37   -------------------------------------   2
    update_result:  obs_id 114  class   hh   36   ------------------------------------   2
    update_result:  obs_id 115  class   hh   36   ------------------------------------   2
    update_result:  obs_id 116  class   hh   38   --------------------------------------   2
    update_result:  obs_id 117  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 118  class   hh   51   ---------------------------------------------------   2
    update_result:  obs_id 119  class   hh   62   --------------------------------------------------------------   2
    update_result:  obs_id 120  class   hh   67   -------------------------------------------------------------------   2
    update_result:  obs_id 121  class   hh   76   ----------------------------------------------------------------------------   2
    update_result:  obs_id 122  class   hh   64   ----------------------------------------------------------------   2
    update_result:  obs_id 123  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id 124  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 125  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 126  class   hh   38   --------------------------------------   2
    update_result:  obs_id 127  class   hh   36   ------------------------------------   2
    update_result:  obs_id 128  class   hh   34   ----------------------------------   2
    update_result:  obs_id 129  class   hh   36   ------------------------------------   2
    update_result:  obs_id 130  class   hh   36   ------------------------------------   2
    update_result:  obs_id 131  class   hh   36   ------------------------------------   2
    update_result:  obs_id 132  class   hh   37   -------------------------------------   2
    update_result:  obs_id 133  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 134  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 135  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 136  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 137  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 138  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 139  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 140  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 141  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 142  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 143  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 144  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id 145  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 146  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 147  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 148  class   hh   35   -----------------------------------   2
    update_result:  obs_id 149  class   hh   32   --------------------------------   2
    update_result:  obs_id 150  class   hh   34   ----------------------------------   2
    update_result:  obs_id 151  class   hh   36   ------------------------------------   2
    update_result:  obs_id 152  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 153  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 154  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 155  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 156  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 157  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 158  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 159  class   hh   38   --------------------------------------   2
    update_result:  obs_id 160  class   hh   37   -------------------------------------   2
    update_result:  obs_id 161  class   hh   37   -------------------------------------   2
    update_result:  obs_id 162  class   hh   37   -------------------------------------   2
    update_result:  obs_id 163  class   hh   38   --------------------------------------   2
    update_result:  obs_id 164  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 165  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 166  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 167  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 168  class   hh   36   ------------------------------------   2
    update_result:  obs_id 169  class   hh   37   -------------------------------------   2
    update_result:  obs_id 170  class   hh   37   -------------------------------------   2
    update_result:  obs_id 171  class   hh   34   ----------------------------------   2
    update_result:  obs_id 172  class   hh   32   --------------------------------   2
    update_result:  obs_id 173  class   hh   34   ----------------------------------   2
    update_result:  obs_id 174  class   hh   34   ----------------------------------   2
    update_result:  obs_id 175  class   hh   33   ---------------------------------   2
    update_result:  obs_id 176  class   hh   34   ----------------------------------   2
    update_result:  obs_id 177  class   hh   35   -----------------------------------   2
    update_result:  obs_id 178  class   hh   38   --------------------------------------   2
    update_result:  obs_id 179  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 180  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 181  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 182  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 183  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 184  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 185  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id 186  class   hh   50   --------------------------------------------------   2
    update_result:  obs_id 187  class   hh   50   --------------------------------------------------   2
    update_result:  obs_id 188  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id 189  class   hh   58   ----------------------------------------------------------   2
    update_result:  obs_id 190  class   hh   59   -----------------------------------------------------------   2
    update_result:  obs_id 191  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 192  class   hh   36   ------------------------------------   2
    update_result:  obs_id 193  class   hh   35   -----------------------------------   2
    update_result:  obs_id 194  class   hh   34   ----------------------------------   2
    update_result:  obs_id 195  class   hh   33   ---------------------------------   2
    update_result:  obs_id 196  class   hh   33   ---------------------------------   2
    update_result:  obs_id 197  class   hh   34   ----------------------------------   2
    update_result:  obs_id 198  class   hh   35   -----------------------------------   2
    update_result:  obs_id 199  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 200  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 201  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 202  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 203  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 204  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 205  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 206  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 207  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 208  class   hh   38   --------------------------------------   2
    update_result:  obs_id 209  class   hh   38   --------------------------------------   2
    update_result:  obs_id 210  class   hh   37   -------------------------------------   2
    update_result:  obs_id 211  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 212  class   hh   38   --------------------------------------   2
    update_result:  obs_id 213  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 214  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 215  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 216  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 217  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 218  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id 219  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 220  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 221  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 222  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 223  class   hh   38   --------------------------------------   2
    update_result:  obs_id 224  class   hh   35   -----------------------------------   2
    update_result:  obs_id 225  class   hh   36   ------------------------------------   2
    update_result:  obs_id 226  class   hh   36   ------------------------------------   2
    update_result:  obs_id 227  class   hh   35   -----------------------------------   2
    update_result:  obs_id 228  class   hh   38   --------------------------------------   2
    update_result:  obs_id 229  class   hh   36   ------------------------------------   2
    update_result:  obs_id 230  class   hh   35   -----------------------------------   2
    update_result:  obs_id 231  class   hh   34   ----------------------------------   2
    update_result:  obs_id 232  class   hh   34   ----------------------------------   2
    update_result:  obs_id 233  class   hh   33   ---------------------------------   2
    update_result:  obs_id 234  class   hh   34   ----------------------------------   2
    update_result:  obs_id 235  class   hh   35   -----------------------------------   2
    update_result:  obs_id 236  class   hh   36   ------------------------------------   2
    update_result:  obs_id 237  class   hh   36   ------------------------------------   2
    update_result:  obs_id 238  class   hh   37   -------------------------------------   2
    update_result:  obs_id 239  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 240  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 241  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 242  class   hh   52   ----------------------------------------------------   2
    update_result:  obs_id 243  class   hh   34   ----------------------------------   2
    update_result:  obs_id 244  class   hh   31   -------------------------------   2
    update_result:  obs_id 245  class   hh   29   -----------------------------   2

    Now look at the results that were returned:

    >>> for res in result: print ' '.join(str(x) for x in res)
    0 sil 28 49
    1 sil 27 2
    2 sil 27 2
    3 sil 26 2
    4 sil 22 2
    5 sil 25 2
    6 sil 23 2
    7 sil 26 2
    8 sil 26 2
    9 sil 24 2
    10 sil 27 2
    11 sil 29 2
    12 sil 31 2
    13 sil 35 2
    14 sil 42 2
    15 sil 46 2
    16 sil 45 2
    17 sil 46 2
    18 sil 42 2
    19 sil 33 2
    20 sil 33 2
    21 sil 36 2
    22 hh 35 2
    23 hh 35 2
    24 hh 37 2
    25 hh 41 2
    26 hh 42 2
    27 hh 48 2
    28 hh 47 2
    29 hh 40 2
    30 hh 38 2
    31 hh 34 2
    32 hh 32 2
    33 hh 34 2
    34 hh 32 2
    35 hh 32 2
    36 hh 32 2
    37 hh 34 2
    38 hh 39 2
    39 hh 47 2
    40 hh 55 2
    41 hh 55 2
    42 hh 56 2
    43 hh 57 2
    44 hh 42 2
    45 hh 32 2
    46 hh 29 2
    47 hh 29 2
    48 hh 28 2
    49 hh 30 2
    50 hh 29 2
    51 hh 29 2
    52 hh 28 2
    53 hh 30 2
    54 hh 37 2
    55 hh 41 2
    56 hh 44 2
    57 hh 48 2
    58 hh 56 2
    59 hh 57 2
    60 hh 50 2
    61 hh 47 2
    62 hh 48 2
    63 hh 48 2
    64 hh 49 2
    65 hh 49 2
    66 hh 47 2
    67 hh 47 2
    68 hh 44 2
    69 hh 37 2
    70 hh 36 2
    71 hh 36 2
    72 hh 38 2
    73 hh 36 2
    74 hh 36 2
    75 hh 37 2
    76 hh 37 2
    77 hh 38 2
    78 hh 43 2
    79 hh 51 2
    80 hh 58 2
    81 hh 60 2
    82 hh 81 2
    83 hh 80 2
    84 hh 44 2
    85 hh 33 2
    86 hh 33 2
    87 hh 36 2
    88 hh 38 2
    89 hh 40 2
    90 hh 38 2
    91 hh 37 2
    92 hh 33 2
    93 hh 33 2
    94 hh 33 2
    95 hh 34 2
    96 hh 36 2
    97 hh 42 2
    98 hh 40 2
    99 hh 37 2
    100 hh 44 2
    101 hh 44 2
    102 hh 38 2
    103 hh 36 2
    104 hh 33 2
    105 hh 32 2
    106 hh 32 2
    107 hh 34 2
    108 hh 38 2
    109 hh 40 2
    110 hh 38 2
    111 hh 38 2
    112 hh 37 2
    113 hh 37 2
    114 hh 36 2
    115 hh 36 2
    116 hh 38 2
    117 hh 43 2
    118 hh 51 2
    119 hh 62 2
    120 hh 67 2
    121 hh 76 2
    122 hh 64 2
    123 hh 49 2
    124 hh 42 2
    125 hh 40 2
    126 hh 38 2
    127 hh 36 2
    128 hh 34 2
    129 hh 36 2
    130 hh 36 2
    131 hh 36 2
    132 hh 37 2
    133 hh 40 2
    134 hh 42 2
    135 hh 44 2
    136 hh 48 2
    137 hh 46 2
    138 hh 43 2
    139 hh 45 2
    140 hh 46 2
    141 hh 48 2
    142 hh 48 2
    143 hh 48 2
    144 hh 47 2
    145 hh 45 2
    146 hh 41 2
    147 hh 39 2
    148 hh 35 2
    149 hh 32 2
    150 hh 34 2
    151 hh 36 2
    152 hh 39 2
    153 hh 40 2
    154 hh 40 2
    155 hh 41 2
    156 hh 43 2
    157 hh 42 2
    158 hh 43 2
    159 hh 38 2
    160 hh 37 2
    161 hh 37 2
    162 hh 37 2
    163 hh 38 2
    164 hh 39 2
    165 hh 44 2
    166 hh 48 2
    167 hh 45 2
    168 hh 36 2
    169 hh 37 2
    170 hh 37 2
    171 hh 34 2
    172 hh 32 2
    173 hh 34 2
    174 hh 34 2
    175 hh 33 2
    176 hh 34 2
    177 hh 35 2
    178 hh 38 2
    179 hh 44 2
    180 hh 45 2
    181 hh 46 2
    182 hh 44 2
    183 hh 40 2
    184 hh 41 2
    185 hh 47 2
    186 hh 50 2
    187 hh 50 2
    188 hh 49 2
    189 hh 58 2
    190 hh 59 2
    191 hh 46 2
    192 hh 36 2
    193 hh 35 2
    194 hh 34 2
    195 hh 33 2
    196 hh 33 2
    197 hh 34 2
    198 hh 35 2
    199 hh 40 2
    200 hh 42 2
    201 hh 44 2
    202 hh 42 2
    203 hh 40 2
    204 hh 41 2
    205 hh 41 2
    206 hh 39 2
    207 hh 39 2
    208 hh 38 2
    209 hh 38 2
    210 hh 37 2
    211 hh 39 2
    212 hh 38 2
    213 hh 41 2
    214 hh 44 2
    215 hh 48 2
    216 hh 46 2
    217 hh 43 2
    218 hh 49 2
    219 hh 46 2
    220 hh 39 2
    221 hh 40 2
    222 hh 42 2
    223 hh 38 2
    224 hh 35 2
    225 hh 36 2
    226 hh 36 2
    227 hh 35 2
    228 hh 38 2
    229 hh 36 2
    230 hh 35 2
    231 hh 34 2
    232 hh 34 2
    233 hh 33 2
    234 hh 34 2
    235 hh 35 2
    236 hh 36 2
    237 hh 36 2
    238 hh 37 2
    239 hh 40 2
    240 hh 43 2
    241 hh 46 2
    242 hh 52 2
    243 hh 34 2
    244 hh 31 2
    245 hh 29 2


    >>> print 'num_observations', len(result), ' elapsed_time', round(elapsed_time, 6), ' observations/second', int(len(result)/elapsed_time)  #doctest: +ELLIPSIS
    num_observations 246  elapsed_time ...  observations/second ...
    """

    (model_dirname, model_filename), (audio_dirname, audio_filename), thresholds = parameter
    
    full_model_filename = path.join(model_dirname, model_filename)
    model = real_model(full_model_filename)
    
    full_audio_filename = path.join(audio_dirname, audio_filename)
    observation_stream, sample_period, kind, qualifiers = get_acoustic_observations(full_audio_filename)

    observation_source = IteratorSource(observation_stream)
##     observation_source = IteratorSource(xrange(100))
    frame_synchronous_decoder = FrameSynchronousProcessor(model, thresholds)
    observation_source.set_sendee(frame_synchronous_decoder.process)
    # XXX I tried, but failed, to use a chain processor here - why didn't this work ?!? (see below) - KJB
    # HSW says, an IteratorSource is not really a processor... we have to revisit the Processor framework
    # as per discussions in Baltimore in late 2008
    # chain = ChainProcessor((observation_source, frame_synchronous_decoder))
    # chain.graph.dot_display(globals=['rankdir=LR'])

    result = list()
    from sys import stdout
    def catch_it(res):
        result.append(res)
        # writing to stdout here appears to have contention with buffers that
        # dcheck/dprint use, and so leads to non-determinism in the log
        #stdout.write("%s\n" % (' '.join(str(x) for x in res),))

    # observation_source.set_sendee(catch_it)
    # XXX This is the part that didn't work - no output showed up
    # chain.set_sendee(catch_it)
    frame_synchronous_decoder.set_sendee(catch_it)

    assert len(result) == 0

    start_time = time.time()
    observation_source.start()
    # hum dee dum dee dum-dum
    observation_source.wait_iter_stopped()
    observation_source.stop(flush=True)
    observation_source.wait()
    observation_source.done()

    return result, time.time() - start_time
    


def _real_example(parameter):
    """
    A decoding example that uses real models from HTK and a real utterance file from HTK.

    XXX need to make this use build dir when SCons is running things
    >>> model_dirname = path.normpath(path.join(onyx.home, 'py/onyx/htkfiles'))
    >>> model_filename = 'monophones4.mmf'

    XXX need to make this use build dir when SCons is running things
    >>> audio_dirname = path.normpath(path.join(onyx.home, 'data/htk_r1_adg0_4'))
    >>> audio_filename = "adg0_4_sr089.mfc"

    >>> prune_rank = 2
    >>> seed_threshold_rank = 1
    >>> seed_threshold_scale = 1

    >>> parameter = (model_dirname, model_filename), (audio_dirname, audio_filename), (prune_rank, seed_threshold_rank, seed_threshold_scale)

    >>> with DebugPrint("update_result"):
    ...     result, elapsed_time = _real_example(parameter)
    real_model():
      filename '#/py/onyx/htkfiles/monophones4.mmf'
      num_models 49
      arity sets:  input (1,)  output (1,)
      model classes: aa ae ah ao aw ax ay b ch d dd dh dx eh el en er ey f g hh ih iy jh k kd l m n ng ow oy p pd r s sh sil sp t td th ts uh uw v w y z
      skipable classes: sp
    get_acoustic_observations():
      filename '#/data/htk_r1_adg0_4/adg0_4_sr089.mfc'
      samp_period 100000
      kind: (6, 'MFCC', 'mel-frequency cepstral coefficients')
      qualifiers: (('A', 'has acceleration coefficients'), ('E', 'has energy'), ('D', 'has delta coefficients'))
      shape: (246, 39)
    update_result:  obs_id   0  class  sil   28   ----------------------------   49
    update_result:  obs_id   1  class  sil   27   ---------------------------   2
    update_result:  obs_id   2  class  sil   27   ---------------------------   2
    update_result:  obs_id   3  class  sil   26   --------------------------   2
    update_result:  obs_id   4  class  sil   22   ----------------------   2
    update_result:  obs_id   5  class  sil   25   -------------------------   2
    update_result:  obs_id   6  class  sil   23   -----------------------   2
    update_result:  obs_id   7  class  sil   26   --------------------------   2
    update_result:  obs_id   8  class  sil   26   --------------------------   2
    update_result:  obs_id   9  class  sil   24   ------------------------   2
    update_result:  obs_id  10  class  sil   27   ---------------------------   2
    update_result:  obs_id  11  class  sil   29   -----------------------------   2
    update_result:  obs_id  12  class  sil   31   -------------------------------   2
    update_result:  obs_id  13  class  sil   35   -----------------------------------   2
    update_result:  obs_id  14  class  sil   42   ------------------------------------------   2
    update_result:  obs_id  15  class  sil   46   ----------------------------------------------   2
    update_result:  obs_id  16  class  sil   45   ---------------------------------------------   2
    update_result:  obs_id  17  class  sil   46   ----------------------------------------------   2
    update_result:  obs_id  18  class  sil   42   ------------------------------------------   2
    update_result:  obs_id  19  class  sil   33   ---------------------------------   2
    update_result:  obs_id  20  class  sil   33   ---------------------------------   2
    update_result:  obs_id  21  class  sil   36   ------------------------------------   2
    update_result:  obs_id  22  class   hh   35   -----------------------------------   2
    update_result:  obs_id  23  class   hh   35   -----------------------------------   2
    update_result:  obs_id  24  class   hh   37   -------------------------------------   2
    update_result:  obs_id  25  class   hh   41   -----------------------------------------   2
    update_result:  obs_id  26  class   hh   42   ------------------------------------------   2
    update_result:  obs_id  27  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  28  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  29  class   hh   40   ----------------------------------------   2
    update_result:  obs_id  30  class   hh   38   --------------------------------------   2
    update_result:  obs_id  31  class   hh   34   ----------------------------------   2
    update_result:  obs_id  32  class   hh   32   --------------------------------   2
    update_result:  obs_id  33  class   hh   34   ----------------------------------   2
    update_result:  obs_id  34  class   hh   32   --------------------------------   2
    update_result:  obs_id  35  class   hh   32   --------------------------------   2
    update_result:  obs_id  36  class   hh   32   --------------------------------   2
    update_result:  obs_id  37  class   hh   34   ----------------------------------   2
    update_result:  obs_id  38  class   hh   39   ---------------------------------------   2
    update_result:  obs_id  39  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  40  class   hh   55   -------------------------------------------------------   2
    update_result:  obs_id  41  class   hh   55   -------------------------------------------------------   2
    update_result:  obs_id  42  class   hh   56   --------------------------------------------------------   2
    update_result:  obs_id  43  class   hh   57   ---------------------------------------------------------   2
    update_result:  obs_id  44  class   hh   42   ------------------------------------------   2
    update_result:  obs_id  45  class   hh   32   --------------------------------   2
    update_result:  obs_id  46  class   hh   29   -----------------------------   2
    update_result:  obs_id  47  class   hh   29   -----------------------------   2
    update_result:  obs_id  48  class   hh   28   ----------------------------   2
    update_result:  obs_id  49  class   hh   30   ------------------------------   2
    update_result:  obs_id  50  class   hh   29   -----------------------------   2
    update_result:  obs_id  51  class   hh   29   -----------------------------   2
    update_result:  obs_id  52  class   hh   28   ----------------------------   2
    update_result:  obs_id  53  class   hh   30   ------------------------------   2
    update_result:  obs_id  54  class   hh   37   -------------------------------------   2
    update_result:  obs_id  55  class   hh   41   -----------------------------------------   2
    update_result:  obs_id  56  class   hh   44   --------------------------------------------   2
    update_result:  obs_id  57  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  58  class   hh   56   --------------------------------------------------------   2
    update_result:  obs_id  59  class   hh   57   ---------------------------------------------------------   2
    update_result:  obs_id  60  class   hh   50   --------------------------------------------------   2
    update_result:  obs_id  61  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  62  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  63  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id  64  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id  65  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id  66  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  67  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id  68  class   hh   44   --------------------------------------------   2
    update_result:  obs_id  69  class   hh   37   -------------------------------------   2
    update_result:  obs_id  70  class   hh   36   ------------------------------------   2
    update_result:  obs_id  71  class   hh   36   ------------------------------------   2
    update_result:  obs_id  72  class   hh   38   --------------------------------------   2
    update_result:  obs_id  73  class   hh   36   ------------------------------------   2
    update_result:  obs_id  74  class   hh   36   ------------------------------------   2
    update_result:  obs_id  75  class   hh   37   -------------------------------------   2
    update_result:  obs_id  76  class   hh   37   -------------------------------------   2
    update_result:  obs_id  77  class   hh   38   --------------------------------------   2
    update_result:  obs_id  78  class   hh   43   -------------------------------------------   2
    update_result:  obs_id  79  class   hh   51   ---------------------------------------------------   2
    update_result:  obs_id  80  class   hh   58   ----------------------------------------------------------   2
    update_result:  obs_id  81  class   hh   60   ------------------------------------------------------------   2
    update_result:  obs_id  82  class   hh   81   ---------------------------------------------------------------------------------   2
    update_result:  obs_id  83  class   hh   80   --------------------------------------------------------------------------------   2
    update_result:  obs_id  84  class   hh   44   --------------------------------------------   2
    update_result:  obs_id  85  class   hh   33   ---------------------------------   2
    update_result:  obs_id  86  class   hh   33   ---------------------------------   2
    update_result:  obs_id  87  class   hh   36   ------------------------------------   2
    update_result:  obs_id  88  class   hh   38   --------------------------------------   2
    update_result:  obs_id  89  class   hh   40   ----------------------------------------   2
    update_result:  obs_id  90  class   hh   38   --------------------------------------   2
    update_result:  obs_id  91  class   hh   37   -------------------------------------   2
    update_result:  obs_id  92  class   hh   33   ---------------------------------   2
    update_result:  obs_id  93  class   hh   33   ---------------------------------   2
    update_result:  obs_id  94  class   hh   33   ---------------------------------   2
    update_result:  obs_id  95  class   hh   34   ----------------------------------   2
    update_result:  obs_id  96  class   hh   36   ------------------------------------   2
    update_result:  obs_id  97  class   hh   42   ------------------------------------------   2
    update_result:  obs_id  98  class   hh   40   ----------------------------------------   2
    update_result:  obs_id  99  class   hh   37   -------------------------------------   2
    update_result:  obs_id 100  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 101  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 102  class   hh   38   --------------------------------------   2
    update_result:  obs_id 103  class   hh   36   ------------------------------------   2
    update_result:  obs_id 104  class   hh   33   ---------------------------------   2
    update_result:  obs_id 105  class   hh   32   --------------------------------   2
    update_result:  obs_id 106  class   hh   32   --------------------------------   2
    update_result:  obs_id 107  class   hh   34   ----------------------------------   2
    update_result:  obs_id 108  class   hh   38   --------------------------------------   2
    update_result:  obs_id 109  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 110  class   hh   38   --------------------------------------   2
    update_result:  obs_id 111  class   hh   38   --------------------------------------   2
    update_result:  obs_id 112  class   hh   37   -------------------------------------   2
    update_result:  obs_id 113  class   hh   37   -------------------------------------   2
    update_result:  obs_id 114  class   hh   36   ------------------------------------   2
    update_result:  obs_id 115  class   hh   36   ------------------------------------   2
    update_result:  obs_id 116  class   hh   38   --------------------------------------   2
    update_result:  obs_id 117  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 118  class   hh   51   ---------------------------------------------------   2
    update_result:  obs_id 119  class   hh   62   --------------------------------------------------------------   2
    update_result:  obs_id 120  class   hh   67   -------------------------------------------------------------------   2
    update_result:  obs_id 121  class   hh   76   ----------------------------------------------------------------------------   2
    update_result:  obs_id 122  class   hh   64   ----------------------------------------------------------------   2
    update_result:  obs_id 123  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id 124  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 125  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 126  class   hh   38   --------------------------------------   2
    update_result:  obs_id 127  class   hh   36   ------------------------------------   2
    update_result:  obs_id 128  class   hh   34   ----------------------------------   2
    update_result:  obs_id 129  class   hh   36   ------------------------------------   2
    update_result:  obs_id 130  class   hh   36   ------------------------------------   2
    update_result:  obs_id 131  class   hh   36   ------------------------------------   2
    update_result:  obs_id 132  class   hh   37   -------------------------------------   2
    update_result:  obs_id 133  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 134  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 135  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 136  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 137  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 138  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 139  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 140  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 141  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 142  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 143  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 144  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id 145  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 146  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 147  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 148  class   hh   35   -----------------------------------   2
    update_result:  obs_id 149  class   hh   32   --------------------------------   2
    update_result:  obs_id 150  class   hh   34   ----------------------------------   2
    update_result:  obs_id 151  class   hh   36   ------------------------------------   2
    update_result:  obs_id 152  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 153  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 154  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 155  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 156  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 157  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 158  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 159  class   hh   38   --------------------------------------   2
    update_result:  obs_id 160  class   hh   37   -------------------------------------   2
    update_result:  obs_id 161  class   hh   37   -------------------------------------   2
    update_result:  obs_id 162  class   hh   37   -------------------------------------   2
    update_result:  obs_id 163  class   hh   38   --------------------------------------   2
    update_result:  obs_id 164  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 165  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 166  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 167  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 168  class   hh   36   ------------------------------------   2
    update_result:  obs_id 169  class   hh   37   -------------------------------------   2
    update_result:  obs_id 170  class   hh   37   -------------------------------------   2
    update_result:  obs_id 171  class   hh   34   ----------------------------------   2
    update_result:  obs_id 172  class   hh   32   --------------------------------   2
    update_result:  obs_id 173  class   hh   34   ----------------------------------   2
    update_result:  obs_id 174  class   hh   34   ----------------------------------   2
    update_result:  obs_id 175  class   hh   33   ---------------------------------   2
    update_result:  obs_id 176  class   hh   34   ----------------------------------   2
    update_result:  obs_id 177  class   hh   35   -----------------------------------   2
    update_result:  obs_id 178  class   hh   38   --------------------------------------   2
    update_result:  obs_id 179  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 180  class   hh   45   ---------------------------------------------   2
    update_result:  obs_id 181  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 182  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 183  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 184  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 185  class   hh   47   -----------------------------------------------   2
    update_result:  obs_id 186  class   hh   50   --------------------------------------------------   2
    update_result:  obs_id 187  class   hh   50   --------------------------------------------------   2
    update_result:  obs_id 188  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id 189  class   hh   58   ----------------------------------------------------------   2
    update_result:  obs_id 190  class   hh   59   -----------------------------------------------------------   2
    update_result:  obs_id 191  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 192  class   hh   36   ------------------------------------   2
    update_result:  obs_id 193  class   hh   35   -----------------------------------   2
    update_result:  obs_id 194  class   hh   34   ----------------------------------   2
    update_result:  obs_id 195  class   hh   33   ---------------------------------   2
    update_result:  obs_id 196  class   hh   33   ---------------------------------   2
    update_result:  obs_id 197  class   hh   34   ----------------------------------   2
    update_result:  obs_id 198  class   hh   35   -----------------------------------   2
    update_result:  obs_id 199  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 200  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 201  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 202  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 203  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 204  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 205  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 206  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 207  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 208  class   hh   38   --------------------------------------   2
    update_result:  obs_id 209  class   hh   38   --------------------------------------   2
    update_result:  obs_id 210  class   hh   37   -------------------------------------   2
    update_result:  obs_id 211  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 212  class   hh   38   --------------------------------------   2
    update_result:  obs_id 213  class   hh   41   -----------------------------------------   2
    update_result:  obs_id 214  class   hh   44   --------------------------------------------   2
    update_result:  obs_id 215  class   hh   48   ------------------------------------------------   2
    update_result:  obs_id 216  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 217  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 218  class   hh   49   -------------------------------------------------   2
    update_result:  obs_id 219  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 220  class   hh   39   ---------------------------------------   2
    update_result:  obs_id 221  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 222  class   hh   42   ------------------------------------------   2
    update_result:  obs_id 223  class   hh   38   --------------------------------------   2
    update_result:  obs_id 224  class   hh   35   -----------------------------------   2
    update_result:  obs_id 225  class   hh   36   ------------------------------------   2
    update_result:  obs_id 226  class   hh   36   ------------------------------------   2
    update_result:  obs_id 227  class   hh   35   -----------------------------------   2
    update_result:  obs_id 228  class   hh   38   --------------------------------------   2
    update_result:  obs_id 229  class   hh   36   ------------------------------------   2
    update_result:  obs_id 230  class   hh   35   -----------------------------------   2
    update_result:  obs_id 231  class   hh   34   ----------------------------------   2
    update_result:  obs_id 232  class   hh   34   ----------------------------------   2
    update_result:  obs_id 233  class   hh   33   ---------------------------------   2
    update_result:  obs_id 234  class   hh   34   ----------------------------------   2
    update_result:  obs_id 235  class   hh   35   -----------------------------------   2
    update_result:  obs_id 236  class   hh   36   ------------------------------------   2
    update_result:  obs_id 237  class   hh   36   ------------------------------------   2
    update_result:  obs_id 238  class   hh   37   -------------------------------------   2
    update_result:  obs_id 239  class   hh   40   ----------------------------------------   2
    update_result:  obs_id 240  class   hh   43   -------------------------------------------   2
    update_result:  obs_id 241  class   hh   46   ----------------------------------------------   2
    update_result:  obs_id 242  class   hh   52   ----------------------------------------------------   2
    update_result:  obs_id 243  class   hh   34   ----------------------------------   2
    update_result:  obs_id 244  class   hh   31   -------------------------------   2
    update_result:  obs_id 245  class   hh   29   -----------------------------   2

    >>> print 'num_observations', len(result), ' elapsed_time', round(elapsed_time, 6), ' observations/second', int(len(result)/elapsed_time)  #doctest: +ELLIPSIS
    num_observations 246  elapsed_time ...  observations/second ...
    >>> print ' '.join(sorted(set(item[0] for item in result)))
    hh sil
    """

    (model_dirname, model_filename), (audio_dirname, audio_filename), thresholds = parameter
    
    full_model_filename = path.join(model_dirname, model_filename)
    model = real_model(full_model_filename)
    
    full_audio_filename = path.join(audio_dirname, audio_filename)
    observation_stream, sample_period, kind, qualifiers = get_acoustic_observations(full_audio_filename)

    context, result, elapsed_time = frame_synchronous_decode(model, thresholds, observation_stream)

    return result, elapsed_time


def _test1():
    print
    print '_test1():'
    
    # XXX need to make this use build dir when SCons is running things
    model_dirname = path.normpath(path.join(onyx.home, 'py/onyx/htkfiles'))
    model_filename = 'monophones4.mmf'

    audio_dirname = path.normpath(path.join(onyx.home, 'data/htk_r1_adg0_4'))
    audio_filename = "adg0_4_sr089.mfc"

    if False:
        prune_rank = 6
        seed_threshold_rank = 4
        seed_threshold_scale = 1
    else:
        prune_rank = 25
        seed_threshold_rank = 15
        seed_threshold_scale = 2

    print '  model', repr(model_filename)
    print '  audio', repr(audio_filename)
    print '  prune_rank', prune_rank
    print '  seed_threshold_rank', seed_threshold_rank
    print '  seed_threshold_scale', seed_threshold_scale

    parameters = (model_dirname, model_filename), (audio_dirname, audio_filename), (prune_rank, seed_threshold_rank, seed_threshold_scale)

    with DebugPrint("update_result"):
        result, elapsed_time = _real_example(parameters)
    #print 'observations/second', int(len(result)/elapsed_time)


def result_to_string(result):
    """
    >>> result_to_string([(1, 2), (2, 2), (3, 3), (3, 4)])
    '1 2 3'

    """
    res = [str(label) for label, _ in result]
    res2 = [res[0]]
    old = res[0]
    for label in res:
        if label != old:
            res2.append(label)
            old = label
    return " ".join(res2)
                
    
if __name__ == '__main__':
    from sys import argv
    args = argv[1:]

    from onyx import onyx_mainstartup

    if not args:
        onyx_mainstartup()

    if '--logreftest' in args:
        _test1()
        
    if '--interactive' in args:
        model_dirname = path.normpath(path.join(onyx.home, 'py/onyx/htkfiles'))
        model_filename = 'monophones4.mmf'

        audio_dirname = path.normpath(path.join(onyx.home, 'data/htk_r1_adg0_4'))
        audio_filename = "adg0_4_sr089.mfc"

        if False:
            prune_rank = 6
            seed_threshold_rank = 4
            seed_threshold_scale = 1
        else:
            prune_rank = 25
            seed_threshold_rank = 15
            seed_threshold_scale = 2

        print '  model', repr(model_filename)
        print '  audio', repr(audio_filename)
        print '  prune_rank', prune_rank
        print '  seed_threshold_rank', seed_threshold_rank
        print '  seed_threshold_scale', seed_threshold_scale

        parameters = (model_dirname, model_filename), (audio_dirname, audio_filename), (prune_rank, seed_threshold_rank, seed_threshold_scale)
    
        thresholds = parameters[2]
        full_model_filename = path.join(model_dirname, model_filename)
        model = real_model(full_model_filename)
    
        full_audio_filename = path.join(audio_dirname, audio_filename)
        observation_stream, sample_period, kind, qualifiers = get_acoustic_observations(full_audio_filename)

        def go():
            context, result, elapsed_time = frame_synchronous_decode(model, thresholds, observation_stream)
            print 'Elapsed time: %s seconds' % elapsed_time
            print result_to_string(result)

        def godb():
            with DebugPrint("update_result"):
                go()
        
