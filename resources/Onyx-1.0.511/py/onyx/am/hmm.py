###########################################################################
#
# File:         hmm.py
# Date:         Tue 18 Mar 2008 15:06
# Author:       Ken Basye
# Description:  Hidden Markov Models
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
Hidden Markov Models

This module provides the Hmm class, which supports modeling with hidden Markov
models.
"""
from __future__ import with_statement, division
import sys
from onyx.am.gaussian import GaussianModelBase, SimpleGaussianModel, GaussianMixtureModel, DummyModel
from onyx.am.gaussian import find_cumulative_index
from onyx.util.floatutils import float_to_readable_string
import onyx.util.mathutils as mathutils
from onyx.util.debugprint import dcheck, DebugPrint
from onyx.util.safediv import safely_divide_float_array
import numpy
from numpy import array, newaxis, zeros
from itertools import izip, repeat, count
from onyx.am.modelmgr import GmmMgr

class Hmm(object):
    """
    Each Hmm is constructed with number of states.  Three functions support Hmm building,
    build_forward_model_compact and build_forward_model_exact, which construct forward models with
    arbitrary skipping, and build_model, which supports arbitrary topologies.

    >>> num_states = 3
    >>> hmm0 = Hmm(num_states)
    >>> print hmm0
    Hmm: num_states = 3
    >>> hmm0_log = Hmm(num_states, log_domain=True)
    >>> dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4) )
    >>> mm = GmmMgr(dummies)
    >>> models = range(3)
    
    >>> hmm1 = Hmm(num_states)
    >>> hmm1.build_forward_model_compact(mm, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    >>> hmm1_log = Hmm(num_states, log_domain=True)
    >>> hmm1_log.build_forward_model_compact(mm, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    >>> hmm2 = Hmm(num_states)
    >>> hmm2.build_forward_model_compact(mm, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    >>> hmm1 == hmm2
    True
    
    >>> hmm0state = Hmm(0)
    >>> hmm1state = Hmm(1)
    >>> hmm2state = Hmm(2)
    >>> hmm3state = Hmm(3)
    >>> hmm4state = Hmm(4)
    >>> with DebugPrint("hmm_bfm") if False else DebugPrint():
    ...      numpy.set_printoptions(linewidth=200)
    ...      hmm0state.build_forward_model_exact(mm, range(0), 5, ((), (), (), (), ()))
    ...      hmm1state.build_forward_model_exact(mm, range(1), 5, ((0.5,), (0.125,), (0.125,), (0.125,), (0.125,)))
    ...      hmm2state.build_forward_model_exact(mm, range(2), 5, ((0.5, 0.5), (0.125, 0.125), (0.125, 0.125), (0.125, 0.125), (0.125, 0.125)))
    ...      hmm3state.build_forward_model_exact(mm, range(3), 5, ((0.5, 0.5, 0.5), (0.125, 0.125, 0.125), (0.125, 0.125, 0.125), (0.125, 0.125, 0.125), (0.125, 0.125, 0.125)))
    ...      hmm4state.build_forward_model_exact(mm, range(4), 5, ((0.5, 0.5, 0.5, 0.5), (0.125, 0.125, 0.125, 0.125), (0.125, 0.125, 0.125, 0.125), (0.125, 0.125, 0.125, 0.125), (0.125, 0.125, 0.125, 0.125)))
    >>> numpy.set_printoptions(linewidth=75)

    >>> mm.set_adaptation_state("INITIALIZING")
    >>> mm.clear_all_accumulators()
    >>> hmm1.begin_adapt("STANDALONE")
    >>> hmm1_log.begin_adapt("STANDALONE")
    >>> mm.set_adaptation_state("ACCUMULATING")
    >>> obs = tuple(repeat(array((0,0)), 12))
    >>> hmm1.adapt_one_sequence(obs)
    >>> hmm1_log.adapt_one_sequence(obs)

    >>> mm.set_adaptation_state("APPLYING")
    >>> hmm1.end_adapt()

    >>> hmm1_log.end_adapt()
    >>> mm.set_adaptation_state("NOT_ADAPTING")
    >>> print hmm1.to_string(full=True)
    Hmm: num_states = 3
      Models (dim = 2):
      GmmMgr index list: [0, 1, 2]
    DummyModel (Dim = 2) always returning a score of 0.1
    DummyModel (Dim = 2) always returning a score of 0.2
    DummyModel (Dim = 2) always returning a score of 0.4
     Transition probabilities:
    [[ 0.          1.          0.          0.          0.        ]
     [ 0.          0.2494527   0.7505473   0.          0.        ]
     [ 0.          0.          0.49667697  0.50332303  0.        ]
     [ 0.          0.          0.          0.88480382  0.11519618]
     [ 0.          0.          0.          0.          0.        ]]

    >>> print hmm1_log.to_string(full=True)
    Hmm: num_states = 3
      Models (dim = 2):
      GmmMgr index list: [0, 1, 2]
    DummyModel (Dim = 2) always returning a score of 0.1
    DummyModel (Dim = 2) always returning a score of 0.2
    DummyModel (Dim = 2) always returning a score of 0.4
     Transition probabilities:
    [[ 0.          1.          0.          0.          0.        ]
     [ 0.          0.2494527   0.7505473   0.          0.        ]
     [ 0.          0.          0.49667697  0.50332303  0.        ]
     [ 0.          0.          0.          0.88480382  0.11519618]
     [ 0.          0.          0.          0.          0.        ]]

    As might be expected, log-domain and probability-domain processing don't
    give exactly the same results, but they're very close in this simple
    example:
    
    >>> hmm1 == hmm1_log
    False
    >>> numpy.allclose(hmm1.transition_matrix, hmm1_log.transition_matrix)
    True

    >>> durs = hmm1.find_expected_durations(12)
    >>> print "Expected durations = %s" % (durs,)
    Expected durations = [ 1.33236099  1.98543653  5.62090219  3.06130028]

    >>> hmm1 == hmm2
    False

    ===================  NETWORK ADAPTATION INTERFACE =========================

    Here's a very simple example of using the network adaptation interface.  In this case, the
    network consists of only one Hmm.  Begin by getting the GmmMgr into the right state:

    >>> mm.set_adaptation_state("INITIALIZING")
    >>> mm.clear_all_accumulators()

    Now get the Hmm into network adaptation mode:

    >>> hmm2.begin_adapt("NETWORK")

    Get the GmmMgr into the accumulation state:

    >>> mm.set_adaptation_state("ACCUMULATING")

    Some trivial observations sice we're using dummy models:

    >>> num_obs = 12
    >>> obs = tuple(repeat(array((0,0)), num_obs))

    Set up for a forward pass.  The second argument says whether this Hmm is at the end of the
    network.  In this case we have only one Hmm, so it is a terminal:

    >>> context = hmm2.init_for_forward_pass(obs, terminal = True)

    Add some mass into the system for the forward pass.  To match the behavior of
    standalone adaptation, we divide an initial mass of 1 evenly across the inputs

    >>> hmm2.accum_input_alphas(context, array(tuple(repeat(1.0/hmm2.num_inputs,
    ...                                                     hmm2.num_inputs))))

    Actually do the forward pass.  Note that we must process one more frame than
    the number of observations - this is because an extra frame is automatically
    added which scores 1 on the exit states of the Hmm (and 0 on all real
    states).  XXX we might want clients to do this for themselves at some point
    rather than this automatic behavior:

    >>> for frame in xrange(num_obs + 1):
    ...    output_alphas = hmm2.process_one_frame_forward(context)

    In this case, these will be None

    >>> output_alphas is None
    True

     Likewise, we initialize and then make the backward pass:

    >>> hmm2.init_for_backward_pass(context)
    >>> hmm2.accum_input_betas(context, array(tuple(repeat(1.0, hmm2.num_outputs))))
    >>> for frame in xrange(num_obs + 1):
    ...    output_betas = hmm2.process_one_frame_backward(context)
    >>> output_betas
    array([  6.80667969e-09])

    Now collect all the gamma sums; here there's only one:

    >>> gamma_sum = hmm2.get_initial_gamma_sum()
    >>> hmm2.add_to_gamma_sum(gamma_sum, context)
    >>> gamma_sum.value
    array([  1.36333398e-09,   1.36333398e-09,   1.36333398e-09,
             1.36333398e-09,   1.36333398e-09,   1.36333398e-09,
             1.36333398e-09,   1.36333398e-09,   1.36333398e-09,
             1.36333398e-09,   1.36333398e-09,   1.36333398e-09,
             1.36333398e-09])

    Here's where the actual accumulation happens:

    >>> hmm2.do_accumulation(context, gamma_sum)

    Finally, get the GmmMgr into the correct state and apply the accumulators to
    actually adapt the model.  Because we're using dummy models, only the
    transition probabilities change in this example:

    >>> mm.set_adaptation_state("APPLYING")
    >>> hmm2.end_adapt()
    >>> mm.apply_all_accumulators()
    >>> print hmm2.to_string(full=True)
    Hmm: num_states = 3
      Models (dim = 2):
      GmmMgr index list: [0, 1, 2]
    DummyModel (Dim = 2) always returning a score of 0.1
    DummyModel (Dim = 2) always returning a score of 0.2
    DummyModel (Dim = 2) always returning a score of 0.4
     Transition probabilities:
    [[ 0.          1.          0.          0.          0.        ]
     [ 0.          0.2494527   0.7505473   0.          0.        ]
     [ 0.          0.          0.49667697  0.50332303  0.        ]
     [ 0.          0.          0.          0.88480382  0.11519618]
     [ 0.          0.          0.          0.          0.        ]]

    >>> durs = hmm1.find_expected_durations(12, verbose=True)
    Occupancy probs after step 0:   [ 1.  0.  0.  0.]  (sum = 1.0)
    Occupancy probs after step 1:   [ 0.2494527  0.7505473  0.         0.       ]  (sum = 1.0)
    Occupancy probs after step 2:   [ 0.06222665  0.56000561  0.37776774  0.        ]  (sum = 1.0)
    Occupancy probs after step 3:   [ 0.01552261  0.32484593  0.61611406  0.0435174 ]  (sum = 1.0)
    Occupancy probs after step 4:   [ 0.00387216  0.17299394  0.70864251  0.11449139]  (sum = 1.0)
    Occupancy probs after step 5:   [ 0.00096592  0.08882834  0.71408144  0.1961243 ]  (sum = 1.0)
    Occupancy probs after step 6:   [  2.40951302e-04   4.48439616e-02   6.76531333e-01   2.78383754e-01]  (sum = 1.0)
    Occupancy probs after step 7:   [  6.01059535e-05   2.24538084e-02   6.21168505e-01   3.56317580e-01]  (sum = 1.0)
    Occupancy probs after step 8:   [  1.49935925e-05   1.11974019e-02   5.60913784e-01   4.27873820e-01]  (sum = 1.0)
    Occupancy probs after step 9:   [  3.74019217e-06   5.57274505e-03   5.01934568e-01   4.92488947e-01]  (sum = 1.0)
    Occupancy probs after step 10:   [  9.33001044e-07   2.77066132e-03   4.46918514e-01   5.50309892e-01]  (sum = 1.0)
    Occupancy probs after step 11:   [  2.32739632e-07   1.37682393e-03   3.96829745e-01   6.01793198e-01]  (sum = 1.0)
    >>> print "Expected durations = %s" % (durs,)
    Expected durations = [ 1.33236099  1.98543653  5.62090219  3.06130028]

    FORWARD SEQUENCE INTERFACE TESTS START HERE

    Create a Hmm with single-state skipping and observation likelihoods of one
    so that no probability leaks out

    >>> fs = Hmm(10)
    >>> dummies = tuple(repeat(DummyModel(2, 1.0), 10))
    >>> mm = GmmMgr(dummies)
    >>> models = range(10)
    
    >>> fs.build_forward_model_compact(mm, models, 3, ((0.0625,) * 10, (0.75,) * 10, (0.1875,) * 10))

    >>> fs.num_states
    10

    >>> print fs.to_string(full=True)
    Hmm: num_states = 10
      Models (dim = 2):
      GmmMgr index list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
    DummyModel (Dim = 2) always returning a score of 1.0
     Transition probabilities:
    [[ 0.      0.      1.      0.      0.      0.      0.      0.      0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.      1.      0.      0.      0.      0.      0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.0625  0.75    0.1875  0.      0.      0.      0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.0625  0.75    0.1875  0.      0.      0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.0625  0.75    0.1875  0.      0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.0625  0.75    0.1875  0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.0625  0.75    0.1875
       0.      0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.      0.0625  0.75
       0.1875  0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.0625
       0.75    0.1875  0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.
       0.0625  0.75    0.1875  0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
       0.0625  0.75    0.1875  0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
       0.      0.0625  0.75    0.1875]
     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
       0.      0.      0.      0.    ]
     [ 0.      0.      0.      0.      0.      0.      0.      0.      0.      0.
       0.      0.      0.      0.    ]]


    Get a decoding context

    >>> fsc = fs.get_decode_context('A')
    >>> fsc.hmm is fs
    True
    >>> fsc.user_data == 'A'
    True

    Inject some mass, twice, e.g. as at a join, total injected mass sums to one

    >>> fsc.activate((0.0, 0.75))
    >>> fsc.activate((0.25, 0))
    >>> sum(fsc.scores) == 0
    True

    We can decode six times total without any mass leaking out of the right-hand
    side because that's how long it takes for the mass to get to the (virtual)
    output states.

    >>> fsc.pass_tokens()
    >>> fsc.has_output and fsc.output
    False
    >>> dummy_obs = (0,0)
    >>> fsc.apply_likelihoods(dummy_obs)
    >>> sum(fsc.scores)
    1.0

    >>> for i in xrange(4): fsc.pass_tokens(); fsc.apply_likelihoods(dummy_obs)
    >>> sum(fsc.scores)
    1.0

    And here's the non-zero output

    >>> fsc.pass_tokens()
    >>> fsc.has_output and fsc.output #doctest: +ELLIPSIS
    (0.00353407859802..., 0.000173807144165...)

    One more set of likelihoods and we're dropping the output mass on the floor

    >>> fsc.apply_likelihoods(dummy_obs)
    >>> sum(fsc.scores) < 1
    True

    >>> num_states = 1
    >>> out_order = 2

    >>> fs = Hmm(num_states)
    >>> dummies = [DummyModel(2, 1.0)] * num_states
    >>> mm = GmmMgr(dummies)
    >>> models = range(num_states)
    
    >>> fs.build_forward_model_compact(mm, models, 2, tuple([p] * num_states for p in toy_probs(out_order)))

    >>> fsc = fs.get_decode_context()
    >>> fsc.activate(toy_probs(out_order - 1))
    >>> sum(fsc.scores)
    0.0

    >>> fsc.pass_tokens()
    >>> fsc.has_output and fsc.output
    False

    >>> fsc.apply_likelihoods(dummy_obs)
    >>> sum(fsc.scores)
    1.0
    >>> fsc.pass_tokens()
    >>> fsc.has_output and fsc.output
    (0.75,)
    >>> fsc.apply_likelihoods(dummy_obs)

    See that we've lost the output mass

    >>> sum(fsc.scores)
    0.25


    Note that a zero-state Hmm makes sense, even with state skipping.  It's just an epsilon

    >>> num_states = 0
    >>> out_order = 4

    >>> fs = Hmm(num_states)
    >>> dummies = [DummyModel(2, 1.0)]
    >>> mm = GmmMgr(dummies)
    >>> models = range(num_states)
    
    >>> fs.build_forward_model_compact(mm, models, out_order, tuple([p] * num_states for p in toy_probs(out_order)))

    >>> fsc = fs.get_decode_context()
    >>> fsc.activate(toy_probs(out_order - 1))
    >>> len(fsc.scores)
    0
    >>> sum(fsc.scores)
    0

    >>> fsc.pass_tokens()
    >>> fsc.has_output and fsc.output
    (0.25, 0.625, 0.125)
    >>> sum(fsc.output)
    1.0
    >>> fsc.apply_likelihoods(())
    >>> sum(fsc.scores)
    0

    >>> fs = Hmm(num_states)
    >>> dummies = [DummyModel(2, 1.0)] * 1
    >>> mm = GmmMgr(dummies)
    >>> models = range(num_states)
    
    >>> fs.build_forward_model_compact(mm, models, 1, (tuple(xrange(10)), ))
    Traceback (most recent call last):
      ...
    ValueError: expected transition order of at least 2, but got 1
    """


    # ============================ CONSTANTS ===================================

    # dict of states you can come from
    _LEGAL_STATE_DICT = {'start': (),
                         'activate' : ('start', 'apply_likelihoods', 'activate'),
                         'pass_tokens' : ('activate','apply_likelihoods'),
                         'apply_likelihoods' : ('pass_tokens',) }

    _ADAPTATION_MODES = ('STANDALONE', 'NETWORK')

    # ==================== CONSTRUCTION AND BASIC PROPERTIES  ====================

    def __init__(self, num_states, log_domain=False, user_data=None):
        assert num_states >= 0
        self._num_states = num_states
        self._log_domain = log_domain
        self._user_data = user_data
        self._models = None
        self._gmm_mgr = None
        self._transition_matrix = None
        self._working_transition_matrix = None
        self._working_trans_mat_swapped = None
        self._dimension = None
        self._num_inputs = None
        self._num_outputs = None
        self.have_models = False        
        self._num_frames_accumulated = 0
        self._activation = None
        self._adapt_mode = None
        self._trans_accums = None
        self._trans_norms = None
        self._covariance_type = None
        self._set_domain_specific_functions()

    @property
    def dimension(self):
        """
        A dimension of None means we're an epsilon model, i.e. we have no real
        states, so we don't care what the dimensions of observations passed to
        us is.
        """
        return self._dimension

    @property
    def num_states(self):
        return self._num_states

    @property
    def log_domain(self):
        return self._log_domain

    @property
    def user_data(self):
        return self._user_data

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def covariance_type(self):
        return self._covariance_type

    @property
    def gmm_manager_id(self):
        return id(self._gmm_mgr)

    @property
    def gmm_manager(self):
        return self._gmm_mgr

    @property
    def adapting(self):
        return self._adapt_mode is not None

    @property
    def models(self):
        return self._models

    @property
    def transition_matrix(self):
        return self._transition_matrix

    # ================================ CONSTRUCTION =======================================


    def build_forward_model_compact(self, gmm_mgr, models, transition_order, transition_prob_sequences):
        """
        Build a forward-transitioning Hmm from models and transition probabilities.  Models must be
        as long as num_states and must contain valid indicies for gmm_mgr.  Transition order must
        be at least 2, which corresponds to a Bakis model (self-loops and 1-forward arcs).
        transition_prob_sequences must have transition-order elements, each if which is a sequence
        of transition probabilities of length num_states.  The first sequence is the self-loop
        probs, the second is the 1-forward probs, and so on.  The model built is compact in the
        sense that multiple input and output transitions are allowed to 'share' a common virtual
        input or output, so the number of virtual input/output nodes is linear in the order.  When
        used in higher-level graph structures (like TrainingGraph), models of this type result in
        training which approximates, but does not capture exactly, the results of using a single
        standalone model.
        """
        ns = self.num_states
        if len(models) != ns:
            raise ValueError("expected %d models, but got %d" % (ns, len(models)))
        if transition_order < 2:
            raise ValueError("expected transition order of at least 2, but got %d" % (transition_order))
        transition_prob_sequences = tuple(transition_prob_sequences)
        if len(transition_prob_sequences) != transition_order:
            raise ValueError("expected transition_prob_sequences of length %d, but got %d" %
                             (transition_order, len(transition_prob_sequences)))

        num_inputs = num_outputs = transition_order - 1
        n = num_inputs + ns +  num_outputs
        trans = zeros((n,n), dtype = float)

        # Virtual input states will have p=1 transitions into the first transition_order-1 model states.
        # Note that if the order is high enough and/or there are very few (or even 0) model states, there
        # may be transitions directly from input to output states; that's OK.
        skip = transition_order - 1
        for i in xrange(num_inputs):
            trans[i, i+skip] = 1.0

        trans_models = trans[num_inputs:, num_inputs:]
        for order, seq in enumerate(transition_prob_sequences):
            if len(seq) != ns:
                raise ValueError("expected every transition prob sequence to be of length %d, but got one of length %d" % (ns, len(seq)))
            for i, tp in enumerate(seq):
                trans_models[i,i+order] = tp

        self.build_model(gmm_mgr, models, num_inputs, num_outputs, trans)


    def build_forward_model_exact(self, gmm_mgr, models, transition_order, transition_prob_sequences):
        """
        Build a forward-transitioning Hmm from models and transition probabilities.  Models must be
        as long as num_states and must contain valid indicies for gmm_mgr.  Transition order must
        be at least 2, which corresponds to a Bakis model (self-loops and 1-forward arcs).
        transition_prob_sequences must have transition-order elements, each if which is a sequence
        of transition probabilities of length num_states.  The first sequence is the self-loop
        probs, the second is the 1-forward probs, and so on.  The model built is exact in the sense
        that input and output transitions are not allowed to share a common virtual input or output,
        so the number of virtual input/output nodes is quadratic in the order.  When used in
        higher-level graph structures (like TrainingGraph), models of this type result in training
        which exactly captures the results of using a single standalone model.
        """
        dc = dcheck("hmm_bfm")
        ns = self.num_states
        if len(models) != ns:
            raise ValueError("expected %d models, but got %d" % (ns, len(models)))
        if transition_order < 2:
            raise ValueError("expected transition order of at least 2, but got %d" % (transition_order,))
        if len(transition_prob_sequences) != transition_order:
            raise ValueError("expected transition_prob_sequences of length %d, but got %d" %
                             (transition_order, len(transition_prob_sequences)))

        num_inputs = num_outputs = int((transition_order * (transition_order - 1)) / 2)
        n = num_inputs + ns +  num_outputs
        trans = zeros((n,n), dtype = float)

        # Virtual input states will have p=1 transitions into the first transition_order-1 model
        # states, but note that there will be transition_order-2 transitions into the first state,
        # transition_order-3 into the second state, and so on.  This is to facilitate having unique
        # paths between the end states of one model and the start states of another model.  Note
        # that if the order is high enough and/or there are very few (or even 0) model states, there
        # may be transitions directly from input to output states; that's OK, but makes for
        # trickiness in the code.  Similar trickiness occurs in the code below assigning other
        # transition probabilities.  Readers who wish to understand this code are advised to
        # activate "hmm_bfm" and run the doctest for this module, which may shed some light on what
        # we're trying to end up with.
        curr_in = 0
        shift = 0
        for i in xrange(transition_order):
            gap = 0
            for j in xrange(i):
                target = num_inputs + j
                s = shift if  target >= num_inputs + ns else 0
                trans[curr_in, target + s] = 1.0
                gap += 1 if  target >= num_inputs + ns else 0
                curr_in += 1
            shift += gap


        trans_models = trans[num_inputs:, num_inputs:]
        pre_shift = max(0, (transition_order - ns)*(transition_order - ns - 1) / 2)
        for order, seq in enumerate(transition_prob_sequences):
            if len(seq) != ns:
                raise ValueError("expected all elements of transition_prob_sequences to be length %d, but got one of length %d" %
                             (ns, len(seq)))
            shift = pre_shift
            for i, tp in enumerate(seq):
                # We only want to shift transitions to output states
                s = shift if i+order >= ns else 0
                trans_models[i,i+order+s] = tp
                gap = transition_order - ns + i
                shift += gap
                
        dc and dc("trans = \n%s" % (trans,))
        self.build_model(gmm_mgr, models, num_inputs, num_outputs, trans)


    def build_model(self, gmm_mgr, models, num_inputs, num_outputs, transition_probs):
        """
        Build an Hmm from models and transition probabilities.  num_inputs and
        num_outputs specify the number of ways into and out of the model.
        transition_probs is a Numpy array with shape (n,n), where n = num_inputs
        + self.num_states + num_outputs.  Transitions into the (virtual) input
        states must be zero, transitions out of the (virtual) output states must
        be zero.
        """
        assert len(models) == self.num_states
        self._num_inputs = ni = num_inputs
        self._num_outputs = num_outputs
        self._gmm_mgr = gmm_mgr
        m0 = gmm_mgr.get_model(models[0]) if self.num_states != 0 else None
        self._dimension = m0.dimension if self.num_states != 0 else None
        self._covariance_type = m0.covariance_type if self.num_states != 0 else None
        
        n = num_inputs + self.num_states +  num_outputs
        
        self._models = []
        for mi in models:
            m = gmm_mgr.get_model(mi)
            assert m.dimension == self.dimension
            assert m.covariance_type == self._covariance_type
            self._models.append(mi)
        self._transition_matrix = numpy.zeros((n, n), dtype=float)
        if transition_probs is None or transition_probs.shape != (n, n):
            raise ValueError("build_model requires a transition array with shape %s" % ((n,n),))
        self._transition_matrix[:,:] = transition_probs[:,:]
        self._working_transition_matrix = self._to_working_domain(self._transition_matrix)
        self.verify_transition_map()  # Throws if bad
        self._working_trans_mat_swapped = self._working_transition_matrix.swapaxes(0,1)
        self.have_models = True

    def sample(self, start_state_dist=None):
        """
        Sample an Hmm

        Returns a tuple of samples from the states of the Hmm; each sample will
        be a pair (state_index, obs) where state_index is the 0-based index of
        the real state used to sample and obs is the sampled observation.  Note
        that this will be of no particular length unless the transition matrix
        of the Hmm is deterministic in the sense that a certain number of
        transitions is guaranteed.  If provided, start_state_dist is a
        distribution over the input state of the Hmm (so it must sum to 1).  If
        not provided, a uniform distribution over input states will be used
        """
        dc = dcheck("hmm_sample")
        ni = self.num_inputs
        ns = self.num_states
        if start_state_dist is None:
            start_state_dist = array([(1.0 / ni) for _ in xrange(ni)])
        elif  start_state_dist.sum() != 1.0:
            raise("expected start_state_dist to sum to one, but got %s" % (start_state_sum))
        dc and dc("start_state_dist = %s" % (start_state_dist,))
        result = []
        r = GaussianModelBase.rand.randrange(ni)
        current_state = find_cumulative_index(start_state_dist, r)
        # Note that there's no check here that the transition matrix has not been
        # constructed with a deterministic loop that never gets to an output state;
        # sampling Hmms so constructed is not recommended :->.
        while( current_state < ni + ns):
            dc and dc("current_state = %s" % (current_state,))
            if current_state >= ni:
                mi = self._models[current_state - ni]
                m = self._gmm_mgr.get_model(mi)
                samp = m.sample()
                result.append((current_state - ni, samp))
            r = GaussianModelBase.rand.random()
            out_trans_dist = self.transition_matrix[current_state]
            current_state = find_cumulative_index(out_trans_dist, r)
            dc and dc("new current_state = %s" % (current_state,))
        return result
        
                
    # ================================ PRINTING AND DIAGNOSTICS ================================
        
    def __str__(self):
        return self.to_string()

    def to_string(self, full = False):
        ret = "Hmm: num_states = %d" % (self.num_states,)
        if self.have_models and self.num_states > 0:
            if full:
                ret += "\n  Models (dim = %s):" % (self.dimension)
                ret += "\n  GmmMgr index list: %s\n" % (self._models)
                for mi in self._models:
                    m = self._gmm_mgr.get_model(mi)
                    ret += str(m) + "\n"
                ret += " Transition probabilities:\n"
                ret += str(self._transition_matrix)
            else:
                ret += ", model dim = %d" % (self.dimension)
        return ret
        
    def __eq__(self, other):
        if ((self.num_states, self.num_inputs, self.num_outputs) != (other.num_states, other.num_inputs, other.num_outputs) or
            (self.dimension, self.covariance_type) != (other.dimension, other.covariance_type) or
            (self.models != other.models) or
            any((self.gmm_manager[m] != other.gmm_manager[m] for m in self.models)) or
            (self.transition_matrix != other.transition_matrix).any()):
            return False
        return True
        
    def __ne__(self, other):
        return not self.__eq__(other)

    def verify_transition_map(self):
        tm = self._transition_matrix
        n = self.num_inputs + self.num_states + self.num_outputs
        if tm.shape != (n, n):
            raise ValueError("bad transition map shape")
        if not (tm[:,:self.num_inputs] == 0.0).all():
            raise ValueError("non-zero transition into some input state; trans = \n%s" % (tm,))
        if not (tm[-self.num_outputs] == 0.0).all():
            raise ValueError("non-zero transition out of some output state")
        n_real = self.num_inputs + self.num_states
        if not numpy.allclose(tm[:n_real].sum(1), numpy.ones((n_real,), dtype=float)):
            # Pinpoint the problems
            prob_list = list()
            for i in xrange(n_real):
                out_prob_sum = tm[i].sum()
                if out_prob_sum < .999 or out_prob_sum > 1.001:
                    prob_list.append("State %d with sum %s" % (i, out_prob_sum))
            problems = '\n'.join(prob_list)
                    
            raise ValueError("outputs from some state don't sum to 1.0 (or even very close)\n" + 
                             "Details: \n" + problems)

    def validateDataSequence(self, sequence):
        if self.num_states > 0:
            for point in sequence:
                if not isinstance(point, numpy.ndarray) or point.shape != (self.dimension,):
                    raise ValueError("Data sequence must be a sequence of ndarrays of shape (%d,)" % self.dimension) 

    def find_expected_durations(self, frame_count, verbose = False):
        """ Find the expected duration in each state for a given frame count.
            Includes an expected duration for each virtual 'exit' state"""
        # XXX Find a way to write this function using the decoding interface,
        # or better still, factor out the common code for the decoding
        # interface and the forward alpha calculations.
        input_alphas = array(tuple(repeat(self._one_prob, self.num_inputs)))
        n = self.num_states + self.num_outputs
        scores = tuple(repeat(self._one_prob, n))
        alphas = self._make_zero_prob_array((n,), dtype = float)
        sum = self._make_zero_prob_array((n,), dtype = float)
        # We set a p = 1 self-loop on each exit state here so we can see
        # that our expected durations sum correctly to the caller's frame count
        out_map = self._working_transition_matrix[-self.num_outputs:,-self.num_outputs:]
        for i in xrange(self.num_outputs):
            out_map[i,i] = self._one_prob
        for i in xrange(frame_count):
            alphas = self._get_alphas_for_frame(alphas, input_alphas, scores)
            # We really only need to reset the input_alphas to zero once, but it has to be between
            # the first and second call to _get_alphas_for_frame, so I'm going to do it here for
            # convenience
            input_alphas[:] = self._zero_prob
            if verbose:
                print "Occupancy probs after step %d:   %s  (sum = %s)" % (i, alphas, self._sum_prob_array(alphas))
            sum = self._add_prob(sum, alphas)
        for i in xrange(self.num_outputs):
            out_map[i,i] = self._zero_prob
        return sum

    def _default_state_attribute_callback(index, not_used, state_type):
        attribs = ('label="%s%d", style=bold' % (state_type, index))
        if state_type == 'i':
            attribs += ', shape=circle, fixedsize=true, height=0.4, color=green'
        elif state_type == 'o':
            attribs += ', shape=circle, fixedsize=true, height=0.4, color=red'
        return attribs
    
    _default_transition_attribute_callback = lambda i,j,tp: 'label="%s"' % (tp,)
    def dot_iter(self, hmm_label='', graph_type='digraph', globals=(), start_state_num=0,
                 state_callback=_default_state_attribute_callback,
                 transition_callback=_default_transition_attribute_callback):
        gen = self.dot_iter_generator(hmm_label, graph_type, globals, start_state_num,
                                      state_callback, transition_callback)
        
        in_names = tuple("state%d" % n for n in xrange(start_state_num, start_state_num + self.num_inputs))
        out_start_idx = start_state_num + self.num_inputs + self.num_states
        out_names = tuple("state%d" % n for n in xrange(out_start_idx, out_start_idx + self.num_inputs))
        return (gen, in_names, out_names)


    def dot_iter_generator(self, hmm_label='', graph_type='digraph', globals=(), start_state_num=0,
                           state_callback=_default_state_attribute_callback,
                           transition_callback=_default_transition_attribute_callback):
        """
        Returns a generator that yields lines of text, including newlines.  The text represents the
        Hmm in the DOT language for which there are many displayers.  See also dot_display().

        Optional argument hmm_label is a string label for the Hmm.  Optional argument graph_type
        defaults to 'digraph', can also be 'graph' and 'subgraph'

        Optional state_callback is a function that will be called with the index of each state and a
        string from the set {'i', 's', 'o'} indicating that the state in question is an input state,
        a real state, or an output state, respectively.  The function should return a valid DOT
        language a_list of attributes for drawing the node.  By default, each node will be labelled
        with the type character followed by the index.

        Optional transition_callback is a function that will be called with the index of the source
        state, the index of the destination state, and the probability of the transition.  It should
        return a valid DOT language a_list of attributes for drawing the transition.  By default,
        each transition will be labelled with its probability.

        Optional globals is an iterable that yields strings to put in the globals section of the DOT
        file.

        >>> dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4) )
        >>> mm = GmmMgr(dummies)
        >>> models = range(4)
        >>> hmm = Hmm(4)
        >>> hmm2 = Hmm(4)
        >>> hmm3 = Hmm(4)
        >>> hmm.build_forward_model_compact(mm, models, 3, ((0.25, 0.25, 0.25, 0.4),
        ...                                                 (0.5, 0.5, 0.5, 0.4),
        ...                                                 (0.25, 0.25, 0.25, 0.2)))
        >>> hmm2.build_forward_model_exact(mm, models, 3, ((0.25, 0.25, 0.25, 0.4),
        ...                                                 (0.5, 0.5, 0.5, 0.4),
        ...                                                 (0.25, 0.25, 0.25, 0.2)))
        >>> trans = array(((0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        ...                (0.0, 0.0, 0.5, 0.25, 0.25, 0.0),
        ...                (0.0, 0.0, 0.5, 0.25, 0.0, 0.25),
        ...                (0.0, 0.5, 0.25, 0.25, 0.0, 0.0),
        ...                (0.0, 0.5, 0.25, 0.25, 0.0, 0.0),
        ...                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
        >>> hmm3.build_model(mm, models, 1, 1, trans)

        >>> line_iter, in_names, out_names = hmm.dot_iter()
        >>> print in_names
        ('state0', 'state1')
        >>> print out_names
        ('state6', 'state7')
        >>> print ''.join(line_iter)
        digraph  { 
          rankdir=LR;
          state0  [label="i0", style=bold, shape=circle, fixedsize=true, height=0.4, color=green];
          state1  [label="i1", style=bold, shape=circle, fixedsize=true, height=0.4, color=green];
          state2  [label="s0", style=bold];
          state3  [label="s1", style=bold];
          state4  [label="s2", style=bold];
          state5  [label="s3", style=bold];
          state6  [label="o0", style=bold, shape=circle, fixedsize=true, height=0.4, color=red];
          state7  [label="o1", style=bold, shape=circle, fixedsize=true, height=0.4, color=red];
          state0 -> state2 [label="1.0"];
          state1 -> state3 [label="1.0"];
          state2 -> state2 [label="0.25"];
          state2 -> state3 [label="0.5"];
          state2 -> state4 [label="0.25"];
          state3 -> state3 [label="0.25"];
          state3 -> state4 [label="0.5"];
          state3 -> state5 [label="0.25"];
          state4 -> state4 [label="0.25"];
          state4 -> state5 [label="0.5"];
          state4 -> state6 [label="0.25"];
          state5 -> state5 [label="0.4"];
          state5 -> state6 [label="0.4"];
          state5 -> state7 [label="0.2"];
        }
        <BLANKLINE>

        >>> line_iter, in_names, out_names = hmm.dot_iter(hmm_label='Foo', state_callback=lambda i,gi,t: 'label="%s%d", style=bold' % (t,i), globals=('rankdir=LR;',))
        >>> print ''.join(line_iter)
        digraph Foo { 
          rankdir=LR;
          rankdir=LR;
          state0  [label="i0", style=bold];
          state1  [label="i1", style=bold];
          state2  [label="s0", style=bold];
          state3  [label="s1", style=bold];
          state4  [label="s2", style=bold];
          state5  [label="s3", style=bold];
          state6  [label="o0", style=bold];
          state7  [label="o1", style=bold];
          state0 -> state2 [label="1.0"];
          state1 -> state3 [label="1.0"];
          state2 -> state2 [label="0.25"];
          state2 -> state3 [label="0.5"];
          state2 -> state4 [label="0.25"];
          state3 -> state3 [label="0.25"];
          state3 -> state4 [label="0.5"];
          state3 -> state5 [label="0.25"];
          state4 -> state4 [label="0.25"];
          state4 -> state5 [label="0.5"];
          state4 -> state6 [label="0.25"];
          state5 -> state5 [label="0.4"];
          state5 -> state6 [label="0.4"];
          state5 -> state7 [label="0.2"];
        }
        <BLANKLINE>

        # >>> hmm.dot_display()
        # >>> hmm.dot_display(hmm_label='FourStateForward',
        # ...                 globals=('label="Compact Four-state HMM with 1 node skipping"',
        # ...                          'labelloc=top'))
        # >>> hmm2.dot_display(hmm_label='FourStateForward',
        # ...                 globals=('label="Exact Four-state HMM with 1 node skipping"',
        # ...                          'labelloc=top'))
        # >>> hmm3.dot_display()
        
        """

        # By default we use a left-to-right layout, but we prepend this so that if the caller's
        # globals includes a different rankdir attribute it will be effective.
        globals = ('rankdir=LR;',) + globals
        # opening
        yield "%s %s { \n" % (graph_type, hmm_label)

        for line in globals:
            yield '  %s\n' % (line,)

        # states
        idx = start_state_num
        for i in xrange(self.num_inputs):
            yield '  state%d  [%s];\n' % (idx, state_callback(i, idx, 'i'))
            idx += 1
        for i in xrange(self.num_states):
            yield '  state%d  [%s];\n' % (idx, state_callback(i, idx, 's'))
            idx += 1
        for i in xrange(self.num_outputs):
            yield '  state%d  [%s];\n' % (idx, state_callback(i, idx, 'o'))
            idx += 1

        # transitions
        n = self.num_inputs + self.num_states +  self.num_outputs
        for i in xrange(n):
            for j in xrange(n):
                tp = self._transition_matrix[i,j]
                if tp != 0.0:
                    yield '  state%d -> state%d [%s];\n' % (i + start_state_num,
                                                            j + start_state_num,
                                                            transition_callback(i, j, tp))

        # closing                    
        yield "}\n"


    def dot_display(self, temp_file_prefix='hmm_',
                    display_tool_format="open -a /Applications/Graphviz.app %s", **kwargs):
        """
        Display a dot-generated representation of the Hmm.  Returns the name of the temporary file
        that the display tool is working from.  The caller is responsible for removing this file.

        Optional temp_file_prefix is the prefix used for the filename.

        Optional display_tool_format is formatting string, with %s where the filename goes, used to
        generate the command that will display the file.  By default it assumes you're on a Mac and
        have Graphviz.app installed in the /Applications directory.

        Remaining keyword arguments are handed to the dot_iter function.
        """
        import os
        from onyx.util import opentemp
        with opentemp('wb', suffix='.dot', prefix=temp_file_prefix) as (name, outfile):
            line_gen, dummy1, dummy2 = self.dot_iter(**kwargs)
            for line in line_gen:
                outfile.write(line)
        os.system(display_tool_format % (name,))
        return name


    # ==================== DECODING (FORWARD SEQUENCE INTERFACE) =================

    # Decoding state is mangaged by a decode_context.  This allows a given Hmm
    # be used for decoding at different parts of a decoding network.  Each place
    # in the network has its own instance of a decode_context to handle the
    # details of the likelihoods at that place in the network.

    def get_decode_context(self, user_data=None):
        return Hmm.decode_context(self, user_data)

    # context for use of an Hmm at a particular place in a decoding network
    class decode_context(object):
        def __init__(self, hmm, user_data=None):
            assert isinstance(hmm, Hmm)
            self._hmm = hmm
            self._user_data = user_data
            self._current_state = 'start'

            # Internal state is just one row vector with room for the virtual
            # states at each end and the actual states in the middle.  The
            # virtual states are used for the passing of likelihood scores
            # between Hmms in the decoding network.  The actual states hold the
            # conditional likelihoods for the Hmm at this place in the network
            shape = hmm.num_inputs + hmm.num_states + hmm.num_outputs ,
            state = self._state = zeros(shape, dtype=float)
            # views of the internal state
            self._state_newaxis = state[:,newaxis]
            self._input = state[:hmm.num_inputs]
            self._scores = state[hmm.num_inputs:-hmm.num_outputs]
            self._output = state[-hmm.num_outputs:]

            self._temp = numpy.empty((hmm.num_states,))

        @property
        def hmm(self):
            return self._hmm

        @property
        def user_data(self):
            return self._user_data

        def _update_state(self, new_state):
            # we'll get a KeyError if new_state isn't a legal state
            assert self._current_state in Hmm._LEGAL_STATE_DICT[new_state], str(self._current_state) + ' -> ' + str(new_state) + ' : ' + str(self.user_data)
            self._current_state = new_state

        def activate(self, scores):
            """
            Add mass to this sequence's inputs.
            """
            self._update_state('activate')
            if len(scores) != self.hmm.num_inputs:
                raise ValueError("expected %d activations scores, got %d" % (self.hmm.num_inputs, len(scores)))
            # XXX should require scores be a numpy array
            self._input += numpy.fromiter(scores, dtype=float)

        def pass_tokens(self):
            """
            Move tokens (scores) along transitions.
            """
            self._update_state('pass_tokens')
            # XXX HSW thinks this could be made less costly: one way would be to have a temp
            # to receive the multiply; but it seems like dot() should be our friend here....
            (self._state_newaxis * self.hmm._transition_matrix).sum(axis=0, out=self._state)            
            assert (self._input[:] == 0.0).all()

        def apply_likelihoods(self, observation, normalization=1):
            """
            Scale state likelihoods by the observation likelihood and multiply
            by the normalization.
            """
            self._update_state('apply_likelihoods')
            new_scores = self._temp
            assert len(observation) == self.hmm.dimension or self.hmm.dimension is None
            self.hmm._get_total_scores_for_observation(observation, new_scores)
            if normalization != 1:
                assert normalization >= 2
                new_scores *= normalization
            self._scores *= new_scores

        @property
        def scores(self):
            """
            The internal likelihood scores.
            """
            return tuple(self._scores)

        @property
        def max_score(self):
            """
            The best internal likelihood scores.
            """
            return self._scores.max() if len(self._scores) > 0 else 0

        @property
        def has_output(self):
            """
            True if the output is active, that is if it has any probability mass.
            """
            return (self._output != 0.0).any()

        @property
        def output(self):
            """
            The output scores for activation of successors.
            """
            return tuple(self._output)

        @property
        def outputX(self):
            """
            The output scores for activation of successors.
            """
            return self._output

        @property
        def max_output(self):
            """
            The best internal likelihood scores.
            """
            return self._output.max()
    
    def forward_score_sequence(self, sequence):
        """
        Score a sequence of datapoints.

        *sequence* should be a sequence of Numpy arrays.  The return is a Numpy
         array with shape (n, s) where n is the number of frames in the sequence
         and s is num_states + num_outputs.  This gives the score of each state
         on each frame, that is, the forward (or alpha) scores.  The total exit
         score for the HMM is thus ret[-1, num_states:].sum().
        """
        self.validateDataSequence(sequence)
        scores, _not_used = self._get_scores_for_adapting(sequence, terminal=True)
        alphas = self._get_alphas_for_sequence(scores)
        return alphas

    # ================================ ADAPTATION ================================


    def train(self, data):
        """
        Train a model on a set of data sequences

        *data* should be an iterable of data sequences; each sequence should
        contain Numpy arrays representing the datapoints.  Training is done in
        standalone mode.
        """
        self.begin_adapt("STANDALONE")
        self.clear_accumulators()
        for seq in data:
            self.adapt_one_sequence(seq)
        self.end_adapt()
        

    # Both modes use these functions to begin and end adaptation
    def begin_adapt(self, mode):
        """
        Initialize accumulators for this HMM if they haven't already been initialized.  Existing
        accumulators are not affected by this call.
        """
        if mode not in self._ADAPTATION_MODES:
            raise ValueError("Unknown adaptation mode: %s" % (mode,))
        elif self._adapt_mode is not None and mode != self._adapt_mode:
            raise ValueError("Change of adaptation mode not allowed; current mode is %s" % (self._adapt_mode,))
        
        self._gmm_mgr.ensure_accumulators(self._models)
        if self._trans_accums is None:
            assert self._trans_norms is None
            s = self.num_states + self.num_outputs
            self._trans_accums = self._make_zero_prob_array((s, s), dtype=float)
            self._trans_norms = self._make_zero_prob_array((s,), dtype=float)
            self._num_frames_accumulated = 0
        self._adapt_mode = mode

    def clear_accumulators(self):
        """
        Clear accumulators for this HMM.
        """
        self._num_frames_accumulated = 0
        self._trans_accums[:] = self._zero_prob
        self._trans_norms[:] = self._zero_prob

    def end_adapt(self):
        """
        Apply transition accumulators if they have anything in them.  This
        function will clear existing accumulators, so it may safely be called
        more than one time.
        """
        dc = dcheck("hmm_ea")
        if self._num_frames_accumulated == 0:
            return
        ni = self.num_inputs
        dc and dc("_trans_accums = \n%s" % self._trans_accums)
        dc and dc("_trans_norms = \n%s" % self._trans_norms)

        # Norm values for some output states may be 0, in which case the accumulators
        # will also be 0, so we use safely_divide_float_array here
        numerator = self._trans_accums
        denom = self._trans_norms[:, newaxis]
        dc and dc("denom = \n%s" % denom)
        self._transition_matrix[ni:,ni:] = self._from_working_domain(self._safe_div_prob_array(numerator, denom))
        dc and dc("_transition_matrix = \n%s" % self._transition_matrix)
        self.clear_accumulators()
        self._adapt_mode = None


    # STANDALONE ADAPTATION MODE
    def adapt_one_sequence(self, sequence):
        """
        Accumulate an Hmm in standalone mode on one sequence.  That is, assume this model is all
        there is; we're not feeding to or from another model, so we start with hardwired inputs
        on both forward and backward passes.
        """
        dc = dcheck("hmm_aos")
        if self._adapt_mode != "STANDALONE":
            raise RuntimeError("No previous call to begin_adapt or wrong adapt mode")
        self.validateDataSequence(sequence)
        scores, comp_scores = self._get_scores_for_adapting(sequence, terminal=True)
        dc and dc("Scores:\n%s" % (scores,))
        alphas = self._get_alphas_for_sequence(scores)
        dc and dc("Alphas:\n%s" % (alphas,))
        betas = self._get_betas_for_sequence(scores)
        dc and dc("Betas:\n%s" % (betas,))
        gammas = self._mult_prob(alphas, betas)
        dc and dc("Gammas (pre-norm):\n%s" % (gammas,))
        norm = self._sum_prob_array(gammas,axis=1)[:, newaxis]
        dc and dc("norm:\n%s" % (norm,))
        # Because the norm array can have 0s, we use self._safe_div_prob_array here.
        norm_gammas = self._safe_div_prob_array(gammas, norm)
        dc and dc("Normalized Gammas:\n%s" % (norm_gammas,))
        xis = self._get_xis_for_sequence( scores, alphas, betas, norm_gammas)
        dc and dc("Xis:\n%s" % (xis,))
        dc and dc("xis.sum(0):\n%s" % (self._sum_prob_array(xis, axis=0),))
        dc and dc("norm_gammas.sum(0):\n%s" % (self._sum_prob_array(norm_gammas, axis=0),))
        self._trans_accums = self._add_prob(self._trans_accums, self._sum_prob_array(xis, axis=0))
        self._trans_norms = self._add_prob(self._trans_norms, self._sum_prob_array(norm_gammas, axis=0))
        dc and dc("_trans_accums = \n%s" % self._trans_accums)
        dc and dc("_trans_norms = \n%s" % self._trans_norms)
        self._num_frames_accumulated += len(sequence)
        for i, m in enumerate(self._models):
            self._gmm_mgr.accum_sequence(m, self._from_working_domain(norm_gammas[:,i]), comp_scores[i], sequence)



    # NETWORK ADAPTATION MODE

    # XXX Rewrite this to deal correctly with contexts!!

    # Network adaptation supports BW adaptation of a collection of Hmms connected
    # together as a lattice (see bwtrainer.py for an example).  The sequence of operations is as
    # follows:
    #  0) Make a pass over the network (any order is fine), calling begin_adapt() on every Hmm
    #  1) For each sequence:
    #    2) Make a pass over the network (any order is fine), calling init_for_forward_pass() on every
    #       Hmm - this is where we're going to initialize the alpha storage.  We also do
    #       scoring here, too.  Each model can be initialized as left-edge and/or right edge,
    #       indicating that the model has, respectively, a position in the lattice at the
    #       beginning or the end.
    #    3) Set the input alphas for the entry models to 1
    #    4) Do num_frame + 1 times:
    #         For each model (now iterating in forward topological order!):
    #           Call process_one_frame_forward, grabbing the output alpha values (note that this
    #           clears the alpha input accumulators)
    #           Accum the output values into the input alphas of all successor models
    #    5) Make a pass over the network (any order is fine), calling init_for_backward_pass() on
    #       every Hmm - this is where we initialize beta storage.  We don't do scoring here, though.
    #    6) Set the input betas for the exit models to 1
    #    7) Do num_frame + 1 times:
    #         For each model (now iterating in *backward* topological order!):
    #           Call process_one_frame_backward, grabbing the output beta values (note that this
    #           clears the input beta accumulators
    #           Accum the output values into the input betas of all predecessor models
    #    ... At this point, each model will have a complete set of alpha and beta values for
    #    each of its states for every frame...
    #    8) Call get_initial_gamma_sum(), then make a pass over the network (any
    #    order is fine), calling add_to_gamma_sum() on every Hmm and
    #    accumulating the result - this will be a framewise array of
    #    normalization values.  9) Make a pass over the network (any order is
    #    fine), calling do_accumulation() on every Hmm, passing in the global
    #    gamma normalization vector accumulated in step 7.
    #  10) Make a pass over the network (any order is fine), calling end_adapt() on every Hmm

    
    # XXX We probably want a state machine to enforce correct sequencing of operations here.

    class HmmTrainingContext(object):
        def __init__(self, terminal, seq_len, alphas, input_alphas, betas, input_betas, scores):
            self.terminal = terminal
            self.current_sequence_len = seq_len
            self.alphas = alphas
            self.input_alphas = input_alphas
            self.betas = betas
            self.input_betas = input_betas
            self.scores = scores
            self.current_frame_index = 0

    def init_for_forward_pass(self, sequence, terminal = False):
        """
        Initialize this model for a forward BW pass on the sequence.  Note that scoring is done in
        this call, but not in init_for_backward_pass, so be sure to do forward passes before
        backward passes.
        """
        dc = dcheck("hmm_iffp")
        self._verify_network_mode()

        self._current_sequence = seq = tuple(sequence)
        scores, self._comp_scores = self._get_scores_for_adapting(sequence, terminal)
        dc and dc("Scores = \n%s" % self._scores)

        n = len(seq) + 1
        s = self.num_states + self.num_outputs
        alphas = self._make_zero_prob_array((n,s), dtype = float)
        input_alphas = self._make_zero_prob_array((self.num_inputs,), dtype = float)
        betas = self._make_zero_prob_array((n,s), dtype = float)
        input_betas = self._make_zero_prob_array((self.num_outputs,), dtype = float)
        self._current_sequence_len = len(seq)
        ret = Hmm.HmmTrainingContext(terminal, self._current_sequence_len, alphas,
                                     input_alphas, betas, input_betas, scores)
        return ret

    def init_for_backward_pass(self, context):
        """
        Initialize this model for a backward BW pass on the sequence.  Note that scoring is *not*
        done in this call, only in init_for_forward_pass, so be sure to do forward passes before
        backward passes.
        """
        self._verify_network_mode()
        context.current_frame_index = self._current_sequence_len


    def accum_input_alphas(self, context, alphas, inter_model_trans_probs=None):
        """
        Add to the incoming alpha values for this model.  alphas must be a Numpy
        array of length self.num_inputs and must be in the working domain.
        Usually it will be the output of some previous call to
        process_one_frame_forward.
        """
        dc = dcheck("hmm_aia")
        self._verify_network_mode()
        if alphas is None:
            alphas = numpy.array(tuple(repeat(self._div_prob(self._one_prob, self.num_inputs), self.num_inputs)))
        elif len(alphas) != self.num_inputs:
            raise ValueError("expected %d alphas, got %d" % (self.num_inputs, len(alphas)))

        if inter_model_trans_probs is not None:
            to_accum = self._mult_prob(alphas, self._to_working_domain(inter_model_trans_probs))
        else:
            to_accum = alphas
        dc and dc("context.alphas[ci] = %s" % (context.alphas[ci],))
        context.input_alphas[:] = self._add_prob(context.input_alphas[:], to_accum[:])


    def process_one_frame_forward(self, context):
        """
        Do one frame of forward processing and return the output alpha values.  Resets the input
        alpha values in context to 0 for the next frame.  This function should be called one time
        more than the length of sequence passed to init_for_forward_pass; the return from that final
        call will be None.
        """
        dc = dcheck("hmm_poff")
        self._verify_network_mode()
        ci = context.current_frame_index
        scores = context.scores[ci]
        prev_alphas = context.alphas[0] if ci == 0 else context.alphas[ci-1]
        dc and dc("prev_alphas = %s" % (prev_alphas,))
        context.alphas[ci] = self._get_alphas_for_frame(prev_alphas, context.input_alphas, scores)
        dc and dc("context.alphas[ci] = %s" % (context.alphas[ci],))
        context.current_frame_index += 1
        context.input_alphas[:] = self._zero_prob
        if context.current_frame_index > context.current_sequence_len:
            assert context.current_frame_index == context.current_sequence_len + 1
            return None
        else:
            output = array(context.alphas[ci, -self.num_outputs:], copy=True)
            return output


    def accum_input_betas(self, context, betas, inter_model_trans_probs=None):
        """
        Add to the incoming beta values for this model.  betas must be a Numpy array of
        length self.num_outputs and it must be in the working domain.
        Usually it will be the output of some previous call to
        process_one_frame_backward.
        """
        self._verify_network_mode()
        if betas is None:
            betas = numpy.array(tuple(repeat(self._one_prob, self.num_outputs)))
        elif len(betas) != self.num_outputs:
            raise ValueError("expected %d betas, got %d" % (self.num_outputs, len(betas)))
        if inter_model_trans_probs is not None:
            to_accum = self._mult_prob(betas, self._to_working_domain(inter_model_trans_probs))
        else:
            to_accum = betas
        context.input_betas[:] = self._add_prob(context.input_betas[:], to_accum[:])


    def process_one_frame_backwardXXX(self, context):
        """
        Do one frame of backward processing and return the output beta values.  Resets the input
        beta values in context to 0 for the next frame.  This function should be called one time
        more than the length of sequence passed to init_for_forward_pass; the return from the first
        call will be None.
        """
        dc = dcheck("hmm_pofb")
        self._verify_network_mode()
        ci = context.current_frame_index
        prev_betas, input_betas, scores = ((None, None, None) if ci ==  context.current_sequence_len else
                                           (context.betas[ci+1], context.input_betas, context.scores[ci+1]))
        dc and dc("prev_betas = %s" % (prev_betas,))
        context.betas[ci], output = self._get_betas_for_frame(prev_betas, input_betas, scores)
        dc and dc("context.betas[ci] = %s" % (context.betas[ci],))
        context.input_betas[:] = 0
        context.current_frame_index -= 1

        dc and dc("END OF FRAME %d\n" % (ci))
        if ci == context.current_sequence_len:
            return None
        else:
            return output



    def process_one_frame_backward(self, context):
        """
        Do one frame of backward processing and return the output beta values.  Resets the input
        beta values in context to 0 for the next frame.  This function should be called one time
        more than the length of sequence passed to init_for_forward_pass; the return from the first
        call will be None.
        """
        dc = dcheck("hmm_pofb")
        self._verify_network_mode()
        ci = context.current_frame_index
        prev_betas, input_betas, scores = ((None, None, None) if ci ==  context.current_sequence_len else
                                           (context.betas[ci+1], context.input_betas, context.scores[ci+1]))
        dc and dc("prev_betas = %s" % (prev_betas,))
        context.betas[ci], prod = self._get_betas_for_frame(prev_betas, input_betas, scores)
        dc and dc("context.betas[ci] = %s" % (context.betas[ci],))
        context.input_betas[:] = 0
        context.current_frame_index -= 1
        no = self.num_outputs
        ns = self.num_states
        ni = self.num_inputs
        a = self._working_trans_mat_swapped
        dc and dc("END OF FRAME %d\n" % (ci))
        if ci == context.current_sequence_len:
            return None

        temp_betas = self._make_zero_prob_array((ni + ns + no), dtype = float)
        temp_betas[:-no] = prod[:-no]
            
        output = self._sum_prob_array(self._mult_prob(a, temp_betas[:,newaxis]), axis=0)[:ni]
        return output



    class TotalGammaSum(object):
        def __init__(self, initializer):
            self.value = initializer

    def get_initial_gamma_sum(self):
        return self.TotalGammaSum(self._make_zero_prob_array((self._current_sequence_len + 1,)))

    def add_to_gamma_sum(self, gamma_sum, context):
        """
        Get the sum of gammas over all states in this context for each frame and
        add it to the gamma_sum provided.
        """
        dc = dcheck("hmm_atgs")
        self._verify_network_mode()
        if not isinstance(gamma_sum, self.TotalGammaSum):
            raise ValueError('gamma_sum must be of type TotalGammaSum, use get_initial_gamma_sum()')
        gammas = self._mult_prob(context.alphas, context.betas)
        dc and dc("Scores:\n%s" % (context.scores,))
        dc and dc("Alphas:\n%s" % (context.alphas,))
        dc and dc("Betas:\n%s" % (context.betas,))
        # There may be non-zero gamma values on virtual outputs of non-terminal
        # states, but we don't want to include them in normalizing
        ns = self.num_states
        if not context.terminal:
            gammas[:, ns:] = self._zero_prob
        dc and dc("Gammas (pre-norm):\n%s" % (gammas,))
        norm = self._sum_prob_array(gammas, axis=1)
        dc and dc("Gammas (post-norm):\n%s" % (norm,))
        gamma_sum.value = self._add_prob(gamma_sum.value, norm)
        dc and dc("gamma_sum (after summing) :\n%s" % (gamma_sum.value,))


    def _get_output_gammas_and_betas(self, context, total_gamma_sum):
        dc = dcheck("hmm_gogab")
        self._verify_network_mode()
        ni = self.num_inputs
        no = self.num_outputs
        ns = self.num_states
        
        gammas = self._div_prob(self._mult_prob(context.alphas, context.betas), total_gamma_sum.value[:,newaxis])
        # Now push gammas on real states through transition probs to outputs 
        a = self._working_transition_matrix[ni:ni+ns,-no:]
        dc and dc("ni = %d, ns = %d, no = %d" % (ni, ns, no))
        dc and dc("a.shape = %s" % (a.shape,))
        dc and dc("gammas.shape = %s" % (gammas.shape,))
        dc and dc("gammas[:,:ns].shape = %s" % (gammas[:,:ns].shape,))
        out_gammas = self._dot_prob_array(gammas[:,:ns], a)
        # Now push betas on real states through transition probs to outputs 
        out_betas = self._dot_prob_array(context.betas[:,:ns], a)
        return out_gammas, out_betas

    def _get_input_betas(self, context):
        self._verify_network_mode()
        ni = self.num_inputs
        # Now push betas on real states through transition probs to inputs
        # this is a bit trickier than the previous cases because we are
        # pushing backward.
        temp = self._mult_prob(context.betas, context.scores)
        a = self._working_transition_matrix[:ni,ni:].swapaxes(0,1)
        result = self._dot_prob_array(temp, a)
        return result

    def get_accum_values(self, total_gamma_sum, pred_context, succ_hmm, succ_context):
        dc = dcheck("hmm_gav")
        self._verify_network_mode()

        if not isinstance(total_gamma_sum, self.TotalGammaSum):
            raise ValueError('gamma_sum must be of type TotalGammaSum, use get_initial_gamma_sum()')
        gammas, betas_out = self._get_output_gammas_and_betas(pred_context, total_gamma_sum)
        # Note that we're calling this on some *other* Hmm, which the caller has
        # provided.  This avoids having these values go through the caller, who
        # shouldn't have to care in which domain we're working.
        betas_in = succ_hmm._get_input_betas(succ_context)

        (num_frames, n) = gammas.shape
        dc and dc("Gammas = \n%s" % (gammas))
        dc and dc("Betas_out = \n%s" % (betas_out))
        dc and dc("Betas_in = \n%s" % (betas_in))
        assert gammas.shape == betas_out.shape == betas_in.shape
        assert n == self.num_outputs == succ_hmm.num_inputs

        ret = numpy.zeros((num_frames, n), dtype=float)
        for t in xrange(num_frames-1):
            numerator = gammas[t] * betas_in[t+1]
            denom = betas_out[t]
            x = safely_divide_float_array(numerator, denom)
            ret[t,:] = x
            dc and dc("For frame %d, gammas[t] = %s, betas_in[t+1] = %s, betas_out[t] = %s, x = %s" %
                      (t, gammas[t], betas_in[t+1], betas_out[t], x))
        return ret


    def do_accumulation(self, context, total_gamma_sum):
        """
        Accumlate the results of processing the current sequence.
        """
        dc = dcheck("hmm_da")
        self._verify_network_mode()
        if not isinstance(total_gamma_sum, self.TotalGammaSum):
            raise ValueError('gamma_sum must be of type TotalGammaSum, use get_initial_gamma_sum()')
        dc and dc("total_gamma_sum:\n%s" % (total_gamma_sum.value,))

        prod = self._mult_prob(context.alphas, context.betas)
        # See add_to_gamma_sum() - we have to do this before the division since the denominator has
        # already had this treatment.
        ns = self.num_states
        if not context.terminal:
            prod[:, ns:] = self._zero_prob
        dc and dc("prod:\n%s" % (prod,))
        denom = total_gamma_sum.value[:,newaxis]
        norm_gammas = self._safe_div_prob_array(prod, denom)
        dc and dc("Norm_gammas:\n%s" % (norm_gammas,))
        xis = self._get_xis_for_sequence(context.scores, context.alphas, context.betas, norm_gammas)
        dc and dc("Xis:\n%s" % (xis,))
        dc and dc("xis.sum(0):\n%s" % (self._sum_prob_array(xis, axis=0),))
        dc and dc("norm_gammas.sum(0):\n%s" % (self._sum_prob_array(norm_gammas, axis=0),))
        self._trans_accums = self._add_prob(self._trans_accums, self._sum_prob_array(xis, axis=0))
        self._trans_norms = self._add_prob(self._trans_norms, self._sum_prob_array(norm_gammas, axis=0))

        self._num_frames_accumulated += context.current_sequence_len
        for i, m in enumerate(self._models):
            self._gmm_mgr.accum_sequence(m, self._from_working_domain(norm_gammas[:,i]),
                                         self._comp_scores[i], self._current_sequence)
        

    # ================================ INTERNAL FUNCTIONS ================================

    def _set_domain_specific_functions(self):
        # This function sets up a collection of functions and constants that let
        # us use either an ordinary probability domain or a logprob domain in
        # our computations.
        if self.log_domain:
            self._zero_prob = mathutils.quiet_log(0.0)
            self._one_prob = 0.0
            self._make_zero_prob_array = mathutils.log_zeros
            self._add_prob = mathutils.logsumexp
            self._mult_prob = numpy.add
            self._div_prob = mathutils.safe_log_divide
            self._sum_prob_array = mathutils.logsumexp_array
            self._dot_prob_array = mathutils.log_dot
            self._safe_div_prob_array = mathutils.safe_log_divide
            self._to_working_domain = mathutils.quiet_log
            self._from_working_domain = numpy.exp
        else:
            self._zero_prob = 0.0
            self._one_prob = 1.0
            self._make_zero_prob_array = numpy.zeros
            self._add_prob = numpy.add
            self._mult_prob = numpy.multiply
            self._div_prob = numpy.divide
            self._sum_prob_array = numpy.sum
            self._dot_prob_array = numpy.dot
            self._safe_div_prob_array = safely_divide_float_array
            self._to_working_domain = lambda (x) : x
            self._from_working_domain = lambda (x) : x

    def _get_scores_for_frame(self, frame, end_one = True):
        # comp_scores is a Python list of Numpy arrays.  The arrays will be of
        # different lengths in case the underlying models have different numbers
        # of components, which is why we don't use a 2-d array.
        get_gmm_model = self._gmm_mgr.get_model
        comp_scores = list(get_gmm_model(mi).score_components(frame) for mi in self._models)
        # Adding 1 as the score of the exit states is a convenience sometimes;
        # it allows us to treat the alpha value of the exit states as the exit
        # likelihoods of the model.  We add output scores to the component
        # scores so we can use them in the summation below.
        comp_scores.extend(list(repeat(array((1.0 if end_one else 0.0,)), self.num_outputs)))
        right_len = self.num_states + self.num_outputs
        assert len(comp_scores) == right_len
        # Sum component scores to get scores for each real and output state
        scores = numpy.fromiter((cs.sum() for cs in comp_scores), dtype = float)
        assert scores.shape == (right_len,)
        # Note that we leave comp_scores in the probablility domain, since these
        # are only used later as inputs to the GmmMgr, which expects them in
        # that domain always.
        return self._to_working_domain(scores), comp_scores

    def _get_total_scores_for_observation(self, observation, out):
        get_gmm_model = self._gmm_mgr.get_model
        # need a better way to overwrite from an iterator
        out[:] = tuple(get_gmm_model(mi).score(observation) for mi in self._models)

    def _get_scores_for_adapting(self, sequence, terminal):
        n = len(sequence)
        s = self.num_states + self.num_outputs
        scores = self._make_zero_prob_array((n+1, s), dtype = float)
        comp_scores = [list() for i in xrange(self.num_states)]
        for t, obs in enumerate(sequence):
            scores[t], cs = self._get_scores_for_frame(obs, end_one = not terminal)
            for i in xrange(self.num_states):
                comp_scores[i].append(cs[i])
        if terminal:
            scores[n, self.num_states:s] = self._one_prob
        assert len(comp_scores) == self.num_states
        for cs in comp_scores:
            assert len(cs) == n
        return scores, comp_scores

    def _get_alphas_for_sequence(self, scores):
        n, s = scores.shape
        assert s == self.num_states + self.num_outputs
        alpha = self._make_zero_prob_array((n, s), dtype = float)
        # Note that we always divide the initial mass evenly across all
        # virtual inputs here
        input_alphas = array(tuple(repeat(self._div_prob(self._one_prob, self.num_inputs), self.num_inputs)))
        # We use alpha[0] as input on the first frame since we just need some
        # array of the right length filled with zeros.
        alpha[0] = self._get_alphas_for_frame(alpha[0], input_alphas, scores[0])
        
        # No more input mass after the first frame
        input_alphas[:] = self._zero_prob
        for t in xrange(1,n):
            alpha[t] = self._get_alphas_for_frame(alpha[t-1], input_alphas, scores[t])
        return alpha

    # Get a new vector of alphas from a current vector and the scores for one frame.  current_alphas
    # should be an appropriate-length vector of 0s for the first frame.
    def _get_alphas_for_frame(self, current_alphas, input_alphas, scores):
        dc = dcheck("hmm_gaff")

        ni = self.num_inputs
        ns = self.num_states
        no = self.num_outputs

        # Short-circuit if all alphas are 0
        if (current_alphas == self._zero_prob).all() and (input_alphas == self._zero_prob).all():
            return self._make_zero_prob_array((ns+no), dtype = float)

        assert len(scores) == ns + no
        a = self._working_transition_matrix[:,ni:]

        alphas = self._make_zero_prob_array((ni+ns+no), dtype = float)
        alphas[:ni] = input_alphas
        alphas[ni:] = current_alphas

        prod = self._mult_prob(alphas[:,newaxis], a)
        prod_sum = self._sum_prob_array(prod, axis=0)
        new_alphas = self._mult_prob(prod_sum, scores)
        dc and (dc("ca = %s, scores = %s" % (current_alphas, scores)) and
                dc("prod = %s" % (prod,)) and
                dc("prod_sum = %s" % (prod_sum,)) and
                dc("new_alphas = %s" % (new_alphas,)))
        return new_alphas

    def _get_betas_for_sequence(self, scores):
        n, s = scores.shape
        assert s == self.num_states + self.num_outputs
        input_betas = self._make_zero_prob_array((self.num_outputs,), dtype = float)
        beta = self._make_zero_prob_array((n, s), dtype = float)
        beta[-1], _ = self._get_betas_for_frame(None, None, None)
        for t in xrange(n-2, -1, -1):
            beta[t], _ = self._get_betas_for_frame(beta[t+1], input_betas, scores[t+1])
        return beta

    # Get a new vector of betas from a current vector and the scores for one frame.
    def _get_betas_for_frame(self, current_betas, input_betas, scores):
        dc = dcheck("hmm_gbff")
        dc and dc("cb = %s, scores = %s" % (current_betas, scores))
        no = self.num_outputs
        ni = self.num_inputs
        ns = self.num_states
        assert current_betas is None or len(current_betas) == ns + no
        if current_betas is None:
            assert scores is None
            assert input_betas is None
            new_betas = numpy.ones((ns + no,), dtype = float)
            temp = None
        else:
            assert len(scores) == ns + no
            len(input_betas) == no
            current_betas[-no:] = self._add_prob(current_betas[-no:], input_betas)
            temp = self._make_zero_prob_array((ni+ns+no), dtype = float)
            temp[ni:] = self._mult_prob(scores, current_betas)
            temp2 = self._sum_prob_array(self._mult_prob(self._working_trans_mat_swapped,
                                                             temp[:, newaxis]), axis=0)
            new_betas = temp2[ni:]
            output = temp2[:ni]
            assert (temp[:ni] == self._zero_prob).all()
            
        assert new_betas.shape == (ns + no,)
        dc and dc("new_betas = %s" % (new_betas[ni:],))
        return new_betas, temp

    def _get_xis_for_sequence(self, scores, alphas, betas, gammas):
        dc = dcheck("hmm_gxfs")
        assert alphas.shape == betas.shape == gammas.shape == scores.shape
        n, s = gammas.shape
        assert s == self.num_states + self.num_outputs
        ni = self.num_inputs
        a = self._working_transition_matrix[ni:,ni:]
        assert a.shape == (s, s)
        xis = self._make_zero_prob_array((n, s, s), dtype = float)

        dc and dc("a[:,:] = \n%s" % (a[:,:],))
        for t in xrange(0,n-1):
            # All operations in this loop are done in i,j space.
            # For all gamma, beta, and scores, which are in have dimensions
            # of (time, state), we get, e.g., beta_j(t) by taking
            # betas[t, newaxis, :] and beta_i(t) by taking betas[t, :, newaxis]
            # The key is the position of the newaxis token.
            dc and dc(DebugPrint.NEWLINE_PREFIX, "Frame %d:" % (t,))
            dc and dc("gammas[t,:,newaxis] = \n%s\nscores[t+1,newaxis,:] = \n%s\nbetas[t+1,newaxis,:] = \n%s" %
                   (gammas[t,:,newaxis], scores[t+1,newaxis,:], betas[t+1,newaxis,:]))
            prod0 = self._mult_prob(gammas[t,:,newaxis], a[:,:])
            prod1 = self._mult_prob(scores[t+1,newaxis,:], betas[t+1,newaxis,:])
            dc and dc("gammas[t,:,newaxis] * a[:,:] = \n%s\nscores[t+1,newaxis,:] * betas[t+1,newaxis,:] = \n%s" %
                   (prod0, prod1))
            prod2 = self._mult_prob(prod0, prod1)
            dc and dc("prod2 = \n%s" % (prod2,))
            denom = betas[t, :, newaxis]
            dc and dc("denom = %s" % (denom,))
            # Because the beta array can have 0s, we use self._safe_div_prob_array here.
            xis[t] = self._safe_div_prob_array(prod2, denom)
            dc and dc("xis[%d] = \n%s" % (t, xis[t]))
                
        return xis


    def _verify_network_mode(self):
        if self._adapt_mode is None:
            raise RuntimeError("No previous call to begin_adapt")
        elif self._adapt_mode != "NETWORK":
            raise RuntimeError("Wrong adapt mode")
        
# ================================ TEST FUNCTIONS AND HELPERS ================================

def toy_probs(count):
    """
    Create a toy distribution of out-arc weights given a count
    of out arcs.  Make forward-one arc get the most weight.
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
            

def test1(num_obs, num_passes):
    num_states = 3
    dimension = 2
    
    # Data generator setup
    target_means = ((1,1), (2,2), (3,3))
    target_vars = ((0.1,0.1), (0.2,0.2), (0.3,0.3))
    target_durations = (2, 3, 6)
    num_steps = sum(target_durations)
    generators = [SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE) for i in xrange(num_states)]
    [m.set_model(tm, tv) for (m, tm, tv) in izip(generators, target_means, target_vars)]

    SimpleGaussianModel.seed(0)
    GaussianMixtureModel.seed(0)
    # Model setup
    hmm0 = Hmm(num_states)
    # Build mixtures with 2 full-covariance gaussians in each state
    num_comps = 2
    gmms = [GaussianMixtureModel(dimension, GaussianMixtureModel.FULL_COVARIANCE, num_comps) for i in xrange(num_states)]
    mm = GmmMgr(gmms)
    models = range(num_states)
    trans = zeros((num_states + 2, num_states + 2), dtype=float)
    trans[0,1] = 1.0
    for i in xrange(1, num_states+1):
        trans[i,i] = 0.5
        trans[i,i+1] = 0.5
    
    hmm0.build_model(mm, models, 1, 1, trans)

    durs = hmm0.find_expected_durations(num_steps)
    print ("After initialization, expected durations for %d steps are %s (sum = %s), targets are %s"
           % (num_steps, durs, durs.sum(), target_durations))
    print hmm0.to_string(True)


    # Try some adaptation
    for p in xrange(num_passes):
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        obs_gen = obs_generator(generators, target_durations)
        for i in xrange(num_obs):
            obs = obs_gen.next()
            hmm0.adapt_one_sequence(obs)
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")

    print hmm0.to_string(True)
    
    durs = hmm0.find_expected_durations(num_steps)
    print ("After training, expected durations for %d steps are %s (sum = %s), targets were %s"
           % (num_steps, durs, durs.sum(), target_durations))



def obs_generator(generators, target_durations):
    assert len(generators) == len(target_durations)
    while True:
        yield [m.sample() for m, d in izip(generators, target_durations) for i in xrange(d)]



def test2():
    print "================ TEST 2 ================"
    num_states = 5
    dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4), DummyModel(2, 0.4) )
    mm = GmmMgr(dummies)
    models = range(num_states)
    hmm2 = Hmm(num_states)
    order = 4
    prob = 1.0 / order
    hmm2.build_forward_model_compact(mm, models, order, ((prob,) * num_states,) * order)
    print hmm2.to_string(full=True)

    mm.set_adaptation_state("INITIALIZING")
    mm.clear_all_accumulators()
    hmm2.begin_adapt("NETWORK")
    mm.set_adaptation_state("ACCUMULATING")
    num_obs = 12
    obs = [array((0,0))] * num_obs
    context = hmm2.init_for_forward_pass(obs, terminal = True)
                               
    hmm2.accum_input_alphas(context, array([1.0] * hmm2.num_inputs))
    for frame in xrange(num_obs + 1):
        alpha_outs = hmm2.process_one_frame_forward(context)
        # print alpha_outs
    hmm2.init_for_backward_pass(context)
    hmm2.accum_input_betas(context, array([1.0] * hmm2.num_outputs))
    for frame in xrange(num_obs + 1):
        beta_outs = hmm2.process_one_frame_backward(context)
        # print beta_outs
        
    norm = hmm2.get_initial_gamma_sum()
    hmm2.add_to_gamma_sum(norm, context)
    hmm2.do_accumulation(context, norm)
    mm.set_adaptation_state("APPLYING")
    hmm2.end_adapt()
    mm.apply_all_accumulators()
    mm.set_adaptation_state("NOT_ADAPTING")
    print hmm2.to_string(full=True)



def test3():
    # Same as test2 but using standalone interface
    print "================ TEST 3 ================"
    num_states = 5
    dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4), DummyModel(2, 0.4) )
    mm = GmmMgr(dummies)
    models = range(num_states)
    hmm2 = Hmm(num_states)
    order = 4
    prob = 1.0 / order
    hmm2.build_forward_model_compact(mm, models, order, ((prob,) * num_states,) * order)
    print hmm2.to_string(full=True)

    mm.set_adaptation_state("INITIALIZING")
    mm.clear_all_accumulators()
    hmm2.begin_adapt("STANDALONE")
    mm.set_adaptation_state("ACCUMULATING")
    obs = [array((0,0))]*12
    hmm2.adapt_one_sequence(obs)
    mm.set_adaptation_state("APPLYING")
    hmm2.end_adapt()
    mm.apply_all_accumulators()
    mm.set_adaptation_state("NOT_ADAPTING")
    print hmm2.to_string(full=True)


def logreftest():
    num_obs = 100
    num_passes = 5
    test1(num_obs, num_passes)
    test2()
    test3()


def temp():
    import sys
    with DebugPrint("hmm_gxfs"):
        test2()
        # test3()
    sys.exit()

    
if __name__ == '__main__':
    
    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    args = argv[1:]
    if '--logreftest' in args:
        logreftest()
