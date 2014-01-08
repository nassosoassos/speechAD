###########################################################################
#
# File:         testbwt.py
# Date:         Tue 5 Aug 2008 15:21
# Author:       Ken Basye
# Description:  Some test drivers for the TrainingGraph class
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
>>> True
True
"""

from __future__ import with_statement
import numpy
from numpy import array, eye
from onyx.am.gaussian import GaussianMixtureModel, SimpleGaussianModel, DummyModel
from onyx.am.hmm import Hmm
from onyx.util.debugprint import DebugPrint
from onyx.am.modelmgr import GmmMgr
from onyx.am.hmm_mgr import HmmMgr
from onyx.am.bwtrainer import TrainingGraph
from onyx.graph.graphtools import FrozenGraph, GraphBuilder
from random import seed, random, randint
from itertools import izip, imap, repeat
from onyx.am.hmm import toy_probs


def train_hmm(hmm, obs_list, num_passes):
    gmm_mgr = hmm.gmm_manager
    for p in xrange(num_passes):
        gmm_mgr.set_adaptation_state("INITIALIZING")
        gmm_mgr.clear_all_accumulators()
        hmm.begin_adapt("STANDALONE")
        gmm_mgr.set_adaptation_state("ACCUMULATING")

        for obs in obs_list:
            hmm.adapt_one_sequence(obs)
        
        gmm_mgr.set_adaptation_state("APPLYING")
        hmm.end_adapt()
        gmm_mgr.apply_all_accumulators()
        gmm_mgr.set_adaptation_state("NOT_ADAPTING")


def validate_training_graph(tg, gmm_mgr, hmm_mgr, seq_list, num_passes, alt_gmm_mgr=None):
    ret = ""
    result_hmm = tg.convert_to_standalone_hmm(alt_gmm_mgr)
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    train_hmm(result_hmm, seq_list, 1)

    # Now adapt original TrainingGraph
    for i in xrange(num_passes):
        gmm_mgr.set_adaptation_state("INITIALIZING")
        hmm_mgr.set_adaptation_state("INITIALIZING")
        gmm_mgr.clear_all_accumulators()
        tg.begin_training()
        gmm_mgr.set_adaptation_state("ACCUMULATING")
        hmm_mgr.set_adaptation_state("ACCUMULATING")
        for seq in seq_list:
            tg.train_one_sequence(seq)
        tg.end_training()
        gmm_mgr.set_adaptation_state("APPLYING")
        hmm_mgr.set_adaptation_state("APPLYING")
        gmm_mgr.apply_all_accumulators()
        hmm_mgr.apply_all_accumulators()
        gmm_mgr.set_adaptation_state("NOT_ADAPTING")
        hmm_mgr.set_adaptation_state("NOT_ADAPTING")

    # Now convert TG to Hmm again and see how they line up
    result_hmm2 = tg.convert_to_standalone_hmm()

    model1str = result_hmm.to_string(full=True)
    model2str = result_hmm2.to_string(full=True)
    ret += "\n\n========= CONVERTED THEN ADAPTED AS Hmm =========\n\n" + model1str
    ret += "\n\n========= ADAPTED AS TG THEN CONVERTED =========\n\n" + model2str
    valid = (model1str == model2str)
    ret += "\n model1str == model2str is: %s" % (valid,)
    valid2 = (result_hmm == result_hmm2)
    ret += "\n result_hmm == result_hmm2 is: %s" % (valid2,)
    return valid, ret

def make_forward_hmm(gmm_mgr, num_states, order, models=None, exact=False):
    hmm0 = Hmm(num_states)
    # generate a set of random indices from the GmmMgr
    models = tuple(randint(0, gmm_mgr.num_models-1) for i in xrange(num_states)) if models is None else models
    trans = tuple([p] * num_states for p in toy_probs(order))
    if exact:
        hmm0.build_forward_model_exact(gmm_mgr, models, order, trans)
    else:
        hmm0.build_forward_model_compact(gmm_mgr, models, order, trans)

    return hmm0

def obs_generator(generators, target_durations):
    while True:
        yield [m.sample() for m, d in izip(generators, target_durations) for i in xrange(d)]


def make_data_generator(dimension):
    # Data generator setup and data generation
    target_means = ((1,1), (2,2), (3,3))
    target_vars = ((0.1,0.1), (0.2,0.2), (0.3,0.3))
    target_durations = (2, 6, 6)
    assert len(target_means) == len(target_vars) == len(target_durations)
    num_generators = len(target_means)
    generators = [SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE) for i in xrange(num_generators)]
    for (m, tm, tv) in izip(generators, target_means, target_vars):
        m.set_model(tm, tv) 
    SimpleGaussianModel.seed(0)
    def obs_generator():
        while True:
            yield [m.sample() for m, d in izip(generators, target_durations) for i in xrange(d)]
    return obs_generator()


def make_standard_gmms(dimension, num_models):
    models = []
    for i in xrange(num_models):
        gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.DIAGONAL_COVARIANCE, 1)
        gmm.set_weights(array((1.0,)))
        mu = array(((0.0,0.0),))
        v = array(((1.0,1.0),))
        gmm.set_model(mu, v)
        models.append(gmm)
    return models

def test0(num_obs):
    # 1 node contains a 0-node order-2 Hmm (i.e. an epsilon node in the training graph)
    dimension = 2
    obs_gen = make_data_generator(dimension)
    obs_list = [obs_gen.next() for i in xrange(num_obs)]

    # GmmMgr setup
    num_models = 20
    models = make_standard_gmms(dimension, num_models)
    gmm_mgr1 = GmmMgr(models[0:10])
    gmm_mgr2 = GmmMgr(models[10:20])

    # Hmm setup
    # Make an Hmm with no states and order 2
    num_states = 0
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr1, num_states, 2, exact=True) 
    hmm_mgr = HmmMgr((hmm0,))

    # TrainingGraph setup
    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    gr0 = FrozenGraph(gb)
    tg0 = TrainingGraph(gr0, hmm_mgr, dict())

    valid, ret = validate_training_graph(tg0, gmm_mgr1, hmm_mgr, obs_list, 1, gmm_mgr2)
    return ret


def test1(num_obs):
    # 1 node contains a 4-node order-2 Hmm
    dimension = 2
    obs_gen = make_data_generator(dimension)
    obs_list = [obs_gen.next() for i in xrange(num_obs)]

    # GmmMgr setup
    num_models = 20
    models = make_standard_gmms(dimension, num_models)
    gmm_mgr1 = GmmMgr(models[0:10])
    gmm_mgr2 = GmmMgr(models[10:20])

    # Hmm setup
    # Make one Hmm with 4 states and order 2 (self loop, forward 1)
    num_states = 4
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr1, num_states, 2, exact=True) 
    hmm_mgr = HmmMgr((hmm0,))

    # TrainingGraph setup
    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    gr0 = FrozenGraph(gb)
    tg0 = TrainingGraph(gr0, hmm_mgr, dict())

    valid, ret = validate_training_graph(tg0, gmm_mgr1, hmm_mgr, obs_list, 1, gmm_mgr2)
    return ret


def test2(num_obs):
    # Each of the 2 nodes contains a 4-node order-2 Hmm; the nodes are connected in single chain
    dimension = 2

    obs_gen = make_data_generator(dimension)
    obs_list = [obs_gen.next() for i in xrange(num_obs)]

    # GmmMgr setup
    num_models = 20
    models = make_standard_gmms(dimension, num_models)
    gmm_mgr1 = GmmMgr(models[0:10])
    gmm_mgr2 = GmmMgr(models[10:20])

    # Hmm setup
    # Make two Hmms with 4 states and order 2 (self loop, forward 1)
    num_states = 4
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr1, num_states, 2, exact=True) 
    hmm1 = make_forward_hmm(gmm_mgr1, num_states, 2, exact=True) 
    hmm_mgr = HmmMgr((hmm0,hmm1))

    # TrainingGraph setup
    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    arc_id = gb.new_arc(node_id0, node_id1)
    gr0 = FrozenGraph(gb)
    tg0 = TrainingGraph(gr0, hmm_mgr, dict())

    valid, ret = validate_training_graph(tg0, gmm_mgr1, hmm_mgr, obs_list, 1, gmm_mgr2)
    return ret


def test3(num_obs):
    # Each of the 4 nodes contains a 4 (or 6)-node order-3 Hmm; the nodes are connected in a
    # diamond pattern
    dimension = 2

    obs_gen = make_data_generator(dimension)
    obs_list = [obs_gen.next() for i in xrange(num_obs)]

    # GmmMgr setup
    num_states = 4
    num_models = 20
    models = make_standard_gmms(dimension, num_models)
    gmm_mgr1 = GmmMgr(models[0:10])
    gmm_mgr2 = GmmMgr(models[10:20])

    # Hmm setup
    # Make four Hmms with 4 (or 6) states and order 3 (self loop, forward 1, forward 2)
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr1, num_states, 3, exact=True) 
    # NB: the asymetry between the two successors is a key part of this test; otherwise,
    # there are no differences between the transition probs going to these successors,
    # which is the tricky case
    hmm1 = make_forward_hmm(gmm_mgr1, num_states + 2, 3, exact=True) 
    hmm2 = make_forward_hmm(gmm_mgr1, num_states, 3, exact=True) 
    hmm3 = make_forward_hmm(gmm_mgr1, num_states, 3, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2, hmm3))

    # TrainingGraph setup
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
    spd[(0,1)] = (0.4, 0.3, 0.8)
    spd[(0,2)] = (0.6, 0.7, 0.2)
    tg0 = TrainingGraph(gr0, hmm_mgr, spd)

    valid, ret = validate_training_graph(tg0, gmm_mgr1, hmm_mgr, obs_list, 1, gmm_mgr2)
    return ret


def test4(num_passes, num_obs):
    # Each of the 4 nodes contains a 4 (or 6)-node order-3 Hmm; the nodes are connected in a
    # diamond pattern
    ret = ""

    dimension = 2

    # Data generator setup and data generation
    obs_gen = make_data_generator(dimension)
    obs_list = [obs_gen.next() for i in xrange(num_obs)]

    # GmmMgr setup
    num_models = 10
    models = make_standard_gmms(dimension, num_models)
    gmm_mgr = GmmMgr(models)

    # Hmm setup
    # Make three Hmms with 4 (or 6) states and order 3 (self loop, forward 1, forward 2)
    num_states = 4
    seed(0)
    hmm0 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    hmm1 = make_forward_hmm(gmm_mgr, num_states + 2, 3, exact=True) 
    hmm2 = make_forward_hmm(gmm_mgr, num_states, 3, exact=True) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    # TrainingGraph setup
    gb = GraphBuilder()
    # Note that here we are using the same HMM in two different TG nodes
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    node_id2 = gb.new_node((2,2))
    node_id3 = gb.new_node((3,0))
    arc_id = gb.new_arc(node_id0, node_id1)
    arc_id = gb.new_arc(node_id0, node_id2)
    arc_id = gb.new_arc(node_id1, node_id3)
    arc_id = gb.new_arc(node_id2, node_id3)
    gr0 = FrozenGraph(gb)
    
    spd = {}
    spd[(0,1)] = (0.4, 0.3, 0.8)
    spd[(0,2)] = (0.6, 0.7, 0.2)

    tg0 = TrainingGraph(gr0, hmm_mgr, spd)

    # Now adapt original TrainingGraph
    for i in xrange(num_passes):
        gmm_mgr.set_adaptation_state("INITIALIZING")
        gmm_mgr.clear_all_accumulators()
        tg0.begin_training()
        gmm_mgr.set_adaptation_state("ACCUMULATING")
        for obs in obs_list:
            tg0.train_one_sequence(obs)
        tg0.end_training()
        gmm_mgr.set_adaptation_state("APPLYING")
        gmm_mgr.apply_all_accumulators()
        gmm_mgr.set_adaptation_state("NOT_ADAPTING")


    ret = tg0.to_string(full=True)
    return ret


def test5(num_obs, do_display=False):
    # A test in which one of the HMMs has a transition from an input directly to
    # an output, so it can behave as an epsilon.  This node is between two other
    # nodes in a linear arrangement.
    
    # Data generator setup and data generation
    dimension = 2
    obs_gen = make_data_generator(dimension)
    obs_list = [obs_gen.next() for i in xrange(num_obs)]

    # GmmMgr setup
    num_models = 20
    models = make_standard_gmms(dimension, num_models)
    gmm_mgr1 = GmmMgr(models[0:10])
    gmm_mgr2 = GmmMgr(models[10:20])

    # Hmm setup
    # Make two Hmms with 2 states and order 2 (self loop, forward 1) The model
    # in the middle is special in that it can skip directly from the input state
    # to the output state.
    seed(0)
    num_states = 2
    hmm0 = make_forward_hmm(gmm_mgr1, num_states, 2, exact=False) 
    hmm1 = Hmm(1)
    trans = array(((0.0, 0.5, 0.5),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))
    hmm1.build_model(gmm_mgr1, (0,), 1, 1, trans)
    hmm2 = make_forward_hmm(gmm_mgr1, num_states, 2, exact=False) 
    hmm_mgr = HmmMgr((hmm0, hmm1, hmm2))

    # TrainingGraph setup
    gb = GraphBuilder()
    node_id0 = gb.new_node((0,0))
    node_id1 = gb.new_node((1,1))
    # node_id2 = gb.new_node((2,2))
    arc_id = gb.new_arc(node_id0, node_id1)
    # arc_id = gb.new_arc(node_id1, node_id2)
    gr0 = FrozenGraph(gb)
    tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=dict())


    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    valid, ret = validate_training_graph(tg0, gmm_mgr1, hmm_mgr, obs_list, 1, gmm_mgr2)
    return ret



def _test8():
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

    result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    return ret


def _test9():
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

    result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    return ret


def _test10():
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


    tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=spd)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    return ret

def _test11():
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

    # The topology here is slightly complex than the previous example
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


    tg0 = TrainingGraph(gr0, hmm_mgr, split_prob_dict=spd)

    if do_display:
        tg0.dot_display()
        tg0.dot_display(expand_hmms=True)

    with DebugPrint("bwt_ctsh") if True else DebugPrint():
        result_hmm = tg0.convert_to_standalone_hmm()
    ret += "\n\n========= TG CONVERTED TO Hmm =========\n\n" + result_hmm.to_string(full=True)

    return ret





def logreftest():
    numpy.set_printoptions(linewidth = 300)
    print "================================  TEST 0  ==========================="
    s0 = test0(10)
    print s0
    print "================================  TEST 1  ==========================="
    s1 = test1(10)
    print s1
    print "================================  TEST 2  ==========================="
    s2 = test2(10)
    print s2
    print "================================  TEST 3  ==========================="
    s3 = test3(10)
    print s3
    print "================================  TEST 4  ==========================="
    s4 = test4(4, 10)
    print s4
    print "================================  TEST 5  ==========================="
    s5 = test5(10)
    print s5


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
    
        numpy.set_printoptions(linewidth = 300)

##         print "================================  TEST 0  ==========================="
##         with DebugPrint("bwt_ctsh", "hmm_gxfs", "hmm_da", "hmm_pofb", "hmm_poff") if True else DebugPrint():
##             s0 = test0(10)
##         print s0

##         print "================================  TEST 1  ==========================="
##         with DebugPrint("bwt_ctsh") if True else DebugPrint():
##             s1 = test1(10)
##         print s1

##         print "================================  TEST 2  ==========================="
##         s2 = test2(10)
##         print s2

##         print "================================  TEST 3  ==========================="
##         s3 = test3(10)
##         print s3

##         print "================================  TEST 4  ==========================="
##         s4 = test4(4, 10)
##         print s4
        print "================================  TEST 5  ==========================="
        with DebugPrint("bwt_ctsh", "hmm_gxfs", "hmm_da", "hmm_pofb", "hmm_poff") if True else DebugPrint():
            s5 = test5(1, do_display=True)
        print s5

        

