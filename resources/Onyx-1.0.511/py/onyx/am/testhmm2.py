###########################################################################
#
# File:         testhmm2.py
# Date:         Fri 21 Nov 2008 17:07
# Author:       Ken Basye
# Description:  Various test drivers for Hmm class; these use the log domain
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
    Various test drivers for Hmm class; these use the log domain

    >>> True
    True
"""

from __future__ import with_statement
from numpy import array, eye
from onyx.am.gaussian import GaussianMixtureModel, SimpleGaussianModel, DummyModel
from onyx.am.hmm import Hmm
from onyx.util.debugprint import DebugPrint
from onyx.am.modelmgr import GmmMgr
from itertools import izip

def make_gmm(dimension, num_mixtures):
    gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.FULL_COVARIANCE, num_mixtures)
    w = [1.0 / num_mixtures for n in xrange(num_mixtures)]
    gmm.set_weights(array(w))
    mu = array(((1.5,1.5), (3,3)))
    v = array((eye(2), eye(2)))
    gmm.set_model(mu, v)
    return gmm

def make_gmm_diag(dimension, num_mixtures):
    gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.DIAGONAL_COVARIANCE, num_mixtures)
    w = [1.0 / num_mixtures for n in xrange(num_mixtures)]
    gmm.set_weights(array(w))
    mu = array(((1.5,1.5), (3,3)))
    v = array(((1.0,1.0), (1.0,1.0)))
    gmm.set_model(mu, v)
    return gmm

# test0 verifies that we can deal correctly with an Hmm that has no real states, only inputs and outputs.  
def test0(num_obs, num_passes):
    dimension = 2
    
    # Data generator setup
    target_means = (1,1)
    target_vars = (0.1,0.1)
    generator = SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE)
    generator.set_model(target_means, target_vars)

    SimpleGaussianModel.seed(0)
    GaussianMixtureModel.seed(0)

    mm = GmmMgr(dimension)

    # Hmm setup
    hmm0 = Hmm(0, log_domain=True)

    # A transition probability matrix with no real state.
    # The entry state feeds into the exit state with p=1.
    trans = array(((0.0, 1.0),
                   (0.0, 0.0)))
    
    hmm0.build_model(mm, (), 1, 1, trans)
    print hmm0.to_string(True)

    # Try some adaptation.  Note that we are feeding the entire data set as one stream
    # to the Hmm adaption call.
    data = [generator.sample() for i in xrange(num_obs)]
    for p in xrange(num_passes):
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        with DebugPrint("hmm_gxfs", "hmm_aos") if False else DebugPrint():
            hmm0.adapt_one_sequence(data)
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")

    print hmm0.to_string(True)


# test1 compares the adaptation of a Gmm with the same model embedded in a single-state Hmm.
# The results are identical.  See also test2.
def test1(num_obs, num_passes):
    dimension = 2
    
    # Data generator setup
    target_means = (1,1)
    target_vars = (0.1,0.1)
    generator = SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE)
    generator.set_model(target_means, target_vars)

    SimpleGaussianModel.seed(0)
    GaussianMixtureModel.seed(0)

    # Gmm setup
    num_mixtures = 2
    gmm0 = make_gmm(dimension, num_mixtures)
    gmm1 = make_gmm(dimension, num_mixtures)
    mm = GmmMgr((gmm1,))

    # Hmm setup
    hmm0 = Hmm(1, log_domain=True)

    # A transition probability matrix with a p=1/2 exit for the real state.
    # The entry state feeds into the real state with p=1.
    trans = array(((0.0, 1.0, 0.0),
                   (0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0)))

    
    hmm0.build_model(mm, (0,), 1, 1, trans)
    print hmm0.to_string(True)
    print gmm0

    # Try some adaptation.  Note that we are feeding the entire data set as one stream
    # to the Hmm adaption call.
    data = [generator.sample() for i in xrange(num_obs)]
    for p in xrange(num_passes):
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        hmm0.adapt_one_sequence(data)
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")
    gmm0.adapt(data, max_iters = num_passes)

    print hmm0.to_string(True)
    print gmm0



def test2(num_obs, num_passes):
    dimension = 2
    
    # Data generator setup
    target_means = (1,1)
    target_vars = (0.1,0.1)
    generator = SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE)
    generator.set_model(target_means, target_vars)

    SimpleGaussianModel.seed(0)
    GaussianMixtureModel.seed(0)

    # Gmm setup
    num_mixtures = 2
    gmm0 = make_gmm(dimension, num_mixtures)
    gmm1 = make_gmm(dimension, num_mixtures)
    mm = GmmMgr((gmm1,))

    # Hmm setup
    # A transition probability matrix with a p=1 exit for the real state.
    # The entry state feeds into the real state with p=1.
    trans = array(((0.0, 1.0, 0.0),
                   (0.0, 0.0, 1.0),
                   (0.0, 0.0, 0.0)))
    hmm0 = Hmm(1, log_domain=True)
    hmm0.build_model(mm, (0,), 1, 1, trans)
    print hmm0.to_string(True) + '\n'
    print gmm0
    print '\n\n'

    # Try some adaptation
    data = [generator.sample() for i in xrange(num_obs)]
    for p in xrange(num_passes):
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        for point in data:
            # We treat each point as an entire sequence
            hmm0.adapt_one_sequence((point,))
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")
    gmm0.adapt(data, max_iters = num_passes)

    print hmm0.to_string(True) + '\n'
    print gmm0


# Another test to see that we can get identical results from training a Gmm and the same Gmm embedded
# in a one-state Hmm.
def test2a(num_obs, num_passes):
    dimension = 2
    
    # Data generator setup
    target_means = (1,1)
    target_vars = (0.1,0.1)
    generator = SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE)
    generator.set_model(target_means, target_vars)
    SimpleGaussianModel.seed(0)
    data = [generator.sample() for i in xrange(num_obs)]


    # Gmm setup
    num_mixtures = 2
    gmm0 = make_gmm_diag(dimension, num_mixtures)
    gmm1 = make_gmm_diag(dimension, num_mixtures)
    mm = GmmMgr((gmm1,))

    # Hmm setup
    # A transition probability matrix with a p ~= 1 self-loop for the real state.
    # The entry state feeds into the real state with p=1.  We use p ~= 1 for the
    # second self loop since we need *some* probability of finishing.
    trans = array(((0.0, 1.0, 0.0),
                   (0.0, 0.999999999999, 0.000000000001),
                   (0.0, 0.0, 0.0)))
    hmm0 = Hmm(1, log_domain=True)
    hmm0.build_model(mm, (0,), 1, 1, trans)
    print hmm0.to_string(True) + '\n'
    print gmm0
    print '\n\n'

    # Try some adaptation
    for p in xrange(num_passes):
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        hmm0.adapt_one_sequence(data)
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")

    really_print = False
    with DebugPrint("gaussian", "gaussian_pt", "gaussian_gmm_score") if really_print else DebugPrint():
        gmm0.adapt(data, max_iters = num_passes)

    print hmm0.to_string(True) + '\n'
    print gmm0


# 400 data values from Chris for mimicking HTK training he did
datasets = (((-2.868,  -3.772),
            (-4.064,  -2.739),
            (-2.040,  -2.857),
            (-4.563,  -2.645),
            (-0.667,  -1.865),
            (-5.565,  -2.027),
            (-2.881,  -3.071),
            (-2.391,  -1.065),
            (-2.723,  -1.324),
            (-3.145,  -3.125),
            (-3.526,  -2.755),
            (-1.576,  -2.847),
            (-2.825,  -2.426),
            (-3.424,  -4.692),
            (-3.097,  -2.133),
            (-3.071,  -2.929),
            (-3.421,  -3.762),
            (-2.953,  -4.394),
            (-2.736,  -3.437),
            (-2.050,  -1.408),
            (-2.850,  -0.544),
            (-3.953,  -2.640),
            (-2.270,  -2.553),
            (-4.296,  -2.501),
            (-1.795,  -2.035),
            (-2.619,  -4.058),
            (-3.405,  -3.515),
            (-3.270,  -3.265),
            (-1.998,  -2.403),
            (-1.905,  -3.534),
            (-2.346,  -4.309),
            (-2.631,  -0.856),
            (-2.394,  -2.482),
            (-1.809,  -2.497),
            (-4.082,  -3.400),
            (-2.560,  -6.093),
            (-3.004,  -4.050),
            (-3.158,  -1.626),
            (-2.662,  -3.654),
            (-2.393,  -3.227),
            (-1.855,  -2.799),
            (-4.666,  -1.780),
            (-2.019,  -4.128),
            (-3.432,  -0.746),
            (-1.723,  -3.199),
            (-3.722,  -1.779),
            (-3.318,  -3.643),
            (-1.391,  -2.661),
            (-2.011,  -3.037),
            (-2.693,  -2.517),
            (-4.569,  -2.872),
            (-4.561,  -4.091),
            (-4.823,  -3.302),
            (-5.072,  -2.243),
            (-4.037,  -2.396),
            (-2.707,  -3.360),
            (-1.506,  -2.152),
            (-3.086,  -2.213),
            (-3.035,  -2.883),
            (-2.930,  -3.378),
            (-2.159,  -2.246),
            (-2.383,  -2.550),
            (-1.958,  -4.888),
            (-3.894,  -0.185),
            (-2.235,  -2.530),
            (-3.090,  -3.280),
            (-3.796,  -4.521),
            (-2.715,  -1.607),
            (-2.915,  -4.012),
            (-2.344,  -2.383),
            (-2.210,  -3.274),
            (-1.897,  -3.146),
            (-3.081,  -4.826),
            (-3.818,  -3.447),
            (-0.454,  -2.289),
            (-2.643,  -2.756),
            (-4.614,  -2.156),
            (-1.995,  -1.562),
            (-3.319,  -2.302),
            (-4.075,  -1.186),
            (-3.800,  -2.898),
            (-2.883,  -1.717),
            (-2.614,  -3.471),
            (-2.433,  -2.722),
            (-1.324,  -2.472),
            (-2.582,  -2.782),
            (-2.850,  -2.393),
            (-2.938,  -2.561),
            (-2.643,  -2.849),
            (-0.973,  -2.560),
            (-3.343,  -4.301),
            (-5.160,  -2.794),
            (-2.908,  -2.849),
            (-2.690,  -2.123),
            (-2.832,  -3.456),
            (-3.012,  -2.316),
            (-2.289,  -3.781),
            (-1.859,  -1.387),
            (-5.528,  -2.981),
            (-2.297,  -4.182),
            (-1.646,   1.229),
            (-2.616,   3.092),
            (0.154,   4.225),
            (-0.534,   2.643),
            (-0.222,   5.666),
            (-0.880,   1.422),
            (0.959,   1.858),
            (0.665,   3.093),
            (-0.234,   4.417),
            (0.569,   2.091),
            (-1.766,   2.941),
            (0.166,   3.997),
            (-0.250,   2.918),
            (-0.197,   3.632),
            (0.484,   3.871),
            (0.577,   2.876),
            (-0.133,   3.948),
            (-1.117,   4.001),
            (-2.408,   2.333),
            (0.332,   2.126),
            (-0.714,   3.629),
            (0.001,   0.751),
            (-1.808,   2.683),
            (-1.278,   0.736),
            (-0.671,   2.353),
            (-1.938,   3.402),
            (2.097,   2.290),
            (-0.529,   2.377),
            (-0.448,  -0.234),
            (2.139,   3.570),
            (-0.980,   2.733),
            (0.913,   3.912),
            (-0.356,   1.514),
            (0.179,   3.649),
            (0.540,   3.135),
            (0.210,   2.218),
            (0.070,   5.273),
            (1.599,   4.263),
            (0.477,   3.424),
            (-0.634,   3.055),
            (0.912,   2.992),
            (0.172,   3.424),
            (-1.054,   2.572),
            (-0.424,   2.672),
            (-2.074,   5.146),
            (1.636,   3.199),
            (-1.048,   3.496),
            (-1.310,   2.434),
            (0.473,   2.090),
            (0.842,   3.358),
            (-0.050,   3.174),
            (-0.172,   2.763),
            (0.271,   3.453),
            (0.615,   0.688),
            (-1.322,   4.543),
            (0.300,   2.603),
            (-0.423,   2.535),
            (-0.161,   3.474),
            (-0.787,   3.748),
            (-0.201,   3.914),
            (0.029,   3.738),
            (1.832,   3.353),
            (-1.245,   2.998),
            (-2.087,   2.845),
            (1.039,   1.768),
            (1.288,   2.016),
            (0.494,   3.880),
            (-0.937,   1.533),
            (0.531,   2.177),
            (0.550,   2.855),
            (0.257,   3.538),
            (0.666,   1.511),
            (-0.939,   3.031),
            (-0.150,   2.970),
            (-0.271,   2.949),
            (-1.597,   3.776),
            (0.593,   2.446),
            (-2.038,   2.837),
            (-1.395,   2.913),
            (0.028,   2.786),
            (0.157,   4.140),
            (-1.398,   3.747),
            (-0.449,   3.023),
            (-0.497,   3.275),
            (-0.337,   3.399),
            (0.738,   4.346),
            (-1.363,   3.571),
            (-0.452,   4.519),
            (-1.750,   3.778),
            (0.636,   3.370),
            (-0.457,   3.557),
            (-0.088,   4.657),
            (-0.799,   2.114),
            (-1.898,   2.556),
            (-0.615,   3.405),
            (1.390,   3.192),
            (0.375,   2.797),
            (0.496,   2.381),
            (0.172,   4.258),
            (0.082,   2.105)),

            ((1.240,  -2.898),
             (2.291,  -4.133),
             (3.191,  -1.027),
            (1.436,  -1.598),
            (3.227,  -3.152),
            (2.152,  -3.071),
            (2.941,  -2.355),
            (3.631,  -3.422),
            (5.687,  -2.773),
            (5.163,  -3.550),
            (3.412,  -2.772),
            (3.175,  -3.868),
            (3.553,  -2.608),
            (4.579,  -2.655),
            (3.127,  -2.106),
            (2.706,  -3.049),
            (3.907,  -2.544),
            (1.726,  -3.921),
            (4.229,  -3.911),
            (3.806,  -3.510),
            (3.272,  -2.860),
            (4.021,  -3.588),
            (1.404,  -3.553),
            (2.511,  -0.603),
            (2.303,  -3.849),
            (4.374,  -3.069),
            (2.412,  -2.675),
            (4.918,  -4.468),
            (3.248,  -3.543),
            (0.771,  -0.523),
            (3.168,  -1.546),
            (3.794,  -3.744),
            (2.551,  -1.869),
            (3.853,  -1.689),
            (2.214,  -2.259),
            (3.464,  -2.702),
            (2.244,  -1.514),
            (2.518,  -1.668),
            (2.696,  -3.056),
            (3.971,  -1.471),
            (2.277,  -3.860),
            (2.913,  -3.492),
            (3.166,  -2.703),
            (2.385,  -2.641),
            (4.035,  -2.440),
            (4.090,  -2.334),
            (2.861,  -3.197),
            (3.298,  -3.862),
            (3.831,  -4.422),
            (2.274,  -2.499),
            (2.999,  -3.476),
            (3.645,  -4.989),
            (2.495,  -4.840),
            (3.260,  -2.646),
            (4.217,  -5.474),
            (4.251,  -3.228),
            (4.858,  -2.949),
            (1.805,  -2.971),
            (3.138,  -2.904),
            (1.577,  -3.449),
            (2.254,  -3.233),
            (3.151,  -1.496),
            (4.724,  -0.706),
            (2.540,  -3.232),
            (4.546,  -2.541),
            (3.293,  -3.867),
            (2.988,  -4.385),
            (3.184,  -2.047),
            (4.200,  -4.309),
            (4.159,  -3.492),
            (3.038,  -2.360),
            (1.109,  -3.657),
            (2.863,  -2.360),
            (2.926,  -4.027),
            (2.445,  -4.178),
            (3.705,  -1.201),
            (0.165,  -4.590),
            (2.551,  -4.004),
            (1.773,  -3.418),
            (3.350,  -2.310),
            (3.796,  -2.961),
            (3.233,  -1.696),
            (2.539,  -1.528),
            (4.796,  -2.744),
            (3.736,  -3.736),
            (2.474,  -4.433),
            (2.081,  -3.816),
            (3.992,  -2.816),
            (4.380,  -1.376),
            (2.356,  -2.875),
            (4.446,  -2.760),
            (2.734,  -1.120),
            (2.065,  -2.962),
            (3.343,  -3.578),
            (2.091,  -3.357),
            (1.951,  -2.246),
            (2.872,  -2.456),
            (2.312,  -2.441),
            (2.945,  -2.713),
            (2.898,  -3.064)),

            ((0.512,  -0.725),
            (1.324,  -0.245),
            (0.417,   0.757),
            (0.077,  -0.096),
            (-0.004,  -1.968),
            (-0.209,   1.491),
            (-1.658,  -0.217),
            (1.601,  -1.054),
            (-0.365,  -0.701),
            (1.248,   0.501),
            (1.283,  -0.426),
            (0.589,  -0.063),
            (-0.296,   0.017),
            (-2.220,  -0.073),
            (0.641,  -1.298),
            (0.968,  -0.807),
            (-0.108,  -0.859),
            (1.159,   0.430),
            (-0.465,   0.923),
            (-1.270,   1.176),
            (0.014,   0.529),
            (-0.143,   0.535),
            (-0.025,  -0.342),
            (-0.279,  -0.401),
            (-1.161,  -1.427),
            (-2.084,  -0.542),
            (-1.181,   0.535),
            (0.786,   1.371),
            (-0.138,  -0.517),
            (0.525,   0.224),
            (0.790,   0.729),
            (-0.333,  -0.856),
            (-0.497,  -0.168),
            (0.099,   0.189),
            (0.198,   0.389),
            (1.230,  -0.302),
            (1.220,   1.591),
            (1.191,   1.221),
            (2.345,  -1.565),
            (1.001,  -0.854),
            (0.066,  -0.482),
            (0.815,  -1.817),
            (0.535,   1.534),
            (-0.707,  -0.248),
            (-0.639,   0.178),
            (-0.789,  -1.598),
            (0.480,   0.187),
            (-0.719,   1.163),
            (0.244,   0.605),
            (-0.990,   0.181),
            (-1.258,  -0.108),
            (-0.376,  -0.399),
            (-2.072,  -2.086),
            (-0.783,   1.570),
            (0.308,  -1.380),
            (1.765,   0.947),
            (0.422,   0.352),
            (0.921,   2.176),
            (0.338,  -0.372),
            (0.655,   1.486),
            (0.969,  -0.296),
            (0.853,   0.347),
            (-0.128,   0.485),
            (-0.385,  -1.263),
            (-1.710,  -0.278),
            (-1.171,   1.045),
            (0.580,   0.122),
            (-1.326,   1.791),
            (1.147,  -0.393),
            (-0.331,   2.370),
            (1.449,   0.364),
            (1.794,   0.779),
            (0.578,  -0.849),
            (0.358,  -1.939),
            (0.032,  -0.488),
            (-0.191,   0.300),
            (-0.333,   0.476),
            (1.416,  -1.253),
            (-0.830,  -2.381),
            (0.501,  -0.418),
            (1.426,   0.415),
            (-0.775,   1.837),
            (0.778,  -0.185),
            (0.109,   0.031),
            (0.007,   0.726),
            (1.146,  -0.248),
            (2.290,  -0.366),
            (1.271,  -1.230),
            (1.019,   1.346),
            (-0.057,   0.253),
            (-1.579,   0.773),
            (-1.995,  -1.423),
            (-0.901,   0.425),
            (-1.290,   1.277),
            (-1.223,  -0.096),
            (-0.343,  -0.312),
            (-0.407,  -1.912),
            (-0.328,   0.874),
            (-0.371,   0.321),
            (1.087,  -0.518)))


def test3():
    for i in (0,1,2):
        for passes in (1,4):
            print "=======================  TEST 3 (dataset %d, %d passes)  =================" % (i, passes)
            test3_helper(i, passes)


def test3_helper(dataset_idx, num_passes):
    """
    This tests mimics a run ChrisW did with HTK.  The models are 2-D single-mode Gaussians
    embedded in a 1-state Hmm.  Each data point is taken as a sequence of length 1.    
    """
    dimension = 2

    # Gmm setup
    gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.DIAGONAL_COVARIANCE, 1)
    gmm.set_weights(array((1.0,)))
    mu = array(((0.0,0.0),))
    v = array(((1.0,1.0),))
    gmm.set_model(mu, v)
    mm = GmmMgr((gmm,))

    # Hmm setup
    # A transition probability matrix with a p=1 self-loop for the real state.
    # The entry state feeds into the real state with p=1.
    trans = array(((0.0, 1.0, 0.0),
                   (0.0, 0.0, 1.0),
                   (0.0, 0.0, 0.0)))
    hmm0 = Hmm(1, log_domain=True)
    hmm0.build_model(mm, (0,), 1, 1, trans)
    print hmm0.to_string(True)

    # adaptation
    data = datasets[dataset_idx]
    for p in xrange(num_passes):
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        for point in data:
            s = array(point)
            # We treat each point as an entire sequence
            hmm0.adapt_one_sequence((s,))
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")

    print hmm0.to_string(True)


def test4():
    a0 = array(datasets[0])
    print a0.mean(0)
    print a0.var(0)
    print
    a1 = array(datasets[1])
    print a1.mean(0)
    print a1.var(0)
    print
    a2 = array(datasets[2])
    print a2.mean(0)
    print a2.var(0)
    

def test5(num_obs):
    for passes in (1,4):
        print "=======================  TEST 5 (%d passes)  =================" % (passes,)
        test5_helper(num_obs, passes)


def test5_helper(num_obs, num_passes):
    """
    This tests mimics a run ChrisW did with HTK.  The models are 2-D single-mode Gaussians
    embedded in a 3-state Hmm.  Each observation is a sequence of length 11, taken by sampling
    2, 3, and 6 times, respectively, from three target distributions.    
    """
    import pprint
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

    # Gmm setup
    num_states = 3
    models = []
    for i in xrange(num_states):
        gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.DIAGONAL_COVARIANCE, 1)
        gmm.set_weights(array((1.0,)))
        mu = array(((0.0,0.0),))
        v = array(((1.0,1.0),))
        gmm.set_model(mu, v)
        models.append(gmm)
    
    mm = GmmMgr(models)
    models = range(num_states)

    # Hmm setup
    trans = array(((0.0, 1.0, 0.0, 0.0, 0.0),
                   (0.0, 0.5, 0.5, 0.0, 0.0),
                   (0.0, 0.0, 0.5, 0.5, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0, 0.0, 0.0)))
    hmm0 = Hmm(num_states, log_domain=True)
    hmm0.build_model(mm, models, 1, 1, trans)
    print hmm0.to_string(True)

    for p in xrange(num_passes):
        # Reseeding here ensures we are repeating the same observations in each pass
        SimpleGaussianModel.seed(0)
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")
        obs_gen = obs_generator(generators, target_durations)
        for i in xrange(num_obs):
            obs = obs_gen.next()
            hmm0.adapt_one_sequence(obs)
        
            obs2 = [tuple(a) for a in obs]
            # Uncomment these lines to show observations as nicely formatted sequences; this
            # is what I gave ChrisW to use with his HTK runs.
            # pprint.pprint(obs2)
            # print
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")
        print hmm0.to_string(True)


def test6():
    """
    This test builds an Hmm with dummy models which always give a score of 1, but
    with a somewhat unusual topology in which there are 6 actual states chained together
    with 2 virtual inputs and 3 virtual outputs.  The point is to make sure we can handle
    this asymetric case correctly.
    """
    import pprint
    num_states = 6
    dimension = 2
    
    # GmmMgr setup
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    mm = GmmMgr(models)
    models = range(num_states)

    # Hmm setup T0:  i1   i2   1    2    3    4    5    6    o1   o2   o3    FROM: 
    trans = array(((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # i1
                   (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # i2

                   (0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 1
                   (0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 2
                   (0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),  # 3
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.1, 0.0, 0.0),  # 4
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.1, 0.0),  # 5
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),  # 6

                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # o1
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # o2
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))) # o3
    
    hmm0 = Hmm(num_states, log_domain=True)
    hmm0.build_model(mm, models, 2, 3, trans)
    print hmm0.to_string(True)

    num_passes = 1
    for p in xrange(num_passes):
        # Reseeding here ensures we are repeating the same observations in each pass
        SimpleGaussianModel.seed(0)
        mm.set_adaptation_state("INITIALIZING")
        mm.clear_all_accumulators()
        hmm0.begin_adapt("STANDALONE")
        mm.set_adaptation_state("ACCUMULATING")

        obs = [array((0,0))] * 11  # Dummy sequence of length 11
        hmm0.adapt_one_sequence(obs)
        
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")
        print hmm0.to_string(True)



def test7():
    """
    This test builds an Hmm with dummy models which always give a score of 1, but
    with a somewhat unusual topology in which there are 6 actual states chained together
    with 2 virtual inputs and 3 virtual outputs.  The point is to make sure we can handle
    this asymetric case correctly.  This is the same as test6 except that now we'll use
    the network adaptation interface instead.
    """
    import pprint
    num_states = 6
    dimension = 2
    
    # GmmMgr setup
    models = []
    for i in xrange(num_states):
        dm = DummyModel(dimension, 1.0)
        models.append(dm)
    
    mm = GmmMgr(models)

    # Hmm setup T0:  i1   i2   1    2    3    4    5    6    o1   o2   o3    FROM: 
    trans = array(((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # i1
                   (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # i2

                   (0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 1
                   (0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # 2
                   (0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0),  # 3
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.1, 0.0, 0.0),  # 4
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.1, 0.0),  # 5
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5),  # 6

                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # o1
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # o2
                   (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))) # o3
    
    hmm0 = Hmm(num_states, log_domain=True)
    models = range(num_states)
    hmm0.build_model(mm, models, 2, 3, trans)
    print hmm0.to_string(True)

    num_passes = 1
    for p in xrange(num_passes):
        # Reseeding here ensures we are repeating the same observations in each pass
        SimpleGaussianModel.seed(0)
        mm.set_adaptation_state("INITIALIZING")
        hmm0.begin_adapt("NETWORK")
        mm.set_adaptation_state("ACCUMULATING")

        num_obs = 11
        obs = [array((0,0))] * num_obs  # Dummy sequence

        context = hmm0.init_for_forward_pass(obs, terminal = True)

        # Add some mass into the system for the forward pass.  To match the behavior of
        # standalone adaptation, we divide an initial mass of 1 evenly across the inputs
        hmm0.accum_input_alphas(context, array([1.0/hmm0.num_inputs] * hmm0.num_inputs))

        # Actually do the forward pass.  Note that we must process one more frame than the number of
        # observations - this is because an extra frame is automatically added which scores 1 on the exit
        # states of the Hmm (and 0 on all real states).  XXX we might want clients to do this for
        # themselves at some point rather than this automatic behavior:

        for frame in xrange(num_obs + 1):
            output_alphas = hmm0.process_one_frame_forward(context)
            print output_alphas

        # Likewise, we initialize and then make the backward pass:
        hmm0.init_for_backward_pass(context)
        hmm0.accum_input_betas(context, array([1.0] * hmm0.num_outputs))
        for frame in xrange(num_obs + 1):
            output_betas = hmm0.process_one_frame_backward(context)
            print output_betas

        # Now collect all the gamma sums; here there's only one:
        norm = hmm0.get_initial_gamma_sum()
        hmm0.add_to_gamma_sum(norm, context)

        # Here's where the actual accumulation happens:
        hmm0.do_accumulation(context, norm)
        
        mm.set_adaptation_state("APPLYING")
        hmm0.end_adapt()
        mm.apply_all_accumulators()
        mm.set_adaptation_state("NOT_ADAPTING")
        print hmm0.to_string(True)



def test8(num_obs):
    for passes in (1,4):
        print "=======================  TEST 8 (%d passes)  =================" % (passes,)
        test8_helper(num_obs, passes)


def test8_helper(num_obs, num_passes):
    """
    This tests mimics a run ChrisW did with HTK.  The models are 2-D single-mode Gaussians embedded
    in a 3-state Hmm.  Each observation is a sequence of length 11, taken by sampling 2, 3, and 6
    times, respectively, from three target distributions.  This is identical to test5 except that
    here I have built the Hmm with only one Gmm, which is shared by all three states.
    """
    import pprint
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

    # Gmm setup
    num_states = 3

    gmm = GaussianMixtureModel(dimension, GaussianMixtureModel.DIAGONAL_COVARIANCE, 1)
    gmm.set_weights(array((1.0,)))
    mu = array(((0.0,0.0),))
    v = array(((1.0,1.0),))
    gmm.set_model(mu, v)
    models = (gmm,)
    
    mm = GmmMgr(models)
    # Here's where we're using the same Gmm in all three states of this Hmm.
    models = (0, 0, 0)

    # Hmm setup
    trans = array(((0.0, 1.0, 0.0, 0.0, 0.0),
                   (0.0, 0.5, 0.5, 0.0, 0.0),
                   (0.0, 0.0, 0.5, 0.5, 0.0),
                   (0.0, 0.0, 0.0, 0.5, 0.5),
                   (0.0, 0.0, 0.0, 0.0, 0.0)))
    hmm0 = Hmm(num_states, log_domain=True)
    hmm0.build_model(mm, models, 1, 1, trans)
    print hmm0.to_string(True)

    for p in xrange(num_passes):
        # Reseeding here ensures we are repeating the same observations in each pass
        SimpleGaussianModel.seed(0)
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



def obs_generator(generators, target_durations):
    assert len(generators) == len(target_durations)
    while True:
        yield [m.sample() for m, d in izip(generators, target_durations) for i in xrange(d)]


def logreftest():
    print "================================  TEST 0  ==========================="
    with DebugPrint("hmm_ea", "hmm_aos", "hmm_gxfs") if False else DebugPrint():
        test0(100, 5)
    print "================================  TEST 1  ==========================="
    test1(100, 5)
    print "================================  TEST 2  ==========================="
    test2(100, 1)
    print "================================  TEST 2a  ==========================="
    test2a(100, 5)
    print "================================  TEST 3  ==========================="
    test3()
    print "================================  TEST 4  ==========================="
    test4()
    print "================================  TEST 5  ==========================="
    test5(400)
    print "================================  TEST 6  ==========================="
    test6()
    print "================================  TEST 7  ==========================="
    test7()
    print "================================  TEST 8  ==========================="
    test8(100)


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
    
        print "================================  TEST 0  ==========================="
        test0(100, 5)
        # print "================================  TEST 1  ==========================="
        # test1(100, 5)
        # print "================================  TEST 2  ==========================="
        # test2(100, 1)
        # print "================================  TEST 2a  ==========================="
        # test2a(100, 5)
        # print "================================  TEST 3  ==========================="
        # test3()
        # print "================================  TEST 4  ==========================="
        # test4()
        # print "================================  TEST 5  ==========================="
        # test5(1)
        # print "================================  TEST 6  ==========================="
        # with DebugPrint("hmm_aos"):
        # test6()
        # print "================================  TEST 7  ==========================="
        # with DebugPrint("hmm_gbff", "hmm_iffp", "hmm_pofb"):
        # test7()
        # test8(400)
