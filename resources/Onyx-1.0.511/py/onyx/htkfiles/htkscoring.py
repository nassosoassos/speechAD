###########################################################################
#
# File:         htkscoring.py (directory: ./py/onyx/hktfiles)
# Date:         17-Nov-2008
# Author:       Hugh Secker-Walker
# Description:  Test of scoring of acoustic models with feature data
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
    Quick test of scoring some acoustic models with feature data.

    >>> module_dir, module_name = os.path.split(__file__)

    >>> am_filename = os.path.join(module_dir, "monophones.mmf")
    >>> with open(am_filename) as f:
    ...     models_dict, hmm_mgr, gmm_mgr = read_htk_mmf_file(f)
    >>> print ' '.join(sorted(models_dict))
    aa ae ah ao aw ax ay b ch d dd dh dx eh el en er ey f g hh ih iy jh k kd l m n ng ow oy p pd r s sh sil sp t td th ts uh uw v w y z

    >>> print hmm_mgr
    HmmMgr with 49 models
    >>> hmm_mgr.num_models
    49
    >>> hmm_mgr.get_adaptation_state()
    'NOT_ADAPTING'

    >>> print gmm_mgr
    GmmMgr with 145 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 39, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)
    >>> gmm_mgr.num_models
    145
    >>> gmm_mgr.get_adaptation_state()
    'NOT_ADAPTING'

    >>> audio_filename = onyx.home + '/data/htk_r1_adg0_4/' + "adg0_4_sr009.mfc"
    >>> with open(audio_filename, 'rb') as f:
    ...    (audio_data, (kind, qualifiers), samp_period) = read_htk_audio_file(f)
    >>> kind
    (6, 'MFCC', 'mel-frequency cepstral coefficients')
    >>> qualifiers
    (('A', 'has acceleration coefficients'), ('E', 'has energy'), ('D', 'has delta coefficients'))

    >>> audio_data.shape
    (449, 39)
    >>> len(audio_data)
    449
    >>> models = tuple(gmm_mgr.get_model(i) for i in xrange(gmm_mgr.num_models))
    >>> global_stats = simple_stats()
    >>> for obs_id, observation in enumerate(audio_data):
    ...   #obs_stats = simple_stats()
    ...   for model in models:
    ...     score = model.score(observation)
    ...     #obs_stats(score)
    ...     global_stats(score)
    ...   #print ' ', obs_id, 'obs_stats:',  obs_stats

    >>> print 'global_stats:', global_stats #doctest: +ELLIPSIS
    global_stats: count 65105  min +(-0918)0x7b58d14eca40...  max +(-0072)0xc1d3354e8e44...  sum +(-0070)0x91f1fafccc92...  sumsq +(-0142)0x927de65a98b...
    >>> global_stats.count == len(audio_data) * gmm_mgr.num_models
    True

    Same test but with adapted models
    
    >>> am_filename = os.path.join(module_dir, "monophones4.mmf")
    >>> with open(am_filename) as f:
    ...     models_dict, hmm_mgr, gmm_mgr = read_htk_mmf_file(f)
    >>> print ' '.join(sorted(models_dict))
    aa ae ah ao aw ax ay b ch d dd dh dx eh el en er ey f g hh ih iy jh k kd l m n ng ow oy p pd r s sh sil sp t td th ts uh uw v w y z

    >>> models = tuple(gmm_mgr.get_model(i) for i in xrange(gmm_mgr.num_models))
    >>> global_stats = simple_stats()
    >>> for obs_id, observation in enumerate(audio_data):
    ...   for model in models:
    ...     score = model.score(observation)
    ...     global_stats(score)

    Three of the stats differ in the LSB on 64-bit platform.
    >>> print 'global_stats:', global_stats #doctest: +ELLIPSIS
    global_stats: count 65105  min +(-1023)0x0000000000000  max +(-0061)0xbf7e129869...  sum +(-0060)0x8e9d15018691...  sumsq +(-0121)0xcd929a6bbdb...
    >>> global_stats.count == len(audio_data) * gmm_mgr.num_models
    True

"""

from __future__ import with_statement
import os
import onyx
from onyx.am.modelmgr import GmmMgr
from onyx.am.hmm_mgr import HmmMgr
from onyx.htkfiles.htkmmf import read_htk_mmf_file
from onyx.htkfiles.htkaudio import read_htk_audio_file
from onyx.util.floatutils import float_to_readable_string

class simple_stats(object):
    def __init__(self):
        self.count = 0
        self.sum = self.sumsq = 0
        self.min = self.max = None
    def __call__(self, sample):
        if self.min is None:
            self.min = self.max = sample
        self.count += 1
        if sample < self.min:
            self.min = sample
        elif sample > self.max:
            self.max = sample
        self.sum += sample
        self.sumsq += sample * sample
    def __str__(self):
        return '  '.join(name + ' ' + formatter(value) for name, value, formatter in (
            ('count', self.count, str),
            ('min', self.min, float_to_readable_string),
            ('max', self.max, float_to_readable_string),
            ('sum', self.sum, float_to_readable_string),
            ('sumsq', self.sumsq, float_to_readable_string),
            ))

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
