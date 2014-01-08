###########################################################################
#
# File:         htkmmf.py (directory: py/onyx/htkfiles)
# Date:         Fri 18 Apr 2008 15:21
# Author:       Ken Basye
# Description:  Read HTK mmf files into our model structure
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
    Read HTK mmf files into our model structure.

    >>> module_dir, module_name = os.path.split(__file__)

    >>> filename = os.path.join(module_dir, "start.mmf")
    >>> with open(filename) as f:
    ...     models_dict, hmm_mgr, gmm_mgr = read_htk_mmf_file(f)
    >>> print sorted(models_dict)
    ['class1', 'class2', 'class3']

    >>> print gmm_mgr
    GmmMgr with 3 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 2, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)

    >>> for name, index in models_dict.items():
    ...     print "Built model %s at index %d:" % (name,index)
    ...     print hmm_mgr[index].to_string()
    Built model class2 at index 1:
    Hmm: num_states = 1, model dim = 2
    Built model class3 at index 2:
    Hmm: num_states = 1, model dim = 2
    Built model class1 at index 0:
    Hmm: num_states = 1, model dim = 2


    >>> filename = os.path.join(module_dir, "monophones.mmf")
    >>> with open(filename) as f:
    ...     models_dict, hmm_mgr, gmm_mgr = read_htk_mmf_file(f)
    >>> print ' '.join(sorted(models_dict))
    aa ae ah ao aw ax ay b ch d dd dh dx eh el en er ey f g hh ih iy jh k kd l m n ng ow oy p pd r s sh sil sp t td th ts uh uw v w y z

    >>> print gmm_mgr
    GmmMgr with 145 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 39, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)
"""

from __future__ import with_statement
import os
from itertools import izip
from numpy import array
from pprint import pformat

from onyx.htkfiles import mmf
from onyx.util.debugprint import dcheck, DebugPrint

def read_htk_mmf_file(file, log_domain=False):
    """
    Read HTK mmf files into our model structure.

    file should be a filestream opened on an HTK MMF file.  hmm_mgr and gmm_mgr
    must be HmmMgr and GmmMgr instances, respectivly, and both must be in the
    NOT_ADAPTING state.  The return is a dict with the names of the models as
    keys and the model indices in hmm_mgr as values.
    """
    # in general these imports should be at module scope, but doing so causes an
    # import circularity as hmm_mgr.py needs to import from *this* module
    # (htkmmf), so we delay things and do it in the function that needs these
    # symbols
    from onyx.am.gaussian import GaussianMixtureModel
    from onyx.am.modelmgr import GmmMgr
    from onyx.am.hmm import Hmm
    from onyx.am.hmm_mgr import HmmMgr

    dc = dcheck("htkmmf_read")
    covar_map = {'diagc' : GaussianMixtureModel.DIAGONAL_COVARIANCE,
                 'fullc' : GaussianMixtureModel.FULL_COVARIANCE }
    contents = file.read()
    dc and dc("contents = \n%s" % (contents,))
    try:
        result = mmf.parse('top', contents)
    except Exception, any:
        print "HTK MMF parse error: " + str(any)
        return None
    
    dc and dc("result = \n%s" % (pformat(result),))
    
    # XXX we may want to store and return some of the meta-information in these
    # files; right now we don't do that.
    if 'options' not in result:
        raise IOError("No global options found in %s" % (filename,))
    opts = result['options']
    if 'models' not in result:
        raise IOError("No models found in %s!" % (filename,))
    models = result['models']
    if 'vecsize' not in opts:
        raise IOError("No vecsize option found in %s" % (filename,))
    dim = opts['vecsize']
    if 'covar' not in opts:
        covar_type = GaussianMixtureModel.DIAGONAL_COVARIANCE
    else:
        if opts['covar'] not in covar_map:
            raise IOError("Unknown covar option %s found in %s" % (opts['covar'], filename,))
        covar_type = covar_map[opts['covar']]
    dim = opts['vecsize']
    hmm_mgr = HmmMgr(dim)
    gmm_mgr = GmmMgr(dim)
    hmms = []
    names = []
    unnamed_index = 0
    for label, m in models:
        assert label == 'HMM'
        dc and dc("m = \n%s" % (pformat(m),))
        dc and dc("m.keys() = \n%s" % (m.keys(),))
        if m.hasattr.decl:
            name = m.decl
        else:
            name = ("UnnamedModel%d" % unnamed_index)
            unnamed_index += 1
        n = m.numstates - 2   # HTK numstates counts virtual entry and exit states
        hmm = Hmm(n, log_domain)
        gmms = []
        for s_label, state in m.states:
            assert s_label == 'state'
            dc and dc("state.keys() = \n%s" % (state.keys(),))
            num_mixtures = 1
            weights = array((1.0,), dtype = float)
            gmm = GaussianMixtureModel(dim, covar_type, num_mixtures)
            gmm.set_weights(weights)
            gmm.set_model(state.mean, state.var)
            dc and dc("gmm = %s" % (gmm,))
            gmms.append(gmm)

        model_indices = gmm_mgr.add_models(gmms)
        hmm.build_model(gmm_mgr, model_indices, 1, 1, m.transp)
        hmms.append(hmm)
        names.append(name)
    indices = hmm_mgr.add_models(hmms)
    return dict(izip(names, indices)), hmm_mgr, gmm_mgr
    
def logreftest():
    module_dir, module_name = os.path.split(__file__)
    files = tuple(os.path.join(module_dir, mmf_file) for mmf_file in ("start.mmf", 'mmf1.mmf', 'mmf4.mmf'))
    for fname in files:
        with DebugPrint('htkmmf_read'):
            with open(fname) as f:
                models, hmm_mgr, gmm_mgr = read_htk_mmf_file(f)
            print gmm_mgr
            for name, index in models.items():
                print "Built model %s at index %d:" % (name,index)
                print hmm_mgr[index].to_string('full')
                print
            print

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    args = argv[1:]
    if '--logreftest' in args:
        logreftest()
