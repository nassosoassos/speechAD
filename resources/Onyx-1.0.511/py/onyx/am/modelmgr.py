###########################################################################
#
# File:         modelmgr.py (directory: ./py/onyx/am)
# Date:         Tue 13 May 2008 09:45
# Author:       Ken Basye
# Description:  Model managers at various levels
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
    Model managers at various levels.

    See in particular ModelManager, a base class for managers, and GmmMgr, a
    manager for Gaussian mixture models.
"""

from __future__ import division
import cStringIO
import numpy
from itertools import izip, chain, repeat
from numpy import array, newaxis, zeros
from onyx.am.gaussian import GaussianMixtureModel, DummyModel, SimpleGaussianModel
from onyx.am.gaussian import GaussianModelBase
from onyx.textdata.onyxtext import OnyxTextReader, OnyxTextWriter
from onyx.textdata.yamldata import YamldataReader, YamldataWriter
from onyx.util.debugprint import dcheck


class ModelManager(object):
    """
    Abstract base class for model managers of various levels.

    This class supports storing a collection of models as a tuple with indexed
    access, as well as transition through a series of states as part of the
    adaptation process.  A state machine enforces the cycle of initialization,
    accumulation, and adaptation.  The intent is that a single high-level client
    signals progress in this cycle via explicit calls to set_adaptation_state,
    while multiple lower-level clients each request initialization, accumulation
    and adaptation of the models they are using.  It is expected that multiple
    low-level clients will share the same models, so more than one client may be
    making such requests for a given model.
    """

    ADAPTATION_STATES = ("NOT_ADAPTING", "INITIALIZING", "ACCUMULATING", "APPLYING")
    # Map from current state to legal next state
    _LEGAL_TRANSITION_MAP = {"NOT_ADAPTING" : ("INITIALIZING",),
                             "INITIALIZING" : ("ACCUMULATING",),
                             "ACCUMULATING" : ("APPLYING", "INITIALIZING"),
                             "APPLYING" : ("NOT_ADAPTING",)}

    def __init__(self, dimension):
        self._dimension = dimension
        self._adaptation_state = 'NOT_ADAPTING'
        
    def get_model(self, index):
        self._verify_index(index)
        return self._models[index]

    def __getitem__(self, index):
        self._verify_index(index)
        return self._models[index]

    def __iter__(self):
        return iter(self._models)
    
    @property
    def num_models(self):
        return len(self._models)

    @property
    def dimension(self):
        return self._dimension

    def get_adaptation_state(self):
        return self._adaptation_state

    def set_adaptation_state(self, new_state):
        if new_state not in self.ADAPTATION_STATES:
            raise ValueError("Unknown adaptation state: %s" % (new_state,))
        elif new_state not in self._LEGAL_TRANSITION_MAP[self._adaptation_state]:
            raise ValueError("Illegal adaptation state transition: from %s to %s" %
                             (self._adaptation_state, new_state))
        self._adaptation_state = new_state


    def _verify_index(self, index):
        assert type(index) == type(0)
        assert index >= 0
        if index >= self.num_models:
            raise IndexError("index was %d, but HmmMgr has only %d models" % (index, len(self._models)))        


    def _require_state(self, state):
        if self._adaptation_state != state:
            raise RuntimeError("Not in %s state" % (state,))



class GmmMgr(ModelManager):
    """
    A container and manager for underlying Gmms.

    E.g., Hmm models use this class to hold the models in each of their states.
    Models must all have the same covariance type.  Accumulation and adaptation
    is handled here.
    
    >>> dim = 12
    >>> covar_type = GaussianModelBase.DIAGONAL_COVARIANCE
    >>> num_comps = 3
    >>> gmm0 = GaussianMixtureModel(dim, covar_type, num_comps)
    >>> gmm1 = gmm0.copy()

    GmmMgrs can be created empty:

    >>> gmm_mgr0 = GmmMgr(dim)
    >>> print gmm_mgr0.to_string()
    GmmMgr with no models (dimension 12)
    
    Or with an iterable of models:
    
    >>> gmm_mgr1 = GmmMgr((gmm0, gmm1))
    >>> print gmm_mgr1.to_string()
    GmmMgr with 2 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 12, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)
    
    Or from another GmmMgr:
    
    >>> gmm_mgr2 = GmmMgr(gmm_mgr1)
    >>> print gmm_mgr2.to_string()
    GmmMgr with 2 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 12, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)

    Or from an iterable of component numbers and other specifications:
    
    >>> num_comp_tuple = (2, 3, 2, 4)
    >>> gmm_mgr3 = GmmMgr(num_comp_tuple, dim, covar_type)
    >>> print gmm_mgr3.to_string()
    GmmMgr with 4 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 12, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)
    
    The other specifications may include primers:

    >>> m0 = SimpleGaussianModel(dim, covar_type)
    >>> m0.set_means(xrange(dim))
    >>> m0.set_vars([1.0] * dim)
    >>> m1 = SimpleGaussianModel(dim, covar_type)
    >>> m1.set_means(xrange(dim, 0, -1))
    >>> m1.set_vars([1.0] * dim)
    >>> primers = (m0, m1, m1, m0)
    >>> gmm_mgr3 = GmmMgr(num_comp_tuple, dim, covar_type, primers)
    >>> print gmm_mgr3.to_string(full=True)
    GmmMgr with 4 models of type <class 'onyx.am.gaussian.GaussianMixtureModel'>
      (dimension 12, covariance_type onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE)
    
    """

    def __init__(self, *args):
        """
        Initialization can take four forms.  First, one int argument constructs
        an empty GmmMgr for models with the given dimension (number of
        features).  Second, another GmmMgr can be passed in, in which case the
        new GmmMgr is a deep copy of the argument.  Third, an iterable of models
        can be passed in, in which case the new GmmMgr will have those models,
        in the order iterated.  In this case, the iterable should return
        instances of either GaussianMixtureModel or DummyModel instances, and
        models must all have the same covariance type.  Finally, arguments may
        be provided in the form (num_comps, dimension, covar_type priming=None),
        that is, with 3 or 4 arguments, where num_comps is an iterable of the
        number of components for each model, dimension is a positive integer,
        and covar_type is either SimpleGaussianModel.DIAGONAL_COVARIANCE or
        SimpleGaussianModel.FULL_COVARIANCE .  New GaussianMixtureModels will be
        created for each element returned by num_comps.  Priming, if it is
        provided, is an iterable of SimpleGaussianModels which will be used to
        initialize all the components of each model, so the priming argument
        must be as long as the num_comps argument, and the priming models should
        have the same covariance type as covar_type.
        """
        
        if not len(args) in set((1, 3, 4)):
            raise ValueError("expected 1, 3, or 4 arguments, but got %d" % (len(args),))

        
        self._models = list()
        self._covariance_type = None
        self._accums = dict()

        if len(args) == 1:
            if isinstance(args[0], GmmMgr):
                other = args[0]
                super(GmmMgr, self).__init__(other.dimension)
                for model in other:
                    self._models.append(model.copy())
                self._covariance_type = other._covariance_type
            elif isinstance(args[0], int):
                super(GmmMgr, self).__init__(args[0])
            else:
                models = tuple(args[0])
                if len(models) == 0:
                    raise ValueError("can't construct from an empty iterable")
                super(GmmMgr, self).__init__(models[0].dimension)
                self.add_models(models)

        else:
            assert(3 <= len(args) <= 4)
            num_comps, dimension, covar_type = args[0], args[1], args[2]
            super(GmmMgr, self).__init__(dimension)
            num_comps = tuple(num_comps)
            priming = tuple(args[3]) if len(args) == 4 else None
            if priming is not None:
                if len(priming) < len(num_comps):
                    raise ValueError("not enough priming models were provided - expected %d but got %d" %
                                     (len(num_comps), len(priming)))
            else:
                priming = repeat(None)
            for nc, primer in izip(num_comps, priming):
                self._models.append(GaussianMixtureModel(dimension, covar_type, nc, primer))
            self._covariance_type = covar_type


    def __eq__(self, other):
        if ((self._covariance_type != other._covariance_type) or
            (self.dimension != other.dimension) or
            (len(self._models) != len(other._models)) or
            any((m1 != m2 for m1,m2 in izip(self._models, other._models)))):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return self.to_string()
    
    def to_string(self, full=False):
        if self.num_models == 0:
            return "GmmMgr with no models (dimension %d)" % self.dimension
        else:
            return ("GmmMgr with %d models of type %s\n  (dimension %d, covariance_type %s)" %
                    (self.num_models, type(self._models[0]), self.dimension, self._covariance_type.name))

    def add_models(self, models):
        """
        Add some models to a manager.

        *models* should be an iterable of either GaussianMixtureModels or
        DummyModels, which must all have the same dimension and covariance type.
        """
        self._require_state("NOT_ADAPTING")
        models = tuple(models)
        if not all((isinstance(m, GaussianMixtureModel) or
                   isinstance(m, DummyModel)) for m in models):
            raise ValueError("Some model had unexpected type - all models should be GaussianMixtureModel or DummyModel")

        if not self._covariance_type:
            self._covariance_type = models[0].covariance_type if len(models) > 0 else None
        if not all((m.covariance_type == self._covariance_type for m in models)):
            raise ValueError("Some model had unexpected covariance type - all models should have one type")

        if not all((m.dimension == self.dimension for m in models)):
            raise ValueError("Some model had unexpected dimension - all models should have dimension %d" %
                             self.dimension)

        start = len(self._models)
        end = start + len(models)
        temp_models = list(self._models)
        temp_models += models
        self._models = tuple(temp_models)
        return range(start, end)

        

    def ensure_accumulators(self, models):
        """
        Make sure there are accumulators for the given models.  If they need to be created, they
        will also be cleared, but this call will have no effect on accumulators that already exist.
        This call can only be made when the adaptation state is INITIALIZING.  Note that this call
        can be made more than once with overlapping sets of models or with the same model occurring
        more than once in a single call.
        """
        self._require_state("INITIALIZING")
        for mi in frozenset(models):
            self._verify_index(mi)
            if not self._accums.has_key(mi):
                m = self.get_model(mi)
                self._accums[mi] = self.GmmAccumSet(m.num_components, m.dimension, self._covariance_type)
                self._accums[mi].clear()


    def clear_all_accumulators(self):
        """
        Clear all existing accumulators.  This call can only be made when the adaptation state is
        INITIALIZING.
        """
        self._require_state("INITIALIZING")
        for mi,accum in self._accums.items():
            accum.clear()


    def apply_all_accumulators(self):
        """
        Apply all active accumulators.  This call can only be made when the adaptation state is
        APPLYING.
        """
        self._require_state("APPLYING")
        for mi in self._accums.keys():
            self._apply_one_accum_set(mi)


    def accum_sequence(self, mi, gamma, comp_scores, seq):
        """
        Accumulate for a sequence of datapoints.  mi is a model index; gamma is a Numpy array of
        shape (N+1,) where N = len(seq).  comp_scores is a list of length N containing .  seq is an
        iterable of observations in the form of Numpy arrays.  This call can only be made when the
        adaptation state is ACCUMULATING.
        """
        dc = dcheck("mm_as")
        self._verify_index_for_accum(mi)
        self._require_state("ACCUMULATING")
        seq = tuple(seq)
        assert len(seq) == len(gamma) - 1
        assert len(seq) == len(comp_scores)
        dc and dc("Accumulating sequence: %s\n  gamma = %s\n  comp_scores = %s" % (seq, gamma, comp_scores))
        self._accums[mi].accum_sequence(gamma, comp_scores, seq)

    def accum_one_frame(self, mi, gamma, comp_scores, frame):
        """
        Accumulate for a single datapoint.  This call can only be made when the adaptation state is
        ACCUMULATING
        """
        self._verify_index_for_accum(mi)
        self._require_state("ACCUMULATING")
        self._accums[mi].accum_one_frame(gamma, comp_scores, frame)

    def get_norm_accum(self, mi):
        """
        Get the value of the normalizing accumulator for a model.  This call can only be made when
        the adaptation state is APPLYING.
        """
        self._verify_index_for_accum(mi)
        self._require_state("APPLYING")
        return self._accums[mi].norm_accum


################### INTERNAL FUNCTIONS #################

    class GmmAccumSet(object):
        def __init__(self, num_comps, dim, covariance_type):
            assert covariance_type in GaussianModelBase.LEGAL_COVARIANCE_TYPES
            self._covariance_type = covariance_type
            self._num_components = num_comps
            self._dimension = dim
            self._num_frames = 0
            self.norm_accum = 0.0
            self.zeroth_accum = zeros((num_comps), dtype=float)
            self.first_accum = zeros((num_comps, dim), dtype=float)
            if self._covariance_type is GaussianModelBase.FULL_COVARIANCE:
                self.second_accum = zeros((num_comps, dim, dim), dtype=float)
            else:
                assert (self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE or
                        self._covariance_type is GaussianModelBase.DUMMY_COVARIANCE)
                self.second_accum = zeros((num_comps, dim), dtype=float)
                

        def __str__(self):
            ret = "GmmAccumSet - num_components = %d, dimension = %d, covariance_type = %s\n" % (self.num_components,
                                                                                                 self.dimension,
                                                                                                 self._covariance_type)
            ret += " num_frames_accumulated = %d, norm_accum = %s\n" % (self.num_frames_accumulated,
                                                                        self.norm_accum)
            ret += " Accums:\n"
            ret += "  Zeroth: %s\n" % (self.zeroth_accum,)
            ret += "  First: %s\n" % (self.first_accum,)
            ret += "  Second: %s\n" % (self.second_accum,)
            return ret

        @property
        def num_components(self):
            return self._num_components

        @property
        def dimension(self):
            return self._dimension

        @property
        def num_frames_accumulated(self):
            return self._num_frames

        def clear(self):
            self._num_frames = 0
            self.norm_accum = 0.0
            # This seems to be the approved way to zero out a numpy array.
            self.zeroth_accum[:] = 0.0 
            self.first_accum[:] = 0.0 
            self.second_accum[:] = 0.0 

        def accum_one_frame(self, gamma, comp_scores, frame):
            comp_gammas = (gamma * comp_scores) / comp_scores.sum()
            self._num_frames += 1
            self.norm_accum += gamma
            self.zeroth_accum += comp_gammas
            self.first_accum += comp_gammas[:, newaxis] * frame
            if self._covariance_type is GaussianModelBase.FULL_COVARIANCE:
                self.second_accum += comp_gammas[:, newaxis, newaxis] * numpy.outer(frame, frame)
            else:
                assert (self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE or
                        self._covariance_type is GaussianModelBase.DUMMY_COVARIANCE)
                self.second_accum += comp_gammas[:, newaxis] * (frame**2)

        def accum_one_frame_XXX(self, gamma, comp_scores, frame):
            comp_gammas = (gamma * comp_scores) / comp_scores.sum()
            self._num_frames += 1
            self.norm_accum += gamma
            self.zeroth_accum += comp_gammas
            temp = comp_gammas[:, newaxis] * frame
            self.first_accum += temp
            if self._covariance_type is GaussianModelBase.FULL_COVARIANCE:
                self.second_accum += comp_gammas[:, newaxis, newaxis] * numpy.outer(frame, frame)
            else:
                assert (self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE or
                        self._covariance_type is GaussianModelBase.DUMMY_COVARIANCE)
                self.second_accum += temp * frame

    
        def accum_sequence(self, gamma, comp_scores, seq):
            for g, c, d in izip(iter(gamma), iter(comp_scores), iter(seq)):
                self.accum_one_frame(g, c, d)


    def _verify_index_for_accum(self, index):
        self._verify_index(index)
        if not self._accums.has_key(index):
            raise ValueError("Accumulation requested without initialization for model index %d", (index,))
                    
        

    def _apply_one_accum_set(self, mi):
        dc = dcheck("mm_aoas")
        m, a = self._models[mi], self._accums[mi]
        dc and dc("For model %d, accums are %s" % (mi, a))
        if a.num_frames_accumulated == 0:
            return
        est_weights = a.zeroth_accum / a.norm_accum
        est_means = a.first_accum / a.zeroth_accum[:,newaxis]
        mu_sq = numpy.zeros_like(a.second_accum)
        for ci in xrange(a.num_components):
            if self._covariance_type is GaussianModelBase.FULL_COVARIANCE:
                mu_sq[ci] = numpy.outer(est_means[ci], est_means[ci])
            else:
                assert (self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE or
                        self._covariance_type is GaussianModelBase.DUMMY_COVARIANCE)
                mu_sq[ci] = est_means[ci] ** 2
                    
        if self._covariance_type is GaussianModelBase.FULL_COVARIANCE:
            divisor = a.zeroth_accum[:,newaxis,newaxis]
        else:
            assert (self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE or
                    self._covariance_type is GaussianModelBase.DUMMY_COVARIANCE)
            divisor = a.zeroth_accum[:,newaxis]

        dc and dc("mu_sq = %s" % (mu_sq,))
        div = a.second_accum / divisor
        dc and dc("div = %s" % (div,))
        est_vars =  div - mu_sq
        # XXX I don't know what the right thing to do here is for the full covariance case
        # Also, we may want to move this clipping to gaussian.py
        if self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
            MIN_DEV = 2.0e-20
            MAX_DEV = 2.0e+20
            est_vars.clip(min=MIN_DEV, max=MAX_DEV, out=est_vars)
        m.set_weights(est_weights)
        m.set_means(est_means)
        dc and dc("Estimated vars = %s" % (est_vars,))
        m.set_vars(est_vars)
            


# XXX Future: fill this class in to support sharing of Gaussians across mixtures, which Gmms
# currently do not support.
class GaussianMgr(ModelManager):
    """
    A container and manager for GuassianModels.
    """

    def __init__(self, models):
        """
        models is an iterable returning instances.  Models must all have the same covariance type.
        """
        self._models = tuple(models)
        all_true = [(isinstance(m, SimpleGaussianModel) or isinstance(m, DummyModel)) for m in self._models]
        if False in all_true:
            raise ValueError("Some model had unexpected type - all models should be SimpleGaussianModel or DummyModel")

        self._covariance_type = self._models[0].covariance_type if len(self._models) > 0 else None
        all_true = [m.covariance_type == self._covariance_type for m in self._models]
        if False in all_true:
            raise ValueError("Some model had unexpected covariance type - all models should have one type")


    def get_model(self, index):
        self._verify_index(index)
        return self._models[index]


    @property
    def num_models(self):
        return len(self._models)


# Serialization

def read_model_dict(file, yaml_reader=None):
    # Create YamldataReader if needed, hook up and do version checking
    if yaml_reader is None:
        yaml_reader = YamldataReader(file, stream_type=OnyxTextReader.STREAM_TYPE,
                                     stream_version=OnyxTextReader.STREAM_VERSION)

    # create OnyxTextReader and hook YamldataReader up to it
    stream = OnyxTextReader(yaml_reader, data_type="ModelDict", data_version="0")
    
    # read document contents
    v,num_models = stream.read_scalar("num_models", int)
    # Note that keys are always strings here
    v, keys = stream.read_list("model_keys", count=num_models)
    v, model_indices = stream.read_list("model_indices", int, count=num_models)
    return dict(izip(keys, model_indices))

def write_model_dict(model_dict, file):
    # create YamldataWriter on file
    yw = YamldataWriter(file, stream_type=OnyxTextWriter.STREAM_TYPE, stream_version=OnyxTextWriter.STREAM_VERSION)

    stream = OnyxTextWriter()

    hdr_gen = stream.gen_header("ModelDict", "0")
    nm_gen = stream.gen_scalar("num_models", len(model_dict))
    key_gen = stream.gen_list("model_keys", model_dict.keys())
    idx_gen = stream.gen_list("model_indices", model_dict.values())

    yw.write_document(chain(hdr_gen, nm_gen, key_gen, idx_gen))


def read_gmm_mgr(file, yaml_reader=None):
    """
    >>> dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4) )
    >>> gmm_mgr = GmmMgr(dummies)
    >>> doc = '''
    ... ---
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : 'OnyxText'
    ...   __onyx_yaml__stream_version : '0'
    ... -
    ...   # OnyxText header looks like this:
    ...   - stream_type OnyxText stream_version 0 data_type GmmManager data_version 0
    ...   # format for GmmManager
    ...   - GmmMgr IndexedCollection Gmm 1
    ...   - Gmm 0
    ...   - covariance_type Singleton onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE
    ...   - dimension 2
    ...   - num_components 3
    ...   - relevances List 3
    ...   - 0.0 0.0 0.0
    ...   - weights Array 1 3
    ...   - +(-0002)0x0000000000000 +(-0001)0x0000000000000 +(-0002)0x0000000000000
    ...   - Gaussians IndexedCollection SimpleGaussian 3
    ...   - SimpleGaussian 0
    ...   - means Array 1 2
    ...   - +(+0000)0x0000000000000 +(+0000)0x0000000000000
    ...   - vars Array 1 2
    ...   - +(+0000)0x0000000000000 +(+0000)0x0000000000000
    ...   - SimpleGaussian 1
    ...   - means Array 1 2
    ...   - +(+0001)0x0000000000000 +(+0001)0x0000000000000
    ...   - vars Array 1 2
    ...   - +(+0000)0x0000000000000 +(+0000)0x0000000000000
    ...   - SimpleGaussian 2
    ...   - means Array 1 2
    ...   - +(+0001)0x8000000000000 +(+0001)0x8000000000000
    ...   - vars Array 1 2
    ...   - +(+0000)0x0000000000000 +(+0000)0x0000000000000
    ... ...
    ...   '''
    >>> f = cStringIO.StringIO(doc)

    # If this works we have done lots of checking, but it would be nice to be able to print something out
    >>> gmm_mgr0 = read_gmm_mgr(f)

    # write-read-write round-trip test
    >>> f_out0 = cStringIO.StringIO()
    >>> f_out1 = cStringIO.StringIO()
    >>> write_gmm_mgr(gmm_mgr0, f_out0)

    # rewind and read
    >>> f_out0.seek(0)
    >>> gmm_mgr1 = read_gmm_mgr(f_out0)

    >>> gmm_mgr0 == gmm_mgr1
    True
    
    >>> write_gmm_mgr(gmm_mgr1, f_out1)
    >>> f_out0.getvalue() == f_out1.getvalue()
    True
    """
    
    # create and hook up YamldataReader, do version checking
    if yaml_reader is None:
        yaml_reader = YamldataReader(file, stream_type=OnyxTextReader.STREAM_TYPE, stream_version=OnyxTextReader.STREAM_VERSION)

    # create OnyxTextReader and hook YamldataReader up to it
    stream = OnyxTextReader(yaml_reader, data_type="GmmManager", data_version="0")
    
    # read document contents
    v,gmms = stream.read_indexed_collection(read_gmm, None, name="GmmMgr")
    
    # finalize Reader and Stream?
    # construct and return GmmMgr object
    return GmmMgr(gmms)

def read_gmm(stream, *_):
    v,covariance_type = stream.read_singleton("covariance_type")
    v,dimension = stream.read_scalar("dimension", int)
    v,num_components = stream.read_scalar("num_components", int)
    v,relevances = stream.read_list("relevances", float)
    v,gmm_weights = stream.read_array("weights", rtype=float, dim=1, shape=(num_components,))    
    v,smms = stream.read_indexed_collection(read_smm, (covariance_type, num_components, dimension), name="Gaussians")
    
    gmm_means = numpy.zeros((num_components,dimension), dtype=float)
    if covariance_type is GaussianMixtureModel.FULL_COVARIANCE:
        var_shape = (num_components, dimension, dimension)
    else:
        assert(covariance_type is GaussianMixtureModel.DIAGONAL_COVARIANCE)
        var_shape = (num_components, dimension)
    gmm_vars = numpy.zeros(var_shape, dtype=float)

    assert(len(smms) == num_components)
    for i in xrange(num_components):
        gmm_means[i] = smms[i].means
        gmm_vars[i] = smms[i].vars
        
    # Construct and return Gmm object
    ret = GaussianMixtureModel(dimension, covariance_type, num_components)
    ret.set_weights(gmm_weights)
    ret.set_means(gmm_means)
    ret.set_vars(gmm_vars)
    ret.set_relevances(relevances)
    return ret


def read_smm(stream, params, _):
    covariance_type, num_components, dimension = params
    v,smm_means = stream.read_array("means", rtype=float, dim=1, shape=(dimension,))
    if covariance_type is GaussianMixtureModel.FULL_COVARIANCE:
        var_dim = 2
        var_shape = (dimension, dimension)
    else:
        assert(covariance_type is GaussianMixtureModel.DIAGONAL_COVARIANCE)
        var_dim = 1
        var_shape = (dimension,)

    v,smm_vars = stream.read_array("vars", rtype=float, dim=var_dim, shape=var_shape)    
    ret = SimpleGaussianModel(dimension, covariance_type)
    ret.set_model(smm_means, smm_vars)
    return ret
             
def write_gmm_mgr(gmm_mgr, file):
    """
    >>> f = cStringIO.StringIO()
    >>> dim = 2
    >>> num_components = 3
    >>> weights = numpy.array((0.25, 0.5, 0.25), dtype=float)
    >>> mu = numpy.array(((1, 1), (2, 2), (3, 3)), dtype=float)
    >>> v = numpy.array(((1, 1), (1, 1), (1, 1)), dtype=float)
    >>> gmm0 = GaussianMixtureModel(dim, GaussianMixtureModel.DIAGONAL_COVARIANCE, num_components)
    >>> gmm0.set_weights(weights)
    >>> gmm0.set_means(mu)
    >>> gmm0.set_vars(v)
    >>> gmm0.set_relevances((10.0, 10.0, 10.0))

    >>> gmm1 = GaussianMixtureModel(dim, GaussianMixtureModel.DIAGONAL_COVARIANCE, num_components)
    >>> gmm1.set_weights(weights)
    >>> gmm1.set_means(mu)
    >>> gmm1.set_vars(v)

    >>> gmm_mgr0 = GmmMgr((gmm0, gmm1))
    >>> write_gmm_mgr(gmm_mgr0, f)
    >>> print f.getvalue()
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "OnyxText"
    - - stream_type OnyxText stream_version 0 data_type GmmManager data_version 0
      - GmmMgr IndexedCollection Gmm 2
      - Gmm 0
      - covariance_type Singleton onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE
      - dimension 2
      - num_components 3
      - relevances List 3
      - 10.0 10.0 10.0
      - weights Array 1 3
      - +(-0002)0x0000000000000 +(-0001)0x0000000000000 +(-0002)0x0000000000000
      - Gaussians IndexedCollection SimpleGaussian 3
      - SimpleGaussian 0
      - means Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - vars Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - SimpleGaussian 1
      - means Array 1 2
      - +(+0001)0x0000000000000 +(+0001)0x0000000000000
      - vars Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - SimpleGaussian 2
      - means Array 1 2
      - +(+0001)0x8000000000000 +(+0001)0x8000000000000
      - vars Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - Gmm 1
      - covariance_type Singleton onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE
      - dimension 2
      - num_components 3
      - relevances List 3
      - 0.0 0.0 0.0
      - weights Array 1 3
      - +(-0002)0x0000000000000 +(-0001)0x0000000000000 +(-0002)0x0000000000000
      - Gaussians IndexedCollection SimpleGaussian 3
      - SimpleGaussian 0
      - means Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - vars Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - SimpleGaussian 1
      - means Array 1 2
      - +(+0001)0x0000000000000 +(+0001)0x0000000000000
      - vars Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
      - SimpleGaussian 2
      - means Array 1 2
      - +(+0001)0x8000000000000 +(+0001)0x8000000000000
      - vars Array 1 2
      - +(+0000)0x0000000000000 +(+0000)0x0000000000000
    <BLANKLINE>

    # Round-trip test
    >>> f.seek(0)  # rewind
    >>> gmm_mgr1 = read_gmm_mgr(f)

    >>> gmm_mgr0 == gmm_mgr1
    True
    """
    # create YamldataWriter on file
    yw = YamldataWriter(file, stream_type=OnyxTextWriter.STREAM_TYPE, stream_version=OnyxTextWriter.STREAM_VERSION)

    stream = OnyxTextWriter()

    hdr_gen = stream.gen_header("GmmManager", "0")
    # The first arg is the collection name, the second arg is the object name, the third arg gives
    # the objects to be written and the fourth arg is called with each object and returns a
    # generator of tuples to be written for that object.  The generator must yield at least one
    # tuple, which will be written as part of the name/index line; subsequent tuples go on their own
    # line.
    all_gmms = [gmm for gmm in gmm_mgr]
    mgr_gen = stream.gen_indexed_collection("GmmMgr", "Gmm", all_gmms, gen_gmm)
 
    yw.write_document(chain(hdr_gen, mgr_gen))


def gen_gmm(stream, gmm):
    return chain(repeat((), 1),  # We don't add anything to the name/index line
                 stream.gen_singleton("covariance_type", gmm.covariance_type),
                 stream.gen_scalar("dimension", gmm.dimension),
                 stream.gen_scalar("num_components", gmm.num_components),
                 stream.gen_list("relevances", gmm.relevances),
                 stream.gen_array("weights", gmm.weights),
                 stream.gen_indexed_collection("Gaussians", "SimpleGaussian", gmm.models, gen_smm))

def gen_smm(stream, smm):
    return chain(repeat((), 1),  # We don't add anything to the name/index line
                 stream.gen_array("means", smm.means),
                 stream.gen_array("vars", smm.vars))



if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

