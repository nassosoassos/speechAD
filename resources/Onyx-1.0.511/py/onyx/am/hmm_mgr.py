###########################################################################
#
# File:         hmm_mgr.py (directory: ./py/onyx/am)
# Date:         Wed 13 Aug 2008 10:20
# Author:       Ken Basye
# Description:  Manager for Hmms
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
    A manager for Hmms.  Also read and write functions for acoustic models.
    
    Here's a simple round-trip test starting from an HTK model file. 

    XXX need to make this use build dir when SCons is running things

    >>> model_dirname = path.normpath(path.join(onyx.home, 'py/onyx/htkfiles'))
    >>> model_filename = path.join(model_dirname, 'monophones.mmf')

    >>> with open(model_filename) as f:
    ...     model_dict, hmm_mgr, gmm_mgr = htkmmf.read_htk_mmf_file(f)

    Write this model out to a file

    >>> with opentemp('wb', suffix='.tmp', prefix='onyx_hmm_mgr_test_') as (filename, outfile):
    ...     write_acoustic_model(model_dict, gmm_mgr, hmm_mgr, outfile)
    
    Now read it back in

    >>> with open(filename, 'rb') as infile:
    ...     model_dict2, gmm_mgr2, hmm_mgr2 = read_acoustic_model(infile, log_domain=True)
    
    >>> os.remove(filename)

    Now compare

    >>> model_dict == model_dict2
    True
    >>> gmm_mgr == gmm_mgr2
    True
    >>> hmm_mgr == hmm_mgr2
    True
    
"""
from __future__ import with_statement
import onyx
import numpy
import os, cStringIO
from itertools import izip, chain, repeat
from numpy import array, newaxis, zeros
from os import path
from onyx.am.gaussian import DummyModel
from onyx.am.hmm import Hmm
from onyx.am.modelmgr import ModelManager, GmmMgr, read_gmm_mgr, write_gmm_mgr, read_model_dict, write_model_dict
from onyx.textdata.onyxtext import OnyxTextReader, OnyxTextWriter
from onyx.textdata.yamldata import YamldataReader, YamldataWriter, YamldataGenerator
from onyx.util import opentemp
from onyx.util.debugprint import dcheck
from onyx.htkfiles import htkmmf

class HmmMgr(ModelManager):
    """
    A container and manager for Hmms.

    E.g., TrainingGraphs use this class to hold the models which comprise the
    graph.  Models must all have the same covariance type.  Accumulation and
    adaptation of transition probabilities is handled here.
    """

    def __init__(self, arg):
        """
        Initialization can take three forms.  First, an int argument constructs
        an empty HmmMgr for models with the given dimension (number of
        features).  Second, another HmmMgr can be passed in, in which case the
        new HmmMgr is a deep copy of the argument.  Third, an iterable of models
        can be passed in, in which case the new HmmMgr will have those models,
        in the order iterated.  In this case, the iterable should return
        instances of Hmm.
        """
        
        self._epsilon_model_map = dict()
        self._adaptation_state = "NOT_ADAPTING"
        self._models = list()
        self._gmm_mgr_set = set()
        self._log_domain = None

        if isinstance(arg, HmmMgr):
            super(HmmMgr, self).__init__(arg.dimension)
            for model in arg:
                self._models.append(model.copy())
            self._gmm_mgr_set.update(arg._gmm_mgr_set)
            self._log_domain = arg.log_domain
        elif isinstance(arg, int):
            super(HmmMgr, self).__init__(arg)
        else:
            models = tuple(arg)
            if len(models) == 0:
                raise ValueError("can't construct from an empty iterable")
            super(HmmMgr, self).__init__(models[0].dimension)
            self.add_models(models)
            


    def __str__(self):
        return self.to_string()
    
    def to_string(self, full=False):
        if len(self._models) == 0:
            return "HmmMgr with no models"
        else:
            ret = "HmmMgr with %d models" % (self.num_models,)
        return ret


    def __eq__(self, other):
        if ((len(self._models) != len(other._models)) or
            any((m1 != m2 for m1,m2 in izip(self._models, other._models)))):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def log_domain(self):
        return self._log_domain

    @property
    def input_arity_set(self):
        return set(m.num_inputs for m in self._models)

    @property
    def output_arity_set(self):
        return set(m.num_outputs for m in self._models)

    def add_models(self, models):
        """
        Add some models to this manager
        
        *models* is an iterable returning instances of Hmm.  Returns the indices
        of the models added.  This call can only be made when the adaptation
        state is NOT_ADAPTING.
        """
        self._require_state("NOT_ADAPTING")
        models = tuple(models)
        if len(models) == 0: return
        # NB: If you get this error for reasons you don't understand, make sure you are importing
        # Hmm from onyx.am.hmm and not just hmm - it matters.
        if any((not isinstance(m, Hmm) for m in models)):
            raise ValueError("Some model had unexpected type - all models should be Hmm")

        if not all((m.dimension == self.dimension for m in models)):
            raise ValueError("Some model had unexpected dimension - all models should have dimension %d" %
                             self.dimension)
        self._log_domain = models[0].log_domain
        if not all((m.log_domain == self.log_domain for m in models)):
            raise ValueError("Some model had unexpected log_domain - all models should have log_domain %d" %
                             self.log_domain)
        start = len(self._models)
        end = start + len(models)
        temp_models = list(self._models)
        temp_models += models
        self._models = tuple(temp_models)
        self._gmm_mgr_set.add(m.gmm_mgr for m in models)
        return range(start, end)

    def add_epsilon_model(self, gmm_mgr, arity, log_domain=False):
        """
        Add an epsilon model with the given arity if it doesn't already exist.
        
        arity is an int with value > 0.  Returns the index of the model added.
        This call can only be made when the adaptation state is NOT_ADAPTING.

        >>> hmm_mgr0 = HmmMgr(12)
        >>> hmm_mgr0.add_epsilon_model(None, 1)
        0
        
        >>> hmm_mgr0.add_epsilon_model(None, 2)
        1
        
        >>> hmm_mgr0.add_epsilon_model(None, 4)
        2

        >>> print hmm_mgr0
        HmmMgr with 3 models

        >>> hmm_mgr0.input_arity_set
        set([1, 2, 4])

        >>> hmm_mgr0.output_arity_set
        set([1, 2, 4])

        >>> idx = hmm_mgr0.get_epsilon_model_index(2)

        # >>> hmm_mgr0[idx].dot_display()
        
        """
        self._require_state("NOT_ADAPTING")
        if arity <= 0:
            raise ValueError("Expected arity to be > 0, but got %d" % (arity,))

        if self.has_epsilon_model(arity):
            return self.get_epsilon_model_index()
        
        epsilon_model = Hmm(0, log_domain)
        order = arity + 1
        epsilon_model.build_forward_model_compact(gmm_mgr, (), order, repeat((), order))

        temp_models = list(self._models)
        temp_models.append(epsilon_model)
        self._models = tuple(temp_models)
        new_index = len(self._models) - 1 
        self._epsilon_model_map[arity] = new_index
        return new_index

    def has_epsilon_model(self, arity):
        """
        See if this HmmMgr has an epsilon model with the given arity.

        """
        if arity <= 0:
            raise ValueError("Expected arity to be > 0, but got %d" % (arity,))

        return  self._epsilon_model_map.has_key(arity)


    def get_epsilon_model_index(self, arity):
        """
        Get the index of an epsilon model with the given arity or None if there isn't one.

        """
        return self._epsilon_model_map[arity] if self.has_epsilon_model(arity) else None


    def ensure_accumulators(self, models):
        """
        Construct and initialize accumulators for the given models if they don't already have them.
        If a model has accumulators, this call doesn't do anything.  This call can only be made when
        the adaptation state is INITIALIZING.
        """
        self._require_state("INITIALIZING")
        for mi in models:
            self._verify_index(mi)
            self.get_model(mi).begin_adapt("NETWORK") # XXX Either make both modes of adaptation
                                                       # available from the Mgr or do away with the
                                                       # difference.
    def clear_all_accumulators(self):
        """
        Clear the accumulators for all models.  This call can only be made when the adaptation state
        is INITIALIZING.
        """
        self._require_state("INITIALIZING")
        for m in self._models:
            m.clear_accumulators()


    def apply_all_accumulators(self):
        """
        Apply the accumulators for all models.  This call can only be made when the adaptation state
        is APPLYING.
        """
        self._require_state("APPLYING")
        for m in self._models:
            m.end_adapt()



def read_hmm_mgr(file, gmm_mgr, log_domain=False, yaml_reader=None):
    """
    Read a serialized HmmMgr object in from file.  gmm_mgr must be a GmmMgr with models
    corresponding to all those used in the HmmMgr's Hmms.  If yaml_reader is not None, it will be
    used as the source of data, in which case file will be ignored completely.
    
    >>> dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4) )
    >>> gmm_mgr = GmmMgr(dummies)
    >>> doc = '''
    ... ---
    ... - __onyx_yaml__meta_version : '1'
    ...   __onyx_yaml__stream_type : 'OnyxText'
    ...   __onyx_yaml__stream_version : '0'
    ... -
    ...   # OnyxText header looks like this:
    ...   - stream_type OnyxText stream_version 0 data_type HmmManager data_version 0
    ...   # format for HmmManager
    ...   - HmmMgr IndexedCollection Hmm 1
    ...   - Hmm 0
    ...   - num_inputs 1
    ...   - num_states 1
    ...   - num_outputs 1
    ...   - transition_matrix Array 2 3 3
    ...   - +(-1023)0x0000000000000 +(+0000)0x0000000000000 +(-1023)0x0000000000000
    ...   - +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000
    ...   - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
    ...   - models List 1
    ...   - 0
    ... ...
    ...   '''
    >>> f = cStringIO.StringIO(doc)
    >>> hmm_mgr = read_hmm_mgr(f, gmm_mgr)
    """
    
    # Create YamldataReader if needed, hook up and do version checking
    if yaml_reader is None:
        yaml_reader = YamldataReader(file, stream_type=OnyxTextReader.STREAM_TYPE,
                                     stream_version=OnyxTextReader.STREAM_VERSION)

    # create OnyxTextReader and hook YamldataReader up to it
    stream = OnyxTextReader(yaml_reader, data_type="HmmManager", data_version="0")
    
    # read document contents
    v,hmms = stream.read_indexed_collection(read_hmm, (gmm_mgr, log_domain), name="HmmMgr")
    
    # finalize Reader and Stream?
    # construct and return HmmMgr object
    return HmmMgr(hmms)


def read_hmm(stream, user_data, header_tokens):
    gmm_mgr, log_domain = user_data
    v,num_inputs = stream.read_scalar("num_inputs", int)
    v,num_states = stream.read_scalar("num_states", int)
    v,num_outputs = stream.read_scalar("num_outputs", int)
    s = num_inputs + num_states + num_outputs
    v,trans_array = stream.read_array("transition_matrix", rtype=float, dim=2, shape=(s,s))    
    v,models = stream.read_list("models", rtype=int, count=num_states)
    # Construct and return Hmm object
    ret = Hmm(num_states, log_domain=log_domain)
    ret.build_model(gmm_mgr, models, num_inputs, num_outputs, trans_array)
    return ret

def write_hmm_mgr(hmm_mgr, file):
    """
    >>> f = cStringIO.StringIO()
    >>> dummies = ( DummyModel(2, 0.1), DummyModel(2, 0.2), DummyModel(2, 0.4), DummyModel(2, 0.4) )
    >>> mm = GmmMgr(dummies)
    >>> models = range(3)
    
    >>> hmm0 = Hmm(3)
    >>> hmm0.build_forward_model_compact(mm, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    >>> hmm1 = Hmm(3)
    >>> hmm1.build_forward_model_compact(mm, models, 2, ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    >>> hmm_mgr0 = HmmMgr((hmm0, hmm1))
    >>> write_hmm_mgr(hmm_mgr0, f)
    >>> print f.getvalue()
    ---
    - __onyx_yaml__stream_version: "0"
      __onyx_yaml__meta_version: "1"
      __onyx_yaml__stream_type: "OnyxText"
    - - stream_type OnyxText stream_version 0 data_type HmmManager data_version 0
      - HmmMgr IndexedCollection Hmm 2
      - Hmm 0
      - num_inputs 1
      - num_states 3
      - num_outputs 1
      - transition_matrix Array 2 5 5
      - +(-1023)0x0000000000000 +(+0000)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
      - +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
      - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000 +(-1023)0x0000000000000
      - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000
      - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
      - models List 3
      - 0 1 2
      - Hmm 1
      - num_inputs 1
      - num_states 3
      - num_outputs 1
      - transition_matrix Array 2 5 5
      - +(-1023)0x0000000000000 +(+0000)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
      - +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
      - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000 +(-1023)0x0000000000000
      - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-0001)0x0000000000000 +(-0001)0x0000000000000
      - +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000 +(-1023)0x0000000000000
      - models List 3
      - 0 1 2
    <BLANKLINE>

    # Round-trip test
    >>> f.seek(0)  # rewind
    >>> hmm_mgr1 = read_hmm_mgr(f, mm)

    >>> hmm_mgr0 == hmm_mgr1
    True
    """
    # create YamldataWriter on file
    yw = YamldataWriter(file, stream_type=OnyxTextWriter.STREAM_TYPE, stream_version=OnyxTextWriter.STREAM_VERSION)

    stream = OnyxTextWriter()

    hdr_gen = stream.gen_header("HmmManager", "0")
    # The first arg is the collection name, the second arg is the object name, the third arg gives
    # the objects to be written and the fourth arg is called with each object and returns a
    # generator of tuples to be written for that object.  The generator must yield at least one
    # tuple, which will be written as part of the name/index line; subsequent tuples go on their own
    # line.
    all_hmms = [hmm for hmm in hmm_mgr]
    mgr_gen = stream.gen_indexed_collection("HmmMgr", "Hmm", all_hmms, gen_hmm)
 
    yw.write_document(chain(hdr_gen, mgr_gen))


    
def gen_hmm(stream, hmm):
    assert hmm.num_states == len(hmm.models)
    return chain(repeat((), 1),  # We don't add anything to the name/index line
                 stream.gen_scalar("num_inputs", hmm.num_inputs),
                 stream.gen_scalar("num_states", hmm.num_states),
                 stream.gen_scalar("num_outputs", hmm.num_outputs),
                 stream.gen_array("transition_matrix", hmm.transition_matrix),
                 stream.gen_list("models", hmm.models))



def write_acoustic_model(model_dict, gmm_mgr, hmm_mgr, file):
    write_model_dict(model_dict, file)
    write_gmm_mgr(gmm_mgr, file)
    write_hmm_mgr(hmm_mgr, file)


def read_acoustic_model(file, log_domain=False):
    gen = YamldataGenerator(file)
    model_dict_yaml_reader = YamldataReader(gen, stream_type=OnyxTextReader.STREAM_TYPE,
                                       stream_version=OnyxTextReader.STREAM_VERSION)
    model_dict = read_model_dict(None, model_dict_yaml_reader)

    gmm_yaml_reader = YamldataReader(gen, stream_type=OnyxTextReader.STREAM_TYPE,
                                     stream_version=OnyxTextReader.STREAM_VERSION)
    gmm_mgr = read_gmm_mgr(None, yaml_reader=gmm_yaml_reader)

    hmm_yaml_reader = YamldataReader(gen, stream_type=OnyxTextReader.STREAM_TYPE,
                                     stream_version=OnyxTextReader.STREAM_VERSION)
    hmm_mgr = read_hmm_mgr(None, gmm_mgr, log_domain=log_domain, yaml_reader=hmm_yaml_reader)
    return model_dict, gmm_mgr, hmm_mgr


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
