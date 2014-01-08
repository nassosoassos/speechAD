###########################################################################
#
# File:         gaussian.py
# Date:         31-Oct-2007
# Author:       Ken Basye
# Description:  Simple Gaussian models
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 The Johns Hopkins University
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
Simple Gaussian models and mixtures

This module provides the classes GaussianModelBase, DummyModel,
SimpleGaussianModel, and GaussianMixtureModel.
"""

from __future__ import division
import types, random
from itertools import izip
import numpy as np

from onyx.util.floatutils import float_to_readable_string
from onyx.util.debugprint import dcheck
from onyx.util.singleton import Singleton
from onyx.util.safediv import safely_divide_float_array
from onyx.util.mathutils import distance, find_cumulative_index


class GaussianModelBase(object):
    """
    >>> m = GaussianModelBase(4, GaussianModelBase.DUMMY_COVARIANCE)
    >>> try:
    ...    m.dimension = 5
    ... except AttributeError:
    ...    print "OK, dimension not settable"
    ... else:
    ...    print "Problem! dimension was settable"
    OK, dimension not settable
    >>> try:
    ...    m.covariance_type = GaussianModelBase.FULL_COVARIANCE
    ... except AttributeError:
    ...    print "OK, covariance_type not settable"
    ... else:
    ...    print "Problem! covariance_type was settable"
    OK, covariance_type not settable
    """
    REASONABLE_NUMERIC_TYPES = frozenset(np.dtype(typ) for typ in ('object', 'int', 'float',
                                                                       'int32', 'float32', 'float64'))

    DUMMY_COVARIANCE = Singleton("onyx.am.gaussian.GaussianModelBase.DUMMY_COVARIANCE")
    FULL_COVARIANCE = Singleton("onyx.am.gaussian.GaussianModelBase.FULL_COVARIANCE")
    DIAGONAL_COVARIANCE = Singleton("onyx.am.gaussian.GaussianModelBase.DIAGONAL_COVARIANCE")
    LEGAL_COVARIANCE_TYPES = frozenset((DUMMY_COVARIANCE, FULL_COVARIANCE, DIAGONAL_COVARIANCE))
    COVARIANCE_TYPE_NAMES = dict(((DUMMY_COVARIANCE, "dummy"),
                                  (FULL_COVARIANCE, "full"),
                                  (DIAGONAL_COVARIANCE, "diagonal")))

    # Use single random number generator for the class.  If this is a
    # problem, it's trivial to make it a per-instance member
    rand = random.Random()
    def __init__(self, dimension, covariance_type):
        assert type(dimension) is int and dimension > 0
        if not covariance_type in self.LEGAL_COVARIANCE_TYPES:
            raise ValueError("expected a covariance type from %s, but got %s" %
                             (self.LEGAL_COVARIANCE_TYPES, covariance_type))
        self._dimension = dimension
        self._covariance_type = covariance_type
    
    @property
    def dimension(self):
        return self._dimension

    @property
    def covariance_type(self):
        return self._covariance_type
    
    def _verify_reasonable(self, candidate):
        assert len(candidate) == self.dimension
        assert np.dtype(type(candidate[0])) in self.REASONABLE_NUMERIC_TYPES
        return True

    @staticmethod
    def seed(seed=None):
        """
        Use a seed other than None for reproducible randomness in the sample() function.
        """
        GaussianModelBase.rand.seed(seed)

    def begin_adapting(self):
        raise NotImplementedError("classes derived from GaussianModelBase must implement this for themselves")

    def add_adaptation_data(self, data):
        raise NotImplementedError("classes derived from GaussianModelBase must implement this for themselves")

    def end_adapting(self):
        raise NotImplementedError("classes derived from GaussianModelBase must implement this for themselves")

class DummyModel(GaussianModelBase):
    """
    DummyModel - return a constant score

    >>> dm = DummyModel(3)
    >>> dm2 = DummyModel(3)
    >>> dm == dm2
    True
    >>> x = np.array([0, 0, 0])
    >>> dm.score(x)
    0.0
    >>> dm.set_value(3.141)
    >>> dm == dm2
    False
    >>> dm.score(x)
    3.141
    >>> dm.score_components(x)
    array([ 3.141])
    """
    def __init__(self, dimension, value = 0.0):
        super(DummyModel, self).__init__(dimension, self.DUMMY_COVARIANCE)
        self._value = value

    def copy(self):
        return DummyModel(self.dimension, self._value)

    def __str__(self):
        return "DummyModel (Dim = %d) always returning a score of %s" % (self.dimension, self._value)

    def __eq__(self, other):
        if (not isinstance(other, DummyModel) or
            (self.dimension != other.dimension) or
            (self._value != other._value)):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def num_components(self):
        return 1

    def set_value(self, v):
        self._value = v
        
    def score(self, x):
        self._verify_reasonable(x)
        return self._value

    def score_components(self, x):
        """
        Return a numpy vector of weight * likelihood products
        """
        self._verify_reasonable(x)
        return np.array([self._value], dtype = float)

    def set_weights(self, w):
        pass

    def set_means(self, m):
        pass

    def set_vars(self, v):
        pass

    def begin_adapting(self):
        pass

    def add_adaptation_data(self, data):
        pass

    def end_adapting(self):
        pass

    def copy(self):
        """
        Return a deep copy of this model
        """
        return DummyModel(self.dimension, self._value)


class SimpleGaussianModel(GaussianModelBase):
    """ 
    Simple Gaussian models

    >>> m1 = SimpleGaussianModel(2, GaussianModelBase.DIAGONAL_COVARIANCE)
    >>> m1.set_model([0.0, 0.0], [1.0, 1.0])
    >>> m2 = SimpleGaussianModel(2, GaussianModelBase.DIAGONAL_COVARIANCE)
    >>> m2.set_model([0.0, 0.0], [1.0, 1.0])
    >>> m1 == m2
    True
    >>> print m2
    SGM: (Type = diagonal, Dim = 2)  Means: 0.0000 0.0000    Vars: 1.0000 1.0000
    >>> m2.means
    array([ 0.,  0.])
    >>> m2.vars
    array([ 1.,  1.])
    >>> m2.setup_for_scoring()
    >>> s0 = m2.score([0.0, 0.0])
    >>> ls0 = m2.log_score([0.0, 0.0])
    >>> float_to_readable_string(s0)
    '+(-0003)0x45f306dc9c883'
    >>> float_to_readable_string(ls0)
    '+(+0000)0xd67f1c864beb4'
    >>> s1 = m2.score([-0., 0.1])
    >>> ls1 = m2.log_score([-0.0, 0.1])
    >>> float_to_readable_string(s1)
    '+(-0003)0x4452da5c54a72'
    >>> float_to_readable_string(ls1)
    '+(+0000)0xd7c6ca9ac6cc8'
    >>> d0 = np.log(s0) + ls0
    >>> d1 = np.log(s1) + ls1
    >>> float_to_readable_string(d0)
    '+(-1023)0x0000000000000'
    >>> float_to_readable_string(d1)
    '+(-1023)0x0000000000000'
    
    >>> m2.seed(0)
    >>> s0 = m2.sample()
    >>> s1 = m2.sample()
    >>> float_to_readable_string(s0[0])
    '-(-0003)0x788fe6e2f434f'
    >>> float_to_readable_string(s0[1])
    '+(-0005)0x0a46196bf10fd'
    >>> float_to_readable_string(s1[0])
    '+(-0001)0x65ccbe5021434'
    >>> float_to_readable_string(s1[1])
    '-(-0004)0x8ab9026ba02b8'
    >>> data = (s0, s1)
    >>> m2.adapt(data)
    >>> m1 == m2
    False
    >>> s0 = m2.score([0.0, 0.0])
    >>> float_to_readable_string(s0)
    '+(+0002)0x0b369542f517e'
    >>> s1 = m2.score([-0.0, 0.1])
    >>> float_to_readable_string(s1) #doctest: +ELLIPSIS
    '+(-0001)0x2921c332de77...'
    >>> print m2
    SGM: (Type = diagonal, Dim = 2)  Means: 0.2575 -0.0319    Vars: 0.1948 0.0042
    >>> print m2.full_string()
    SGM: (Type = diagonal, Dim = 2)  Means: +(-0002)0x07a8c49764360 -(-0005)0x0595f5b5a7a3a    Vars: +(-0003)0x8eed05646a1e9 +(-0008)0x101af0dd95558  recip = +(+0002)0x662bcaf94c16d  log_recip = +(+0000)0xb8dd5b2bb2947  var_inv: +(-0003)0x8eed05646a1e9 +(-0008)0x101af0dd95558

    >>> m3 = SimpleGaussianModel(2, GaussianModelBase.FULL_COVARIANCE)
    >>> m3.set_model([0.0, 0.0], [[1.0, 0.2], [0.3, 1.0]])
    >>> print m3     
    SGM: (Type = full, Dim = 2)  Means: 0.0000 0.0000    Vars:
     1.0000 0.2000
     0.3000 1.0000
    >>> data = (np.array((1, 10)), np.array((1.1, 1)), np.array((1.3, 8)))
    >>> m3.adapt(data)
    >>> print m3
    SGM: (Type = full, Dim = 2)  Means: 1.1333 6.3333    Vars:
     0.0156 -0.0111
     -0.0111 14.8889
    >>> m2.adapt(data)
    >>> print m2
    SGM: (Type = diagonal, Dim = 2)  Means: 1.1333 6.3333    Vars: 0.0156 14.8889

    >>> m2.set_relevances((10, 10))
    >>> data = (np.array((5, 5)), np.array((5, 5)), np.array((6, 6)))
    >>> m2.adapt(data)
    >>> print m2
    SGM: (Type = diagonal, Dim = 2)  Means: 1.5152 6.2424    Vars: 0.0343 13.5556
    >>> m3 = m2.copy()
    >>> print m3
    SGM: (Type = diagonal, Dim = 2)  Means: 1.5152 6.2424    Vars: 0.0343 13.5556
    
    """

    def __init__(self, dimension, covariance_type):
        super(SimpleGaussianModel, self).__init__(dimension, covariance_type)
        if self.covariance_type is GaussianModelBase.DUMMY_COVARIANCE:
            raise ValueError("dummy covariance type not allowed in this context")
        self._means = None
        self._vars = None
        self._var_inv = None
        self._denominator = 0.0
        self._two_pi_to_dim = pow((2 * np.pi), self.dimension)
        self._cholesky = None
        self._score_tuple = None
        self._mean_relevance_factor = 1.0
        self._var_relevance_factor = 1.0
        self._num_adaptation_frames = 0
        self._first_accum = np.zeros((dimension,), dtype=float)
        if self.covariance_type is GaussianModelBase.FULL_COVARIANCE:
            self._second_accum = np.zeros((dimension, dimension), dtype=float)
        else:
            assert self._covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE
            self._second_accum = np.zeros((dimension,), dtype=float)
                

    @property
    def means(self):
        return self._means

    @property
    def vars(self):
        return self._vars

    def copy(self):
        """
        Return a deep copy of this model

        """
        ret = SimpleGaussianModel(self.dimension, self.covariance_type)
        ret.set_model(self.means, self.vars)
        return ret

    def setup_for_scoring(self):
        if self._score_tuple is None:
            recip = 1 / self._denominator
            log_recip = np.log(recip)
            temp1 = np.empty(self.dimension, dtype=float)
            self._score_tuple = self._means, self._var_inv, recip, log_recip, self._var_recip_scaled, temp1

    def __str__(self):
        return self._to_string()
        
    def full_string(self):
        ret = self._to_string(float_to_readable_string)
        if self._score_tuple is not None:
            means, var_inv, recip, log_recip, var_recip_scaled, temp1 = self._score_tuple
            ret += "  recip = %s" % (float_to_readable_string(recip))
            ret += "  log_recip = %s" % (float_to_readable_string(log_recip))
            ret += "  var_inv:"
            ret += self._vars_to_string(self._vars, float_to_readable_string)
        return ret
            

    def _to_string(self, format_float = None):
        def format_mean_or_var(x):  return "%.4f" % (x,)
        if format_float is None:
            format_float = format_mean_or_var
        ret = "SGM: (Type = %s, Dim = %d)" % (self.COVARIANCE_TYPE_NAMES[self.covariance_type], self.dimension)
        ret += "  Means:"
        if self._means is not None:
            for m in self._means:
                ret += " " + format_float(m)
        else:
            ret += " Not set"
        ret += "    Vars:"
        if self._vars is not None:
            ret += self._vars_to_string(self._vars, format_float)
        else:
            ret += " Not set"
        return ret

    def _vars_to_string(self, arr, format_float):
        ret = ""
        if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
            for v in self._vars:
                ret += " " + format_float(v)
        else:
            for row in self._vars:
                ret += '\n'
                for v in row: 
                    ret += " " + format_float(v)
        return ret

    def __eq__(self, other):
        dc = dcheck("gmm_eq")
        if (self.dimension, self.covariance_type) != (other.dimension, other.covariance_type):
            dc and dc("dimensions or covariance_types are not eq")
            return False
        if (self.means != other.means).any():
            dc and dc("means are not eq: self.means = %s; other.means = %s, cmp = %s" %
                      (self.means, other.means, self.means == other.means))
            return False
        if (self.vars != other.vars).any():
            dc and dc("vars are not eq: self.vars = %s; other.vars = %s" % (self.vars, other.vars))
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def set_model(self, m=None, v=None):
        self._score_tuple = self._cholesky = None
        if m is not None:
            self.set_means(m)
        if v is not None:
            self.set_vars(v)

    def set_means(self, m, rel_factor=1.0):
        self._verify_reasonable(m)
        assert 0.0 <= rel_factor <= 1.0
        self._score_tuple = self._cholesky = None
        est_means = np.array(m, dtype=float)
        self._means = est_means if rel_factor == 1.0 else rel_factor * est_means + (1 - rel_factor) * self.means

    def set_vars(self, v, rel_factor=1.0):
        dc = dcheck("gaussian_numeric_error")
        self._verify_reasonable(v)
        assert 0.0 <= rel_factor <= 1.0
        self._score_tuple = self._cholesky = None
        if self.covariance_type is GaussianModelBase.FULL_COVARIANCE:
            assert all(len(x) == self.dimension for x in v)
            est_vars = np.array(v, dtype=float)
            self._vars = est_vars if rel_factor == 1.0 else rel_factor * est_vars + (1 - rel_factor) * self.vars
            try:
                self._var_recip_scaled = None
                self._var_inv = np.linalg.inv(self._vars)
                var_det = np.linalg.det(self._vars)
                self._denominator = np.sqrt(pow((2 * np.pi), self.dimension) * var_det) 
            except np.linalg.LinAlgError, e:
                dc and dc("Error setting full covar matrix: %s\nself._vars = %s" % (e, self._vars))
                raise e
        else:
            assert self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE
            est_vars = np.array(v, dtype=float)
            self._vars = est_vars if rel_factor == 1.0 else rel_factor * est_vars + (1 - rel_factor) * self.vars
            try:
                self._var_recip_scaled = -0.5 / self._vars
                self._var_inv = np.diagflat(1.0/self._vars)
                # Attempt to avoid underflow by using 128-bit floats here
                temp = np.array(self._vars, dtype=np.longfloat)
                var_det = temp.prod()  # product of elements
                # Must not have any 0s in this case
                assert var_det != 0
                # Now go back to regular float value here
                self._denominator = np.float(np.sqrt(self._two_pi_to_dim * var_det))
            except np.linalg.LinAlgError, e:
                dc and dc("Error setting diagonal var vector: %s\nself._vars = %s" % (e, self._vars))
                raise e
            
    def set_relevances(self, values):
        """
        Set relevances for adaptation - see adapt()

        *values* must be a tuple of two non-negative numbers, the first is used
        for means, the second for variances.
        """
        if len(values) != 2 or values[0] < 0 or values[1] < 0:
            raise ValueError("relevance values must be a tuple of two non-negative numbers")
        # Rather than keep the actual relevance values, we precompute the
        # factors we're going to need.  Since we have only one component, it's
        # kind of a stretch to call this 'relevance' adaptation, but it matches
        # what happens in one-component mixture models
        self._mean_relevance_factor = 1.0 / (1.0 + values[0])
        self._var_relevance_factor = 1.0 / (1.0 + values[1])
        
            
    def begin_adapting(self):
        self._num_adaptation_frames = 0
        self._first_accum[:] = 0.0
        self._second_accum[:] = 0.0

    def add_adaptation_data(self, data):
        data = tuple(data)
        num_frames = len(data)
        verify_reasonable = self._verify_reasonable
        if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
            square_op = np.multiply
        else:
            assert self.covariance_type is GaussianModelBase.FULL_COVARIANCE
            square_op = np.outer

        for frame in data:
            verify_reasonable(frame)
            self._first_accum += frame
            self._second_accum += square_op(frame, frame)

        self._num_adaptation_frames += num_frames

    def end_adapting(self):
        if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
            square_op = np.multiply
        else:
            assert self.covariance_type is GaussianModelBase.FULL_COVARIANCE
            square_op = np.outer

        est_means = self._first_accum / self._num_adaptation_frames
        est_vars = self._second_accum / self._num_adaptation_frames - square_op(est_means, est_means)

        self.set_means(est_means, self._mean_relevance_factor)
        self.set_vars(est_vars, self._var_relevance_factor)
        self.setup_for_scoring()

    def adapt(self, data):
        """
        Adjust model to fit data while taking current state into account.
        
        Uses relevance-based adaptation to balance the contribution of new data,
        *data* is a set of frames to train on; each frame is a numpy vector
        """
        self.begin_adapting()
        self.add_adaptation_data(data)
        self.end_adapting()
    
    def score(self, x):
        """ Return likelihood of *x* """
        assert self._verify_reasonable(x)
        assert self._score_tuple is not None, 'setup_for_scoring() needs to be called'
        means, var_inv, recip, log_recip, var_recip_scaled, temp1 = self._score_tuple
        # diff -> temp1
        np.subtract(x, means, temp1)
        if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
            # diff^2 -> temp1
            np.multiply(temp1, temp1, temp1)
            return np.exp(np.dot(temp1, var_recip_scaled)) * recip
        else:
            return np.exp(-0.5 * np.dot(np.dot(temp1, var_inv), temp1)) * recip

    def log_score(self, x):
        """ Find negative log-likelihood of *x* """
        assert self._verify_reasonable(x)
        assert self._score_tuple is not None, 'setup_for_scoring() needs to be called'
        means, var_inv, recip, log_recip, var_recip_scaled, temp1 = self._score_tuple
        diff = np.subtract(x, means)
        return 0.5 * np.dot(np.dot(diff, var_inv), diff) - log_recip

    def sample(self):
        assert self._means is not None
        # This is kind of expensive, so we only do it if needed.
        if self._cholesky is None:
            if self.covariance_type is GaussianModelBase.FULL_COVARIANCE:
                self._cholesky = np.linalg.cholesky(self._vars)
            else:
                assert self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE
                self._cholesky = np.linalg.cholesky(np.diagflat(self._vars))
        normalvariate = self.rand.normalvariate
        rand = np.fromiter((normalvariate(0,1) for x in range(self.dimension)), None)
        return self._means + np.dot(rand, self._cholesky)


# TODO:
#    better stopping criteria
#    better initialization

class GaussianMixtureModel(GaussianModelBase):
    """
    Gaussian mixture models

    >>> m0 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    >>> m1 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    >>> m0.set_weights(np.array((0.75, 0.25)))
    >>> m1.set_weights(np.array((0.75, 0.25)))
    >>> mu = np.array(((1, 1, 1), (3, 3, 3)))
    >>> m0.set_means(mu)
    >>> m1.set_means(mu)
    >>> v = np.array(((1, 1, 1), (1, 1, 1)))
    >>> m0.set_vars(v)
    >>> m1.set_vars(v)
    >>> print m0
    Gmm: (Type = diagonal, Dim = 3, NumComps = 2)
    Weights   Models
     0.7500     SGM: (Type = diagonal, Dim = 3)  Means: 1.0000 1.0000 1.0000    Vars: 1.0000 1.0000 1.0000
     0.2500     SGM: (Type = diagonal, Dim = 3)  Means: 3.0000 3.0000 3.0000    Vars: 1.0000 1.0000 1.0000
    >>> print m1
    Gmm: (Type = diagonal, Dim = 3, NumComps = 2)
    Weights   Models
     0.7500     SGM: (Type = diagonal, Dim = 3)  Means: 1.0000 1.0000 1.0000    Vars: 1.0000 1.0000 1.0000
     0.2500     SGM: (Type = diagonal, Dim = 3)  Means: 3.0000 3.0000 3.0000    Vars: 1.0000 1.0000 1.0000
    >>> m0 == m1
    True
    >>> s0 = m0.score([0,0,0])
    >>> float_to_readable_string(s0)    
    '+(-0007)0x5c2d69462ba21'
    >>> s1 = m0.score([1,1,1])
    >>> float_to_readable_string(s1)    
    '+(-0005)0x866d5e87388e3'
    >>> s2 = m0.score([2,2,2])
    >>> float_to_readable_string(s2)    
    '+(-0007)0xd03c4e0dff270'
    >>> comps = m0.score_components([2,2,2])
    >>> [float_to_readable_string(x) for x in comps]
    ['+(-0007)0x5c2d3a8a7f5d4', '+(-0009)0xd03c4e0dff270']
    >>> comps.sum() == s2
    True

    >>> m0.seed(0)
    >>> data0 = [m0.sample() for i in xrange(200)]
    >>> [float_to_readable_string(x) for x in data0[0]]
    ['+(+0001)0xe1c1f726d4fce', '+(+0001)0x13adff266bf7a', '+(+0001)0x1f71f9c100f8e']
    >>> m1 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    >>> m0 == m1
    False
    >>> print m1
    Gmm: (Type = diagonal, Dim = 3, NumComps = 2)
    Weights   Models
     0.5000     SGM: (Type = diagonal, Dim = 3)  Means: -0.3997 0.1183 0.9441    Vars: 0.7000 0.7000 0.7000
     0.5000     SGM: (Type = diagonal, Dim = 3)  Means: -1.0006 1.5618 -0.7573    Vars: 0.7000 0.7000 0.7000
    >>> m1.train(data0)
    >>> print m1
    Gmm: (Type = diagonal, Dim = 3, NumComps = 2)
    Weights   Models
     0.4231     SGM: (Type = diagonal, Dim = 3)  Means: 2.0489 2.3079 2.2403    Vars: 1.9840 1.7728 1.2277
     0.5769     SGM: (Type = diagonal, Dim = 3)  Means: 0.6909 0.9010 0.8748    Vars: 0.8548 0.7503 0.8175
    >>> m2 = GaussianMixtureModel(3, GaussianModelBase.FULL_COVARIANCE, 2)
    >>> print m2
    Gmm: (Type = full, Dim = 3, NumComps = 2)
    Weights   Models
     0.5000     SGM: (Type = full, Dim = 3)  Means: 0.7995 -0.5587 0.0207    Vars:
     0.7000 0.0000 0.0000
     0.0000 0.7000 0.0000
     0.0000 0.0000 0.7000
     0.5000     SGM: (Type = full, Dim = 3)  Means: -0.1678 1.9985 -1.6805    Vars:
     0.7000 0.0000 0.0000
     0.0000 0.7000 0.0000
     0.0000 0.0000 0.7000
    >>> data1 = [m0.sample() for i in xrange(800)]
    >>> m2.train(data1)
    >>> print m2
    Gmm: (Type = full, Dim = 3, NumComps = 2)
    Weights   Models
     0.1912     SGM: (Type = full, Dim = 3)  Means: 0.2810 1.3328 0.9262    Vars:
     0.3892 -0.1290 -0.0582
     -0.1290 0.6084 -0.1546
     -0.0582 -0.1546 0.7614
     0.8088     SGM: (Type = full, Dim = 3)  Means: 1.7744 1.5895 1.6766    Vars:
     1.7091 0.9436 0.7874
     0.9436 1.9551 0.9047
     0.7874 0.9047 1.7681

    Here's an example with priming of the means and variances.  A single SimpleGaussianModel
    should be provided; this is used for all components.

    >>> mu_primer = np.array((10, 20, 1000))
    >>> var_primer = np.array((1, 22, 10000))
    >>> primer = SimpleGaussianModel(3, GaussianModelBase.DIAGONAL_COVARIANCE)
    >>> primer.set_model(mu_primer, var_primer)

    >>> m3 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2, primer=primer)
    >>> print m3
    Gmm: (Type = diagonal, Dim = 3, NumComps = 2)
    Weights   Models
     0.5000     SGM: (Type = diagonal, Dim = 3)  Means: 10.8238 18.5308 1074.7759    Vars: 1.0000 22.0000 10000.0000
     0.5000     SGM: (Type = diagonal, Dim = 3)  Means: 10.0038 18.4424 938.7859    Vars: 1.0000 22.0000 10000.0000

    >>> m4 = m3.copy()
    >>> print m4
    Gmm: (Type = diagonal, Dim = 3, NumComps = 2)
    Weights   Models
     0.5000     SGM: (Type = diagonal, Dim = 3)  Means: 10.8238 18.5308 1074.7759    Vars: 1.0000 22.0000 10000.0000
     0.5000     SGM: (Type = diagonal, Dim = 3)  Means: 10.0038 18.4424 938.7859    Vars: 1.0000 22.0000 10000.0000

"""

    def __init__(self, dimension, covariance_type, num_components, primer=None):
        super(GaussianMixtureModel, self).__init__(dimension, covariance_type)
        if self.covariance_type is GaussianModelBase.DUMMY_COVARIANCE:
            raise ValueError("dummy covariance type not allowed in this context")
        assert type(num_components) is int and num_components > 0
        self._num_components = num_components
        self._models = [SimpleGaussianModel(dimension, covariance_type) for i in xrange(num_components)]
        if primer is not None:
            if (not isinstance(primer, SimpleGaussianModel) or
                primer.dimension != dimension or
                primer.covariance_type != covariance_type):
                raise ValueError("invalid primer - %s" % (primer,))
        self._init_models(primer)

        self._last_observation = self._last_score = None
        self.temp1 = np.empty((num_components,))
        self._weight_relevance = 0.0
        self._mean_relevance = 0.0
        self._var_relevance = 0.0
            
        
    @property
    def num_components(self):
        return self._num_components

    @property
    def weights(self):
        return self._weights

    @property
    def models(self):
        return self._models

    @property
    def relevances(self):
        return (self._weight_relevance, self._mean_relevance,  self._var_relevance)


    def copy(self):
        """
        Return a deep copy of this model

        """
        ret = GaussianMixtureModel(self.dimension, self.covariance_type, self.num_components)
        
        for i, m in enumerate(self.models):
            ret._models[i].set_model(m.means, m.vars)
        ret._weights[:] = self._weights[:]
        ret._weight_relevance = self._weight_relevance
        ret._mean_relevance = self._mean_relevance
        ret._var_relevance = self._var_relevance
        # XXX We're not copying the score-caching stuff here.
        return ret

    def setup_for_scoring(self):
        for m in self._models:
            m.setup_for_scoring()

    def __str__(self):
        def format_weight(x):  return "%.4f" % (x,)
        ret = "Gmm: (Type = %s, Dim = %d, NumComps = %s)\n" % (self.COVARIANCE_TYPE_NAMES[self.covariance_type],
                                                               self.dimension,
                                                               self.num_components)
        ret += "Weights   Models"
        for i in xrange(self.num_components):
            ret += "\n"
            if self._weights is not None:
                ret += " " + format_weight(self._weights[i]) + "     "
            else:
                ret += " Not set"
            if self._models is not None:
                ret += str(self._models[i])
            else:
                ret += " Not set"
        return ret
        
        
    def __eq__(self, other):
        if ((self.dimension, self.covariance_type) != (other.dimension, other.covariance_type) or
            (self.num_components != other.num_components) or 
            (self.weights != other.weights).any() or 
            any((m1 != m2 for (m1,m2) in izip(self.models, other.models)))):
            return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


    def set_model(self, m=None, v=None):
        if m is not None:
            self.set_means(m)
        if v is not None:
            self.set_vars(v)

    def set_weights(self, w, rel_factors = 1.0):
        expected_shape = (self.num_components,)
        if not w.shape == expected_shape:
            raise ValueError("Bad argument to set_weights; expected a np.array with shape %s, but got %s" %
                             (expected_shape, w.shape))
        self._weights = rel_factors * w + (1 - rel_factors) * self._weights

        # self._weights = self._weights / self._weights.sum()
        # Instead of the above, do division in place
        np.divide(self._weights, self._weights.sum(), self._weights)

    def set_means(self, m, rel_factors=None):
        expected_shape = (self.num_components, self.dimension)
        if not m.shape == expected_shape:
            raise ValueError("Bad argument to set_means; expected a np.array with shape %s, but got %s" %
                             (expected_shape, m.shape))
                
        rel_factors = np.ones((self.num_components,), dtype=float) if rel_factors is None else rel_factors
        for i in xrange(self.num_components):
            self._models[i].set_means(m[i], rel_factors[i])


    def set_vars(self, v, rel_factors=None):
        if (self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE):
            expected_shape = (self.num_components, self.dimension)
        else:
            assert(self.covariance_type is GaussianModelBase.FULL_COVARIANCE)
            expected_shape = (self.num_components, self.dimension, self.dimension)
            
        if not v.shape == expected_shape:
            raise ValueError("Bad argument to set_vars; expected a np.array with shape %s, but got %s" %
                             (expected_shape, v.shape))

        rel_factors = np.ones((self.num_components,), dtype=float) if rel_factors is None else rel_factors
        for i in xrange(self.num_components):
            self._models[i].set_vars(v[i], rel_factors[i])

            
    def set_relevances(self, values):
        """
        Set relevances for adaptation - see adapt()

        values must be a tuple of three non-negative numbers, the first is used
        for weights, the second for means, and the third for variances.
        """
        if len(values) != 3 or values[0] < 0 or values[1] < 0 or values[2] < 0:
            raise ValueError("relevance values must be a tuple of three non-negative numbers")
        self._weight_relevance = values[0]
        self._mean_relevance = values[1]
        self._var_relevance = values[2]


    def _init_models(self, primer = None, mode = 'random'):
        if mode == 'random':
            self._init_models_rand(primer)
        elif mode == 'none':
            pass
        elif mode == 'kmeans':
            raise NotImplemented("kmeans initialization not implemented yet")
        else:
            assert(False)

    def _init_models_rand(self, primer):
        k = self.num_components
        d = self.dimension
        self._weights = np.ones(k, dtype=float) / k
        dc1 = dcheck("gaussian_priming")
        if primer is None:
            means = np.array([self.rand.normalvariate(0,1) for x in range(d) for y in range(k)])
            self.set_means(np.reshape(means, (k, d)))
            if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
                vars = np.array([0.7] * d * k)
                # vars = np.array([self.rand.normalvariate(0,1) ** 2 for x in range(d) for y in range(k)])
                self.set_vars(np.reshape(vars, (k, d)))
            else:
                assert self.covariance_type is GaussianModelBase.FULL_COVARIANCE
                self.set_vars(np.resize(np.eye(d) * 0.7, (k, d, d)))
        else: # use primer
            means = []
            vars = []
            dc1 and dc1("priming with model = %s" % (primer,))
            mean_primer = primer.copy()
            mean_primer.set_vars(0.25 * primer.vars)
            dc1 and dc1("priming means with model = %s" % (mean_primer,))
            for y in xrange(k):    # loop over components
                means.append(mean_primer.sample())
                vars.append(primer.vars)
            means = np.array(means).reshape(k, d)
            dc1 and dc1("primed means = %s" % (means,))
            self.set_means(means)
            if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
                vars = np.array(vars).reshape(k, d)
            else:
                vars = np.array(vars).reshape(k, d, d)
            dc1 and dc1("primed vars = %s" % (vars,))
            self.set_vars(vars)



    def adapt(self, data, max_iters = 10):
        """
        Adapt models, taking both their current state and new data into account.

        *data* is an iterable of frames to train on; each frame is a numpy vector
        """
        data = tuple(data)
        assert len(data) != 0
        for i in xrange(max_iters):
            est_w, est_m, est_v, norm = self._get_EM_updates(data)
            w_rel_factors = norm / ( norm + self._weight_relevance)
            m_rel_factors = norm / ( norm + self._mean_relevance)
            v_rel_factors = norm / ( norm + self._var_relevance)
            self.set_weights(est_w, w_rel_factors)
            self.set_means(est_m, m_rel_factors)
            self.set_vars(est_v, v_rel_factors)


    def train(self, data, max_iters = 10, init_first=True):
        """
        Train models from scratch, using one set of data.  

        *data* is an iterable of frames to train on; each frame is a numpy vector
        """
        if init_first:
            self._init_models()
        
        data = tuple(data)
        assert len(data) != 0
        for i in xrange(max_iters):
            w, mu, var, not_used = self._get_EM_updates(data)
            self.set_weights(w)
            self.set_means(mu)
            self.set_vars(var)
    
    def get_estimate(self, x):
        comp_scores = self._weights * self.get_likelihoods(x)
        score_sum = np.sum(comp_scores)
        # assert score_sum != 0
        norm = safely_divide_float_array(comp_scores, score_sum)
        return norm, comp_scores, score_sum

    def get_estimate_log(self, x):
        comp_scores = np.log(self._weights) + self.get_loglikelihoods(x)
        score_sum = logsumexp_array(comp_scores)
        norm = comp_scores - score_sum
        return norm, comp_scores, score_sum

    def _get_EM_updates(self, data):
        dc1 = dcheck("gaussian")
        dc2 = dcheck("gaussian_pt")
        n = len(data)
        k = self.num_components
        d = self.dimension
        norm_sum = np.zeros((k), dtype=float)
        mean_sum = np.zeros((k, d), dtype=float)
        colon_slice = slice(None)

        # There's some Numpy cleverness here to allow us to treat the two
        # covariance cases with the same code.  In the diagonal case, things are
        # pretty much simple operations on vectors, although we do have to
        # broadcast the data across components.  In the full case, we have to
        # take outer products of the data at one point and of the means at
        # another, and we have to broadcast the norm across two dimensions when
        # dividing by it in the var calculation.  Some of this is accomplished
        # by using variables which are tuples of slice objects.  Note that if s0
        # and s1 are slice objects, then arr[s0, s1] == arr[(s0, s1)], and that
        # the colon_slice constructed here is the equivalent of using a ':' in
        # square brackets

        if self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE:
            var_sum = np.zeros((k, d), dtype=float)
            square_data_op = np.multiply
            norm_slices = (colon_slice, np.newaxis)
            square_mean_slices1 = (colon_slice, )
            square_mean_slices2 = (colon_slice, )
        else:
            assert self.covariance_type is GaussianModelBase.FULL_COVARIANCE
            var_sum = np.zeros((k, d, d), dtype=float)
            square_data_op = np.outer
            norm_slices = (colon_slice, np.newaxis, np.newaxis)
            square_mean_slices1 = (colon_slice, colon_slice, np.newaxis)
            square_mean_slices2 = (colon_slice, np.newaxis, colon_slice)

        for x in data:
            dc2 and dc2("Processing frame %s" % (x,))
            norm, raw, ssum = self.get_estimate(x)
            dc2 and dc2("norm = %s, raw = %s, ssum = %s" % (norm, raw, ssum))
            norm_sum += norm
            mean_sum += norm[:, np.newaxis] * x
            var_sum += norm[norm_slices] * square_data_op(x, x)

        w = norm_sum / n
        assert (norm_sum != 0).all()
        mu = mean_sum[:,:] / norm_sum[:,np.newaxis]
        dc1 and dc1("w update = %s" % (w,))
        dc1 and dc1("norm_sum = %s" % (norm_sum,))
        dc1 and dc1("mean_sum = %s" % (mean_sum,))
        dc1 and dc1("var_sum = %s" % (var_sum,))
        dc1 and dc1("mu.shape = %s" % (mu.shape,))
        dc1 and dc1("mean_sum.shape = %s" % (mean_sum.shape,))
        dc1 and dc1("norm_sum.shape = %s" % (norm_sum.shape,))
        assert (norm_sum != 0).all()
        var = var_sum / norm_sum[norm_slices] - (mu[square_mean_slices1] * mu[square_mean_slices2])

        if (self.covariance_type is GaussianModelBase.DIAGONAL_COVARIANCE):
            # XXX I don't know what the right thing to do here is for the full covariance case
            # Also, we may want to move this clipping to the set_vars function rather than here;
            # note that this code was copied from modelmgr.py which has a similar issue
            MIN_DEV = 2.0e-20
            MAX_DEV = 2.0e+20
            var.clip(min=MIN_DEV, max=MAX_DEV, out=var)
            assert (var != 0).all()

        return w, mu, var, norm


    def score(self, x):
        # trivial score caching -- is x same as last time
        # XXX won't work if user mutates and reuses x -- numpy needs frozen semantics
        if x is not self._last_observation:
            # cache miss
            # XXX need to make sure that ._last_observation is set to None
            # whenever model state is changed
            self._last_observation = x
            self._last_score = self.score_components(x, self.temp1).sum()
        assert self._last_score is not None
        return self._last_score

    def score_components(self, x, out=None):
        """ Return a numpy vector of weight * likelihood products """
        assert self._verify_reasonable(x)
        return np.multiply(self._weights, self.get_likelihoods(x), out)

    def get_likelihoods(self, x):
        assert self._verify_reasonable(x)
        self.setup_for_scoring()
        scores = np.fromiter((m.score(x) for m in self._models), dtype=float)
        # This is called too often to leave these in all the time
        # dc = dcheck("gaussian_gmm_score")
        # dc and dc("get_likelihoods returning %s" % (ret,))
        return scores

    def get_loglikelihoods(self, x):
        assert self._verify_reasonable(x)
        self.setup_for_scoring()
        return np.fromiter((m.log_score(x) for m in self._models), dtype=float)

    def sample(self):
        # Randomly sample i with p(i) = w_i
        r = GaussianModelBase.rand.random()
        i = find_cumulative_index(self.weights, r)
        return self._models[i].sample()
        
    
def _test0():
    print '============== test0 ============'
    m0 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    m0.set_weights(np.array((0.75, 0.25)))
    mu = np.array(((1, 1, 1), (2, 3, 4)))
    m0.set_means(mu)
    v = np.array(((1, 1, 1), (0.5, 0.5, 1)))
    m0.set_vars(v)
    print m0
    m0.seed(0)
    data0 = (m0.sample() for i in xrange(300))
    m1 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    print m1
    m1.train(data0, 20)
    print m1
    print

def _test1():
    print '============== test1 ============'
    m0 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    m0.set_weights(np.array((0.75, 0.25)))
    mu = np.array(((1, 1, 1), (2, 3, 4)))
    m0.set_means(mu)
    v = np.array(((1, 1, 1), (0.5, 0.5, 1)))
    m0.set_vars(v)
    print m0
    m0.seed(0)
    data0 = (m0.sample() for i in xrange(1080))
    m1= GaussianMixtureModel(3, GaussianModelBase.FULL_COVARIANCE, 2)
    print m1
    m1.train(data0, 20)
    print m1
    print

def _test2():
    print '============== test2 ============'
    m0 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    m0.set_weights(np.array((0.75, 0.25)))
    mu = np.array(((1, 1, 1), (2, 3, 4)))
    m0.set_means(mu)
    v = np.array(((1, 1, 1), (0.5, 0.5, 1)))
    m0.set_vars(v)
    print m0
    m0.seed(0)
    m1 = GaussianMixtureModel(3, GaussianModelBase.DIAGONAL_COVARIANCE, 2)
    m1.set_relevances((10, 10, 10))
    print m1
    for not_used in xrange(5):
        data0 = (m0.sample() for i in xrange(100))
        m1.adapt(data0, 20)
    print m1
    print

def _logreftest():
    _test0()
    _test1()
    _test2()

if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    args = argv[1:]
    if '--logreftest' in args:
        _logreftest()
        
