###########################################################################
#
# File:         datagen.py (directory: ./projects/ddt)
# Date:         Thu 29 Nov 2007 14:54
# Author:       Ken Basye
# Description:  Artificial data generation
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
    Artificial data generation
   
    >>> seed(0)
    >>> SimpleGaussianModel.seed(0)
    >>> dg = DataGenerator(4,2,(0.0, 1.0), (0.0, 0.0))
    >>> data1 = dg.generate_simulated_data(5)    
    >>> for d in data1: print d
    ('label1', array([ 0.25506799,  0.28978996]))
    ('label3', array([ 1.26645019,  0.21366939]))
    ('label2', array([ 1.48253327, -0.08142416]))
    ('label0', array([ 0.98825779, -0.29010103]))
    ('label2', array([ 0.4747599 ,  0.31283595]))

    >>> data2 = dg.generate_simulated_data_per_phoneme(2)
    >>> for d in data2: print d
    ('label0', array([ 0.68896674, -1.07414585]))
    ('label1', array([ 0.49684315,  0.85441203]))
    ('label2', array([ 1.66518734,  0.66961399]))
    ('label3', array([-0.81511777, -0.83660221]))
    ('label0', array([ 1.3446109 , -0.16231805]))
    ('label1', array([ 0.60313069, -1.1606823 ]))
    ('label2', array([ 1.04930696,  0.77732584]))
    ('label3', array([ 0.97084463,  0.55312036]))

    >>> data3, num_errors = dg.add_errors_to_data(data2, 0.25)
    >>> num_errors
    2
    >>> for d in data3: print d
    ('label1', array([ 0.68896674, -1.07414585]))
    ('label3', array([ 0.49684315,  0.85441203]))
    ('label2', array([ 1.66518734,  0.66961399]))
    ('label3', array([-0.81511777, -0.83660221]))
    ('label0', array([ 1.3446109 , -0.16231805]))
    ('label1', array([ 0.60313069, -1.1606823 ]))
    ('label2', array([ 1.04930696,  0.77732584]))
    ('label3', array([ 0.97084463,  0.55312036]))
"""

from onyx.am.gaussian import SimpleGaussianModel
import numpy
from random import random, choice, uniform, seed

class DataGenerator(object):
    def __init__(self, num_phonemes, num_features, var_diag_interval,
                 var_offdiag_interval):
        self.num_features = num_features
        self.num_phonemes = num_phonemes
        self._labels = tuple("label%d" % i for i in range(num_phonemes))
        # Randomly choose num_phonemes vectors
        # KJB - need something more sophisticated here to ensure minimum
        # distance between phonemes  
        self._targets = tuple(tuple(random() for i in range(num_features))
                              for j in range(num_phonemes))

        def buildOneCovarMatrix():
            uni = (uniform(var_offdiag_interval[0], var_offdiag_interval[1])
                   for i in xrange(num_features**2))
            mat = numpy.fromiter(uni, 'float64').reshape(num_features, num_features)
            # Make symetric
            lower_tri = numpy.tril(mat) # lower triangular
            mat = lower_tri + lower_tri.transpose()
            # Now clobber diagonal with other values
            for i in range(num_features):
                mat[i,i] = uniform(var_diag_interval[0], var_diag_interval[1])
            return mat

        # Build covar matrices
        covars = tuple(buildOneCovarMatrix() for i in range(num_phonemes))

        # Create a dictionary from _labels to SimpleGaussianModels
        self._models = dict()
        for i,label in enumerate(self._labels):
            m = SimpleGaussianModel(num_features, SimpleGaussianModel.FULL_COVARIANCE)
            m.set_model(self._targets[i], covars[i])
            self._models[label] = m

    def get_labels(self):
        return self._labels

    def generate_simulated_data(self, num_frames):
        # sequence of labels
        correct_labels = tuple(choice(self._labels) for i in range(num_frames))
        # a parallel sequence of data points
        frames = (self._models[lab].sample() for lab in correct_labels)
        return zip(correct_labels, frames)  # return list of pairs


    def generate_simulated_data_per_phoneme(self, num_frames_per_phoneme):
        assert(num_frames_per_phoneme > 0)
        correct_labels = self._labels * num_frames_per_phoneme
        # a parallel sequence of data points
        frames = (self._models[lab].sample() for lab in correct_labels)
        return zip(correct_labels, frames)  # return list of pairs
        

    # data is a list of (label, frame) pairs as above
    def add_errors_to_data(self, data, error_rate):  
        # Select subset frames to change to incorrect labels
        assert 0.0 <= error_rate < 1.0
        num_errors = int(error_rate * len(data) + 0.5)
        # NOTE: this code puts all the errors at the beginning of the list!!
        result = list()
        for old_label, frame in data[:num_errors]:
            new_label = old_label
            # until we get a new label
            while new_label == old_label:
                new_label = choice(self._labels)
            result.append((new_label, frame))
            # print "Replaced correct %s with error %s" % (old_label, new_label)
        result.extend(data[num_errors:])  # concatenate remainder of list
        return result, num_errors

if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()
