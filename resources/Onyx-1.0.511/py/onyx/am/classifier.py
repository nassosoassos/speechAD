###########################################################################
#
# File:         classifier.py
# Date:         Fri Nov 9 15:06:50 2007
# Author:       Ken Basye
# Description:  Classifiers using SimpleGaussianModels and GaussianMixtureModels
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
Classifiers using SimpleGaussianModels and GaussianMixtureModels


"""

from __future__ import division
import numpy
from itertools import izip, repeat, count
from collections import defaultdict
from cStringIO import StringIO
from onyx.am.gaussian import GaussianModelBase, SimpleGaussianModel, GaussianMixtureModel, distance
from onyx.am.modelmgr import GmmMgr, read_model_dict, write_model_dict, read_gmm_mgr, write_gmm_mgr
from onyx.util.streamprocess import ProcessorBase
from onyx.util.floatutils import float_to_readable_string
from onyx.textdata.yamldata import YamldataReader, YamldataGenerator
from onyx.textdata.onyxtext import OnyxTextReader

class ClassifierBase(object):
    """
    Base class for classifiers

    """
    def __init__(self, labels, dimension):
        self._labels = set(labels)
        self._num_classes = len(self._labels)
        self._dimension = dimension
        self._models = dict()

    @property
    def dimension(self):
        return self._dimension
    
    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        return self._labels

    def _score_all(self, datum):
        ret = {}
        for label in self._labels:
            model = self.get_model(label)
            model.setup_for_scoring()
            ret[label] = model.score(datum)
        return ret

    def classify_one(self, datum):
        """
        Classify a single point.

        Returns a tuple of (score, label) items sorted by score.
        """
        ret = tuple((s, l) for (l, s) in self._score_all(datum).items())
        return sorted(ret, reverse=True)


class SimpleClassifier(ClassifierBase):
    """
    A classifier based on SimpleGaussianModels

    >>> labels = ('A', 'B', 'C')
    >>> c = SimpleClassifier(labels, 2)
    >>> print c
    SimpleClassifier (num_classes = 3, dimension = 2)
       Labels/Models:
    Label: A  Model:SGM: (Type = diagonal, Dim = 2)  Means: Not set    Vars: Not set
    Label: B  Model:SGM: (Type = diagonal, Dim = 2)  Means: Not set    Vars: Not set
    Label: C  Model:SGM: (Type = diagonal, Dim = 2)  Means: Not set    Vars: Not set
    >>> p0 = numpy.zeros(2)
    >>> p1 = numpy.ones(2)
    >>> p2 = p1 * 0.1
    >>> data = (('A', p0), ('B', p1), ('A', p1), ('B', 2*p1), ('C', 4*p1), ('C', 5*p1))
    >>> c.train_all_classes(data)
    >>> print c
    SimpleClassifier (num_classes = 3, dimension = 2)
       Labels/Models:
    Label: A  Model:SGM: (Type = diagonal, Dim = 2)  Means: 0.5000 0.5000    Vars: 0.2500 0.2500
    Label: B  Model:SGM: (Type = diagonal, Dim = 2)  Means: 1.5000 1.5000    Vars: 0.2500 0.2500
    Label: C  Model:SGM: (Type = diagonal, Dim = 2)  Means: 4.5000 4.5000    Vars: 0.2500 0.2500

    >>> result = c.classify_one(p0)
    >>> print tuple(((label, float_to_readable_string(score)) for score, label in result))
    (('A', '+(-0003)0xdfa3e572aa124'), ('B', '+(-0014)0x4986a82011d6e'), ('C', '+(-0118)0x6796d08b3cfa2'))

    """

    def __init__(self, labels, dimension):
        super(SimpleClassifier, self).__init__(labels, dimension)
        for label in labels:
            m = SimpleGaussianModel(dimension, SimpleGaussianModel.DIAGONAL_COVARIANCE)
            self._models[label] = m

    def __str__(self):
        ret = ("SimpleClassifier (num_classes = %d, dimension = %d)\n" %
               (self.num_classes, self.dimension))
        ret += "   Labels/Models:"
        for l in sorted(self.labels):
            ret += "\nLabel: %s  Model:" % (l,)
            ret += str(self._models[l])
        return ret

    def train_one_class(self, label, data):
        """
        Train the model for one class on given data.
        
        arguments are a single label and a sequence of data points
        """
        self._models[label].adapt(data)
        self._models[label].setup_for_scoring()

    def train_all_classes(self, labeled_data):
        """
        Train the models for several classes on given data.

        labeled_data is a sequence of (label, point) pairs.  All the data for a
        given label will be agglomerated before training.
        """
        # partition data
        data_dict = defaultdict(list)
        for (l,p) in labeled_data:
            # Some checks to catch problems early
            assert type(p) == type(numpy.array([]))
            assert len(p) == self.dimension
            assert l in self.labels
            data_dict[l].append(p)
        for (label, points) in data_dict.items():
            self.train_one_class(label, points)


    def get_model(self, label):
        return self._models[label]

    def _get_distances(self):
        models = self._models.values()
        ret = []
        for (i,m0) in enumerate(models):
            for m1 in models[i+1:]:
                ret.append(distance(x._means,y._means))
        return ret

    def get_distance_dict(self):
        ret = {}
        for l1 in self.labels:
            for l2 in self.labels:
                ret[(l1,l2)] = distance(self._models[l1]._means,
                                        self._models[l2]._means)
        return ret

    def get_min_distance(self):
        """ Get minimum distance between any two distinct models """
        return self._get_distances().min()

    def get_max_distance(self):
        """ Get maximum distance between any two distinct models """
        return self._get_distances().max()

    def get_mean_distance(self):
        """ Get mean distance between distinct models """
        from numpy import array
        a = array(self._get_distances())
        return a.mean()



class AdaptingGmmClassifier(ClassifierBase):
    """
    A classifier based on GaussianMixtureModels

    >>> dimension = 2
    >>> covar_type = GaussianModelBase.DIAGONAL_COVARIANCE
    >>> labels = ('A', 'B', 'C')
    >>> ncomps = (3, 2, 4)

    >>> GaussianMixtureModel.seed(0)
    >>> gmm_mgr0 = GmmMgr(ncomps, dimension, covar_type)
    >>> c0 = AdaptingGmmClassifier(gmm_mgr0, izip(labels, count()))
    >>> print c0
    AdaptingGmmClassifier (num_classes = 3, dimension = 2)
       Labels/Models:
    Label: A  Model:Gmm: (Type = diagonal, Dim = 2, NumComps = 3)
    Weights   Models
     0.3333     SGM: (Type = diagonal, Dim = 2)  Means: -0.1839 0.0325    Vars: 0.7000 0.7000
     0.3333     SGM: (Type = diagonal, Dim = 2)  Means: 0.6988 -0.0964    Vars: 0.7000 0.7000
     0.3333     SGM: (Type = diagonal, Dim = 2)  Means: 1.4135 -1.5326    Vars: 0.7000 0.7000
    Label: B  Model:Gmm: (Type = diagonal, Dim = 2, NumComps = 2)
    Weights   Models
     0.5000     SGM: (Type = diagonal, Dim = 2)  Means: 0.2709 -1.2055    Vars: 0.7000 0.7000
     0.5000     SGM: (Type = diagonal, Dim = 2)  Means: -0.0531 -0.2902    Vars: 0.7000 0.7000
    Label: C  Model:Gmm: (Type = diagonal, Dim = 2, NumComps = 4)
    Weights   Models
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: -0.2928 -2.1074    Vars: 0.7000 0.7000
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: 0.0847 0.6270    Vars: 0.7000 0.7000
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: 1.6793 0.8341    Vars: 0.7000 0.7000
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: -2.3151 -1.2254    Vars: 0.7000 0.7000

    Here's an example with priming.  The priming value is an iterable with as
    many elements as there are labels.  Each element should be a
    SimpleGaussianModel Each classifier will be initialized from the model
    provided; see GaussianMixtureModel for details on how this is done.

    >>> means = numpy.array((1, 5))
    >>> vars = numpy.array((5, 5))
    >>> priming = tuple([SimpleGaussianModel(dimension, covar_type) for _ in xrange(3)])
    >>> priming[0].set_model(means, vars)
    >>> priming[1].set_model(means*10, vars*2)
    >>> priming[2].set_model(means*2, vars*0.5)
    
    >>> gmm_mgr0 = GmmMgr(ncomps, dimension, covar_type, priming)
    >>> c0 = AdaptingGmmClassifier(gmm_mgr0, izip(labels, count()))
    >>> print c0
    AdaptingGmmClassifier (num_classes = 3, dimension = 2)
       Labels/Models:
    Label: A  Model:Gmm: (Type = diagonal, Dim = 2, NumComps = 3)
    Weights   Models
     0.3333     SGM: (Type = diagonal, Dim = 2)  Means: 2.0534 3.8165    Vars: 5.0000 5.0000
     0.3333     SGM: (Type = diagonal, Dim = 2)  Means: 1.2268 3.3290    Vars: 5.0000 5.0000
     0.3333     SGM: (Type = diagonal, Dim = 2)  Means: 1.8754 6.3120    Vars: 5.0000 5.0000
    Label: B  Model:Gmm: (Type = diagonal, Dim = 2, NumComps = 2)
    Weights   Models
     0.5000     SGM: (Type = diagonal, Dim = 2)  Means: 10.4282 50.4246    Vars: 10.0000 10.0000
     0.5000     SGM: (Type = diagonal, Dim = 2)  Means: 10.2892 48.9640    Vars: 10.0000 10.0000
    Label: C  Model:Gmm: (Type = diagonal, Dim = 2, NumComps = 4)
    Weights   Models
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: 2.4455 9.9650    Vars: 2.5000 2.5000
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: 2.0905 10.3844    Vars: 2.5000 2.5000
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: 3.3061 11.4513    Vars: 2.5000 2.5000
     0.2500     SGM: (Type = diagonal, Dim = 2)  Means: 1.8027 10.4335    Vars: 2.5000 2.5000

    >>> c0.set_num_em_iterations(5)

    >>> f = StringIO()
    >>> write_gmm_classifier(c0, f)

    # >>> print f.getvalue()
    
    >>> f.seek(0)
    >>> c1 = read_gmm_classifier(f)
    >>> str(c1) == str(c0)
    True

    """

    def __init__(self, gmm_mgr, label_index_pairs):
        self._label_index_map = dict(label_index_pairs)
        super(AdaptingGmmClassifier, self).__init__(self._label_index_map, gmm_mgr.dimension)
        self._gmm_mgr = gmm_mgr
        self._num_em_iters = 10

    def __str__(self):
        ret = ("AdaptingGmmClassifier (num_classes = %d, dimension = %d)\n" %
               (self.num_classes, self.dimension))
        ret += "   Labels/Models:"
        for l in sorted(self.labels):
            ret += "\nLabel: %s  Model:" % (l,)
            ret += str(self.get_model(l))
        return ret


    def get_model(self, label):
        return self._gmm_mgr[self._label_index_map[label]]
    
    def set_relevance(self, relevance):
        """
        Set the relevances for all models in this classifier

        Note that we use just one number for all models and for all parameters
        (i.e. for weights, means, and vars).
        """
        if relevance < 0:
            raise ValueError("relevance must be non-negative, got %s" % (relevance,))
        for l in self._label_index_map.keys():
            self.get_model(l).set_relevances((relevance, relevance, relevance))

    def set_num_em_iterations(self, num_iters):
        if int(num_iters) < 1:
            raise ValueError("expected a num_iters >= 1 , but got %s" % (num_iters,))
        self._num_em_iters = int(num_iters)

    def adapt_one_class(self, label, points):
        """
        Adapt one class on a set of datapoints
        
        Arguments are a single label and a sequence of points.
        """
        self.get_model(label).adapt(points, max_iters=self._num_em_iters)

    def adapt_all_classes(self, labeled_data):
        """
        Adapt the models for several classes on given data

        labeled_data is a sequence of (label, point) pairs.  All the data for a
        given label will be agglomerated before training.
        """
        # partition data
        data_dict = defaultdict(list)
        for (l,p) in labeled_data:
            # Some checks to catch problems early
            assert type(p) == type(numpy.array([]))
            assert len(p) == self.dimension
            assert l in self.labels
            data_dict[l].append(p)
        for (label, points) in data_dict.items():
            self.adapt_one_class(label, points)
    

def write_gmm_classifier(classifier, file):
    write_model_dict(classifier._label_index_map, file)
    write_gmm_mgr(classifier._gmm_mgr, file)


def read_gmm_classifier(file):
    gen = YamldataGenerator(file)
    model_dict_yaml_reader = YamldataReader(gen, stream_type=OnyxTextReader.STREAM_TYPE,
                                       stream_version=OnyxTextReader.STREAM_VERSION)
    model_dict = read_model_dict(None, model_dict_yaml_reader)

    gmm_yaml_reader = YamldataReader(gen, stream_type=OnyxTextReader.STREAM_TYPE,
                                     stream_version=OnyxTextReader.STREAM_VERSION)
    gmm_mgr = read_gmm_mgr(None, yaml_reader=gmm_yaml_reader)

    return AdaptingGmmClassifier(gmm_mgr, model_dict.items())


class AdaptingGmmClassProcessor(ProcessorBase):
    """
    Process labelled data events and train a classifier with them.
    
    Events to be processes should have be pairs of the form (label, data) where
    data is an iterable of data points in the form of Numpy arrays.  The label
    may be None, in which case no adaptation will be performed.  This processor
    will send events of the form (label, ((s0, l0), (s1, l1),...)) where label
    is the label of the incoming event and the s,l pairs are the scores and
    labels of the various classes in order of decreasing score (so l0 is the
    most likely label according to the classifier).  One such event will be
    generated for each data point in the incoming event, so note that this
    processor can generate many outgoing events for each incoming event.
    """
    def __init__(self, classifier, sendee=None, sending=True):
        super(AdaptingGmmClassProcessor, self).__init__(sendee, sending=sending)
        self._classifier = classifier

    def process(self, labeled_data):
        """
        Process labelled data events.

        labeled_data is a pair of (label, data) where data is an iterable of
        data points in the form of Numpy arrays.
        """
        if len(labeled_data) != 2:
            raise ValueError("expecting events in the form (label, data), but got %s" % (labeled_data,))
        label, data = labeled_data
        data = tuple(data)
        for point in data:
            pairs = self._classifier.classify_one(point)
            self.send((label, pairs))
        if label is not None:
            self._classifier.adapt_one_class(label, data)



class HmmClassifier(ClassifierBase):
    """
    A classifier using HMMs

    
    """

    def __init__(self, hmm_mgr, label_index_pairs):
        self._label_index_map = dict(label_index_pairs)
        super(HmmClassifier, self).__init__(self._label_index_map, hmm_mgr.dimension)
        self._hmm_mgr = hmm_mgr

    def __str__(self):
        ret = ("HmmClassifier (num_classes = %d, dimension = %d)\n" %
               (self.num_classes, self.dimension))
        ret += "   Labels/Models:"
        for l in sorted(self.labels):
            ret += "\nLabel: %s  Model:" % (l,)
            ret += str(self._models[l])
        return ret

    def get_model(self, label):
        return self._hmm_mgr[self._label_index_map[label]]
    
    def begin_training(self):
        gmm_mgr = self._hmm_mgr[0].gmm_manager
        gmm_mgr.set_adaptation_state("INITIALIZING")
        hmm_mgr.set_adaptation_state("INITIALIZING")
        gmm_mgr.clear_all_accumulators()
        
        tg0.begin_training()
        gmm_mgr.set_adaptation_state("ACCUMULATING")
        hmm_mgr.set_adaptation_state("ACCUMULATING")
        tg0.train_one_sequence(obs_seq)
        gmm_mgr.set_adaptation_state("APPLYING")
        hmm_mgr.set_adaptation_state("APPLYING")
        gmm_mgr.apply_all_accumulators()
        hmm_mgr.apply_all_accumulators()
        tg0.end_training()
        gmm_mgr.set_adaptation_state("NOT_ADAPTING")
        hmm_mgr.set_adaptation_state("NOT_ADAPTING")


    def train_one_class(self, label, data):
        """
        Train the model for one class on given data.
        
        arguments are a single label and a sequence of training instances
        """
        self.get_model(label).train(data)

    def train_all_classes(self, labeled_data):
        """
        Train the models for several classes on given data.

        labeled_data is a sequence of (label, instance) pairs.  All the data for a
        given label will be agglomerated before training.
        """
        # partition data
        data_dict = defaultdict(list)
        for (l,p) in labeled_data:
            assert l in self.labels
            data_dict[l].append(p)
        for (label, points) in data_dict.items():
            self.train_one_class(label, points)




### Test helper functions ###
from gaussian import GaussianModelBase
def make_target(dimension, num_comps, weights, means, vars):
    ret = GaussianMixtureModel(dimension, GaussianModelBase.DIAGONAL_COVARIANCE, num_comps)
    ret.set_weights(numpy.array(weights))
    ret.set_means(numpy.array(means))
    ret.set_vars(numpy.array(vars))
    return ret
    

def test0():
    print '============== test0 ============'
    dimension = 3
    target0 = make_target(dimension, 2, (0.75, 0.25), ((1, 1, 1), (2, 3, 4)), ((1, 1, 1), (0.5, 0.5, 1)))
    target1 = make_target(dimension, 2, (0.5, 0.5), ((-1, -1, -1), (-2, -3, -4)), ((1, 1, 1), (0.5, 0.5, 1)))
    target2 = make_target(dimension, 2, (0.1, 0.9), ((1, 1, -2), (3, 3, 5)), ((1, 1, 1), (0.5, 0.5, 1)))
    print target0
    print target1
    print target2
    GaussianModelBase.seed(0)
    
    labels = ('A', 'B', 'C')
    ncomps = (1, 2, 2)

    sources = dict((('A', target0), ('B', target1), ('C', target2)))

    GaussianMixtureModel.seed(0)
    gmm_mgr = GmmMgr(ncomps, dimension, GaussianModelBase.DIAGONAL_COVARIANCE)
    c0 = AdaptingGmmClassifier(gmm_mgr, izip(labels, count()))
    print
    print c0


    # Prime things a little bit to try to get a good start
    c0.set_relevance(0.001)
    for i in xrange(1):
        for label in labels:
           target = sources[label]
           data = (target.sample() for i in xrange(100))
           c0.adapt_one_class(label, data)

    # Now adapt on more data
    c0.set_relevance(10)
    for i in xrange(10):
        for label in labels:
           target = sources[label]
           data = (target.sample() for i in xrange(100))
           c0.adapt_one_class(label, data)
           
    print
    print c0
    print

    
def test1():
    print '============== test1 ============'
    dimension = 3
    target0 = make_target(dimension, 2, (0.75, 0.25), ((1, 1, 1), (2, 3, 4)), ((1, 1, 1), (0.5, 0.5, 1)))
    target1 = make_target(dimension, 2, (0.5, 0.5), ((-1, -1, -1), (-2, -3, -4)), ((1, 1, 1), (0.5, 0.5, 1)))
    target2 = make_target(dimension, 2, (0.1, 0.9), ((1, 1, -2), (3, 3, 5)), ((1, 1, 1), (0.5, 0.5, 1)))
    print target0
    print target1
    print target2
    GaussianModelBase.seed(0)
    
    labels = ('A', 'B', 'C')
    ncomps = (1, 2, 2)

    sources = dict((('A', target0), ('B', target1), ('C', target2)))

    GaussianMixtureModel.seed(0)
    gmm_mgr = GmmMgr(ncomps, dimension, GaussianModelBase.DIAGONAL_COVARIANCE)
    c0 = AdaptingGmmClassifier(gmm_mgr, izip(labels, count()))
    print
    print c0

    result = list()
    proc0 = AdaptingGmmClassProcessor(c0, result.append)

    # Prime things a little bit to try to get a good start
    c0.set_relevance(0.001)
    c0.set_num_em_iterations(2)
    for i in xrange(1):
        for label in labels:
           target = sources[label]
           data = (target.sample() for i in xrange(100))
           proc0.process((label, data))

    # Now adapt on more data
    c0.set_relevance(10)
    c0.set_num_em_iterations(2)
    for i in xrange(10):
        for label in labels:
           target = sources[label]
           data = (target.sample() for i in xrange(100))
           proc0.process((label, data))
           
    print
    print c0
    print
    print len(result)
    # XXX Win32 gets values off in the last 2-3 hex digits.  I'm not sure how to account for this in a
    # logref test, so I'm disabling this printing for now.
    
    # for training_label, scores in result[-10:]:
    #     print training_label, tuple(((label, float_to_readable_string(score)) for score, label in scores))
    correct = tuple(label for label, scores in result)
    guessed = tuple(scores[0][1] for l, scores in result)
    print len(correct), len(guessed)
    ind = [c == g for (c, g) in izip(correct, guessed)]
    print ind.count(True)
    print ind.count(True) / len(correct)
    
    
    


def logreftest():
    test0()
    test1()

if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()
    # test0()
    # test1()

    from sys import argv
    args = argv[1:]
    if '--logreftest' in args:
        logreftest()


