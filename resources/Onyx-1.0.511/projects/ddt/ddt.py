###########################################################################
#
# File:         ddt.py (directory: ./projects/ddt)
# Date:         Sat Nov 10 11:36:30 2007
# Author:       Ken Basye
# Description:  Delayed Decision Training classes
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 - 2009 The Johns Hopkins University
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
    Delayed Decision Training

    >>> True
    True
"""

from numpy import array, median, fromiter
from onyx.am.classifier import SimpleClassifier
from onyx.am.gaussian import SimpleGaussianModel
from datagen import DataGenerator


# Utility functions
def measureAccuracy(classifier, labelled_data):
    """
    labelled_data is a list of (label, point) pairs
    returns (ER, results) where results is a list of True/False
    results - True meaning the classifier got the correct label
    """
    results = []
    num_errors = 0
    for (label, point) in labelled_data:
        # Get a list of (label, score) pairs sorted by score
        scores = classifier.classify(point)
        (best, best_score) = scores[0]
        correct = (best == label)
        results.append(correct)
        if not correct: num_errors += 1
    return((float(num_errors) / len(labelled_data)), results)


def measurePrimaryAndVariantAccuracy(allele, labelled_data):
    """
    labelled_data is a list of (label, point) pairs
    returns (ER, results) where results is a list of True/False
    results - True meaning the classifier got the correct label
    """
    results = []
    p_num_errors = 0
    v_num_errors = 0
    p_score_higher = 0
    v_score_higher = 0
    score_identical = 0
    for (label, point) in labelled_data:
        # Get a list of (label, score) pairs sorted by score
        primary_scores = allele.classify_with_primary(point)
        variant_scores = allele.classify_with_variants(point)
        (p_best, p_best_score) = primary_scores[0]
        (v_best, v_best_score) = variant_scores[0]

        if p_best_score > v_best_score:
            p_score_higher += 1
            assert(p_best in allele._variant_labels)
        elif v_best_score > p_best_score:
            v_score_higher += 1
            assert(v_best in allele._variant_labels)
        else:
            score_identical += 1

        p_correct = (p_best == label)
        v_correct = (v_best == label)
        if not p_correct: p_num_errors += 1
        if not v_correct: v_num_errors += 1
    p_rate = float(p_num_errors) / len(labelled_data)
    v_rate = float(v_num_errors) / len(labelled_data)
    return(p_rate, v_rate, p_score_higher, v_score_higher, score_identical)


def dump_data(data):
    ret = ""
    for d in data:
        ret += "%s - %s\n" %(d[0], d[1])
    return ret

def make_summary_string(exp_name, rate, results, c, test_data, gen):
    ret = ("Experiment: %s\n   Error rate = %.2f%%\n   Test set size = %d" %
           (exp_name, rate*100, len(test_data)))
    return ret


def make_all_runs_summary_string(expt, results):
    """
    results is a list of (name, rate) pairs
    """
    rates = array(tuple(r for (n,r) in results))
    ret = str(expt) + '\n'
    ret += ("Summary for %d runs:  Median = %.2f%%, Mean = %.2f%%, Std. Dev. = %.4f" % 
            (len(results), median(rates) * 100, rates.mean() * 100, rates.std()))
    return ret


def process_frame_result(frame_result):
    p_acc = frame_result[0]
    v_acc = frame_result[1]
    p_score_higher_count = frame_result[2]
    v_score_higher_count = frame_result[3]
    same_score_count = frame_result[4]
    delta = p_score_higher_count - v_score_higher_count
    status = "ACCEPTED" if delta >= 0 else "REJECTED"
    ret = (status, delta, p_score_higher_count, v_score_higher_count, same_score_count, p_acc, v_acc)
    s = "(%s  delta = %d, p = %d, v = %d, s = %d, p_acc = %.3f, v_acc = %.3f)" % ret
    return ret, s    


def make_training_data(gen, expt):
    # There's a problem here if there's only one data point, since
    # then we end up with a variance of 0.  We currently hack around
    # this problem by guaranteeing more than one point.  We could
    # change the models to allow zero variance but this will mean not
    # being able to make samples from the models without some extra
    # work. and we'd still have to deal with not getting *any* data
    # for some class.  Note that we don't care at all about order of
    # training data in these experiments, so we just build our
    # training data in two parts and cat them together.  If you hit
    # either of these asserts, you're asking for an error rate that's
    # too high and/or a training data size that's too low.  We need
    # three correct samples per phoneme so we can exclude one in some
    # alleles
    num_secondary_frames  =  expt.num_training_frames - expt.num_phonemes * 3
    num_errorful_frames = expt.num_training_frames * expt.training_error_rate 
    assert expt.num_training_frames >= expt.num_phonemes * 3
    assert num_secondary_frames >  num_errorful_frames
    errorless_training_data = gen.generate_simulated_data_per_phoneme(3)
    secondary_training_data = gen.generate_simulated_data(num_secondary_frames)
    
    # Slight trickiness to get a correct error rate for this subset of the data
    subset_error_rate = float(num_errorful_frames) / num_secondary_frames
    errorful_training_data, num_errors_added = gen.add_errors_to_data(secondary_training_data, subset_error_rate)
    return errorful_training_data + errorless_training_data, num_errors_added
    
# Take a list of (label,point) pairs and return a dictionary in which the keys
# are the labels and a list of points is the value for each key.
def partition_data(data):
    ret = {}
    for (l,p) in data:
        if ret.has_key(l):
            ret[l].append(p)
        else:
            ret[l] = [p]
    return ret

class SimpleAllele(object):
    """
    An allele is a set of classifiers which have been trained on
    variants of the same training data.  For example, a simple allele
    might have two classifiers, one with all the data and one with one
    training data point left out.

    This implementation uses two SimpleClassifiers whose models
    overlap except for a single variant.  Training can be done either
    on the primary classifier or on the variant; training on the
    variant only affects one model.
    """
    def __init__(self, primary_classifier, variant_labels):
        self._variant_labels = variant_labels
        self._variant_models = {}
        for label in variant_labels:
            model = SimpleGaussianModel()
            # KJB - this guy shouldn't have to know so much about models -
            # maybe do some kind of cloning from models in the classifier?
            model.set_model(primary_classifier.num_features, 'diagonal')
            self._variant_models[label] = model
            
        self._primary = primary_classifier
        self._variant_classifier = None

    def train_primary(self, data):
        self._primary.train_all(data)

    def train_variants(self, data):
        # partition data
        data_dict = {}
        for (l,p) in data:
            # We only want to train variants here
            if l not in self._variant_labels:
                continue
            if data_dict.has_key(l):
                data_dict[l].append(p)
            else:
                data_dict[l] = [p]
        for (label, points) in data_dict.items():
            self._variant_models[label].train(points)

    def classify_with_primary(self, datum):
        return self._primary.classify(datum)

    def classify_with_variants(self, datum):
        ret = self.classify_with_primary(datum)
        variant_scores = {}
        for label in self._variant_labels:
            variant_scores[label] = self._variant_models[label].score(datum)
        # Now clobber primary scores with variants
        for i, (label, score) in enumerate(ret):
            if variant_scores.has_key(label):
                ret[i] = (label, variant_scores[label])
        ret.sort(reverse=True)
        return ret

    def make_details_string(self):
        ret = "Allele: variant labels = %s\n" % (self._variant_labels,)
        for label in self._variant_labels:
            primary = self._primary.get_model(label)
            variant = self._variant_models[label]
            ret += "For %s:\nprimary is %s\nvariant is %s\n" % (label, primary.to_string(), variant.to_string())
        return ret

class DDTExperimentPars(object):
    def __init__(self, num_phonemes, num_features, var_offdiag_interval, var_diag_interval,
                 num_test_frames, num_training_frames, num_practice_frames,
                 training_error_rate, practice_error_rate, num_runs):
        self.num_phonemes = num_phonemes
        self.num_features = num_features
        self.var_offdiag_interval = var_offdiag_interval
        self.var_diag_interval = var_diag_interval
        self.num_test_frames = num_test_frames
        self.num_training_frames = num_training_frames
        self.num_practice_frames = num_practice_frames
        self.training_error_rate = training_error_rate
        self.practice_error_rate = practice_error_rate
        self.num_runs = num_runs
    def __str__(self):
        ret = "num_phonemes = %s\n" %  (self.num_phonemes,)
        ret += "num_features = %s\n" % (self.num_features,)
        ret += "var_offdiag_interval = %s\n" % (repr(self.var_offdiag_interval),)
        ret += "var_diag_interval = %s\n" % (repr(self.var_diag_interval),)
        ret += "num_test_frames = %s\n" % (self.num_test_frames,)
        ret += "num_training_frames = %s\n" % (self.num_training_frames,)
        ret += "num_practice_frames = %s\n" % (self.num_practice_frames,)
        ret += "training_error_rate = %s\n" % (self.training_error_rate,)
        ret += "practice_error_rate = %s\n" % (self.practice_error_rate,)
        ret += "num_runs = %s" % (self.num_runs,)
        return ret


# Top level experimental functions

def do_ddt_runs(expt):
    gen = DataGenerator(expt.num_phonemes, expt.num_features,
                        expt.var_diag_interval, expt.var_offdiag_interval)

    perfect_practice_data = gen.generate_simulated_data(expt.num_practice_frames)
    practice_data, num_practice_errors = gen.add_errors_to_data(perfect_practice_data, expt.practice_error_rate)
    practice_data_dict = partition_data(practice_data)
    # We got some practice data for every point, right?
    assert( len(practice_data_dict.keys() == expt.num_phonemes))

    test_data = gen.generate_simulated_data(expt.num_test_frames)

    n = expt.num_training_frames
    assert( n * expt.training_error_rate >= 5)   # number of errorful points
    assert( n * (1-expt.training_error_rate) > 5)  # number of correct points
    error_training_frame_indices = range(0,5)
    correct_training_frame_indices = range(n-5, n)

    all_results = {}
    all_results['Error'] = []
    all_results['Correct'] = []
    for run_idx in range(0, expt.num_runs):
        training_data, num_errors = make_training_data(gen, expt)
        c = SimpleClassifier(gen.get_labels(), gen.num_features)
        c.train_all(training_data)

        def run_some_frames(frame_indices):
            frame_results = []
            for i in frame_indices:
                label = training_data[i][0]
                a = SimpleAllele(c, [label])
            
                # subtract (label, frame) from training_data for active phoneme
                alt_data = training_data[:i] + training_data[i+1:]
            
                # train alternate model in allele on alternate data
                a.train_variants(alt_data)
                # print a.make_details_string()

                # Construct a subset of the practice data with only the points
                # which are labelled with the active label of the allele (see comments below).
                data = [(label, point) for point in practice_data_dict[label]]
                results = measurePrimaryAndVariantAccuracy(a, data)

                # KJB - here's the original version, in which we just
                # used all the practice data This essential means we
                # aren't using the practice data labels at all, which
                # might be an interesting variation, but isn't the
                # original intention.
                #results = measurePrimaryAndVariantAccuracy(a, practice_data)

                frame_results.append(results)
            return frame_results

        error_results = run_some_frames(error_training_frame_indices)
        all_results['Error'].append(error_results)
        correct_results = run_some_frames(correct_training_frame_indices)
        all_results['Correct'].append(correct_results)
    return all_results


# Notes about baseline experiments:
# We should establish a baseline by training on the entire training
# set and by also training on the training data and the practice data.
# Also useful would be the "oracle" rates gotten by having all perfect
# training data and by training on exactly the correct subset of the
# training data.  We can get the first easily, and use those numbers
# to get the second by considering smaller amounts of training data.
def do_baseline_runs(expt):
    gen = DataGenerator(expt.num_phonemes, expt.num_features,
                        expt.var_diag_interval, expt.var_offdiag_interval)

    all_results = []
    for run_idx in range(expt.num_runs):
        test_data = gen.generate_simulated_data(expt.num_test_frames)

        # There's a problem here if there's only one data point, since
        # then we end up with a variance of 0.  We currently hack
        # around this problem by guaranteeing more than one point.  We
        # could change the models to allow zero variance but this will
        # mean not being able to make samples from the models without
        # some extra work.  Note that we don't care at all about order
        # of training data in these experiments, so we just build our
        # training data in two parts and cat them together.  If you
        # hit either of these asserts, you're asking for an error rate
        # that's too hig and/or a training data size that's too low.
        # We need two correct samples per phoneme.
        num_secondary_frames  =  expt.num_training_frames - expt.num_phonemes * 2
        num_errorful_frames = expt.num_training_frames * expt.training_error_rate 
        assert expt.num_training_frames >= expt.num_phonemes * 2
        assert num_secondary_frames >  num_errorful_frames
        errorless_training_data = gen.generate_simulated_data_per_phoneme(2)
        secondary_training_data = gen.generate_simulated_data(num_secondary_frames)

        # Slight trickiness to get a correct error rate for this subset of the data
        subset_error_rate = float(num_errorful_frames) / num_secondary_frames
        errorful_training_data, num_errors = gen.add_errors_to_data(secondary_training_data, subset_error_rate)

        practice_data = gen.generate_simulated_data(expt.num_practice_frames)
        errorful_practice_data, num_errors = gen.add_errors_to_data(practice_data, expt.practice_error_rate)

        training_data = errorless_training_data + errorful_training_data + errorful_practice_data

        c = SimpleClassifier(gen.get_labels(), gen.num_features)
        c.train_all(training_data)

        (rate, results) = measureAccuracy(c, test_data)
        name = "Baseline 0.%d" % (run_idx,)
        summary = make_summary_string(name, rate, results, c, test_data, gen)
        all_results.append((name, rate))

        # print "Classifier:\n"
        # print c.to_string()
        # print summary
    print "\n--------------------------Summary-----------------------"
    print make_all_runs_summary_string(expt, all_results)


def simple_test(expt):
    # Build a generator and a classifier that are perfectly matched
    # with respect to means and see what sort of error rate we get for
    # various variance values in the generator.
    gen = DataGenerator(expt.num_phonemes, expt.num_features,
                        expt.var_diag_interval, expt.var_offdiag_interval)
    test_data = gen.generate_simulated_data(expt.num_test_frames)

    # Make perfect "training data" in the form of two points for each
    # class whose mean is exactly the mean for that class.  Training
    # on this will give a correct mean for the model, but with some
    # non-zero variance

    labels = gen.get_labels()
    means = [array(target) for target in gen._targets]

    # Construct a list of (label, point) pairs with two points for each label
    delta = [0.1] * expt.num_features
    assert len(labels) == len(means)
    data = zip(labels, (m+delta for m in means)) + zip(labels, (m-delta for m in means))
    # print dump_data(data)

    c = SimpleClassifier(labels, gen.num_features)
    c.train_all(data)

    (rate, results) = measureAccuracy(c, test_data)
    summary = make_summary_string("Simple test", rate, results, c, test_data, gen)
    print summary
    
def do_complete_baseline_sweep(expt):
    var_diag_interval_sweep = [(0.03, 0.06), (0.05, 0.1), (0.05, 0.2)]
#    var_diag_interval_sweep = [(0.05, 0.1)]
    num_training_frames_per_phoneme_sweep = [3, 5, 7, 10, 20, 30]
#    num_training_frames_per_phoneme_sweep = [10, 20]
    training_error_rate_sweep = [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
#    training_error_rate_sweep = [0.0, 0.05, 0.2]
    num_practice_frames_per_phoneme_sweep = [0, 100, 500, 1000]
#    num_practice_frames_per_phoneme_sweep = [0, 500]
    practice_error_rate_sweep = [0.05, 0.1, 0.2, 0.3, 0.4]
#    practice_error_rate_sweep = [0.1, 0.2]

    for vdi in var_diag_interval_sweep:
        expt.var_diag_interval = vdi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    for per in practice_error_rate_sweep:
                        expt.practice_error_rate = per
                        do_baseline_runs(expt)

def baseline_sweep2(expt):
    var_diag_interval_sweep = [(0.05, 0.1)]
    num_training_frames_per_phoneme_sweep = [5, 10, 20]
    training_error_rate_sweep = [0.0, 0.01, 0.05, 0.1, 0.2]
    num_practice_frames_per_phoneme_sweep = [0, 100, 1000]
    practice_error_rate_sweep = [0.1, 0.2, 0.4]
    dimension_sweep = [3,5,10,20]

    for d in dimension_sweep:
        expt.num_features = d
        for vdi in var_diag_interval_sweep:
            expt.var_diag_interval = vdi
            for ntfpp in num_training_frames_per_phoneme_sweep:
                expt.num_training_frames = ntfpp * expt.num_phonemes
                for ter in training_error_rate_sweep:
                    expt.training_error_rate = ter
                    for npfpp in num_practice_frames_per_phoneme_sweep:
                        expt.num_practice_frames = npfpp * expt.num_phonemes
                        for per in practice_error_rate_sweep:
                            expt.practice_error_rate = per
                            do_baseline_runs(expt)


def baseline_sweep5(expt):
    var_intervals_sweep = [
                           ((0.08, 0.1), (0.2, 0.4)),
                           ((0.1, 0.15), (0.3, 0.5)),
                           ((0.1, 0.2), (0.5, 0.7)),
                           ((0.1, 0.3), (0.5, 1.0))
                           ]

    num_training_frames_per_phoneme_sweep = [5, 10, 50, 100, 500]
    training_error_rate_sweep = [0.0, 0.1, 0.2]
    num_practice_frames_per_phoneme_sweep = [0]

    for (vodi, vdi) in var_intervals_sweep:
        expt.var_diag_interval = vdi
        expt.var_offdiag_interval = vodi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    do_baseline_runs(expt)


def ddt_sweep0(expt):
    var_intervals_sweep = [((0.0, 0.0), (0.01, 0.03)),
                           ((0.0, 0.01), (0.03, 0.06)),
                           ((0.0, 0.01), (0.05, 0.1)),
                           ((0.0, 0.01), (0.05, 0.2))]
#    num_training_frames_per_phoneme_sweep = [3, 5, 7, 10, 20, 30]
    num_training_frames_per_phoneme_sweep = [5, 10]
#    training_error_rate_sweep = [0.0, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    training_error_rate_sweep = [0.05, 0.10]
#    num_practice_frames_per_phoneme_sweep = [0, 100, 500, 1000]
    num_practice_frames_per_phoneme_sweep = [500]
#    practice_error_rate_sweep = [0.05, 0.1, 0.2, 0.3, 0.4]
    practice_error_rate_sweep = [0.1, 0.2, 0.3, 0.4]

    for (vodi, vdi) in var_intervals_sweep:
        expt.var_diag_interval = vdi
        expt.var_offdiag_interval = vodi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    for per in practice_error_rate_sweep:
                        expt.practice_error_rate = per
                        ddt_results = do_ddt_runs(expt)
                        summarize(expt, ddt_results)

def ddt_sweep5(expt):
    var_intervals_sweep = [
                           ((0.08, 0.1), (0.2, 0.4)),
                           ((0.1, 0.15), (0.3, 0.5)),
                           ((0.1, 0.2), (0.4, 0.7)),
                           ((0.1, 0.3), (0.5, 1.0))
                           ]
    num_training_frames_per_phoneme_sweep = [5, 10, 50, 100]
    training_error_rate_sweep = [0.10, 0.20]
    num_practice_frames_per_phoneme_sweep = [500, 1000]
    practice_error_rate_sweep = [0.1, 0.3, 0.5]

    for (vodi, vdi) in var_intervals_sweep:
        expt.var_diag_interval = vdi
        expt.var_offdiag_interval = vodi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    for per in practice_error_rate_sweep:
                        expt.practice_error_rate = per
                        ddt_results = do_ddt_runs(expt)
                        summarize(expt, ddt_results)

def ddt_sweep4(expt):
    var_intervals_sweep = [((0.0, 0.01), (0.03, 0.06)),
                           ((0.0, 0.01), (0.05, 0.1)),
                           ((0.0, 0.01), (0.05, 0.2)),
                           ((0.02, 0.04), (0.1, 0.2)),
                           ((0.03, 0.06), (0.15, 0.3)),
                           ((0.05, 0.1), (0.2, 0.4))
                           ]
    num_training_frames_per_phoneme_sweep = [5, 10, 20, 40, 100]
    training_error_rate_sweep = [0.05, 0.10]
    num_practice_frames_per_phoneme_sweep = [500]
    practice_error_rate_sweep = [0.1, 0.2, 0.3, 0.4, 0.5]

    for (vodi, vdi) in var_intervals_sweep:
        expt.var_diag_interval = vdi
        expt.var_offdiag_interval = vodi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    for per in practice_error_rate_sweep:
                        expt.practice_error_rate = per
                        ddt_results = do_ddt_runs(expt)
                        summarize(expt, ddt_results)

def ddt_sweep2(expt):
    expt.var_offdiag_interval = (0.0, 0.0)
    var_diag_interval_sweep = [(0.01, 0.03)]
    num_training_frames_per_phoneme_sweep = [10]
    training_error_rate_sweep = [0.10]
    num_practice_frames_per_phoneme_sweep = [500]
    practice_error_rate_sweep = [0.5, 0.6, 0.7, 0.9]

    for vdi in var_diag_interval_sweep:
        expt.var_diag_interval = vdi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    for per in practice_error_rate_sweep:
                        expt.practice_error_rate = per
                        ddt_results = do_ddt_runs(expt)
                        summarize(expt, ddt_results)

def ddt_sweep3(expt):
    expt.var_offdiag_interval = (0.0, 0.0)
    var_diag_interval_sweep = [(0.01, 0.03)]
    num_training_frames_per_phoneme_sweep = [10]
    training_error_rate_sweep = [0.10]
    num_practice_frames_per_phoneme_sweep = [500]
    practice_error_rate_sweep = [0.99]

    for vdi in var_diag_interval_sweep:
        expt.var_diag_interval = vdi
        for ntfpp in num_training_frames_per_phoneme_sweep:
            expt.num_training_frames = ntfpp * expt.num_phonemes
            for ter in training_error_rate_sweep:
                expt.training_error_rate = ter
                for npfpp in num_practice_frames_per_phoneme_sweep:
                    expt.num_practice_frames = npfpp * expt.num_phonemes
                    for per in practice_error_rate_sweep:
                        expt.practice_error_rate = per
                        ddt_results = do_ddt_runs(expt)
                        summarize(expt, ddt_results)

def ddt_profile_sweep0(expt):
    expt.var_offdiag_interval = (0.0, 0.0)
    expt.var_diag_interval = (0.01, 0.03)
    expt.num_training_frames = 10 * expt.num_phonemes
    expt.training_error_rate = 0.10
    expt.num_practice_frames = 100 * expt.num_phonemes
    expt.practice_error_rate = 0.3
    expt.num_runs = 1

    ddt_results = do_ddt_runs(expt)
    summarize(expt, ddt_results)


def do_simple_allele_test(expt):
    gen = DataGenerator(expt.num_phonemes, expt.num_features,
                        expt.var_diag_interval, expt.var_offdiag_interval)
    test_data = gen.generate_simulated_data(expt.num_test_frames)

    for run_idx in range(0, expt.num_runs):
        training_data, num_errors = make_training_data(gen, expt)
        # select training data frames to be tested, put into sample_training_frames
        # sample_training_frames is a subset of the training data consisting of some
        # errorful frames and some correct frames - we hope to identify the
        # incorrect frames

        # For now, use first 5 frames and last 5.  The former will have errors and the
        # latter will be correct
        n = len(training_data)
        assert( n * expt.training_error_rate > 5)   # number of errorful points
        assert( n * (1-expt.training_error_rate) > 5)  # number of correct points
        sample_training_frame_indices = range(0,5) + range(n-5, n)

        c = SimpleClassifier(gen.get_labels(), gen.num_features)
        c.train_all(training_data)

        all_results = []
        for i in sample_training_frame_indices:
            label = training_data[i][0]
            a = SimpleAllele(c, [label])
            
            # subtract (label, frame) from training_data for active phoneme
            alt_data = training_data[:i] + training_data[i+1:]
            
            # train alternate model in allele on alternate data
            a.train_variants(alt_data)
            # print a.make_details_string()
            
            results = measurePrimaryAndVariantAccuracy(a, test_data)
            print results
            all_results.append(results)
        print 'End run %d \n' % (run_idx,)

def summarize_one(expt, result_list, grep_label):
    num_rejected = 0
    num_accepted = 0
    total = 0
    for i,run in enumerate(result_list):
        print "Run %d" % (i,)
        for frame_result in run:
            ret, printable = process_frame_result(frame_result)
            print printable
            total += 1
            if ret[0] == 'REJECTED':
                num_rejected += 1
            elif ret[0] == 'ACCEPTED':
                num_accepted += 1
            else:
                assert(False)
    percent_accepted = (float(num_accepted) / total) * 100
    percent_rejected = (float(num_rejected) / total) * 100
    print("Accepted: %.2f%% (%d / %d)  Rejected: %.2f%% (%d / %d)" %
          (percent_accepted, num_accepted, total, percent_rejected, num_rejected, total))
    print("%s, %.2f, %.2f, %s, %s, %s, %s, %s" % ( grep_label,
                                                   expt.var_diag_interval[0],
                                                   expt.var_diag_interval[1],
                                                   expt.num_training_frames,
                                                   expt.training_error_rate,
                                                   expt.practice_error_rate,
                                                   percent_accepted,
                                                   percent_rejected))
          
        
def summarize(expt, ddt_results):    
    print str(expt) + '\n'
    print 'Error training frames:'
    summarize_one(expt, ddt_results['Error'], "___ERR")
    print 'Correct training frames:'
    summarize_one(expt, ddt_results['Correct'], "___COR")


def original_main():

    expt = DDTExperimentPars(50,            # num_phonemes
                             24,            # num_features
                             (0.0, 0.0),    # var_offdiag_interval
                             (0.01, 0.05),  # var_diag_interval
                             2000,          # num_test_frames
                             50 * 5,        # num_training_frames
                             10000,          # num_practice_frames
                             0.03,          # training_error_rate
                             0.30,          # practice_error_rate
                             10,             # num_runs
                             )

#    print expt
#    simple_test(expt)
#    do_complete_baseline_sweep(expt)
#    do_simple_allele_test(expt)
#    ddt_results = do_ddt_runs(expt)
#    summarize(ddt_results)
#    ddt_sweep0(expt)
#    baseline_sweep2(expt)
#    ddt_sweep3(expt)
#    ddt_profile_sweep0(expt)
#    ddt_sweep4(expt)
#    ddt_sweep5(expt)
    baseline_sweep5(expt)


if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()
