___________________________________________________________________________
_
_ File:         README.txt
_ Date:         Thu Nov 29 14:24:30 2007
_ Author:       Ken Basye
_ Description:  Information on Delayed Decision Training classes
_
_ This file is part of Onyx   http://onyxtools.sourceforge.net
_
_ Copyright 2007 The Johns Hopkins University
_
_ Licensed under the Apache License, Version 2.0 (the "License").
_ You may not use this file except in compliance with the License.
_ You may obtain a copy of the License at
_   http://www.apache.org/licenses/LICENSE-2.0
_ 
_ Unless required by applicable law or agreed to in writing, software
_ distributed under the License is distributed on an "AS IS" BASIS,
_ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
_ implied.  See the License for the specific language governing
_ permissions and limitations under the License.
_
___________________________________________________________________________


The code supporting preliminary DDT experiments is divided into four
modules in two directories:
  Directory ddt:
    ddt.py - top-level DDT experiment code, also code for constructing
             alleles, which are pairs of classifiers which differ in
             what data was used to train them.  There's some extra
             functionality in SimpleAllele to support having more than
             variant label between the primary and variant model, but
             the current experiments always use just one variant
             label.  An important utility function for generating 
             errorful training data with enough samples for each frame
             is near the top.
    datagen.py - artificial data generation code; this is pretty
                 simple.  Note the two ways to generate data used
                 elsewhere to make sure there's enough data.

  Directory am:
    gaussian.py - Basic multivariate gaussian distributions.  To
                  support use in models it has a scoring function;
                  note that this generates real likelihoods, not
                  log-likelihoods.  To support use in data generation
                  it supports sampling.  This requires Choleski
                  factorization of the variance matrix which in turn
                  requires non-zero variances on the diagonal.  A
                  possibly useful change to this code would be to
                  relax this restriction using the semantics that a
                  model with a zero covariance matrix would not both
                  factoring, but would instead sample by always
                  returning the mean.  Also supports training from
                  multiple examples, but without any support for 
                  relevance, so each training completely clobbers the
                  old values.

    classifier.py - Simple Bayesian classifier using a single
                    diagonal-variance for each class.  Supports
                    training in the obvious fashion (again without
                    relevance).  Some extra code in the middle of the
                    class supports looking at distances between model
                    means. 


Dependencies:  These four modules have a diamond-shape dependency
graph with one extra edge between the top and bottom nodes.  ddt.py,
at the top, imports from all three other modules.  datagen.py and
classifier.py both import from gaussian.py

Various experiments can be run easily by commenting calls in or out in
the last block of ddt.py.  There's some (but not enough) test code in
gaussian.py, and pretty much none in any of the other files :-<.  







