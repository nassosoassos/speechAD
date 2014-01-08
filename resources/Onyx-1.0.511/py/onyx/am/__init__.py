###########################################################################
#
# File:         __init__.py (package: onyx.am)
# Date:         15-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Package initialization for onyx.am
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
Package providing Gaussian-based models

Modeling is done with four classes:

- :class:`~onyx.am.gaussian.DummyModel`
- :class:`~onyx.am.gaussian.SimpleGaussianModel`
- :class:`~onyx.am.gaussian.GaussianMixtureModel`
- :class:`~onyx.am.hmm.Hmm`

The first three are all derived from :class:`~onyx.am.gaussian.GaussianModelBase`.

In addition, there are several other classes supporting operations on
collections of models:

- :class:`~onyx.am.modelmgr.GmmMgr` - a collection of GaussianMixtureModels or DummyModels
- :class:`~onyx.am.hmm_mgr.HmmMgr` - a collection of Hmms
- :class:`~onyx.am.bwtrainer.TrainingGraph` - a structure for doing embeddded training of Hmms
- :class:`~onyx.am.classifier.SimpleClassifier` - a collection of SimpleGaussianModels with associated labels
- :class:`~onyx.am.classifier.AdaptingGmmClassifier` - a collection of GaussianMixtureModels with associated labels
- :class:`~onyx.am.classifier.HmmClassifier` - a collection of Hmms with associated labels
- :class:`~onyx.am.classifier.AdaptingGmmClassProcessor` - a processor wrapper around AdaptingGmmClassifier

Scoring in the Acoustic Modeling package
++++++++++++++++++++++++++++++++++++++++

Say something about scoring here.

Adaptation in the Acoustic Modeling package
+++++++++++++++++++++++++++++++++++++++++++

Adaptation interfaces are available at many levels in the AM package.
Also, there are several styles of interface, reflecting both the
nature of the underlying algorithms and user demand.  

====================================================== ============= =============  =========   ===============  ==============
Class                                                  One-shot?      Begin/End?    Relevance   Algorithm        Notes
====================================================== ============= =============  =========   ===============  ==============
:class:`~onyx.am.gaussian.SimpleGaussianModel`         Yes           Yes            Yes         Closed form      The simplest case
:class:`~onyx.am.gaussian.GaussianMixtureModel`        Yes           No             Yes         EM               (1)
:class:`~onyx.am.hmm.Hmm`                              Yes           Yes            No          Baum-Welch       See also TrainingGraph
:class:`~onyx.am.modelmgr.GmmMgr`                      No            Yes            No          EM               (2)
:class:`~onyx.am.hmm_mgr.HmmMgr`                       No            Partial        No          ??               (3)
:class:`~onyx.am.bwtrainer.TrainingGraph`              Yes           ??             No          Baum-Welch       (4)
:class:`~onyx.am.classifier.AdaptingGmmClassifier`     Yes           No             Yes         EM               (5)
:class:`~onyx.am.classifier.AdaptingGmmClassProcessor` Yes           No             Yes         EM               (6)
====================================================== ============= =============  =========   ===============  ==============

Notes:

 1. The GaussianMixtureModel adapt() function can do multiple EM iterations over
    a single collection of datapoints.  Each iteration re-estimates the weights,
    means, and variances used in the next iteration.
 
 2. The GmmMgr class supports accumulation and the application of accumulators.
    This is used by the Hmm and TrainingGraph classes in their implementation of
    Baum-Welch training of Hmms.  It has the potential to be used for an
    external EM implementation, but currently there isn't one.
 
 3. Note about HmmMgr
 
 4. Note about TrainingGraph

 5. The AdaptingGmmClassifier class uses the one-shot adapt() function in
    GaussianMixtureModel to do adaptation.

 6. The AdaptingGmmClassProcessor class uses the one-shot interface of the
    AdaptingGmmClassifier to do adaptation.  An input event consists of a single
    label and a sequence of training frames; these frames are then use together
    in several EM iterations (see note 1.)

"""

__all__ = []


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
