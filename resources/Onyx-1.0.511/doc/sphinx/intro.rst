..
 ==========================================================================
 =
 = File:         intro.rst
 = Date:         Mon 26 Jan 2009 17:37
 = Author:       Ken Basye
 = Description:  
 =
 = This file is part of Onyx   http://onyxtools.sourceforge.net
 =
 = Copyright 2009 The Johns Hopkins University
 =
 = Licensed under the Apache License, Version 2.0 (the "License").
 = You may not use this file except in compliance with the License.
 = You may obtain a copy of the License at
 =   http://www.apache.org/licenses/LICENSE-2.0
 = 
 = Unless required by applicable law or agreed to in writing, software
 = distributed under the License is distributed on an "AS IS" BASIS,
 = WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 = implied.  See the License for the specific language governing
 = permissions and limitations under the License.
 =
 ==========================================================================


Introduction to Onyx
====================

Onyx is for doing research and development on machine learning
algorithms.  Onyx was orginally developed at the `Human Language Technology Center
of Excellence <http://web.jhu.edu/hltcoe>`_.  Onyx simplifies the process
of taking a computational research idea and implementing a working experiment.
It uses a dataflow model of how data gets manipulated during an experiment.
Using Onyx, it is straightforward to take an machine-learning idea as
outlined on a whiteboard, e.g. with data flowing through arrows and algorithmic
processing happening in blocks, and turn that idea into a working experimental
configuration.  Furthermore, it is easy to take a working configuration and
either extend it or embed it into a larger experiment.

A key design feature of Onyx is very strong support for what are refered
to as **streaming** or **online** machine-learning algorithms.  Online
alogrithms are increasingly necessary to deal with the rapidly growing volume of
data that is available for use by machine-learning technology.  These algorithms
are distinct from more-traditional batch algorithms.  One defining feature of
online algorithms is that they only get to examine a given piece of data for a
limited time -- once they are done with the data they must let it go and cannot
store it for later.  Online algorithms are a rational approach (perhaps the only
rational approach) to the fact that the availability of data is far outstripping
the resources necessary to store the data.

Onyx is written in `Python <http://python.org>`_, a high-level
interpreted language that is very-well suited for use in both exploratory
research and for advanced technology prototyping and development.  Python, and
Onyx, make it very easy to build online algorithms and models.  The
language itself is easy to learn, and Onyx makes it easy to implement
each each step in an algorithm as a simple function or a simple object.  Onyx
is then used to connect these algorithmic blocks into a dataflow graph.
The experiment can be started, and the models and algorithmic state can be
examined and changed *in situ*, that is, while the experiment is running.


Key Features
------------

Features of Onyx include:

* deep support for experiments with online machine-learning algorithms
* a dataflow architecture supports factoring problems into small algorithmic components
* access to the interpreter to examine live objects, models, algorithm state, etc.
* easy to produce reliable experimental frameworks for colleagues to use
* transparent use of multithreading
* transparent access to grid computing (SGE environment)
* high-performance C libraries for numerical work (Numpy and SciPy)
* straightforward mechanisms for integrating external executables into a dataflow
* detailed documentation with verified example-code
* OpenSource access to all Python source code, illustrating numerous best-practices
* tutorial examples


..
 Further Reading
 ---------------

 See also:

 .. toctree::
    :maxdepth: 2

    vision.rst
    faq.rst
