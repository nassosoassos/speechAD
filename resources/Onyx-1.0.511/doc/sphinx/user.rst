..
 ==========================================================================
 =
 = File:         user.rst
 = Date:         11-Aug-2009
 = Author:       Hugh Secker-Walker
 = Description:  Documentation for installing the project as a user
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


User Installation
=================

These instructions are for users users of Onyx who plan to use its packages and
modules for their experimental work, but who don't expect to modify the Onyx
source code.


Prerequisites
-------------

Onyx runs on Linux and Mac OS X (10.4 and 10.5) sytems.

The following packages need to be installed on your system.  Your package
management system should have them readily available.

* `Python 2.5 <http://www.python.org/download/releases/2.5.4/>`_ or the `latest Python 2.* <http://www.python.org/download/>`_  series
* `numpy package <http://www.scipy.org/Download>`_ for Python
* `scipy package <http://www.scipy.org/Download>`_ for Python
* `libsndfile <http://www.mega-nerd.com/libsndfile/>`_
* `mpg123 <http://www.mpg123.de/>`_


Building and testing Onyx relies on some of the standard Unix command-line
toolchains and utilities.  Again, your package management system should have
them readily available.
* ``gcc`` and ``g++`` compilers
* ``md5sum``
* `sphinx documentation generator <http://sphinx.pocoo.org/>`_


Building
--------

* Get the latest tarball
* It will unpack into a directory, e.g. ``Onyx-1.0.497``
* Make sure your PYTHONPATH points to the ``py`` directory of Onyx, e.g. ``$ export PYTHONPATH=~/Onyx-1.0.497/py``

To do the build:

::

  cd ~/Onyx-1.0.497
  bin/scons NO_DOC=1
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  ...  [ for many lines ]
  scons: done building targets.

.. note::

   If the build is successful the SCons output will finish with the
   line ``scons: done building targets.`` If the build failed then the
   SCons output will end with something else, e.g. ``scons: building
   terminated because of errors.`` or ``scons: done building targets
   (errors occurred during build).``

.. note::

   If you have the `Sphinx documentation-building system
   <http://sphinx.pocoo.org>`_ installed, you can run ``bin/scons`` without the
   ``NO_DOC=1`` option, which will also rebuild the project's documentation.


Here's an example of what the SCons output looks like when one of the tests fails.

::

  cd ~/onyx
  bin/scons NO_DOC=1
  scons: Reading SConscript files ...
  scons: done reading SConscript files.
  scons: Building targets ...
  ...  [for many lines]
  scons: *** [build/linux2-posix-x86_64-64bit-le/py/onyx/builtin.log-doctest] Error 1
  scons: building terminated because of errors.

If this happens, you would look in,
e.g. ``build/linux2-posix-x86_64-64bit-le/py/onyx/builtin.log-doctest`` to see
what the problem was.  If you find yourself in this situation and trying to
fiddle code to fix it, **be sure** to make your changes in the source tree and
not in the build tree.  E.g. for the above you might edit the file
``py/onyx/builtin.py``, but you would not edit the file
``build/linux2-posix-x86_64-64bit-le/py/onyx/builtin.py``.

.. note::
   
    Do not edit files in the build tree.  This is where SCons puts its output.
    Your edits will do nothing and will get overwritten.

Using the Installation
----------------------

Once all is well with your local build and testing, then you can use Onyx in
your own modules and scripts.  E.g. here's the top of a complicated script that
pulls in all sorts of acoustic modeling, signal processing, and dataflow
classes::

    from __future__ import division
    import sys, math
    import numpy as N
    from collections import deque
    from itertools import izip, count

    from onyx.am.classifier import AdaptingGmmClassifier, AdaptingGmmClassProcessor
    from onyx.am.gaussian import  GaussianModelBase, SimpleGaussianModel, GaussianMixtureModel
    from onyx.am.modelmgr import  GmmMgr
    from onyx.dataflow.join import SynchronizingSequenceJoin
    from onyx.signalprocessing.htkmfcc import make_fft_abs_processor, make_melcepstral_processor
    from onyx.util.debugprint import DebugPrint, dcheck, dprint
    from onyx.util.streamprocess import FunctionProcessor, SequenceFunctionProcessor, ChainProcessor, SplitProcessor

    etc...

Have fun!
