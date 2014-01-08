///////////////////////////////////////////////////////////////////////////
//
// File:         README.txt (directory: py)
// Date:         21-Sep-2007
// Author:       Hugh Secker-Walker
// Description:  Readme for the Python code in the project
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2007 The Johns Hopkins University
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.  See the License for the specific language governing
// permissions and limitations under the License.
//
///////////////////////////////////////////////////////////////////////////

[ Note: as of August, 2009, this readme is somewhat out of date, but still has
  some info not found elsewhere.  See doc/sphinx/developer.rst and
  doc/sphinx/user.rst for more up-to-date instructions. ]

This is the readme for the top-level Python directory of Onyx .

You need Python 2.5.2 or better to use the project's Python code.  The Python
interpreter should be on your path.

Note that the 'python' command must run the Python 2.5 interpreter.  You may
need to use symbolic links in your bin directory, or in the project's bin
directory to make this work.  For instance, on Mac OS X 10.4, the default
'python' command run's Apple's version of Python 2.3.  So you have to put the
location of Python 2.5 earlier on your PATH, or use a symbolic link.

To use the Python code in the project, you need to put the project's Python
directory on your PYTHONPATH.  The project's Python directory is the directory
in which this readme is found.  E.g., under Linux, you might have the following
in your ~/.profile:

  export PYTHONPATH=~/onyx/my_onyx_code_branch/py

If you have multiple instances of the project installed, you will have to manage
the PYTHONPATH appropriately for each installation.  If you have workable
solution to this awkwardness, feel free to make suggestions.  Having Python
itself be part of the project is one possible approach.

When PYTHONPATH is correctly set up then every instance of Python that you run
will have access to the project's Python packages.  These packages are found in
the directories under the 'onyx' subdirectory.  Note that py/sitecustomize.py
will be run by every Python instance.  At present this script does nothing, but
in the past it fiddled the Python environment.

We also have some third-party pure Python libraries in the py directory in order
to simplify their maintenance.  Like the onyx packages, they will be available
for import via PYTHONPATH.


We endeavor to use Python's doctest module to give combined documentation and
unit testing.  The doctest module makes it possible to write documentation that
contains working examples of the documented features.  Running a module from the
command-line will run the documentation-based unit tests.  E.g., to run the
doctests in the onyx.signalprocessing.spectrum module, run the following, from
any directory:

  $ python -m onyx.signalprocessing.spectrum

To exercise the Onyx builtin objects, do:

  $ python -m onyx.builtin

The goal is to have every released module use this combined
documentation and unit testing facility.

The SCons framework includes tools that help with this.  For instance, from the
top of the project look at ./SConstruct, ./SConscript, and then
e.g. ./py/onyx/signalprocessing/SConscript and you will (begin to) see how
onyx/signalprocessing/spectrum.py gets compiled and tested when you run scons
from the top level.

The top-level ./templates directory has templates for some typical files that
are written when you incorporate new code into the project.  The key steps are
to have a SConscript file that shows your Python filenames to env.PyFiles() and,
if that's a new SConscript file, to edit the top-level ./SConscript to add your
new SConscript file to the list of files that it runs.  Beyond that, use the
doctest facility extensively in your code.


If you're an emacs user, you may want to load python-mode.el which
implements a useful major mode for editing and running python code.
