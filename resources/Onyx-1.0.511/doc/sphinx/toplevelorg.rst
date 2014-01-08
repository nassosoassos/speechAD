..
  =========================================================================
  =
  = File:         toplevelorg.rst
  = Date:         Tue 11 Aug 2009 12:23
  = Author:       Ken Basye
  = Description:  Top-down description of main files and directories
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
  =========================================================================

====================
Project Organization
====================

This file provides an overview of the important files and directories that make
up Onyx.

Files: 
------

* `changes.txt <../../../../changes.txt>`_ - the change log for the process.  Developers add comments to this
  file (in an automated way) whenever they check new work into the project.

* :file:`SConscript` - the top-most of many Scons control files - the purpose of this
  one is mostly to point to the rest

* :file:`SConstruct` - the root configuration file for the build/test system, which is based
  on Scons

Directories:
------------

* :file:`bin` - Developer tools, some binaries used by Onyx for file reading, and
  some demo scripts.  The bin directory contains platform-specific
  subdirectories for binary executables.

* :file:`scons` - A complete Scons package; you probably don't want to change this at
  all except by replacing it with a completely new version.

* :file:`site-scons` - Project-specific additions to the build system

* :file:`templates` - stubs of various file types to use as starting points when adding
  new files to the project

* :file:`build` - destination directory for a Scons build.  If you run 'bin/scons', this
  directory will be created or overwritten.  Although there's a complete version
  of the project source in this directory, you probably don't want to make
  modifications here.  All builds appear in subdirectories named for the build
  platform.

* :file:`doc` - static, i.e. non-built, documentation in various forms, also the sources
  and machinery that support documentation building.  For the built
  documentation, which both much easier to browse and more complete, see
  "build/<platform>/doc/sphinx/_build/html/index.html' (XXX need to make this
  easier to get to).

* :file:`py` - External Python modules and core Python source code for the project.  The
  core Python source code is in the 'onyx' subdirectory, see :ref:`core-label`.

* :file:`cpp` - C++ source code for the project.  This directory also contains the
  'pylib' subdirectory, where binary library files for each supported platform
  are kept.  (XXX need a description of how binary library system works,
  particularly WRT built binary libraries).

* :file:`data` - Sample input data used for testing

* :file:`projects` - Experiments, demos, and other work not part of the project core.

* :file:`sandbox` - Scratch work area that is still checked in and can be tested.  The
  things in here tend to be more half-baked than the things in projects.




