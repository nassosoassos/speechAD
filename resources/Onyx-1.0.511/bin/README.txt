---------------------------------------------------------------------------
-
- File:         README.txt (directory: ./bin)
- Date:         3-Oct-2007
- Author:       Hugh Secker-Walker
- Description:  Readme for the bin directory
-
- This file is part of Onyx   http://onyxtools.sourceforge.net
-
- Copyright 2007 The Johns Hopkins University
-
- Licensed under the Apache License, Version 2.0 (the "License").
- You may not use this file except in compliance with the License.
- You may obtain a copy of the License at
-   http://www.apache.org/licenses/LICENSE-2.0
- 
- Unless required by applicable law or agreed to in writing, software
- distributed under the License is distributed on an "AS IS" BASIS,
- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
- implied.  See the License for the specific language governing
- permissions and limitations under the License.
-
---------------------------------------------------------------------------

This directory contains scripts and executables needed by the project.

For robust behavior, users are encouraged to work in the top directory of the
project and type bin/scons or bin/prep in order to run the commands in this bin
directory.  That way you are sure that the command in the current working tree
is the one that gets used.

It is possible to include this bin directory early on their PATH, but they are
discouraged from so doing because such an approach is error-prone if users have
more than one working tree on their system.

There are also subdirectories, e.g. darwin-posix-i386-32bit-le, that contain
platform-specific binaries that the project's tools use.  These binaries are
used instead requiring the user to install the correct version of these tools on
their platform(s).  The project's tools (SCons and Python) automatically prepend
the correct directory to the PATH that they use.

