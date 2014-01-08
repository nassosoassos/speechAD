===========================================================================
=
= File:         README.txt (directory: cpp/sph2pipe)
= Date:         16-Sep-2009
= Author:       Hugh Secker-Walker
= Description:  Notes regarding the use of sph2pipe
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
===========================================================================

Onyx includes a version of sph2pipe.  This utility is useful for unpacking NIST
Sphere files that are compressed with the shorten codec, the format most often
used by LDC.

The C files in this directory were copied from sph2pipe version 2.5 which was
obtained from the LDC in February 2009:
  ftp://ftp.ldc.upenn.edu/pub/ldc/misc_sw/sph2pipe_v2.5.tar.gz
See also the Onyx file: 
  doc/licenses/LICENSE_sph2pipe

Some of the source files from version 2.5 were then modified slightly so as to
permit compilation under GCC 4.4.1 without errors or warnings, e.g.:
  gcc -Wall -Werror file_headers.c shorten_x.c sph2pipe.c
