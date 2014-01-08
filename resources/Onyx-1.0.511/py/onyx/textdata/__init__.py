###########################################################################
#
# File:         __init__.py (package: onyx.textdata)
# Date:         17-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Package initialization for onyx.textdata
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
The textdata formalisms are used to represent data in human-readable text files.

There are two Python-based implementations of textdata
representations.  There is the stand-alone Textdata format and the
supporting toolset for reading and writing data in this format.  There
is the Yamldata toolset which has support for reading and writing data
using the YAML format.  The Yamldata format is preferred.  However,
the use of YAML by the Yamldata format is strongly influenced by the
basic approach used in Textdata.  At the client level, the usage of
the two sets of tools is almost completely interchangeable.

Textdata is both a format and a set of tools that provide a simple way
to unambiguously declare data as space-separated text tokens in
line-based records.
It was designed to support the diverse needs of researchers to handle
text-based data in a more-or-less structured manner.  As such, it is
minimally intrusive and readily accomodates existing ad hoc plain-text
formatting schemes.
At the same time, Textdata is well-specified so it is suitable for use
as the native serialized representation of complex data structures.
Using a small header, it supports data typing and versioning, a
configurable comment prefix, and configurable escaping of
non-graphical characters.

The Textdata library supports writing and reading textdata, ensuring
that well-formed textdata streams are written, verifying correctness
when streams are being read, and taking care of idiomatic textdata
usage.  The Textdata command-line tools can be used to perform common
operations on text data.  Also, their source code provides real-world
examples of how to use the library.

    >>> True
    True
"""

__all__ = ['yamldata', 'textdata']


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

