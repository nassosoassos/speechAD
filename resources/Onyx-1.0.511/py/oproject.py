###########################################################################
#
# File:         oproject.py (directory: py)
# Date:         15-Sep-2009
# Author:       Hugh Secker-Walker
# Description:  Module for the OnyxProject Python tools
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
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
OnyxProject constants and utility functions

Running this module as a script will print the module's platform string to
stdout.

Module Attributes
-----------------

platform

  A string encoding the runtime platform.  It consists of five parts separated
  by dash: platform, os, architecture, data-width, endianness

  E.g. ``darwin-posix-i386-32bit-le``
"""

def _make_platform():
    # runtime identifier: platform, os, architecture, data-width, endianness
    # e.g. darwin-posix-i386-32bit-le
    import os, sys, platform, re
    machine = platform.machine()
    width = platform.architecture()[0]
    platform = "%s-%s-%s-%s-%s" % (sys.platform,
                                   os.name,
                                   re.compile(r'^i[3456]86$').sub('i386', machine),
                                   width,
                                   'le' if sys.byteorder == 'little' else 'be',)
    platform = platform.lower()
    return platform
platform = _make_platform()
assert len(platform.split('-')) == 5
    
if __name__ == '__main__':
    print platform
