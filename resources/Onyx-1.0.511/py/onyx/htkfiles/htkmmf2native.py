###########################################################################
#
# File:         htkmmf2native.py
# Date:         Fri 23 Jan 2009 11:35
# Author:       Ken Basye
# Description:  Convert a model file from HTK's mmf format to our native format
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

# Note some extra hoops we go through here because we need the name and not the
# file handle from mkstemp.  On Windows, the subsequent unlink fails unless the
# initial descriptor has been closed.
"""
    Convert a model file from HTK's mmf format to our native format

    >>> module_dir, module_name = os.path.split(__file__)

    >>> infilename = os.path.join(module_dir, "start.mmf")
    >>> fd, outfilename = tempfile.mkstemp(prefix='tmp', suffix='am')
    >>> os.close(fd)
    >>> htkmmf2native(infilename, outfilename)
    >>> filecmp.cmp(os.path.join(module_dir, "start.am"), outfilename)
    True
    >>> os.unlink(outfilename)

    >>> infilename = os.path.join(module_dir, "monophones.mmf")
    >>> fd, outfilename = tempfile.mkstemp(prefix='tmp', suffix='am')
    >>> os.close(fd)
    >>> htkmmf2native(infilename, outfilename)
    >>> filecmp.cmp(os.path.join(module_dir, "monophones.am"), outfilename)
    True
    >>> os.unlink(outfilename)

"""
from __future__ import with_statement
import cStringIO
import tempfile
import os
import filecmp
from onyx.htkfiles.htkmmf import read_htk_mmf_file
from onyx.am.hmm_mgr import write_acoustic_model

def htkmmf2native(htk_file_name, native_file_name):
     with open(htk_file_name) as f_in:
         models, hmm_mgr, gmm_mgr = read_htk_mmf_file(f_in)
     with open(native_file_name, 'wb') as f_out:
         write_acoustic_model(models, gmm_mgr, hmm_mgr, f_out)


def usage(prog_name):
    print("Usage: %s [infile outfile]" % (prog_name,))
    print("With no arguments, run doctests; with infile and outfile, convert infile to outfile")

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        from onyx import onyx_mainstartup
        onyx_mainstartup()
    elif len(sys.argv) == 3:
        htkmmf2native(sys.argv[1], sys.argv[2])
    else:
        usage(sys.argv[0])
