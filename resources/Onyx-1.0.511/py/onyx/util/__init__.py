###########################################################################
#
# File:         __init__.py (package: onyx.util)
# Date:         15-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Package initialization for onyx.util
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 - 2009 The Johns Hopkins University
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
    Generic utilities

    >>> with opentemp('wb', suffix='.tmp', prefix='onyx_opentemp_test_') as (filename, outfile):
    ...    outfile.write('foobar')
    >>> with open(filename, 'rb') as infile:
    ...    infile.read()
    'foobar'
    >>> os.remove(filename)

"""

from __future__ import with_statement
import os, contextlib, tempfile

# XXX somehow this decorator blocks doctesting in the docstring for this function! - KJB
@contextlib.contextmanager
def opentemp(mode, **mkstemp_args):
    """
    This implements a Python ContextManager for working with temporary files.
    Open a temporary file with *mode* as in open(), e.g. 'wb'.  The *mkstemp_args*
    keyword arguments are handed to mkstemp from Python's tempfile module::
    tempfile.mkstemp([suffix=''[, prefix='tmp'[, dir=None[, text=False]]]]).

    Yields a tuple, (filename, openfile), for use in a 'with' statement, where
    filename is a string naming the file and openfile is the file object.
    Caller is responsible for deleting the file when it's no longer needed.
    """
    fd, filename = tempfile.mkstemp(**mkstemp_args)
    with os.fdopen(fd, mode) as file:
        yield filename, file


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
