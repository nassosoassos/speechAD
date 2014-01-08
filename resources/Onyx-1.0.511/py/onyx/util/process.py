###########################################################################
#
# File:         process.py (directory: ./py/onyx/util)
# Date:         20-Apr-2009
# Author:       Hugh Secker-Walker
# Description:  Utilities for running processes
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
Utilities for running processes.
"""

from subprocess import Popen, PIPE
import onyx

def subprocess(args):
    """
    Execute a subprocess using the strings in args.

    Returns a triple, (stdout, stderr, cmd), where stdout is the bytes from the
    subprocess standard out, stderr is the bytes from the subprocess standard
    error, and cmd is a single string giving the command and its arguments.

    Raises onyx.SubprocessError if the command fails to execute or if the command
    exits with a non-zero return code.

    >>> stdout, stderr, cmd = subprocess(('true', 'unused'))
    >>> stdout
    ''
    >>> stderr
    ''
    >>> cmd
    'true unused'

    >>> subprocess(('no_such_command', '--version'))
    Traceback (most recent call last):
      ...
    SubprocessError: [Errno 2] got 'No such file or directory' while trying to execute 'no_such_command --version'

    >>> subprocess(('false',))
    Traceback (most recent call last):
      ...
    SubprocessError: [Errno 1] command failed: 'false' : 
    """
    cmd = ' '.join(args)
    try:
        proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False, close_fds=True)
    except OSError, e:
        raise onyx.SubprocessError(e.errno, "got %r while trying to execute %r" % (e.strerror, cmd))
    stdout, stderr = proc.communicate()
    assert proc.returncode is not None
    if proc.returncode != 0:
        raise onyx.SubprocessError(proc.returncode, "command failed: %r : %s" % (cmd, stderr.strip(),))
    return stdout, stderr, cmd

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
