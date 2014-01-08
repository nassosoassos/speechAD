###########################################################################
#
# File:         dotdisplay.py
# Date:         7-Jul-2009
# Author:       Hugh Secker-Walker
# Description:  Tools and mix-in classes for support DOT-based graphical display
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
Tools and mix-in classes for DOT-based graphical display.

:attr:`DISPLAY_COMMAND_FORMAT` is the default command-line for displaying a DOT
file, where %s is replaced with the name of the file to display
"""
from __future__ import with_statement
import os
import onyx.util, onyx.util.process

# XXX need platform-specific default
DISPLAY_COMMAND_FORMAT = "open -a /Applications/Graphviz.app %s"

class DotDisplay(object):
    """
    Mix-in baseclass for displaying a DOT rendering of an object.

    Uses the object's :meth`dot_iter` method to display a DOT rendering of the
    object.
    """
    def dot_display(self, dot_iter=None, temp_file_prefix=None, display_command_format=None, **kwargs):
        r"""
        Display a dot-generated representation of the graph.  

        Optional *dot_iter* is the generator function to pull on for text lines
        of DOT code.  It is called with *kwargs*.  It defaults to the object's
        ``dot_iter`` method

        Optional *temp_file_prefix* is the prefix used for the temporary
        filename.  It defaults to the name of the type of ``self``.

        Optional *display_command_format* is a formatting string, with %s where the
        temporary filename goes, that is used to generate the command that will
        display the file.  It defaults to the the value of the module's
        :attr:`DISPLAY_COMMAND_FORMAT` attribute.

        Remaining keyword arguments, *kwargs*, are handed to the *dot_iter*
        generator function.  In general, no arguments are necessary to get an
        object to display.

        Returns a tuple of four items: (temp_filename, stdout, stderr, cmd),
        where temp_filename is the name of the temporary file that gets created
        for the display command to use, stdout and stderr are the standard-out
        and standard-error from the command, and cmd is a string representing
        the command.  The caller is responsible for removing temp_filename.

        Raises :exc:`AttributeError` if *dot_iter* is not given and the
        object doesn't have a ``dot_iter`` method

        Raises :exc:`~onyx.SubprocessError` if the command fails to execute
        or if the command exits with a non-zero return code.

        """
        if dot_iter is None: dot_iter = self.dot_iter
        if temp_file_prefix is None: temp_file_prefix = type(self).__name__ + '_'
        if display_command_format is None: display_command_format = DISPLAY_COMMAND_FORMAT

        with onyx.util.opentemp('wb', suffix='.dot', prefix=temp_file_prefix) as (name, outfile):
            for line in dot_iter(**kwargs):
                outfile.write(line)
        stdout, stderr, cmd = onyx.util.process.subprocess((display_command_format % (name,)).split())
        return name, stdout, stderr, cmd

def test_DotDisplay():
    r"""
    Function for testing DotDisplay functionality.

    >>> obj = test_DotDisplay()
    >>> temp_filename, stdout, stderr, cmd = obj.dot_display(display_command_format="cat %s")
    >>> temp_filename #doctest: +ELLIPSIS
    '/tmp/TestDotDisplay_....dot'
    >>> cmd #doctest: +ELLIPSIS
    'cat /tmp/TestDotDisplay_....dot'
    >>> stdout
    'digraph { \n  n00 [label="node_00"]; \n  n01 [label="node_01"]; \n  n00 -> n01; \n} \n'
    >>> stderr
    ''
    >>> os.remove(temp_filename)
    """
    class TestDotDisplay(DotDisplay):
        def dot_iter(self, **kwargs):
            yield 'digraph { \n'
            yield '  n00 [label="node_00"]; \n'
            yield '  n01 [label="node_01"]; \n'
            yield '  n00 -> n01; \n'
            yield '} \n'
    return TestDotDisplay()

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
