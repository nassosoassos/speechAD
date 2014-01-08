###########################################################################
#
# File:         command.py
# Date:         Tue 16 Jun 2009 12:38
# Author:       Ken Basye
# Description:  CommandLineProcessor, a dataflow processing element for executing command line subprocesses
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
A dataflow processing element for executing command line subprocesses

The CommandLineProcessor class provides a dataflow processing element that
executes command line subprocesses.  Activation is caused by being sent an event
consisting of a dictionary of variable bindings; any other event type is an
error.  Upon successful completion of the subprocess, the element sends the
dictionary to its successor.  If the command cannot be run, or runs and returns
a non-zero value, an error is thrown and no event is sent to the successor.  The
*last_command_line*, *last_stdout_output*, and *last_stderr_output* properties
can be used to see what command was run and the output to the two streams.
"""

import re
from cStringIO import StringIO
from onyx.util import streamprocess
from onyx.util import process
from onyx.util import timestamp

class CommandLineProcessor(streamprocess.ProcessorBase):
    def __init__(self, command_line, sendee=None, sending=True, variable_dict={}, notify_stream=None):
        """
        CommandLineProcessors are constructed with five arguments.
        *command_line* gives the form of the command line and may be either a
        string or a sequence of strings.  *sendee* and *sending* are the usual
        optional arguments for a dataflow processing element; note that if
        sending is True, the sendee must be set before the element's *process*
        function is called.  *variable_dict* is a dictionary with keys which are
        variable names and values to bind them to; this argument defaults to {}.
        If *notify_stream* is not None, it should be Python file object (like
        sys.stdout), and the processor will write a time-stamped copy of the
        command line to the stream before the command is executed, and a
        time-stamped notice to the stream when it finishes.
        
        >>> result = list()

        >>> clp0 = CommandLineProcessor(r"ls -d /usr", result.append)
        
        When a CommandLineProcessor is sent an event in the form of a
        dictionary, the command is run and a True is sent to the sendee.
        
        >>> clp0.process({})
        >>> result
        [{}]
        
        Other event types are errors and there is no event sent out.

        >>> clp0.process(False)
        Traceback (most recent call last):
        ...
        ValueError: Expected event of type dict, got one of type <type 'bool'>

        >>> result
        [{}]

        The command may refer to variables, which will be looked up in the
        event dictionary.  A variable is a symbol preceded by a '$' or it may be
        written as ${symbol}.
        
        >>> clp1 = CommandLineProcessor("ls -d $dir", result.append)
        >>> clp1.process({"dir":r"/usr"})
        >>> result
        [{}, {'dir': '/usr'}]
        >>> clp1.last_command_line
        'ls -d /usr'

        Variables will also be looked up in the optional *variable_dict*
        argument to the constructor.

        >>> clp2 = CommandLineProcessor("ls -d ${dir}", result.append, variable_dict={"dir":r"/usr"})
        >>> clp2.process({})
        >>> result
        [{}, {'dir': '/usr'}, {}]
        >>> clp2.last_command_line
        'ls -d /usr'

        Commands may be given as sequences of strings.  Also note that bindings
        in the event take precedence over bindings given to the constructor.
        
        >>> clp3 = CommandLineProcessor(("ls", "-d", "${dir}"), result.append, variable_dict={"dir":r"/etc"})
        >>> clp3.process({"dir":r"/usr"})
        >>> result
        [{}, {'dir': '/usr'}, {}, {'dir': '/usr'}]
        >>> clp3.last_command_line
        'ls -d /usr'

        Variable dictionaries may have values which are not strings, the command
        will be built by converting the value to a string.

        >>> note_stream = StringIO()
        >>> var_dict = {"dir":r"/usr", "one":1}
        >>> clp4 = CommandLineProcessor(("ls", "-d", "-$one", "${dir}"), result.append, notify_stream=note_stream)
        >>> clp4.process(var_dict)
        >>> result
        [{}, {'dir': '/usr'}, {}, {'dir': '/usr'}, {'dir': '/usr', 'one': 1}]

        >>> clp4.last_command_line
        'ls -d -1 /usr'
        >>> clp4.last_stdout_output.strip()
        '/usr'
            
        >>> clp4.last_stderr_output
        ''
        >>> print note_stream.getvalue()  # doctest: +ELLIPSIS
        [20...] Starting: ls -d -1 /usr
        [20...] Finished: ls -d -1 /usr
        <BLANKLINE>
        
        """

        self._command_args = command_line.split() if isinstance(command_line, str) else tuple(command_line)
        if len(self._command_args) < 1:
            raise ValueError("Expected a non-empty command line, but got %s" % (command_line,))
        label = '$ ' + self._command_args[0]
        super(CommandLineProcessor, self).__init__(sendee, sending=sending, label=label)
        self._static_variable_dict = variable_dict
        self._last_stdout_output = self._last_stderr_output = self._last_command_line = ""
        self.set_notify_stream(notify_stream)
                
    # XXX Announcing commands, their output, completion, etc.

    @property
    def last_command_line(self):
        return self._last_command_line

    @property
    def last_stdout_output(self):
        return self._last_stdout_output

    @property
    def last_stderr_output(self):
        return self._last_stderr_output

    @property
    def notify_stream(self):
        return self._notify_stream

    def set_notify_stream(self, stream):
        if stream is not None and not isinstance(stream, (file, StringIO().__class__)):
            raise ValueError("expected notify stream to be a file, got %s" % (stream,))
        self._notify_stream = stream

    def _make_command_sequence(self, var_dict):
        var_re = re.compile(r"\$(\w+)|\$\{(\w+)\}")
        def repl_fn(var_match):
            matches = var_match.groups()
            # If we're in this function at all, we had a match, and it had to be
            # on one group or the other.
            varname = matches[1] if matches[0] is None else matches[0]
            if not var_dict.has_key(varname):
                raise ValueError("Expected variable dictionary to have variable %" % (varname,))
            return str(var_dict[varname])
        result = [var_re.sub(repl_fn, arg) for arg in self._command_args]
        return tuple(result)

    def process(self, event):
        if not isinstance(event, dict):
            raise ValueError("Expected event of type dict, got one of type %s" % (type(event),))

        # We want the bindings in the event to override those in the static
        # dict, so we make of copy of the static dict and update the copy with
        # the event
        bindings = dict(self._static_variable_dict)
        bindings.update(event)
        
        command_args = self._make_command_sequence(bindings)
        
        # This will throw with a useful error message if the command can't be
        # run or if it runs and fails.
        if self.notify_stream:
            note = '[' + timestamp.get_timestamp() + '] Starting: ' + ' '.join(command_args) + '\n'
            self.notify_stream.write(note)
        (self._last_stdout_output,
         self._last_stderr_output,
         self._last_command_line) = process.subprocess(command_args)

        # If we got here, the subprocess finished successfully.
        if self.notify_stream:
            note = '[' + timestamp.get_timestamp() + '] Finished: ' + ' '.join(command_args) + '\n'
            self.notify_stream.write(note)
        self.send(event)

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



