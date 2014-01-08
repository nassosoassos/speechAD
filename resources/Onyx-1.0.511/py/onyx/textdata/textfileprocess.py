###########################################################################
#
# File:         textfileprocess.py
# Date:         Wed 23 Jan 2008 12:33
# Author:       Ken Basye
# Description:  Produce streams of line events from textfiles.
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008 The Johns Hopkins University
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
>>> True
True
"""
import os
from onyx.util.streamprocess import ProcessorBase

class TextFileEventBase(object):
    # TODO: Block instantiation?
    pass

class TFNewFileEvent(TextFileEventBase):
    def __init__(self):
        self.filename = None
    def __repr__(self):
        s =  "%s: %s" % (self.__class__.__name__, self.filename)
        return s

class TFLineEvent(TextFileEventBase):
    def __init__(self):
        self.line = None
    def __repr__(self):
        s =  "%s: [%s] (%d)" % (self.__class__.__name__, self.line, len(self.line))
        return s

class TFFinishedEvent(TextFileEventBase):
    # TODO: make this class a singleton
    def __init__(self):
        pass
    def __repr__(self):
        s =  "%s" % (self.__class__.__name__,)
        return s


class TextFileProcessor(ProcessorBase):
    """
    From a list of text files, produce a stream of events corresponding to
    lines in the files.  Separate events are sent for the beginning of each
    file and for reaching the end of the entire set (subsequent calls to process()
    continue to generate this event).  Note that lines are returned with newlines
    still at the end, in the style of Python line-based file reading.

    >>> module_dir, module_name = os.path.split(__file__)

    >>> files = tuple(os.path.join(module_dir, text_file) for text_file in ('foo.txt', 'bar.txt', 'baz.txt'))
    >>> result = []
    >>> tfs = TextFileProcessor(files, result.append)
    >>> while not tfs.finished():
    ...    tfs.process()
    >>> result  #doctest: +ELLIPSIS
    [TFNewFileEvent: ...foo.txt, TFLineEvent: [#!MLF!#
    ] (8), TFLineEvent: [foo 1
    ] (6), TFLineEvent: [foo 2
    ] (6), TFLineEvent: [foo 3
    ] (6), TFNewFileEvent: ...bar.txt, TFLineEvent: [bar 4
    ] (6), TFLineEvent: [bar 5
    ] (6), TFLineEvent: [bar 6
    ] (6), TFNewFileEvent: ...baz.txt, TFLineEvent: [baz 7
    ] (6), TFLineEvent: [baz 8
    ] (6), TFLineEvent: [baz 9
    ] (6), TFFinishedEvent]

    >>> tfs.process()
    >>> tfs.process()
    >>> result[-3:]
    [TFFinishedEvent, TFFinishedEvent, TFFinishedEvent]

    """

    def __init__(self, files, sendee=None, sending=True):
        super(TextFileProcessor, self).__init__(sendee, sending=sending)
        self._file_iter = iter(files)
        self._current_file = None
        self._current_iter = None
        self._finished = False
        if sendee:
            self.set_sendee(sendee)    

    def finished(self):
        return self._finished

    def process(self):
        if self._finished:
            # For persistent callers
            evt = TFFinishedEvent()
        else:
            try:
                if not self._current_iter:
                    raise StopIteration
                line = self._current_iter.next()
                evt = TFLineEvent()
                evt.line = line
            except StopIteration:
                filename = self.open_next_file()
                if filename:
                    evt = TFNewFileEvent()
                    evt.filename = filename
                else:  # end of the line
                    evt = TFFinishedEvent()
                    self._finished = True
        self.send(evt)

    def open_next_file(self):
        if self._current_file is not None:
            self._current_file.close()
        try:
            filename = self._file_iter.next()
        except StopIteration:
            return None
        self._current_file = open(filename, 'rU')
        if self._current_file is None:
            raise ValueError("Unable to open file: %s" % (filename,))
        self._current_iter = iter(self._current_file)
        return filename

if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()

