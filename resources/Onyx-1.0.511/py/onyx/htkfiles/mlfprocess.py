###########################################################################
#
# File:         mlfprocess.py
# Date:         Mon 21 Jan 2008 10:02
# Author:       Ken Basye
# Description:  Event streams for HTK master label files
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
Event streams for HTK master label files

Event stream processors are provided at two levels.  MLFProcessor provides a
low-level stream, while MLFBlockProcessor provides a higher-level stream based
on getting the low-level events.
"""

import os, re, exceptions
from onyx.textdata.textfileprocess import TextFileProcessor
from onyx.textdata.textfileprocess import TextFileEventBase, TFLineEvent, TFNewFileEvent, TFFinishedEvent
from onyx.util.streamprocess import ProcessorBase

class MLFRecordStartEvent(object):
    def __repr__(self):
        return "Record start: %s" % (self.record_filename,)

class MLFLabelEvent(object):
    def __repr__(self):
        return ("%s%s %s%s%s" % (
                '%d' % (self.start,) if self.start is not None else '',
                ' %d' % (self.end,) if self.end is not None else '',
                self.phone_label,
                ' %f' % (self.score,) if self.score is not None else '',
                ' ' + self.word_label if self.word_label else ''
                ))

class MLFRecordEndEvent(object):
    def __repr__(self):
        return "Record end: %s" % (self.record_filename,)


class MLFProcessorError(exceptions.StandardError):
    pass

class MLFProcessor(ProcessorBase):
    """
    Generator for mlf events from a stream of character lines (as produced,
    e.g., from reading an mlf file) or TextFileEvents.  Event types include:

    Event types sent:  MLFRecordStartEvent
                       MLFLabelEvent
                       MLFRecordEndEvent

    >>> lines = ['#!MLF!#',\
                 '"*/en_0638-A-001.rec"',\
                 '0 300000 sil 47.421978',\
                 '300000 1400000 lr7 301.527985',\
                 '1400000 2100000 sil 130.250153',\
                 '.',\
                 '"*/en_0638-A-002.rec"',\
                 '0 400000 sil -8.702469',\
                 '400000 800000 n7 14.018845',\
                 '800000 1100000 I 19.814428',\
                 '1100000 1700000 ng 103.180771',\
                 '1700000 2500000 d 24.743431',\
                 '2500000 3800000 U -52.839127',\
                 '.',]
    >>> result = []
    >>> mes = MLFProcessor(result.append)
    >>> result
    []
    >>> for line in lines:
    ...     mes.process(line)
    >>> len(result)
    13

    >>> for evt in result:
    ...    print evt
    Record start: */en_0638-A-001.rec
    0 300000 sil 47.421978
    300000 1400000 lr7 301.527985
    1400000 2100000 sil 130.250153
    Record end: */en_0638-A-001.rec
    Record start: */en_0638-A-002.rec
    0 400000 sil -8.702469
    400000 800000 n7 14.018845
    800000 1100000 I 19.814428
    1100000 1700000 ng 103.180771
    1700000 2500000 d 24.743431
    2500000 3800000 U -52.839127
    Record end: */en_0638-A-002.rec
    
    >>> lines2 = ['#!MLF!#',
    ...           '"*/en_0638-A-001.rec"',
    ...           '0 300000 r 33.539131 right',
    ...           '300000 1500000 aI 289.424377',
    ...           '1500000 1800000 t 48.881046',
    ...           '1800000 2100000 sil 56.252335 <sil>',
    ...           '.',
    ...           '"*/en_0638-A-002.rec"',
    ...           '0 800000 oU -22.766861 oh',
    ...           '800000 1100000 y -0.764874 you',
    ...           '1100000 1400000 u 20.385105',
    ...           '1400000 1700000 d 47.047123 did',
    ...           '1700000 2000000 I 13.943106',
    ...           '2000000 2500000 d -4.789762',
    ...           '2500000 3800000 sil -126.102829 <sil>',
    ...           '.']
    >>> for line in lines2:
    ...     mes.process(line)
    >>> len(result)
    28
    >>> for evt in result:
    ...    if isinstance(evt, MLFLabelEvent) and  evt.word_label:
    ...        print evt
    0 300000 r 33.539131 right
    1800000 2100000 sil 56.252335 <sil>
    0 800000 oU -22.766861 oh
    800000 1100000 y -0.764874 you
    1400000 1700000 d 47.047123 did
    2500000 3800000 sil -126.102829 <sil>
    
    Test an error condition by starting beyond the record_filename
    marker in a record.  Note that just starting beyond the header
    string isn't an error(!) since we're just feeding more records
    under the "original" header.  
    
    >>> try: 
    ...     for line in lines[2:]:
    ...         mes.process(line)
    ... except MLFProcessorError, e:
    ...     print e
    Unexpected line in stream (file UNKNOWN, line 30): 0 300000 sil 47.421978
    
    Test use of TextFileProcessor as input
    >>> module_dir, module_name = os.path.split(__file__)

    >>> files = tuple(os.path.join(module_dir, mlf_file) for mlf_file in ('f1.mlf', 'f2.mlf'))
    >>> result = []
    >>> mes2 = MLFProcessor(result.append)
    >>> tfs = TextFileProcessor(files, mes2.process)
    >>> while not tfs.finished():
    ...    tfs.process()
    >>> for evt in result:
    ...    print evt
    Record start: */en_0638-A-001.rec
    0 300000 sil 47.421978
    300000 1400000 lr7 301.527985
    1400000 2100000 sil 130.250153
    Record end: */en_0638-A-001.rec
    Record start: */en_0638-A-002.rec
    0 400000 sil -8.702469
    400000 800000 n7 14.018845
    800000 1100000 I 19.814428
    1100000 1700000 ng 103.180771
    1700000 2500000 d 24.743431
    2500000 3800000 U -52.839127
    Record end: */en_0638-A-002.rec
    Record start: */en_0638-A-001.rec
    0 300000 r 33.539131 right
    300000 1500000 aI 289.424377
    1500000 1800000 t 48.881046
    1800000 2100000 sil 56.252335 <sil>
    Record end: */en_0638-A-001.rec
    Record start: */en_0638-A-002.rec
    0 800000 oU -22.766861 oh
    800000 1100000 y -0.764874 you
    1100000 1400000 u 20.385105
    1400000 1700000 d 47.047123 did
    1700000 2000000 I 13.943106
    2000000 2500000 d -4.789762
    2500000 3800000 sil -126.102829 <sil>
    Record end: */en_0638-A-002.rec
    """
        
    def __init__(self, sendee=None, sending=True):
        super(MLFProcessor, self).__init__(sendee, sending=sending)
        self.set_sendee(sendee)
        self._state = 'initial'
        self._record_filename = None
        self._line_count = 0
        self._filename = "UNKNOWN"

    def process(self, event_or_line):
        """
        Process either a line of text or a TextLine event, and perhaps generate
        an MlfEvent and call the sendee with it
        """
        if isinstance(event_or_line, TextFileEventBase):
            if isinstance(event_or_line, TFNewFileEvent):
                # We don't generate an event, but we do note the new filename
                self._filename = event_or_line.filename
                return
            if isinstance(event_or_line, TFLineEvent):
                line = event_or_line.line
            else:
                return
        else:
            line = event_or_line.strip()

        out_event = self.parse_line(line)
        self._line_count += 1
        if out_event is not None:
            self.send(out_event)

    def parse_line(self, line):
        line = line.strip()
        MLF_HEADER_STRING = '#!MLF!#'
        MLF_RECORD_END_STRING = '.'
        MLF_RECORD_START_REGEXP = re.compile('^"(.*)"$')
        if line == MLF_HEADER_STRING:
            self.verify_state(('initial', 'record_ended'), line)
            self._state = 'in_mlf_block'
            return None
        if line == MLF_RECORD_END_STRING:
            self.verify_state(('in_record'), line)
            self._state = 'record_ended'
            ret = MLFRecordEndEvent()
            ret.record_filename = self._record_filename
            return ret
        m = MLF_RECORD_START_REGEXP.match(line)
        if m is not None:
            self._record_filename = m.group(1)
            self.verify_state(('in_mlf_block', 'record_ended'), line)
            self._state = 'in_record'
            ret = MLFRecordStartEvent()
            ret.record_filename = self._record_filename
            return ret
        # Must be a label line
        self.verify_state(('in_record'), line)
        return self.make_label_event(line)

    def make_label_event(self, line):
        """
        Make an event corresponding to the label represented in 'line'.
        
        Currently, we can deal with lines containing 1, 4, or 5 tokens, always
        white-space separated (which is the HTK spec).  The full spec would
        allow other numbers of tokens, and the interpretation of the tokens used
        here matches the files we have at present, but the HTK spec would allow
        other interpretations of lines with 4 or 5 tokens.  Unfortunately, there
        doesn't seem to be any in-band way to disambiguate these cases.
        """
        ret = MLFLabelEvent()
        tokens = line.split()
        n = len(tokens)
        if n == 1:
            ret.word_label = tokens[0]
            ret.start = None
            ret.end = None
            ret.phone_label = None
            ret.score = None
            return ret
        if 4 <= n <= 5:
            ret.start = int(tokens[0])
            ret.end = int(tokens[1])
            ret.phone_label = tokens[2]
            ret.score = float(tokens[3])
            if n == 5:
                ret.word_label = tokens[4]
            else:
                ret.word_label = None
            return ret
        msg = ("Unexpected number of tokens (%d) in stream (file %s, line %d): %s" % (
                n, self._filename, self._line_count, line))
        raise MLFProcessorError(msg)

    def verify_state(self, acceptable, line, filename = None):
        if self._state not in acceptable:
            msg = ("Unexpected line in stream (file %s, line %d): %s" % (
                self._filename, self._line_count, line))
            raise MLFProcessorError(msg)


class MLFBlockEvent(object):
    def __init__(self, record_filename, labels):
        self.record_filename = record_filename
        self.labels = labels
        
    def __repr__(self):
        return "MLFBlock for %s, labels = %s" % (self.record_filename, self.labels)


class MLFBlockProcessor(ProcessorBase):
    """
    Generator for blocks of mlf data from a stream of MLF processor events.
    
    Event types sent:  MLFBlockEvent

    >>> module_dir, module_name = os.path.split(__file__)

    >>> files = tuple(os.path.join(module_dir, mlf_file) for mlf_file in ('f1.mlf', 'f2.mlf', 'mono.mlf'))
    >>> result = []
    >>> def handler0(label_evt): return label_evt.word_label
    >>> mbp0 = MLFBlockProcessor(handler0, result.append)
    >>> mp0 = MLFProcessor(mbp0.process)
    >>> tfs = TextFileProcessor(files, mp0.process)
    >>> while not tfs.finished():
    ...    tfs.process()

    >>> len(result)
    45
    >>> for evt in result[:4]:
    ...    print evt
    MLFBlock for */en_0638-A-001.rec, labels = [None, None, None]
    MLFBlock for */en_0638-A-002.rec, labels = [None, None, None, None, None, None]
    MLFBlock for */en_0638-A-001.rec, labels = ['right', None, None, '<sil>']
    MLFBlock for */en_0638-A-002.rec, labels = ['oh', 'you', None, 'did', None, None, '<sil>']
    >>> print result[4].record_filename
    */adg0_4_sr009.lab
    >>> print result[4].labels[:20]
    ['sil', 'sh', 'ow', 'sp', 'dh', 'ax', 'sp', 'g', 'r', 'ih', 'dd', 'l', 'iy', 'z', 'sp', 'td', 'ch', 'r', 'ae', 'k']
    """

    def __init__(self, label_event_handler=None, sendee=None, sending=True):
        """
        label_event_handler is a callable which takes MLFLabelEvents; the values
        returned will be appended to the list for each record.  If None, the
        label events themselves will be appended.  sendee is a function to call
        with our output events.
        """
        super(MLFBlockProcessor, self).__init__(sendee, sending=sending)
        self.set_sendee(sendee)
        self._current = None
        self._label_event_handler = label_event_handler
        
    def process(self, event):
        """
        Process an event from an MLFProcesser and perhaps generate an MlfBlock
        Event and call the sendee with it
        """
        if isinstance(event, MLFRecordStartEvent):
            assert hasattr(event, 'record_filename')
            self._current = []
        elif isinstance(event, MLFLabelEvent):
            if not self._label_event_handler:
                self._current.append(event)
            else:
                self._current.append(self._label_event_handler(event))
        elif isinstance(event, MLFRecordEndEvent):
            out_event = MLFBlockEvent(event.record_filename, self._current)
            self.send(out_event)
        else:
            raise NotImplemented("wasn't expecting an event of type %s" % (type(event)))


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()




