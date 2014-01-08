###########################################################################
#
# File:         ngramidx.py
# Date:         Tue 29 Jan 2008 15:35
# Author:       Ken Basye
# Description:  N-gram indices for STD
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

from onyx.util.streamprocess import ProcessorBase
from exceptions import ValueError

class QueryEvent(object):
    def __init__(self, seq):
        self.seq = seq

class NGramEvent(object):
    def __init__(self, seq, start, end, document):
        self.seq = seq
        self.start = start
        self.end = end
        self.document = document

class NGramIndexProcessor(ProcessorBase):
    """ A class to represent an NGram-indexed set of documents.  This
    class supports the ProcessorBase streaming interface.  Input
    events should be either new NGrams or queries.  Only the latter
    will generate output events, which are candidate locations for a
    given query (the query itself will be attached to each candidate).

    >>> results = []
    >>> nip = NGramIndexProcessor(3, results.append)
    >>> e1 = NGramEvent(('a', 'b', 'c'), 0, 10, "doc1.wav")
    >>> e2 = NGramEvent(('a', 'b', 'd'), 0, 8, "doc2.wav")
    >>> nip.process(e1)
    >>> nip.process(e2)
    >>> nip
    NGramIndexProcessor with 2 documents and 2 ngrams
    >>> results
    []
    
    """

    def __init__(self, arity, sendee=None, sending=True):
        super(NGramIndexProcessor, self).__init__(sendee, sending=sending)
        if arity <= 0:
            raise ValueError("NGramIndexProcessor needs a positive value of N")
        self.arity = arity
        self._dict = {}
        self._doc_dict = {}
        self._doc_list = []
        self._doc_ngram_count_list = []
        self.num_docs = 0
        self.num_ngrams = 0

    def __repr__(self):
        s = "NGramIndexProcessor with %d documents and %d ngrams" % (self.num_docs, self.num_ngrams)
        return s

    def process(self, event):
        if isinstance(event, QueryEvent):
            self.process_query_event(event)
        elif isinstance(event, NGramEvent):
            self.process_ngram_event(event)
        else:
            raise ValueError("Unexpected event type passed to NGramIndexProcessor")

    def process_ngram_event(self, event):
        assert isinstance(event, NGramEvent)
        if len(event.seq) != self.arity:
            raise ValueError("NGramEvent with wrong arity passed to NGramIndexProcessor")
        if not self._doc_dict.has_key(event.document):
            self._doc_dict[event.document] = self.num_docs
            self.num_docs += 1
            self._doc_list.append(event.document)
            self._doc_ngram_count_list.append(0)
        doc_idx = self._doc_dict[event.document]
        self._doc_ngram_count_list[doc_idx] += 1
        if self._dict.has_key(event.seq):
            self._dict[event.seq].append((event.start, event.end, doc_idx,))
        else:
            self._dict[event.seq] = [(event.start, event.end, doc_idx,)]
        self.num_ngrams += 1
        


    def process_query_event(self, event):
        pass    


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

