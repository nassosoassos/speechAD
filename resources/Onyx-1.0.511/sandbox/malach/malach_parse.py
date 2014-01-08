###########################################################################
#
# File:         malach_parse.py
# Date:         Mon 27 Apr 2009 15:18
# Author:       Ken Basye
# Description:  Utilities to parse Malach transcription files
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
    This module defines some utilities useful for parsing transcription files in
    the Malach dataset.  The transcription files are in XML metaformat; the
    format is fairly simple.  See doc for the function 'malach_parse' for
    details.
"""    

from __future__ import with_statement
import onyx
import xml.etree.ElementTree
import xml.parsers.expat
from itertools import islice, izip
import os.path


def malach_parse(file):
    """
    >>> module_dir, module_name = os.path.split(__file__)
    >>> fname0 = os.path.join(module_dir, "test_transcriptions/00009-003.trs")
    >>> with open(fname0) as f0:
    ...     speakers, turns = malach_parse(f0)
    
    >>> len(speakers)
    3

    >>> speakers[0]
    {'dialect': 'native', 'name': '<spkr1> Interviewer', 'accent': '', 'id': 'spk1', 'scope': 'local', 'check': 'no'}
        
    >>> len(turns)
    27

    >>> [len(parts) for parts, attribs in turns][:10]
    [1, 1, 1, 1, 1, 172, 1, 6, 1, 1]
    
    Each turn consists of a sequence of utterances (parts) and a set of attributes in the form of a dict:
    
    >>> parts, attribs = turns[5]
    >>> attribs
    {'endTime': '760.7', 'speaker': 'spk2', 'startTime': '63.658'}

    >>> len(parts)
    172

    Each part consists of a sync time and the transcript for that turn

    >>> parts[15]
    ('120.089', 'so the plan was that we would crawl out through the little window')

    >>> module_dir, module_name = os.path.split(__file__)
    >>> fname0 = os.path.join(module_dir, "test_transcriptions/35995-002.trs")
    >>> with open(fname0) as f0:
    ...     speakers, turns = malach_parse(f0)

    >>> len(turns)
    79

    Certain turns have a structure in which two people are talking over each
    other.  When that happens, the turn will have a 3-part structure in which
    the first element is the sync time, and the second and third elements are
    both pairs of (nb, trans) where nb is the number of the speaker and trans is
    what that speaker said.  The number of the speaker corresponds to the digit
    at the end of the speaker's 'id' field in the speakers list; see above.

    >>> [len(parts) for parts, attribs in turns][:20]
    [1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 6, 2, 30, 1, 8, 1]

    >>> for parts, attribs in turns:
    ...    for i in xrange(len(parts)):
    ...        if len(parts[i]) == 3:
    ...            print parts[i]
    ('22.334', ('1', 'and'), ('2', 'right'))
    ('29.694', ('1', 'and'), ('2', 'right'))
    ('31.897', ('1', "people's"), ('2', 'right'))
    ('201.601', ('1', 'differently than the'), ('2', 'the Jews'))
    ('379.26', ('1', 'they take you'), ('2', "it's a marine camp"))
    ('466.130', ('1', 'together'), ('2', 'no'))
    ('692.927', ('1', 'head'), ('2', 'yeah also'))
    ('734.16', ('1', 'where did they send you'), ('2', 'a camp like'))
    
    """    

    def raise_error(mesg):
        raise onyx.DataFormatError(mesg + " [File: %s]" % (file.name,))

    try:
        tree = xml.etree.ElementTree.parse(file)
    except xml.parsers.expat.ExpatError, ee:
        raise_error("XML parse failure: %s" % (ee,))

    # Set up some useful constants for the parse
    _TRANS_TAG = "Trans"
    _SPEAKERS_TAG = "Speakers"
    _ONE_SPEAKER_TAG = "Speaker"
    _EPISODE_TAG = "Episode"
    _SECTION_TAG = "Section"
    _TURN_TAG = "Turn"
    _SYNC_TAG = "Sync"
    _WHO_TAG = "Who"

    trans = tuple(tree.getiterator(_TRANS_TAG))
    if len(trans) == 0:
        raise_error("Failed to file a Trans element in this tree")
    elif len(trans) > 1:
        raise_error("Found more than one Trans element in this tree")
    
    spkrs_record = trans[0].find(_SPEAKERS_TAG)
    speakers = list()
    if spkrs_record is not None:
        spkrs_iter = spkrs_record.getiterator(_ONE_SPEAKER_TAG)
        for spkr_record in spkrs_iter:
            assert spkr_record.attrib.has_key('id')
            speakers.append(spkr_record.attrib)
    else:
        pass
    
        # raise_error("Failed to find Speakers subelement in Trans")


    spkrs_id_set = set((attrib['id'] for attrib in speakers))

    # There's one Episode per Trans and one Section per Episode
    episode_record = trans[0].find(_EPISODE_TAG)
    if episode_record is None:
        raise_error("Failed to find an Episode subelement in Trans")

    section_record = episode_record.find(_SECTION_TAG)
    if section_record is None:
        raise_error("Failed to find a Section subelement in Episode")

    turns_iter = section_record.getiterator(_TURN_TAG)
    turns = list()
    for turn_record in turns_iter:
        sync_records = turn_record.getiterator(_SYNC_TAG)
        if len(sync_records) == 0:
            raise_error("Failed to find even one Sync subelement in Turn")
        who_records = turn_record.getiterator(_WHO_TAG)

        # Process speaker attribute.  The 'speaker' attribute is not always
        # present in the files, but it will be in our return
        turn_attribs = dict(turn_record.attrib)
        if not turn_attribs.has_key('speaker'):
            turn_attribs['speaker'] = None
        else:
            # The 'speaker' attribute is a string, sometimes it has more than
            # one speaker ID separated with a space.
            listed_spkrs = set(turn_attribs['speaker'].split())
            if not listed_spkrs <= spkrs_id_set:
                raise_error("'speaker' attribute doesn't match any in Speakers set")

            # When there's more than one speaker listed, each Sync tag is paired
            # with two Who tags - this covers the case when two people are
            # talking over each other.
            if len(listed_spkrs) > 1:
                assert len(listed_spkrs) == 2
                if len(who_records) != 2 * len(sync_records):
                    raise_error("Failed to find two Who subelements per Sync in Turn with two speakers")

        syncs = list()               
        if len(who_records) == 0:
            # Just process a collection of simple syncs
            for sync_record in sync_records:
                # The tail of the sync_record is what we're really after here!
                trans = sync_record.tail.strip()
                sync_attribs = sync_record.attrib
                if not sync_attribs.has_key('time'):
                    raise_error("Failed to find a 'time' attribute in Sync subelement")
                sync_time = sync_attribs['time']
                syncs.append((sync_time, trans))
        else:
            # Process Sync/Who/Who triples
            who_a_iter = islice(who_records, 0, None, 2)
            who_b_iter = islice(who_records, 1, None, 2)
            for sync_record, who_a, who_b in izip(sync_records, who_a_iter, who_b_iter):
                sync_attribs = sync_record.attrib
                if not sync_attribs.has_key('time'):
                    raise_error("Failed to find a 'time' attribute in Sync subelement")
                sync_time = sync_attribs['time']
                # In this case, the transcripts are the tails of the Who tags;
                # the Sync tag has a tail with only whitespace.
                assert sync_record.tail.strip() == ""
                trans_who_a = who_a.tail.strip()
                trans_who_b = who_b.tail.strip()

                who_a_attribs = who_a.attrib
                if not who_a_attribs.has_key('nb'):
                    raise_error("Failed to find an 'nb' attribute in Who subelement")
                who_a_nb = who_a_attribs['nb']

                who_b_attribs = who_b.attrib
                if not who_b_attribs.has_key('nb'):
                    raise_error("Failed to find an 'nb' attribute in Who subelement")
                who_b_nb = who_b_attribs['nb']
                
                syncs.append((sync_time, (who_a_nb, trans_who_a), (who_b_nb, trans_who_b)))
        turns.append((syncs, turn_attribs))
    
    return speakers, turns


def parse_all_files(dirname):
    """
    >>> module_dir, module_name = os.path.split(__file__)
    >>> dirname = os.path.join(module_dir, "test_transcriptions")
    >>> parse_all_files(dirname)
    Parsing 2 files
    Parsing failed on 0 files

##     >>> dirname = "./transcriptions_fixed"
##     >>> parse_all_files(dirname)
##     Parsing 784 files
##     Parsing failed on 0 files

##     >>> dirname = "./transcriptions_fixed/additional"
##     >>> parse_all_files(dirname)
##     Parsing 48 files
##     Parsing failed on 0 files
    """
    import glob
    import os

    fnames = glob.glob(dirname + os.sep + '*.trs')
    print("Parsing %d files" % (len(fnames),))
    failures = 0
    for fname in fnames:
        with open(fname) as f:
            try:
                speakers, turns = malach_parse(f)
            except onyx.DataFormatError, e:
                print e
                failures += 1
    print("Parsing failed on %d files" % (failures,))


# We define and test validation functions only if the platform in question has
# lxml, but this is not required at this point.
def have_lxml():
    try:
        import lxml.etree
        return True
    except ImportError:
        return False

if have_lxml():

    def validate(file):
        """
        >>> module_dir, module_name = os.path.split(__file__)
        >>> fname0 = os.path.join(module_dir, "test_transcriptions/00009-003.trs")
        >>> with open(fname0) as f0:
        ...     validate(f0)
        """
    
        import lxml.etree
        p = lxml.etree.XMLParser(dtd_validation=True)
        tree = lxml.etree.parse(file, parser=p)


    def validate_all_files(dirname):
        """
        >>> module_dir, module_name = os.path.split(__file__)
        >>> dirname = os.path.join(module_dir, "test_transcriptions")
        >>> validate_all_files(dirname)
        Validating 2 files
        Validation failed on 0 files
        
        ##     >>> dirname = "./transcriptions_fixed"
        ##     >>> validate_all_files(dirname)
        ##     Validating 784 files
        ##     Validation failed on 0 files
        
        ##     >>> dirname = "./transcriptions_fixed/additional"
        ##     >>> validate_all_files(dirname)
        ##     Validating 48 files
        ##     Validation failed on 0 files
        """
        import glob
        import os
        fnames = glob.glob(dirname + os.sep + '*.trs')
        print("Validating %d files" % (len(fnames),))
        failures = 0
        for fname in fnames:
            with open(fname) as f:
                try:
                    validate(f)
                except lxml.etree.LxmlSyntaxError, e:
                    print e
                    print fname
                    failures += 1
        print("Validation failed on %d files" % (failures,))
    

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



