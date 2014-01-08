###########################################################################
#
# File:         istext.py
# Date:         15-Aug-2009
# Author:       Hugh Secker-Walker
# Description:  Script to classify the text-ness of files
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

r"""
Classifies file data into text or binary, with line-ending and tab flags.

Use --unix argument to only print the filenames of non-TAB unix files.

>>> module_dir, module_name = os.path.split(__file__)
>>> main([os.path.join(module_dir, 'SConscript'), 'no_such_file']) #doctest: +ELLIPSIS
istext: ...SConscript   unix 
istext: no_such_file   not-a-file 

>>> for text in _test_sets:
...   print repr(text), ' ', classify(text)
'abc'   noends
'a\tbc'   noends TAB
'ab\nc'   unix
'ab\rc'   mac
'ab\r\nc'   dos
'a\tbc'   noends TAB
'a\tb\nc'   unix TAB
'a\tb\rc'   mac TAB
'a\tb\r\nc'   dos TAB
'\x00abc'   binary
'abc\xff'   binary
"""
from __future__ import with_statement
import sys, os

# what we consider to be text
ASCII = frozenset(chr(x) for x in xrange(32, 127))
LF = frozenset('\n')
CR = frozenset('\r')
TAB = frozenset('\t')

# derived sets
TEXT = ASCII | TAB | LF | CR
DOS = CR | LF

# names for line ending disciplines
ENDINGS = 'dos', 'unix', 'mac', 'noends'

# number of characters to examine of large files
CHUNK = 1 << 20

def classify(data):
    chars = frozenset(data)
    if chars <= TEXT:
        # some sort of text file
        non_ascii = chars - ASCII
        endings = non_ascii - TAB
        assert endings <= DOS
        if endings == DOS:
            # note: this isn't actually testing the co-occurence of CF-LR;
            # rather, anything with both CF and LF is called dos
            report = 'dos'
        elif endings == LF:
            report = 'unix'
        elif endings == CR:
            report = 'mac'
        else:
            report = 'noends'
        assert report in ENDINGS
        if TAB & non_ascii:
            report += ' TAB'
    else:
        # file has non-text characters
        report = 'binary'
    return report

def main(args):
    # implement that '--unix' means to only print filenames and only of non-TAB
    # unix files; similarly for --dos and --mac
    if args and args[0][:2] == '--' and args[0][2:] in ENDINGS:
        strict = args[0][2:]
        del args[0]
    else:
        strict = None

    for filename in args:
        if os.path.isfile(filename) and not os.path.islink(filename):
            with open(filename, 'rb') as infile:
                report = classify(infile.read(CHUNK))
        else:
            # not an actual file
            report = 'not-a-file'
            
        if strict is not None:
            if report == strict:
                print filename
        else:
            print 'istext:', filename, ' ', report, ''

    
if __name__ == '__main__':
    _test_sets = (
        'abc',
        'a\tbc',
        'ab\nc',
        'ab\rc',
        'ab\r\nc',
        'a\tbc',
        'a\tb\nc',
        'a\tb\rc',
        'a\tb\r\nc',
        '\x00abc',
        'abc\xff',
        )


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    main(sys.argv[1:])
