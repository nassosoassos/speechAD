###########################################################################
#
# File:         filefix.py
# Date:         Tue 28 Apr 2009 14:51
# Author:       Ken Basye
# Description:  Some general tools for fixing files
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
from __future__ import with_statement
import os.path
import re

"""
    >>> problem_line_re = re.compile("^(.*)<(Trans|Episode|Speaker|Speakers|Turn|Who|Sync)(.*)>(.*)$")


"""

def apply_line_transform_to_dir(transform, dirname, newdirname, glob_str='*'):
    """
    Apply the callable *transform* to each line of each file in *dirname* that
    matches *glob_str* (default '*'), creating new files with the same basename
    in *newdirname*.
    """
    import glob
    import os
    fnames = glob.glob(dirname + os.sep + glob_str)
    print("Reading %d files in %s" % (len(fnames), dirname))
    for fname in fnames:
        dir,base = os.path.split(fname)
        newfname = os.path.join(newdirname, base)
        with open(fname) as f:
            with open(newfname, 'w') as newf:
                for lineno,line in enumerate(f):
                    newf.write(transform(line, lineno))


def fix_all_malach_files(transform, stage):
    """
    Process the given transform on all Malach transcription files.  *stage*
    should be a positive integer; its value will be used to determine both the
    source and target directory names.
    """
    dirname = "./transcriptions%d" % (stage,)
    newdirname = "./transcriptions%d" % (stage + 1,)
    os.mkdir(newdirname)
    apply_line_transform_to_dir(transform, dirname, newdirname, glob_str='*.trs')
    dirname = "./transcriptions%d/additional" % (stage,)
    newdirname = "./transcriptions%d/additional" % (stage + 1,)
    os.mkdir(newdirname)
    apply_line_transform_to_dir(transform, dirname, newdirname, glob_str='*.trs')



def fix_encoding_in_header(line, lineno):
    """
    This line transform fixes a problem in the first line of the file where the
    encoding attribute had been formatted incorrectly.
    """
    _CORRECT = """<?xml version="1.0" encoding="ISO-8859-1"?>\n"""
    if lineno == 0 and line.find("encoding") == -1:
        return _CORRECT
    else:
        return line

def fix_version_date(line, lineno):
    """
    This line transform fixes a problem in the third line of the file where the
    version_date attribute had been misspelt.
    """
    if lineno == 2 and line.find("version_data") != -1:
        return line.replace("version_data", "version_date")
    else:
        return line


def fix_common_bad_tags(line, lineno):
    """
    This line transform fixes a problem in several files where <> was used to
    indicate certain transcription tokens, e.g. '<pause>' Since <> is the XML
    tag syntax, this causes XML parsing to fail in many places.  This transform
    identifies the problem regions and replaces <XXX> with &gt;XXX&lt; which
    will then be parsed correctly.  This transform is limited to replacing a few
    common bad tags just to reduce the remaining problems to a manageable size.
    """
    bases = ("noise", "pause", "um", "UH", "breath", "inhale", "uh", "cough", "laugh", "HM", "emotion", "UH-UH", "UM",
             "unintelligible", "mouth", "silence", "lead_silence", "hm", "uh_hum", "sniff", "exhale", "UH-UH-UH", "uh-uh",
             "cross_talk_begin", "cross_talk_end", "cross_talk_begins", "cross_talk_ends",
             "bkgrd_noise", "cross_talk", "long_pause", "UH_HUH", "uh_huh", "UH_HUM", "UH-HUH", "uh-huh", "UH-HUM", "EH",
             "laugh-laugh", "noise-noise", "cough-cough", "ap-", "uf-", "spk#1", "spk#2")
    pairs = [("<%s>" % (token,), "&lt;%s&gt;" % (token,)) for token in bases]
    for problem, fix in pairs:
        if line.find(problem) != -1:
            line = line.replace(problem, fix)
    return line
    

    
def fix_bad_tags1(line, lineno):
    """
    This line transform fixes a problem in several files where <> was used to
    indicate certain transcription tokens, e.g. '<pause>' Since <> is the XML
    tag syntax, this causes XML parsing to fail in many places.  This transform
    identifies the problem regions and replaces <XXX> with &gt;XXX&lt; which
    will then be parsed correctly.  This transform is limited to replacing
    tokens in <>s with only lower-case letters, and underscores, and will
    only replace one such instance in a line.  This covers many error cases, and
    later transforms can do more work on fewer instances.
    """
    import re
    problem_line_re = re.compile("^(.*)<([a-z_]*)>(.*)$")
    match = problem_line_re.match(line)
    if match is None:
        return line
    else:
        groups = match.groups()
        assert len(groups) == 3
        newline = groups[0] + '&lt;' + groups[1] + '&gt;' + groups[2] + '\n'
        return newline


def fix_bad_tags2(line, lineno):
    """
    This line transform fixes a problem in several files where <> was used to
    indicate certain transcription tokens, e.g. '<pause>' Since <> is the XML
    tag syntax, this causes XML parsing to fail in many places.  This transform
    identifies the problem regions and replaces <XXX> with &gt;XXX&lt; which
    will then be parsed correctly.  Limited to any <> with an a-z character
    immediately after the <.
    """
    import re
    problem_line_re = re.compile("^(.*)<([a-z].*)>(.*)$")
    match = problem_line_re.match(line)
    if match is None:
        return line
    else:
        groups = match.groups()
        assert len(groups) == 3
        newline = groups[0] + '&lt;' + groups[1] + '&gt;' + groups[2] + '\n'
        return newline

def fix_bad_tags3(line, lineno):
    """
    This line transform fixes a problem in several files where <> was used to
    indicate certain transcription tokens, e.g. '<pause>' Since <> is the XML
    tag syntax, this causes XML parsing to fail in many places.  This transform
    identifies the problem regions and replaces <XXX> with &gt;XXX&lt; which
    will then be parsed correctly.  This transform deals with tokens in <>s
    which consist only of capital letters, underscores, and hyphens.
    """
    import re
    problem_line_re = re.compile("^(.*)<([A-Z_/-]*)>(.*)$")
    match = problem_line_re.match(line)
    if match is None:
        return line
    else:
        groups = match.groups()
        assert len(groups) == 3
        newline = groups[0] + '&lt;' + groups[1] + '&gt;' + groups[2] + '\n'
        return newline

def fix_bad_tags4(line, lineno):
    """
    This line transform fixes remaining bad tags, which is anything in <>s that
    doesn't start with a tag we know about.  It prints the line it is fixing,
    and is meant to be used when almost everything has been fixed.
    """
    import re
    ok_line_re = re.compile(r"^(.*)</?(Trans|Episode|Speaker|Speakers|Turn|Who|Sync|Section|\?xml|!DOCTYPE)(.*)>(.*)$")
    ok_match = ok_line_re.match(line)
    problem_line_re = re.compile("^(.*)<(.*)>(.*)$")
    problem_match = problem_line_re.match(line)
    if ok_match is not None:
        return line
    if problem_match is None:
        return line
    else:
        groups = problem_match.groups()
        assert len(groups) == 3
        newline = groups[0] + '&lt;' + groups[1] + '&gt;' + groups[2] + '\n'
        print line
        return newline

def check_for_bad_tags0(line, lineno):
    """
    This line transform just checks for bad tags, which is anything in <>s that
    doesn't start with a tag we know about.  It prints any line which has more than one < in it.
    """
    import re
    ok_line_re = re.compile(r"^(.*)</?(Trans|Episode|Speaker|Speakers|Turn|Who|Sync|Section|\?xml|!DOCTYPE)(.*)>(.*)$")
    ok_match = ok_line_re.match(line)
    problem_line_re = re.compile("^(.*)<(.*)>(.*)$")
    problem_match = problem_line_re.match(line)
    if ok_match is not None:
        return line
    if problem_match is None:
        return line
    else:
        groups = problem_match.groups()
        if line.count('<') > 1:
            print line
        return line

def check_for_bad_tags(line, lineno):
    """
    This line transform just checks for bad tags, which is anything in <>s that
    doesn't start with a tag we know about.
    """
    import re
    ok_line_re = re.compile(r"^(.*)</?(Trans|Episode|Speaker|Speakers|Turn|Who|Sync|Section|\?xml|!DOCTYPE)(.*)>(.*)$")
    ok_match = ok_line_re.match(line)
    problem_line_re = re.compile("^(.*)<(.*)>(.*)$")
    problem_match = problem_line_re.match(line)
    if ok_match is not None:
        return line
    if problem_match is None:
        return line
    else:
        groups = problem_match.groups()
        print line
        return line

if __name__ == '__main__':
    fix_all_malach_files(fix_encoding_in_header, 1)
    fix_all_malach_files(fix_version_date, 2)
    fix_all_malach_files(fix_common_bad_tags, 3)
    # We do two rounds of the next fix since there are several begin/end pairs
    # and each round will only clean up one tag
    fix_all_malach_files(fix_bad_tags1, 4)
    fix_all_malach_files(fix_bad_tags1, 5)
    fix_all_malach_files(fix_bad_tags2, 6)
    fix_all_malach_files(fix_bad_tags3, 7)
    fix_all_malach_files(check_for_bad_tags0, 8)
    fix_all_malach_files(fix_bad_tags4, 9)
    fix_all_malach_files(check_for_bad_tags, 10)
    



