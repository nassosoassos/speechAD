###########################################################################
#
# File:         version.py
# Date:         Fri 21 Aug 2009  13:24
# Author:       Ken Basye
# Description:  Generate a version number for this project
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
The project version number

See the project file doc/versioning.txt for the interpretation of the project's
version numbers.


Module Attributes
-----------------

SHORT_VERSION_NUMBER_STRING
  A string representation of the major and minor version number of the project,
  e.g. `1.0`.

FULL_VERSION_NUMBER_STRING
  A string representation of the full version number of the project,
  e.g. `1.01.503`.  

FULL_VERSION_RE
  A compiled regular expresssion with match groups for major, minor, and revno
  fields of the FULL_VERSION_NUMBER_STRING.


>>> SHORT_VERSION_NUMBER_STRING.split('.') == [str(MAJOR_VERSION), str(MINOR_VERSION)]
True

>>> m = FULL_VERSION_RE.search(FULL_VERSION_NUMBER_STRING)
>>> m is not None
True
>>> major, minor, revno = m.groups()
>>> int(major) == MAJOR_VERSION
True
>>> int(minor) == MINOR_VERSION
True
>>> int(revno) >= 503
True

"""

from __future__ import with_statement
import onyx
import os.path
import re
import sys

MAJOR_VERSION = 1
MINOR_VERSION = 0

SHORT_VERSION_NUMBER_STRING = "%d.%d" % (MAJOR_VERSION, MINOR_VERSION)

# Onyx version numbers are X.Y.Z, where X and Y are defined here, and Z is the
# current Bazaar revision number.  The trick is that we don't want to have to
# require bzrtool or even that this file be in a bzr branch in order to run.  So
# we extract the revision number from changes.txt, and check it against the
# current bzr revision number if we can.

FULL_VERSION_NUMBER_STRING = None
CHANGES_FILENAME = 'changes.txt'
with open(os.path.join(onyx.home, CHANGES_FILENAME)) as change_file:
    # Here's the kind of line we're looking for:
    # Last revno: 495   Last revision id: kbasye1@jhu.edu-20090821152922-5517dmztu546s12e
    REVNO_LINE_RE = re.compile(r'^Last revno: ([0-9]+)   Last revision id: .*$')
    for line in change_file:
        match = REVNO_LINE_RE.match(line)
        if match is not None:
            # The revision number in changes is the *last* revision, which
            # should always be one less than the current revision.
            revno = int(match.group(1)) + 1
            FULL_VERSION_NUMBER_STRING = "%d.%d.%s" % (MAJOR_VERSION, MINOR_VERSION, revno)
            break
if FULL_VERSION_NUMBER_STRING is None:
    raise onyx.DataFormatError("Expected to find Last revno in %s, but did not."
                               % (CHANGES_FILENAME,))
FULL_VERSION_RE = re.compile(r'\b([0-9]+)\.([0-9]+)\.([0-9]+\b)')

# If we can, we'd like to check the revision number against what Bazaar thinks,
# but we can only do this if we're in a Bazaar-controlled project and we have
# access to bzrlib.

sys.path.append(os.path.abspath(os.path.join(onyx.home, 'bin')))
try:
    # if this doesn't work, we can't complain
    import bzrtool
    local_tree, local_revno, local_last_revid = bzrtool.get_local_info()
except:
    local_revno = None

if local_revno is not None and local_revno != revno:
    raise onyx.OnyxException("Revision number from %s (%d) doesn't match the Bazaar revision number (%d)" %
                             (CHANGES_FILENAME, revno, local_revno))

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
