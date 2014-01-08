###########################################################################
#
# File:         timestamp.py
# Date:         Wed 24 Jun 2009 15:07
# Author:       Ken Basye
# Description:  Generation of time stamps for general use
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
Generation of time stamps for general use

"""


import datetime
import time
import re

# XXX This construction seems to be correct, but there are many corner cases
# that can't easily be tested :-<.
class TZ(datetime.tzinfo):
    def utcoffset(self, dt):
        return datetime.timedelta(seconds=-(time.altzone if time.daylight else time.timezone))

    def dst(self, dt):
        return datetime.timedelta(minutes=(0 if time.daylight == 0 else 60))

# This is just for testing
class _DummyTZ(datetime.tzinfo):
    def utcoffset(self, dt):
        return datetime.timedelta(0)

    def dst(self, dt):
        return datetime.timedelta(0)

TIMESTAMP_RE = re.compile(r"(20\d\d)-([01]\d)-([0123]\d)-(Mon|Tue|Wed|Thu|Fri|Sat|Sun)-([012]\d):([012345]\d):([012345]\d)\.(\d\d\d\d\d\d)([\+-][012]\d[012345]\d)")

def get_timestamp(dt=None):
    """
    Return a timestamp string for general use.

    This function returns a timestamp string in the format
    YYYY-MM-DD-ddd-HH:MM:SS.mmmmmmm+HH:MM.  If *dt* is not None, it should be a Python
    datetime.datetime object which will be formatted, otherwise, a datetime
    object is created by calling datetime.now().

    >>> dt = datetime.datetime(2002, 12, 25, 13, 45, 56, 123456, _DummyTZ())
    >>> get_timestamp(dt)
    '2002-12-25-Wed-13:45:56.123456+0000'

    The expected use will be to get a current time stamp, but this is hard to
    test.

    >>> ts0 = get_timestamp()
    >>> len(ts0)
    35
    >>> ts0[0:2], ts0[4], ts0[7]
    ('20', '-', '-')

    Uncomment this to see what time it is :->.
    
    # >>> ts0
    
    The TIMESTAMP_RE object in this module is a compiled regular expression that
    matches a timestamp string and groups it into 9 parts: year, month, date,
    day, hour, minute, second, microsecond, and UTC offset.

    >>> match = TIMESTAMP_RE.match(ts0)
    >>> len(match.groups())
    9

    """
    if dt is None:
        dt = datetime.datetime.now(TZ())
    format_string = '%Y-%m-%d-%a-%H:%M:%S.' + ('%06d' % (dt.microsecond,)) + '%z'
    return dt.strftime(format_string)
    

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



