###########################################################################
#
# File:         delta.py
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Delta calculations for MFCC-type work
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 The Johns Hopkins University
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
Delta calculations.
"""
from onyx.signalprocessing import sigprocbase


class Delta(sigprocbase):
    """
    Single delta.

    >>> type(Delta(None))
    <class '__main__.Delta'>
    """
    SERIAL_VERSION = 0

    def __init__(self, arg):
        super(Delta, self).__init__(arg)

class DeltaDelta(sigprocbase):
    """
    Double delta.

    >>> type(DeltaDelta(None))
    <class '__main__.DeltaDelta'>
    """
    SERIAL_VERSION = 0

    def __init__(self, arg):
        super(DeltaDelta, self).__init__(arg)


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
