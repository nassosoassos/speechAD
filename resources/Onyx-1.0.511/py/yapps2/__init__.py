###########################################################################
#
# File:         __init__.py (directory: ./py/yapps2)
# Date:         22-April-2008
# Author:       Ken Basye
# Description:  Package initialization for Onyx use of YAPPS
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008, 2009 The Johns Hopkins University
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
    This module exists solely to allow yapps2 to be looked up as a
    Python package, e.g. so that symbols from yappsrt can be imported.

    >>> import yapps2

    Note: we'd like to test the following, which do work from the Python
    command-line, but, as of 2008-11-18, fail in the doctest....
    >>> #import yapps2.yappsrt
    >>> #from yapps2 import yappsrt
    >>> #from yapps2.yappsrt import *
"""

__all__ = []

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
