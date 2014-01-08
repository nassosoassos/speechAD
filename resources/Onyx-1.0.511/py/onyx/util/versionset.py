###########################################################################
#
# File:         versionset.py
# Date:         Tue 9 Sep 2008 13:00
# Author:       Ken Basye
# Description:  Keep track of a set of versions, e.g., for making decisions about support
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
Support for keeping track of a set of versions


"""

class VersionSet(tuple):
    """
    >>> vs0 = VersionSet((1, 2, 3, (7, 11)))
    >>> [v in vs0 for v in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)]
    [True, True, True, False, False, False, True, True, True, True, True, False, False]

    """
    def __init__(self, iterable):
        for item in iterable:
            if (not type(item) is int and
                not hasattr(item, "__len__") and
                not len(item) == 2 and
                not type(item[0]) is int and
                not type(item[1]) is int and
                item[0] <= item[1] ):
                raise ValueError("expected every item in iterable to be int or tuple of two ints in nondecreasing order, but got %s" % (item,))
        self._set = frozenset(iterable)
               
        
    def __contains__(self, item):
        """
        Does this set contain a particular version?  Note that versions are integers.
        """
        if not type(item) is int:
            raise ValueError("expected to containment argument to be int, but got %s" % (item,))
        for v in self._set:
            if type(v) is int:
                if v == item:
                    return True
            else:
                assert hasattr(v, "__len__") and len(v) == 2
                if v[0] <= item <= v[1]:
                    return True
        return False


if __name__ == '__main__':
    from onyx import onyx_mainstartup
    onyx_mainstartup()    

