###########################################################################
#
# File:         __init__.py (package: onyx.dataflow)
# Date:         11-Oct-2007
# Author:       Hugh Secker-Walker
# Description:  Package initialization for onyx.dataflow
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
    Support for dataflow elements

    >>> processor()
    Traceback (most recent call last):
      ...
      File "<stdin>", line 31, in __repr__
    NotImplementedError: subclass must override this method

"""

class processor(object):
    """
    Baseclass for objects implementing the processor interface.
    """

    null = object()

    def __init__(self, *_):
        pass

    def __repr__(self):
        raise NotImplementedError("subclass must override this method")

    def configure(self, **kwargs):
        raise NotImplementedError("subclass must override this method")

    def send_many(self, inputs):
        send = self.send
        for input in inputs:
            send(input)


    def process_one(self, item):
        # process a single input item and return a list of results
        # if item is self.null, do any state update and return a list of results
        raise NotImplementedError("subclass must override this method")

    def process_some(self, items):
        """
        Process each of the elements of 'items'.  Return a list of results.
        """
        process_one = self.process_one
        ret = process_one(self.null)
        for item in items:
            ret.extend(process_one(item))
        return ret


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
