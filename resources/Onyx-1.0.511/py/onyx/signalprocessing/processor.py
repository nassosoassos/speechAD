###########################################################################
#
# File:         processor.py
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Processor object pushes data through objects in a graph
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
Processor object pushes data through objects in a graph

    >>> True
    True
"""

from onyx.containers.objectset import IndexedObjectSet

class SignalProcessor(object):
    """
    Signal processing object pushes data through objects in a graph
    """

    def __init__(self, stream):
        self.runtime_objects = IndexedObjectSet(stream)
        self.dataflow_graph = FrozenGraph(stream)

    def serialize(self, stream):
        self.runtime_objects.serialize(stream)
        self.dataflow_graph.serialize(stream)
        


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
