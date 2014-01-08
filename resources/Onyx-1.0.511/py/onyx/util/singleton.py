###########################################################################
#
# File:         singleton.py
# Date:         Tue 2 Sep 2008 15:47
# Author:       Ken Basye
# Description:  Support for singleton objects
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
Support for singleton objects

"""

class Singleton(object):
    """
    Singleton provides a simple way to instantiate singleton objects.

    As a convention, we suggest that you give singletons names which include the modules
    where they're instantiated, to prevent collisions across modules.
    >>> x = Singleton('onyx.util.singleton.name0')
    >>> x
    onyx.util.singleton.Singleton('onyx.util.singleton.name0')
    
    Singletons can be tested for equality with 'is' (and with '==' if you prefer)
    >>> x is Singleton('onyx.util.singleton.name0')
    True
    >>> x is Singleton('onyx.util.singleton.name1')
    False
    >>> y = Singleton('foo')
    >>> y
    onyx.util.singleton.Singleton('foo')
    >>> x == y
    False

    Use the name property to get the name of a singleton
    >>> x.name
    'onyx.util.singleton.name0'
    >>> y.name
    'foo'
    """

    _registry = dict()
    def __new__(cls, name):
        if not cls._registry.has_key(name):
            # register name
            s = super(Singleton, cls).__new__(Singleton)
            s._name = name
            cls._registry[name] = s
            
        return cls._registry[name]

    def __repr__(self):
        return "onyx.util.singleton.Singleton('%s')" % self._name

    @property
    def name(self):
        return self._name

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()


        




