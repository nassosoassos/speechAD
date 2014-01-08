###########################################################################
#
# File:         serializable.py (directory: ./py/onyx/containers)
# Date:         Thu 4 Sep 2008 11:42
# Author:       Ken Basye
# Description:  Mixin class for a set of serializable classes - see also objectset.py for parsing functionality
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
Mixin class for a set of serializable classes


"""

# XXX There's some trickiness with getting good module names in derived classes.
# If you run into such a problem, one workaround seems to be to explicitly
# import the class you want to serialize from the module it's in, even if this
# is otherwise gratuitous.  See the example of the class _Test below.  - KJB

class Serializable(object):
    """
    This mix-in class supports serialization of objects and their subsequent deserialization by
    IndexedObjectSet, q.v..  Classes inheriting from Serializable must override get_serial_version()
    and may also choose to override get_serial_module_name, get_serial_factory_name, and
    get_serial_factory_args.  

    Note that this class can't be actually be used:

    >>> t0 = Serializable()
    >>> t0.serialize()
    Traceback (most recent call last):
        ...
    NotImplementedError: Class 'Serializable' derived from Serializable must implement get_serial_version()
    """

    def serialize(self):
        """
        Return a serialized version of this object as a tuple of strings.  The form of the tuple is:
        (module_name, factory_name, version, arg0, arg1, ...)  where the version and the args will
        be passed to the factory to construct the object.
        """
        module_name = self.get_serial_module_name()
        factory = self.get_serial_factory_name()
        version = str(self.get_serial_version()) 
        args = self.get_serial_factory_args()
        return (module_name, factory, version) + args

    def get_serial_module_name(self):
        return str(type(self).__module__)

    def get_serial_factory_name(self):
        return str(type(self).__name__)

    def get_serial_version(self):
        raise NotImplementedError("Class %r derived from Serializable "
                                  "must implement get_serial_version()" % (type(self).__name__,))
        
    def get_serial_factory_args(self):
        return ()


class _Test(Serializable):
    """
    Test proper usage of Serializable as a mix-in.  We do this here rather than
    in the Serializable doc string because we'd like to get a reasonable module
    name.

    Note: this is a rare instance of allowing an import in a doctest string --
    as a rule we do not use imports in doctest strings because the modulefinder
    doesn't track them for SCons's dependency work.  Since the import is
    importing this very module, there's no dependecy problem.

    >>> from onyx.containers.serializable import _Test
    >>> t0 = _Test()
    >>> t0.serialize()
    ('onyx.containers.serializable', '_Test', '3')
    """
    VERSION = 3
    def get_serial_version(self):
        return self.VERSION


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
