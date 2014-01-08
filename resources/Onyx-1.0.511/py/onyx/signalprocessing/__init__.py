###########################################################################
#
# File:         __init__.py (package: onyx.signalprocessing)
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Support for signal processing objects
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
Signal Processing

This package supports a wide variety of signal processing objects.

"""

from onyx.builtin import attrdict
from onyx.dataflow import processor
from onyx.containers.serializable import Serializable

class sigprocbase(processor, Serializable):
    """
    Baseclass for signal processing objects that adhere to
    string, list, or dict constructor options.

    >>> def printit(spobj):
    ...   for key in sorted(spobj.init_options.iterkeys()):
    ...     print key, ':', spobj.init_options[key]
    ...   print spobj.init_args


    String initialization:

    >>> obj = sigprocbase('foo=23 bar=update 23 56 75')
    >>> printit(obj)
    bar : update
    foo : 23
    ['23', '56', '75']

    List-of-string initialization:

    >>> obj = sigprocbase(['foo=44', 'bar=reset', 'period=2300*usec', '19', '45', 'attack-decay'])
    >>> printit(obj)
    bar : reset
    foo : 44
    period : 2300*usec
    ['19', '45', 'attack-decay']

    Dict initialization:
    
    >>> obj = sigprocbase({'3dB': '1500*hz'})
    >>> printit(obj)
    3dB : 1500*hz
    []
    >>> obj
    sigprocbase({'3dB': '1500*hz'})

    Subclass with defaults and then a string-based override example:
    
    >>> class myfilter(sigprocbase):
    ...   def __init__(self, options=None):
    ...     self.init_options = attrdict({'cutoff': '25*hz', 'slope': '50*dB/oct'})
    ...     super(myfilter, self).__init__(options)

    >>> printit(myfilter())
    cutoff : 25*hz
    slope : 50*dB/oct
    []

    >>> printit(myfilter('cutoff=50hz foo'))
    cutoff : 50hz
    slope : 50*dB/oct
    ['foo']


    >>> obj = sigprocbase(23)
    Traceback (most recent call last):
       ...
    ValueError: expected options to be an instance of str, tuple, list, or dict, but got 'int'


    >>> sigprocbase().configure()
    Traceback (most recent call last):
      ...
      File "<doctest __main__.sigprocbase[12]>", line 1, in <module>
        sigprocbase().configure()
      File "<stdin>", line 110, in configure
    NotImplementedError: subclass must override this method
    """


    def __init__(self, options=None):
        """Updates self.init_options dictionary with option values, creating the dictionary if it doesn't exist."""

        super(sigprocbase, self).__init__()

        self.original_options = options

        if not hasattr(self, 'init_options'):
            self.init_options = attrdict()
        init_options = self.init_options

        if not hasattr(self, 'init_args'):
            self.init_args = list()
        init_args = self.init_args

        if options is None:
            return

        if isinstance(options, str):
            options = options.split()

        if isinstance(options, (list, tuple)):
            # put assignments in the dictionary
            init_options.update(option.split('=') for option in options if option.count('=') == 1)
            # append all others onto the list
            init_args.extend(option for option in options if option.count('=') != 1)
        elif isinstance(options, dict):
            init_options.update(options)
        else:
            raise ValueError("expected options to be an instance of str, tuple, list, or dict, but got %r" %(type(options).__name__,))

        if init_options.has_key('serial_version'):
            v = int(init_options['serial_version'])
            self.check_serial_version(v)

    def __repr__(self):
        return "%s(%s)" % (type(self).__name__, repr(self.original_options),)


    def get_serial_version(self):
        if not hasattr(self, 'SERIAL_VERSION'):
            raise NotImplementedError("subclass must provide a value for SERIAL_VERSION")
        else:
            return self.SERIAL_VERSION

    def check_serial_version(self, version):
        if not hasattr(self, 'SERIAL_VERSION'):
            raise NotImplementedError("subclass must provide a value for SERIAL_VERSION")
        elif self.SERIAL_VERSION != version:
            raise ValueError("expected version %d, but got %d" % (self.SERIAL_VERSION, version))

##     def configure(self, **kwargs):
##         raise NotImplementedError("subclass must override this method")

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
