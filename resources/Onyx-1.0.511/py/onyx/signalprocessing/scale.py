###########################################################################
#
# File:         scale.py (directory: ./py/onyx/signalprocessing)
# Date:         12-Sep-2007
# Author:       Hugh Secker-Walker
# Description:  Numerical data-scaling objects
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 - 2009 The Johns Hopkins University
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
    Numerical data-scaling objects
"""

from __future__ import division
from math import log, sqrt, exp, pow, e as EulersNumber
from onyx.builtin import attrdict
from onyx.signalprocessing import sigprocbase

def number(value):
    """
    Change value from string to complex to float to int if such
    changes do not lose numerical precision.

    >>> number(23.0)
    23
    >>> number(23.5+0j)
    23.5
    >>> number('23.0+0j')
    23
    """

    if isinstance(value, str):
        value = complex(value)
    if isinstance(value, complex):
        if value.imag == 0j:
            value = value.real
    if isinstance(value, float):
        intvalue = int(value)
        if intvalue == value:
            value = intvalue
    return value


class Abs(sigprocbase):
    """
    Takes the absolute value of the components of (possibly
    complex-valued) data vectors.  Returns a tuple of the results.

    >>> a = Abs()
    >>> a([1, 2, -3])
    (1, 2, 3)

    >>> c = tuple(complex(3 * x, -4 * x) for x in xrange(7))
    >>> c
    (0j, (3-4j), (6-8j), (9-12j), (12-16j), (15-20j), (18-24j))
    >>> a(c)
    (0, 5, 10, 15, 20, 25, 30)

    """
    SERIAL_VERSION = 0
    def __init__(self, arg=None):
        super(Abs, self).__init__(arg)
        # XXX have sigprocbase.__init__() take a no_args=True option and throw the error
        # KJB: Now that we have serial_version=N as an option, we can't make this assert
        # assert not arg

    @staticmethod
    def __call__(iterable):
        return tuple(number(abs(value)) for value in iterable)


class Log(sigprocbase):
    """
    Takes the log of the components of data vectors.  Actually
    computes (scale * log_base) of the values given, where scale
    defaults to 1 and base defaults to that of the natural log given
    by math.e in the math module.

    Returns a tuple of the results.

    >>> def roundem(iterable): return tuple(number(round(x, 3)) for x in iterable)

    >>> log2 = Log('base=2')
    >>> log2([1,2,4,8])
    (0.0, 1.0, 2.0, 3.0)

    >>> roundem(log2(sqrt(2)**x for x in xrange(11)))
    (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5)

    >>> loge = Log()
    >>> loge([1, exp(1)])
    (0.0, 1.0)

    >>> dBamp = Log('base=10 scale=20')
    >>> roundem(dBamp(sqrt(10)**x for x in xrange(11)))
    (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

    >>> dBpower = Log('base=10 scale=10')
    >>> roundem(dBpower(10**x for x in xrange(11)))
    (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    """
    
    SERIAL_VERSION = 0
    def __init__(self, args=None):
        self.init_options = attrdict({'base': EulersNumber, 'scale': 1})
        super(Log, self).__init__(args)
        self.logbase = pow(number(self.init_options.base), 1 / number(self.init_options.scale))
        
    def __call__(self, iterable):
        logbase = self.logbase
        return tuple(log(value, logbase) for value in iterable)

    def get_serial_factory_args(self):
        return ("base=%s" % (self.init_options.base,),  "scale=%s" % (self.init_options.scale,))

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
