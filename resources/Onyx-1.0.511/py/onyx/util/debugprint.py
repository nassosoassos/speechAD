###########################################################################
#
# File:         debugprint.py
# Date:         Wed 4 Jun 2008 18:00
# Author:       Ken Basye
# Description:  Support for conditional printing, intended for diagnostic purposes.
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
    Support for conditional printing, intended for diagnostic purposes.

    This module provides support for conditional printing for diagnostic purposes.
    See help(DebugPrint), help(dcheck), and help(dprint) for further information.
    
    >>> with DebugPrint('foo'):
    ...    dc = dcheck("foo")
    ...    dc and dc("Hello =", 3.14159)
    foo: Hello = 3.14159

    No output here

    >>> dprint('foo', "World")

    Nor here

    >>> dprint('never_seen', "World")

    File or cStringIO objects in the list of keys are used to control where
    the output for subsequent keys is sent by dprint.  The default is to
    print to sys.stdout.

    >>> out1 = cStringIO.StringIO()
    >>> out2 = cStringIO.StringIO()
    >>> with DebugPrint('foo', out1, 'bar', out2, 'baz'):
    ...    dprint('foo', "Hello0")
    ...    dprint('bar', "Hello1")
    ...    dprint('baz', "Hello2")
    foo: Hello0
    >>> print(out1.getvalue().strip())
    bar: Hello1
    >>> print(out2.getvalue().strip())
    baz: Hello2
    >>> out1.close()
    >>> out2.close()
"""

    
from __future__ import with_statement
import sys
from functools import partial
from threading import Lock
from collections import defaultdict
from types import FileType
import cStringIO
from onyx.util import timestamp

class DebugPrint(object):
    """
    A context manager for conditional printing::

      with DebugPrint(*keys):
    
    Instances of this class support conditional printing for diagnostic
    purposes.  Creation of an instance in the context of a 'with' statement will
    activate the key or keys used for creation, which must be an immutable
    object.  The end of the with statement will deactivate the key(s), unless it
    was already activated in some outer context.  Key activation controls the
    behavior of dcheck and dprint; see help(dcheck) and help(dprint) for
    documentation on how to do conditional printing.  A key which is a file
    object or cStringIO object will be used as the output target for conditional
    printing for all subsequent keys; keys without an associated output target
    will have sys.stdout as their output target.  Passing no args means 'do not
    activate any key'.

    The usual idiom is to use dcheck(key) to get a value which is False if key
    isn't active, or a callable if key is active.  This allows the 'dc and
    dc(mesg)' construction, which avoids evaluation of the 'mesg' argument if no
    printing will be done.  See help(dcheck) for details on the callable
    returned for an active key.

    >>> with DebugPrint('HI'):
    ...    dc = dcheck('HI')
    ...    dc and dc("Hello")
    HI: Hello

    Adding the object DebugPrint.TIMESTAMP_ON to the list of keys will turn on
    time stamping for output for subsequent keys.  Use DebugPrint.TIMESTAMP_OFF
    to turn it back off for subsequent keys
    
    >>> with DebugPrint(DebugPrint.TIMESTAMP_ON, 'HI', DebugPrint.TIMESTAMP_OFF, 'HI2'):  #doctest: +ELLIPSIS
    ...    dc0 = dcheck('HI')
    ...    dc2 = dcheck('HI2')
    ...    dc0 and dc0("Hello")
    ...    dc2 and dc2("Hello2")
    [20...] HI: Hello
    HI2: Hello2

    Note: remaining examples use dprint directly for illustrative purposes; you
    probably don't want to do this yourself.
    
    >>> with DebugPrint('HI', 'HI2', 4.5):
    ...    dprint('HI', "Hello")
    ...    dprint('HI2', "World")
    ...    dprint(4.5, "!")
    ...    dprint("not_set", "You should not see this")
    HI: Hello
    HI2: World
    4.5: !

    Using various constants as keys will work, but isn't advisable as it leads
    to confusing-looking calls ands output:
    
    >>> with DebugPrint(1):
    ...   dprint(2, "You do not see this")
    >>> with DebugPrint(True):
    ...   dprint(True, "You do see this")
    True: You do see this
    >>> with DebugPrint(None):
    ...   dprint(None, "You do see this")
    None: You do see this

    Non-conventional use of float as key - you probably don't want to do this
    
    >>> with DebugPrint(4.5):
    ...    dprint(4.5, "Hello")
    4.5: Hello

    Here's a simple way to have a secondary local condition for printing:
    
    >>> blah = False
    >>> with DebugPrint('foo') if blah else DebugPrint():
    ...    dprint('foo', "World")
    >>> blah = True
    >>> with DebugPrint('foo') if blah else DebugPrint():
    ...    dprint('foo', "World")
    foo: World

    You can also create DebugPrint objects without using 'with' and turn them on
    and off with the functions 'on' and 'off'.

    >>> dp = DebugPrint('HI')
    >>> dp.on()
    >>> dc = dcheck('HI')
    >>> dc and dc("Hello")
    HI: Hello
    >>> dp.off()
    >>> dc = dcheck('HI')
    >>> dc and dc("Hello")
    False
    """

    _count_dict = defaultdict(int)
    _stream_dict = dict()
    NO_PREFIX = object()
    NEWLINE_PREFIX = object()
    TIMESTAMP_ON = object()
    TIMESTAMP_OFF = object()
    _lock = Lock()
    
    def __init__(self, *keys):
        DebugPrint._lock.acquire()
        self._keys = []
        self._entered = False
        current_stream = sys.stdout
        timestamp = False
        for key in keys:
            if self._is_stream(key):
                current_stream = key
            elif key is DebugPrint.TIMESTAMP_ON:
                timestamp = True
            elif key is DebugPrint.TIMESTAMP_OFF:
                timestamp = False
            else:
                self._keys.append(key)
            DebugPrint._stream_dict[key] = (current_stream, timestamp)
        DebugPrint._lock.release()

    def __enter__(self):
        assert not self._entered
        DebugPrint._lock.acquire()
        for key in self._keys:
            DebugPrint._count_dict[key] += 1
        self._entered = True
        DebugPrint._lock.release()

    # XXX should figure out when the dummy args are not None and do the right
    # thing in that case.
    def __exit__(self, dummy1=None, dummy2=None, dummy3=None):
        assert self._entered
        DebugPrint._lock.acquire()
        for key in self._keys:
            DebugPrint._count_dict[key] -= 1
        self._entered = False
        DebugPrint._lock.release()

    on = __enter__
    off = __exit__

    @staticmethod
    def _is_stream(item):
        return isinstance(item, (file, cStringIO.OutputType))

    @staticmethod
    def active(key):
        # Can't use dprint here :->
        # print("active: key is %s, dict is %s" % (key, DebugPrint._count_dict))
        # have_it = DebugPrint._count_dict.has_key(key)
        # print("active: DebugPrint._count_dict.has_key(%s) is %s" % (key, have_it))
        # if have_it:
        #     print("active: DebugPrint._count_dict[%s] is %d" % (key, DebugPrint._count_dict[key],))
        count = DebugPrint._count_dict.get(key)
        return count > 0 if count is not None else 0

    @staticmethod
    def get_stream(key):
        assert DebugPrint.active(key)
        return DebugPrint._stream_dict[key]



def dcheck(*keys):
    """
    Check to see if debug printing keys are active, get a debug print function
    if so.
    
    Call this function to see if a key or any of several keys is active in
    DebugPrint.  If an active key is not found, the return is False.  If an
    active key is found, the return is a callable with argument (\*things) which
    behaves just like dprint (q.v.). Note that such a callable is considered
    True when evaluated as a boolean, allowing the 'and' construction used
    below, which avoids evaluation of the arguments to the callable.  See
    help(DebugPrint) for documentation on how to activate keys.
    
    >>> with DebugPrint('HI'):
    ...    print('HI %s active' % ('is' if dcheck('HI') else 'is not'))
    ...    print('HI2 %s active' % ('is' if dcheck('HI2') else 'is not'))
    ...    print('HI or HI2 %s active' % ('is' if dcheck('HI', 'HI2') else 'is not'))
    HI is active
    HI2 is not active
    HI or HI2 is active

    >>> with DebugPrint('HI'):
    ...    dc = dcheck('HI')
    ...    dc and dc("Hello")
    HI: Hello
    """
    
    active = DebugPrint.active
    for k in keys:
##         # Avoid call to active for sake of speed?
##         if DebugPrint._count_dict.has_key(k) and DebugPrint._count_dict[k] > 0 :
        if active(k):
            return partial(dprint, k)
    return False

def dprint(key, *things):
    """
    Print str(key) +": " + str(thing[0]) + ' ' + str(thing[1]) + ... if key is active in DebugPrint.
    Output goes to the stream associated with key which is sys.stdout by
    default.  If thing[0] is DebugPrint.NO_PREFIX, suppress both str(key) and
    the colon-space. If thing[0] is DebugPrint.NEWLINE_PREFIX, write a newline
    before the usual output; this is useful for setting off a block of debug
    printing.  See help(DebugPrint) for documentation on how to activate keys
    and how to change the stream associated with a key.

    >>> with DebugPrint('HI'):
    ...    dprint('HI', "Hello,", "World!")
    HI: Hello, World!

    >>> with DebugPrint('HI'):
    ...    dprint('HI', DebugPrint.NO_PREFIX, "Hello")
    Hello

    Turning off the prefix includes turning off timestamps if they were turned
    on:

    >>> with DebugPrint(DebugPrint.TIMESTAMP_ON, 'HI'):
    ...    dprint('HI', DebugPrint.NO_PREFIX, "Hello")
    Hello

    >>> with DebugPrint('HI'):
    ...    dprint('HI', DebugPrint.NEWLINE_PREFIX, "World!")
    <BLANKLINE>
    HI: World!

    A newline is printed before the prefix, including the timestamp if there is
    one:

    >>> with DebugPrint(DebugPrint.TIMESTAMP_ON, 'HI'):  #doctest: +ELLIPSIS
    ...    dprint('HI', DebugPrint.NEWLINE_PREFIX, "World!")
    <BLANKLINE>
    [20...] HI: World!

    """
    if len(things) == 0:
        return
    if DebugPrint.active(key):
        outs = list()
        (stream, use_timestamp) = DebugPrint.get_stream(key)
        thingiter = iter(things)
        thing0 = things[0]
        if thing0 is DebugPrint.NO_PREFIX:
            thingiter.next()
        else:
            if thing0 is DebugPrint.NEWLINE_PREFIX:
                thingiter.next()
                outs.append('\n')
            if use_timestamp:
                outs.append('[' + timestamp.get_timestamp() + '] ')
            prefix = str(key), ': '
            outs.extend(prefix)
        outs.append(' '.join(str(thing) for thing in thingiter))
        outs.append('\n')
        stream.write(''.join(outs))
        # may want to give a control for this flush
        stream.flush()
    
if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
