###########################################################################
#
# File:         audiobase.py (directory: ./py/onyx/audio)
# Date:         8-Nov-2007
# Author:       Hugh Secker-Walker
# Description:  Framework for audio input
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
    Exercise dynamically-linked audio library.

    Check some emprically observed invariants (on the Mac OS X 10.4).

    >>> audio.default_input() > 200
    True

    >>> len(audio.transport_name(audio.default_input())) > 0
    True

    Make two erroring calls

    >>> audio.transport_name(None)
    Traceback (most recent call last):
       ...
    TypeError: an integer is required

    >>> len(audio.device_info(audio.default_input())) == 2
    True

    >>> audio.device_info(None)
    Traceback (most recent call last):
       ...
    TypeError: an integer is required

    >>> min_num_inputs = 2
    >>> inputs = audio.inputs()
    >>> # inputs
    >>> len(inputs) >= min_num_inputs
    True

    Verify structure of inputs

    >>> #setify this guy
    >>> tuple(input for input in inputs if not( len(input) == 4 and map(type, input) == [int, str, str, str]))
    ()

    There are three built-in devices....

    >>> len(tuple(input for input in inputs if input[1] == 'Built-In')) == 3
    True


    We put timeouts in the checks of the deque because some devices
    don't return any buffers, e.g. a disconnected Bluetooth headset.

    Mess around with the devault input device

    >>> source = audiosource_base(audio.default_input())
    >>> source.device
    1
    >>> source.start()
    >>> start = time.time()
    >>> while len(source) < 500 and time.time() - start < 2: time.sleep(0.1)
    >>> source.stop()
    >>> x = source.pop()
    >>> len(x) == 2
    True
    >>> len(x[0]) == len(x[1]) == 512
    True
    >>> set(len(x) == 2 and len(x[0]) == len(x[1]) == 512 for x in source)
    set([True])
    >>> source.clear()
    >>> del source

    Get all the input sources at once.  This can fail if you have non-input
    devices that don't include 'output' in their descriptions, e.g. some (cheap)
    USB speakers expose non-working microphone devices from their chipsets....

    >>> sources = tuple(audiosource_base(input[0]) for input in inputs if input[2].lower().find('output') == -1)
    >>> len(sources) >= 2
    True

    Start them all

    >>> for source in sources: source.start()
    >>> start = time.time()
    >>> while min(*(len(source) for source in sources)) < 200 and time.time() - start < 1: time.sleep(0.1)
    >>> for source in sources: source.stop()

    >>> start = time.time()
    >>> for source in sources: source.start()
    >>> while min(*(len(source) for source in sources)) < 500 and time.time() - start < 5: time.sleep(0.1)
    >>> for source in sources: source.stop()
    
    >>> set(len(x) == 2 and len(x[0]) == len(x[1]) == 512 for source in sources for x in source)
    set([True])

    >>> del sources
"""

import time
from collections import deque

# import onyx so that sys.path gets tweaked to find the built shared objects;
# this permits running this script's doctests stand-alone, e.g. from emacs
import onyx
# _audio is the shared object for access to live audio
import _audio as audio

class audiosource_base(deque):

    def __init__(self, uid, *args):

        # __del__ requires self.device, so we bind it before making
        # any calls that could fail and cause us to be deleted causing
        # __del__ to be called
        self.device = None

        super(audiosource_base, self).__init__(*args)

        verbose = False
        device = audio.new_device(uid, self.appendleft, verbose)

        self.uid = uid
        self.device = device

    def __del__(self):
        if self.device is not None:
            self.stop()
        audio.del_device(self.device)

    def start(self):
        audio.start_device(self.device)

    def stop(self):
        audio.stop_device(self.device)
    


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
