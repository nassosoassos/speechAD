###########################################################################
#
# File:         liveaudio.py (directory: ./py/onyx/audio)
# Date:         7-Jan-2009
# Author:       Hugh Secker-Walker
# Description:  A dataflow source for live audio input
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
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
    Provides LiveAudioSource and some live audio tools from the _audio module.
    A LiveAudioSource is used to push live audio data into a dataflow network.

    >>> output = list()
    >>> verbose = False
    >>> mic = LiveAudioSource(default_input(), sendee=output.append, verbose=verbose)
    >>> mic.uid > 0
    True
    >>> mic_name = str(mic)
    >>> len(mic_name) > 0
    True
    >>> mic_name == 'Built-in Microphone, Apple Inc.' or True
    True
    >>> mic.start()
    >>> time.sleep(1/16)
    >>> mic.stop()
    >>> mic.done()
    >>> len(output) > 1
    True
"""
from __future__ import division

import time

# import onyx so that sys.path gets tweaked to find the built shared objects;
# this permits running this script's doctests stand-alone, e.g. from emacs
import onyx
# _audio is the shared C-based module for access to live audio
import _audio as audio
from _audio import inputs, default_input, transport_name, device_info

from onyx.dataflow.source import DataflowSourceBase

class LiveAudioSource(DataflowSourceBase):
    def __init__(self, uid, sendee=None, verbose=False):
        # __del__ uses self.device, so we bind it before making
        # any calls that could fail and cause us to be deleted causing
        # __del__ to be called
        self._device = None

        super(LiveAudioSource, self).__init__(sendee)
        self._uid = uid

        def process(value):
            # value will be a tuple of buffers, where each buffer is samples
            # from one channel of the input device

            # no error even if we're done, so no check_name argument to _enter()
            if self._enter():
                self._enqueue(value)
                self._exit()

        self._device = audio.new_device(self.uid, process, verbose)

    def __str__(self):
        return ', '.join(device_info(self.uid))
    def __repr__(self):
        return '<%s: %s>' % (type(self).__name__, str(self))

    @property
    def uid(self):
        return self._uid

    def start(self):
        super(LiveAudioSource, self).start()
        audio.start_device(self._device)

    def stop(self):
        audio.stop_device(self._device)
        super(LiveAudioSource, self).stop()

    def done(self):
        # note: this is automatically called by __del__
        if self._device is not None:
            self.stop()
            audio.del_device(self._device)
            self._device = None
        super(LiveAudioSource, self).done()

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
