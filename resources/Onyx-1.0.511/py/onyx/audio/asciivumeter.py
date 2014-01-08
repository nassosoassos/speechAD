###########################################################################
#
# File:         asciivumeter.py
# Date:         12-Nov-2007
# Author:       Hugh Secker-Walker
# Description:  Open live inputs and output chart-plotter lines of VU Meter levels
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007, 2008 The Johns Hopkins University
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
Code that uses the Python live input to create an ascii-based version of a
chart-plotter record of the VU Meter analysis of the live input.

As of December 2008 this only works on Mac OS X.
"""
from __future__ import division

# Code for playing around with a live-input ASCII VU meter.

# import onyx so that sys.path gets tweaked to find the built shared objects;
# this permits running this script's doctests stand-alone, e.g. from emacs
import onyx
# _audio is the shared object for access to live audio
from _audio import inputs, default_input
from audiobase import audiosource_base
from math import sqrt, log10, log
from itertools import repeat, izip
from functools import partial
from sys import argv, stdout
from time import time, sleep
from collections import deque

from onyx.signalprocessing.spectrum import fft


# command-line: positive argv[1] means run for that many seconds; -1 means run forever;
# no command line means run for one second
seconds = int(argv[1]) if len(argv) > 1 else 1

# A second command-line argument of any value means generate lines appropriate
# for a horizontal display - yuck!
vertical_display = (len(argv) <= 2)

# BEGIN configurable constants: fiddle these if you want

# if True, use system's "default" input, e.g. as selected in Preferences->Sound
# if False, display from all input devices; sigh... setting this False can cause
# a failure with the 'some queue is not getting data, stopping' message if there
# are devices that look like input devices, but which don't actually run...
default_only = True

# if True, do not display results for buffers/channels containing a
# single flatline value (e.g. all zeroes)
remove_digitalsilence = True

# if True, do not display duplicate channels; e.g. when Apple's
# "ambient noise reduction" is switched on, the two "stereo" channels
# from the built-in mic eventually have identical samples in them
# (after a few seconds, interesting observation regarding Apple's
# algorithm); setting this True will process and display only a single
# channel when stereo channels are identical
uniquify = True

# if True, compute FFT on each buffer; nothing is done with the
# resulting FFTs, so this merely affects performance
do_ffts = False

# reference amplitude for 0 dB; 0 dB is the minimum displayed
reflevel = 1 / (1 << 17)
#reflevel = 1 / (1 << 18)

# maximum dB; threshold for clipping display in the UI
maxdb = 120
#maxdb = 110

# Endpoint Controls
min_hunt = 0.1
max_hunt = -0.25
min_range = 15
low_per_mil = 300
high_per_mil = 700
start_window = 25
start_count = 10
stop_window = 40
stop_count = 15

show_trackers = False



# END configurable constants


def is_flatline(buf):
    return len(set(buf)) <= 1

# XXX maxdb isn't used here
def db(ref_dev, maxdb, buf):
    recip_ref = 1 / ref_dev
    len_buf = len(buf)
    recip_len = 1 / len_buf
    mean_sample = sum(buf) * recip_len
    mean_energy = sum(i*i for i in buf) * recip_len
    var = mean_energy - mean_sample * mean_sample
    dev = sqrt(var)
    db = int(20 * log10(dev * recip_ref)) if dev > ref_dev else 0
    # return maxdb if db > maxdb else db
    return db
refdb = partial(db, reflevel, maxdb)

dbs_queue = deque()
tag_queue = deque()
from onyx.signalprocessing.endpoint import UtteranceTagger
utt_tagger = UtteranceTagger(min_hunt, max_hunt, min_range, low_per_mil, high_per_mil, start_window, start_count, stop_window, stop_count)

hbar1 = ':' if vertical_display else '-'
hbar2 = '|' if vertical_display else '_'
vbar = '-' if vertical_display else '|'


print 'inputs:'
for input in inputs():
    print ' ', input
print 'default', default_input()

if default_only:
    sources = (audiosource_base(default_input()),)
else:
    sources = tuple(audiosource_base(input[0]) for input in inputs() if input[2].lower().find('output') == -1)

joinnosp = ''.join

# tokens for succesive channels
tokens = ( 'x', 's', 'o', '$',  '+', 'X', '*', '#')
# non-data elements of the "strip recorder"
overwrite = set((' ', hbar2, 'I', '}', hbar1, ';', '.', '=', vbar))
spaces = [' '] * (maxdb+1)
## clip = ['= - '] * ((maxdb+3) // 4 + 1)
## clip = ['| == '] * ((maxdb+4) // 5 + 1)
clip = ['|= == == ='] * ((maxdb+4) // 5 + 1)
clip = list(joinnosp(clip))[:len(spaces)]

for i in xrange(0, (maxdb+1), 10):
    spaces[i] = (hbar1 if i % 20 == 0 else '.')
    if i % 60 == 0:
        spaces[i] = hbar2

assert len(spaces) == len(clip)

start = time()

for source in sources: source.start()

reciplog2 = 1 / log(2)
onethird = 1 / 3
sample_count = 0
    

def get_bufs(sources):
    bufs = list()
    for source in sources:
        # deal with buffered up channels for source
        buffen = list(list(buf) for buf in source.pop())
        # collect any backlog
        while len(source) > 0:
            for buff, buf in izip(buffen, source.pop()):
                buff.extend(buf)
        bufs.extend(buffen)
    return tuple(bufs)

def remove_flatline(bufs):
    return tuple(buf for buf in bufs if not is_flatline(buf))

def remove_paired_duplicates(bufs):
    newbufs = list()
    prev = None
    for buf in bufs:
        if buf != prev:
            newbufs.append(buf)
        prev = buf
    return tuple(newbufs)
    
while seconds < 0 or time() - start < seconds:
    def maxqueu(): return max(len(source) for source in sources)
    def minqueu(): return min(len(source) for source in sources)
    m = minqueu()
    while m > 0:

        bufs = get_bufs(sources)
        if remove_digitalsilence:
            # remove bufs containing a unique sample value
            bufs = remove_flatline(bufs)
        if uniquify:
            # remove bufs that are duplicate stereo bufs
            bufs = remove_paired_duplicates(bufs)

        assert type(bufs) is tuple
        
        sample_count += max(len(buf) for buf in bufs)

        # do the dB calculation on each buffer
        dbs = tuple(refdb(buf) for buf in bufs)
        if do_ffts:
            ffts = tuple(fft(buf) for buf in bufs)

        # need to use dataflow and slot filler
        tag_queue.extend(utt_tagger(dbs[0]))
        if show_trackers:
            dbs = dbs + tuple(int(x) for x in utt_tagger.tracker.range)
        dbs_queue.append(dbs)


        if tag_queue and dbs_queue:
            # build the ascii display

            dbs = dbs_queue.popleft()

            places = spaces[:]
            for db, token in izip(dbs, tokens):
                if db < maxdb:
                    pl = places[db]
                    places[db] = token if pl in overwrite else 'O'
                else:
                    places = clip[:]

            # half-second lines (with assumptions)
            if sample_count >= 22050:
                sample_count = 0
                for i in xrange(2, maxdb, 2):
                    if places[i] == ' ':
                        places[i] = vbar

            # utterance tag
            if tag_queue.popleft():
                places[2:4] = '=='

            # the main part of the display
            main = joinnosp(places)

            # show the buffer backlog
            m = maxqueu()
            backlog = hbar2 * m
            print '\n', main, ' ', backlog, '   ',
            stdout.flush()

        if m > 1000:
            print
            print "minqueu:", m, "too far behind real time, stopping"
            for source in sources: source.stop()
            
        m = minqueu()

    sleep_max = 1.3
    sleep_base = 0.01
    sleeper = sleep_base
    while sleeper <= sleep_max:
        sleep(sleeper)
        m = minqueu()
        if m > 0:
            break
        sleeper *= 2
    if sleeper > sleep_max:
        print
        print 'some queue is not getting data, stopping'
        for source in sources: source.stop()
        break

print    
