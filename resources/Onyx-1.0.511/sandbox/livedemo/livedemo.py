###########################################################################
#
# File:         livedemo.py (directory: ./sandbox/livedemo)
# Date:         8-Jan-2009
# Author:       Hugh Secker-Walker
# Description:  Speech Activity Detection and DDT demo
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
    >>> True
    True
"""
from __future__ import division

import math
import numpy as N
import sys
from collections import deque
from itertools import izip, count

from onyx.am.classifier import AdaptingGmmClassifier, AdaptingGmmClassProcessor
from onyx.am.gaussian import  GaussianModelBase, SimpleGaussianModel, GaussianMixtureModel
from onyx.am.modelmgr import  GmmMgr
from onyx.dataflow.join import SynchronizingSequenceJoin
from onyx.signalprocessing.htkmfcc import make_fft_abs_processor, make_melcepstral_processor
from onyx.util.debugprint import DebugPrint, dcheck, dprint
from onyx.util.streamprocess import FunctionProcessor, SequenceFunctionProcessor, ChainProcessor, SplitProcessor


def open_displays():
    global logstream, displaystream, displaystream2

    args = sys.argv[1:]
    if '-l' in args:
        logstream = open(args[args.index('-l')+1], 'wb', 0)
    else:
        logstream = sys.stdout

    if '-o' in args:
        displaystream = open(args[args.index('-o')+1], 'wb', 0)
    else:
        displaystream = None

    if '-o2' in args:
        displaystream2 = open(args[args.index('-o2')+1], 'wb', 0)
    else:
        displaystream = None

if __name__ == '__main__':
    open_displays()


## def make_dir():
##     olddir = dir
##     def new_dir(*args):
##         print ' '.join(item for item in olddir(*args) if not item.startswith('__'))
##     return new_dir
## # doesn't work, presumably we'd have to put it into the builtins....
## #dir = make_dir()
## def mydir(*args): print ' '.join(dir(*args))
join = ' '.join
import time
from time import sleep
import sys

class doer(object):
    """
    Returns an object whose repr calls the thunk which was handed to the constructor.
    The idea is to put these in the __main__ namespace of the interpreter and then
    typing the bareword attribute in the namespace causes the thunk to get run.
    """
    __slots__ = 'thunk'
    def __init__(self, thunk):
        self.thunk = thunk
    def __repr__(self):
        self.thunk()
        return ' '
        # return self.thunk.__name__

def reload():
    print 'execfile(%r)' % __file__
    execfile(__file__)
reload = doer(reload)

class audio(object):
    """
    Expose a microphone interface for use at the interpreter.
    """
    __slots__ = 'current_mic'
    def __init__(self):
        self.current_mic = None
    @property
    def list(self):
        from onyx.audio.liveaudio import inputs
        for id, chan, name, manu in inputs():
            print name
    def get_mic(self):
        return self.current_mic
    def set_mic(self, mic_name=None):
        # select a new live input
        from onyx.audio.liveaudio import inputs, default_input, LiveAudioSource, device_info
        if mic_name is not None:
            mic_id = None
            if isinstance(mic_name, str):
                for id, chan, name, manu in inputs():
                    if name.lower().find(mic_name.lower()) != -1:
                        mic_id = id
                        break
            if mic_id is None:
                print 'no microphone matching %r' % mic_name
                return
        else:
            mic_id = default_input()
        assert mic_id is not None
        new_mic = LiveAudioSource(mic_id, verbose=False)

        # remove the old input
        if self.current_mic is not None:
            state = self.current_mic.state
            sendee = self.current_mic.sendee
            self.current_mic.done()
        else:
            state = sendee = None

        self.current_mic = new_mic
        if sendee is not None:
            self.current_mic.set_sendee(sendee)
            # XXX knows too much about LiveAudioSource
            if state == 'running':
                self.current_mic.start()
        global start, stop
        start = doer(self.current_mic.start)
        stop = doer(self.current_mic.stop)
        # hack to avoid printing the first time the mic is set, which happens
        # when the module is loaded, and which messes up doctesting verification
        if not (state is sendee is None):
            print ', '.join(device_info(mic_id))

    mic = property(get_mic, set_mic)

    @property
    def on(self):
        if self.current_mic is not None:
            self.current_mic.start()
    @property
    def off(self):
        if self.current_mic is not None:
            self.current_mic.stop()
audio = audio()
# default input
audio.mic = None

    
# XXX LiveAudioSource (and thus audiomodule.cpp) need to be able to provide
# the sample frequency....
sample_nsec =    22676  # 44100 Hz
#frame_nsec =  10000000  # 10 msec
#window_nsec = 25600000  # 25.6 msec
frame_nsec =   8000000  #  8 msec
window_nsec = 24000000  # 24 msec
dither = 1 / (1 << 10)
preemcoef = 0.96875
samples_per_sec, fft_size, fftmag = make_fft_abs_processor(sample_nsec, frame_nsec, window_nsec,
                                                            dither=dither, preemcoef=preemcoef,
                                                            zmeansource=True, usehamming=True,
                                                            max_channels=2)

class DpToggle(object):
    """
    An object with two attributes, on and off, that toggle debug printing
    of the DebugPrint object created from the constructor args.
    """
    # XXX DebugPrint needs work to develop semantics to support multiple
    # instances of DpToggle....
    def __init__(self, *args):
        self.dp_control = DebugPrint(*args)
        self._on = False
    def __del__(self):
        self.off
    @property
    def on(self):
        if not self._on:
            self._on = True
            self.dp_control.on()
            return 'on'
    @property
    def off(self):
        if self._on:
            self._on = False
            self.dp_control.off()
            return 'off'

vumeterlog = 'vumeterlog'
vulog = DpToggle(logstream, vumeterlog)

runslog = 'Coalescer'
runs = DpToggle(sys.stdout, runslog)

endpointdisplay = 'endpointdisplay'
utt = DpToggle(displaystream, endpointdisplay)

utt.scale = 1

def fillon():
    utt.fill = True
def filloff():
    utt.fill = False
filloff()
utt.fillon = doer(fillon)
utt.filloff = doer(filloff)

def trakon():
    utt.trak = True
def trakoff():
    utt.trak = False
trakoff()
utt.trakon = doer(trakon)
utt.trakoff = doer(trakoff)

class Coalescer(object):
    """
    Coalesce runs of items with the same tag.  So, always buffers at least
    one item since it has to decide if the tag has changed....
    """
    def __init__(self, label='<unnamed>'):
        self.result = list()
        self.prev_tag = object()
    def __call__(self, value):
        # XXX gap semantics
        tag, data = value
        result = self.result
        ret = ()
        if tag != self.prev_tag:
            if result:
                # careful: return a one-item sequence of (tag, (events...))
                rettag = self.prev_tag
                retseq =  tuple(result)
                ret = ((rettag, retseq),)
                dc = dcheck(runslog)
                if dc:
                    dc(ret[0])
                    dc(rettag)
                    for index, item in enumerate(retseq):
                        dc(' ', index)
                        for val in item:
                            dc(' ', ' ', val)
                del result[:]
            self.prev_tag = tag
        result.append(data)
        return ret

class SimpleGaussianTrainer(object):
    def __init__(self, labels, nfeatures):
        self.labels = list(labels)
        self.nfeatures = nfeatures
        self.models = tuple(SimpleGaussianModel(nfeatures, GaussianModelBase.DIAGONAL_COVARIANCE) for _ in self.labels)
        self.samples_seen = [0] * len(self.models)
    def __call__(self, labeled_data):
        label, data = labeled_data
        assert label in self.labels
        assert all(len(datum) == self.nfeatures for datum in data)
        index = self.labels.index(label)
        relevance = self.samples_seen[index] / len(data)
        self.samples_seen[index] += len(data)
        self.models[index].set_relevances((relevance, relevance))
        self.models[index].adapt(data)
        dc = dcheck('SimpleGaussianTrainer')
        if dc:
            dc('samples_seen', tuple(self.samples_seen))
            for label, model in izip(self.labels, self.models):
                dc(' ', '%-6s:' % label, model)
        return label

class VuMeter(object):
    """
    Calculate decibels (dB) in a form suitable for VU-meter display and for
    energy-based utterance detection.  This class exposes a lot of controls for
    playing with display.
    """
    def __init__(self, lowfreq=300, highfreq=3000,
                 samples_per_sec=44100, fft_size=2048):
        self.config(lowfreq=lowfreq, highfreq=highfreq,
                    samples_per_sec=samples_per_sec, fft_size=fft_size)
    def config(self, lowfreq=300, highfreq=3000,
               samples_per_sec=44100, fft_size=2048):
        self.data_length = fft_size // 2 + 1
        self.bin_factor = fft_size / samples_per_sec
        self.band = lowfreq, highfreq
        self.ref = 1
        self.db = 40, 4
        self.prev = 0

    def get_band(self):
        return self.low_freq, self.high_freq
    def set_band(self, freqs):
        self.low_freq, self.high_freq = freqs
        # figure out range of fft bins to work with
        low_index = int(math.ceil(self.low_freq * self.bin_factor))
        high_index = int(math.floor(self.high_freq * self.bin_factor))
        # XXX need checks on the range
        if high_index <= low_index:
            high_index = low_index + 1
        # the bin selector
        self.select = slice(low_index, high_index)
    band = property(get_band, set_band)

    def get_ref(self):
        return self.ref_energy
    def set_ref(self, ref_energy):
        assert ref_energy > 0, str(ref_energy)
        self.ref_energy = N.float32(ref_energy)
    ref = property(get_ref, set_ref)

    def __call__(self, data):
        # empirically, (and interestingly): this (near 16-bit) shift factor lets
        # the exposed ref property be 1 while giving occasional 0 dB output on a
        # Mac with "Use ambient noise reduction" set to on and "Input volume"
        # slider set to the minimum (settings which give about the lowest level
        # signal data you can get on the mac)
        ref_factor = N.float32(1 / (1 << 18))
        # this should be an adequate range of dBs
        clip_lo, clip_hi = 0, 140

        assert data.shape[-1] == self.data_length
        band = data[self.select]
        assert len(band) >= 1
        ref = N.float32(self.ref * len(band) * ref_factor)
        band *= band
        bandsum = band.sum()
        # XXX not ndarrays....
        sum = float(bandsum)
        if False:
            # slight smoothing; bizarre: causes Python ncurses to bog down!
            sum = (self.prev + sum) / 2
            self.prev = sum
        # space is 10 * log10 (energy)
        dB = (10 * math.log10(sum / ref)) if sum > ref else 0
        dB = max(clip_lo, dB)
        dB = min(clip_hi, dB)
        dcheck(vumeterlog) and dprint(vumeterlog, 'dB', N.float32(dB), 'channels', len(band), 'ref', self.ref, 'scaled_ref', ref, 'energy', bandsum)
        # we return the dB in decibel scaling
        return dB


def make_endpointer(vumeter, utt_tagger):
    """
    Return a function that computes endpointing tags, and which also generates
    tagged VU display.  This runs two things in parallel and aligns their
    output, and keeps stuff happening frame synchronously, albeit with a
    latency.  Knows a lot about what UtteranceTagger does.
    """
    # XXX should get split and use SynchronizingSequenceJoin
    db_queue = deque()
    tag_queue = deque()
    line_length = 255
    blank_line = (' ') * line_length
    pixel = '*'
    bar_length = 6
    bar = ((pixel + ' ') * bar_length)[:bar_length]
##     barx = ' ' * bar_length
    # subtract 3 so that there's a space between clipped vu and utt flag
    vu_top = line_length - bar_length - 3
    last_tag = [None]
    toggle = [0]
    def process(data):
        db = vumeter(data)
        db_queue.append(db)
        tag_queue.extend(utt_tagger(db))
        if tag_queue and db_queue:
            tags = list()
            #while tag_queue and db_queue:
            if True:
                # XXX the tracker_range info is diagnostic
                tag, tracker_range = tag_queue.popleft()
                db = db_queue.popleft()

                tags.append(tag)
                state_change = (tag != last_tag[0])
                if state_change:
                    last_tag[0] = tag

                dc = dcheck(endpointdisplay)
                if dc:
                    toggle[0] ^= 0x1
                    tog = toggle[0]
                    line = list(blank_line)
                    dbindex = min(vu_top, int(db*utt.scale))
                    if utt.fill and not state_change:
                        nfill = (dbindex+1)//2
                        line[tog:dbindex+tog:2] = pixel * nfill
                    line[dbindex] = pixel
                    if utt.trak:
                        lo, high = tuple(min(vu_top, int(x*utt.scale)) for x in tracker_range)
                        line[lo] = pixel if not utt.fill else ' '
                        line[high] = pixel
                    if tag and tog:
                        line[-bar_length:] = bar
                    if state_change:
                        line[:bar_length] = bar
##                         if not utt.fill:
##                             line[2:bar_length+2] = bar
##                         else:
##                             line[2:bar_length+2] = barx
                    dc(DebugPrint.NO_PREFIX, ' ', ''.join(line))

            return tags
        else:
            return ()
    return process

ddt_scale = 2.0
ddt_accum_scale = ddt_scale / 1000
ddt_sums = [0, 0]
label_stats = [0, 0]
accuracy_stats = [0, 0]
last_label = None
def ddt_accum(value):
    event0, event1 = value
    assert event0[0] == event1[0]
    data0 = event0[1]
    data1 = event1[1]
    assert len(data0) == len(data1) == 2
    line_len = 255
    line = [' '] * line_len
    offset = line_len // 4

    label = event0[0]
    label_stats[0] += 1
    if label == True:
        label_stats[1] += 1
    else:
        assert label == False
    if data0[0][1] == label:
        accuracy_stats[0] += 1
    if data1[0][1] == label:
        accuracy_stats[1] += 1

    global last_label
    if label != last_label:
        dprint('ddt_accum', label)
        dprint('ddt_accum', ' ', label_stats, accuracy_stats)
        last_label = label
    
    if data0[0][1] == True:
        c0True = data0[0][0]
        c0False = data0[1][0]
    else:
        assert data0[0][1] == False
        c0True = data0[1][0]
        c0False = data0[0][0]

    if data1[0][1] == True:
        c1True = data1[0][0]
        c1False = data1[1][0]
    else:
        assert data1[0][1] == False
        c1True = data1[1][0]
        c1False = data1[0][0]

    llrTrue = math.log(c0True / c1True)
    llrFalse = math.log(c0False / c1False)

    if llrTrue > 0:
        ddt_sums[0] += 1
    elif llrTrue < 0:
        ddt_sums[0] -= 1
    else:
        pass

    if llrFalse > 0:
        ddt_sums[1] += 1
    elif llrFalse < 0:
        ddt_sums[1] -= 1
    else:
        pass

    def clip(val, scale):
        val = int(val * scale) + offset
        val = max(0, val)
        val = min(line_len-1, val)
        return val
        
    indexTrue = clip(llrTrue, ddt_scale)
    indexFalse = clip(llrFalse, ddt_scale)
    line[indexTrue] = '*'
    line[indexFalse] = '|'
        
    indexTrue = clip(ddt_sums[0], ddt_accum_scale)
    indexFalse = clip(ddt_sums[1], ddt_accum_scale)
    line[indexTrue] = '0'
    line[indexFalse] = 'X'

    line.append('\n')
    displaystream2.write(''.join(line))


vu = vumeter = VuMeter()

# Endpoint Controls -- not always intuitive
speed='fast'
if speed == 'slow':
    min_hunt = 0.0375
    max_hunt = -0.25
    min_range = 17
    low_per_mil = 250
    high_per_mil = 600
    start_window = 25
    start_count = 10
    stop_window = 45
    stop_count = 17
else:
    assert speed == 'fast'
    min_hunt = 0.0375
    max_hunt = -0.5
    min_range = 17
    low_per_mil = 400
    high_per_mil = 700
    start_window = 25
    start_count = 10
    stop_window = 35
    stop_count = 15
from onyx.signalprocessing.endpoint import UtteranceTagger
utt_tagger = UtteranceTagger(min_hunt, max_hunt, min_range, low_per_mil, high_per_mil, start_window, start_count, stop_window, stop_count)
endpointer = SequenceFunctionProcessor(make_endpointer(vumeter, utt_tagger))

# an MFCC processor
samples_per_sec = 44100
fft_length = 2048
low_cutoff = 300
high_cutoff = 3500
numchans = 24
numceps = 6
c0 = False
ceplifter = 22
mfcc0 = make_melcepstral_processor(samples_per_sec=samples_per_sec, fft_length=fft_length, numchans=numchans, low_cutoff=low_cutoff, high_cutoff=high_cutoff, numceps=numceps, c0=c0, ceplifter=ceplifter)
mfcc1 = make_melcepstral_processor(samples_per_sec=samples_per_sec, fft_length=fft_length, numchans=numchans, low_cutoff=low_cutoff, high_cutoff=high_cutoff, numceps=numceps, c0=c0, ceplifter=ceplifter)

def square(data):
    data *= data
    return data
square = FunctionProcessor(square, label='x^2')
# square.graph.dot_display(globals=['rankdir=LR'])


# model/classifier params
labels = True, False
ncomps = 7, 3
nfeatures = numceps

# mfcc, with c0: gathered from about 60,000 frames of 'train' data from Ken and Hugh skyping
true_means_prime = N.array((-5.5522, -1.0517, 1.0240, 3.1609, 0.5628, -122.5789), dtype=N.float32)
true_vars_prime = N.array((26.2238, 26.4064, 59.7242, 35.1180, 47.2471, 3967.2240), dtype=N.float32)
false_means_prime = N.array((-9.4634, -5.3991, -4.2773, 1.7494, -0.0822, -228.6211), dtype=N.float32)
false_vars_prime = N.array((3.0097, 6.0277, 8.3711, 10.7198, 13.4285, 456.7074), dtype=N.float32)

# mfcc, no c0: 20,000 frames of Hugh talking
true_means_prime = N.array((-4.8087, 3.9863, -0.5217, 1.3076, 0.7514, -4.6497), dtype=N.float32)
true_vars_prime = N.array((26.8496, 32.6631, 32.3662, 24.2963, 36.2244, 34.1555), dtype=N.float32)
false_means_prime = N.array((-6.8806, -1.3424, -3.8147, 0.4520, 0.7129, -3.1560), dtype=N.float32)
false_vars_prime = N.array((2.7468, 6.2286, 7.4355, 10.1530, 13.3865, 15.9309), dtype=N.float32)
true_prime = SimpleGaussianModel(nfeatures, GaussianModelBase.DIAGONAL_COVARIANCE)
true_prime.set_model(true_means_prime, true_vars_prime)
false_prime = SimpleGaussianModel(nfeatures, GaussianModelBase.DIAGONAL_COVARIANCE)
false_prime.set_model(false_means_prime, false_vars_prime)

primer = (true_prime, false_prime)

GaussianMixtureModel.seed(0)
gmm_mgr0 = GmmMgr(ncomps, nfeatures, GaussianModelBase.DIAGONAL_COVARIANCE, primer)
gmm_mgr1 = GmmMgr(ncomps, nfeatures, GaussianModelBase.DIAGONAL_COVARIANCE, primer)
classify0 = AdaptingGmmClassifier(gmm_mgr0, izip(labels, count()))
classify1 = AdaptingGmmClassifier(gmm_mgr1, izip(labels, count()))
classify0.set_relevance(333)
classify1.set_relevance(333)
classify0.set_num_em_iterations(2)
classify1.set_num_em_iterations(2)
classifier0 = AdaptingGmmClassProcessor(classify0)
classifier1 = AdaptingGmmClassProcessor(classify1)

gaussian_trainer = SimpleGaussianTrainer(labels, nfeatures)
trainer = FunctionProcessor(gaussian_trainer)

# audio.mic, fftmag, endpointer, mfcc0, square, mfcc1, classifier0, classifier1, trainer

spectral_split = SplitProcessor()
utt_tag_split = SplitProcessor()

gmm0_join = SynchronizingSequenceJoin()
gmm1_join = SynchronizingSequenceJoin()

coalesce0 = SequenceFunctionProcessor(Coalescer())
coalesce1 = SequenceFunctionProcessor(Coalescer())

ddt_join = SynchronizingSequenceJoin()

# build the network (uggg, by hand)

audio.mic.set_sendee(fftmag.process)

fftmag.set_sendee(spectral_split.process)
spectral_split.add_sendee(endpointer.process)
endpointer.set_sendee(utt_tag_split.process)
spectral_split.add_sendee(mfcc0.process)
spectral_split.add_sendee(square.process)
square.set_sendee(mfcc1.process)

utt_tag_split.add_sendee(gmm0_join.get_process_function())
utt_tag_split.add_sendee(gmm1_join.get_process_function())
mfcc0.set_sendee(gmm0_join.get_process_function())
mfcc1.set_sendee(gmm1_join.get_process_function())

gmm0_join.set_sendee(coalesce0.process)
gmm1_join.set_sendee(coalesce1.process)

coalesce0.set_sendee(classifier0.process)
coalesce1.set_sendee(classifier1.process)
    
classifier0.set_sendee(ddt_join.get_process_function())
classifier1.set_sendee(ddt_join.get_process_function())
ddt_join.set_sendee(ddt_accum)

class Mode(object):
    """
    Give a 'mode' that when set does replumbing on the fly.
    """
    def __init__(self):
        self.vu

    @property
    def vu(self):
        self.mode = 'vu'
        fftmag.set_sendee(endpointer.process)

    @property
    def train(self):
        self.mode = 'train'
        trainer.set_sending(False)

        coalesce0.set_sendee(trainer.process)
        coalesce1.set_sending(False)

        fftmag.set_sendee(spectral_split.process)

    @property
    def classify(self):
        self.mode = 'classify'
        coalesce0.set_sendee(classifier0.process)
        coalesce1.set_sendee(classifier1.process)

        fftmag.set_sendee(spectral_split.process)

    def __str__(self):
        return str(self.mode)
    def __repr__(self):
        return self.mode
mode = Mode()


def on():
    utt.on
    utt.scale = 5
    prints = (
        #'gaussian',
        #'gaussian_pt',
        'SimpleGaussianTrainer',
        'ddt_accum',
        )
    db=DebugPrint(logstream, *prints)
    db.on()
    audio.on
on = doer(on)
def off():
    audio.off
off = doer(off)
    
def snap():
    import time
    audio.on
    time.sleep(1.5)
    audio.off


def startup(args):
    global outfile
    global on, off, dp
    global start, stop
    global processor
    
    from functools import partial
    from onyx.audio.liveaudio import inputs, default_input, LiveAudioSource
    from onyx.signalprocessing.htkmfcc import make_fft_abs_processor
    from onyx.util.debugprint import DebugPrint, dcheck, dprint
    #from onyx.util.streamprocess import ChainProcessor

    if '-o' in args:
        outfile = open(args[args.index('-o')+1], 'wb', 0)
    else:
        outfile = sys.stdout

    if '-l' in args:
        logfile = open(args[args.index('-l')+1], 'wb', 0)
    else:
        logfile = sys.stdout

##     if '-m' in args:
##         mic_name = args[args.index('-m')+1]
##     else:
##         mic_name = 'blue'

    debug = DebugPrint(logfile, 'saddemo')

##     on = doer(debug.on)
##     off = doer(debug.off)
    dp = partial(dprint, 'saddemo', DebugPrint.NO_PREFIX, ' ')

    plot_points = 255
    line = [' '] * plot_points

##     # select a live input
##     mic_id = default_input()
##     for id, chan, name, manu in inputs():
##         if name.lower().find(mic_name.lower()) != -1:
##             mic_id = id
##             break
##     mic = LiveAudioSource(mic_id, verbose=False)    
##     start = doer(mic.start)
##     stop = doer(mic.stop)

    return

    # use the module's mic
    mic = audio.mic
    
    # XXX LiveAudioSource (and thus audiomodule.cpp) need to be able to provide
    # the sample frequency....
    sample_nsec =    22676  # 44100 Hz
    #frame_nsec =  10000000  # 10 msec
    #window_nsec = 25600000  # 25.6 msec
    frame_nsec =   8000000  #  8 msec
    window_nsec = 24000000  # 24 msec
    dither = 1 / (1 << 10)
    preemcoef = 0.96875
    samples_per_sec, fft_size, fftmag = make_fft_abs_processor(sample_nsec, frame_nsec, window_nsec,
                                                                dither=dither, preemcoef=preemcoef,
                                                                zmeansource=True, usehamming=True,
                                                                max_channels=2)

    print 'samples_per_sec', samples_per_sec, ' fft_size', fft_size

    # figure out range of fft bins to work with
    lowfreq = 300
    highfreq = 4000
    import math
    low_index = int(math.ceil(lowfreq * fft_size / samples_per_sec))
    high_index = int(math.floor(highfreq * fft_size / samples_per_sec))
    # the bin selector
    select = slice(low_index, high_index)
    def display(data):
        band = data[select]
        band *= band
        sum = float(band.sum())
        dB = int(dB_scale * (10 * math.log10(sum) + dB_offset))
        dp('%11.6f ' % sum, '%3d' % dB)
        dB = max(1, dB)
        #outfile.write('\n' + ' ' * dB + '|')
        outfile.write( pen * dB + 'O' + '\n')

    mic.set_sendee(fftmag.process)
    fftmag.set_sendee(display)

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
    del onyx_mainstartup

del division


##     startup(args)
##     del args, startup
