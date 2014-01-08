###########################################################################
#
# File:         htkmfcc.py (directory: ./py/onyx/signalprocessing)
# Date:         8-Dec-2008
# Author:       Hugh Secker-Walker
# Description:  Simple processor doing a version of HTK's MFCC signal processing
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
   Implement a simple version of HTK's Mel Cepstral front end.
"""

# XXX grrr, the use of allclose() and 'doctest: +ELLIPSIS' are mostly
# to support Windows..., and also the HPCC linux

# XXX there are a couple of general-purpose utility classes here that should be
# moved to a util directory module

# XXX should we either require that all arguments be of type numpy.float32, or
# automatically convert to this type?

# XXX the Mel filterbanking stuff could be abstracted usefully in several
# different ways: 
#
# First: Any monotonic function and its inverse could be used for the nonlinear
# mapping, and these should be provided in an object, e.g. for (HzToMel,
# MelToHz), so Bark, Uniform, etc would be simple to add.  Doing this would
# require a lot of renaming from things with '*mel*' in their names to something
# more generic concerning the typical ASR filterbanking operations that are
# going on.
#
# Second: The hardwired 50% overlapping should be generalized to an integral
# number of filters in which each sample gets to play: so, 1 for no overlap, 2
# for the current 50%, 3 for 66% overlap, etc.
#
# Third: A pet peeve of mine [HSW], Mel filters are almost always talked and
# written about in terms of psychoacoustic "critical bands".  Yet in ASR
# implementations with which I'm familiar, no provision is made to use such
# bandwidths in the filter design.  Providing for use of filters with
# psychoacoustic bandwidths (or third-octave, or ERBs, equivalent rectangular
# bandwidths, etc.) would be a nice thing.

from __future__ import division
from __future__ import with_statement
import functools
import math
import time
from collections import deque
from itertools import izip, tee
import numpy as N
from cStringIO import StringIO
from onyx.util import checkutils, floatutils
from onyx.util.streamprocess import FunctionProcessor, SequenceFunctionProcessor, ChainProcessor
from onyx.util.duration import time_usec_sec_str
from onyx.builtin import frozendict
from onyx.util.debugprint import DebugPrint, dcheck, dprint


# XXX needs more work on gap semantics, e.g. when argument is None, as
# implemented in MelFilterBank
def htkmfcc():
    # From the HTK Book (for HTK Version 3.4) (c) 2006
    #
    # stuff in parens isn't started yet
    # (dither) (scaling issues int16 vs float32)
    # Preemphasis
    # SlidingWindow
    # remove_dc
    # hamming, kaiser
    # (LPC, REF)
    # (fft)  numpy.fft.rfft(data, power-of-two, axis=-1)
    # (abs) or (abssqr)
    # (warp)
    # MelFilterBank
    # (log10)
    # dct
    # (PLP)
    # (delta, accel, etc)
    pass


# From: "The HTK Book (for HTK Version 3.4)", December 2006, pp. 320 - 322
# Section: 18.1 Configuration Variables used in Library Modules 
# Table 18.1: Library Module Configuration Variables 
# (just the parts that apply to signal processing)

#Name Default Description 
HTK_PARAMS = (
    ('ACCWINDOW', 2, 'Acceleration window size'),
    ('ADDDITHER', 0.0, 'Level of noise added to input signal'),
    ('AUDIOSIG', 0, 'Audio signal number for remote control'),
    ('BYTEORDER', None, 'Define byte order VAX or other'),
    ('CEPLIFTER', 22, 'Cepstral liftering coefficient'),
    ('CMEANDIR', None, 'Directory to find cepstral mean vecotrs'),
    ('CMEANMASK', None, 'Filename mask for cepstral mean vectors'),
    ('CMEANPATHMASK', None, 'Path name mask for cepstral mean vectors, the matched string is used to extend CMEANDIR string'),
    ('COMPRESSFACT', 0.33, 'Amplitude compression factor for PLP'),
    ('DELTAWINDOW', 2, 'Delta window size'),
    ('DOUBLEFFT', False, 'Use twice the required size for FFT'),
    ('ENORMALISE', True, 'Normalise log energy'),
    ('ESCALE', 0.1, 'Scale log energy'),
    ('HEADERSIZE', None, 'Size of header in an alien file'),
    ('HIFREQ', -1.0, 'High frequency cut-off in fbank analysis'),
    ('LINEIN', True, 'Enable audio input from machine line input'),
    ('LINEOUT', True, 'Enable audio output to machine line output'),
    ('LOFREQ', -1.0, 'Low frequency cut-off in fbank analysis'),
    ('LPCORDER', 12, 'Order of lpc analysis'),
    ('MATTRANFN', None, 'Input transformation file'),
    ('MEASURESIL', True, 'Measure background silence level'),
    ('MICIN', False, 'Enable audio input from machine mic input'),
    ('NATURALREADORDER', False, 'Enable natural read order for binary files'),
    ('NATURALWRITEORDER', False, 'Enable natural write order for binary files'),
    ('NSAMPLES', None, 'Num samples in alien file input via a pipe'),
    ('NUMCEPS', 12, 'Number of cepstral parameters'),
    ('NUMCHANS', 20, 'Number of filterbank channels'),
    ('OUTSILWARN', True, 'Print a warning message to stdout before measuring audio levels'),
    ('PHONESOUT', True, 'Enable audio output to machine phones output'),
    ('PREEMCOEF', 0.97, 'Set pre-emphasis coefficient'),
    ('RAWENERGY', True, 'Use raw energy'),
    ('SAVECOMPRESSED', False, 'Save the output file in compressed form'),
    ('SAVEWITHCRC', True, 'Attach a checksum to output parameter file'),
    ('SILENERGY', 0.0, 'Average background noise level (in dB) will normally be measured rather than supplied in configuration'),
    ('SILFLOOR', 50.0, 'Energy silence floor in dBs'),
    ('SILGLCHCOUNT', 2, 'Maximum number of frames marked as silence in window which is classified as speech whilst expecting silence'),
    ('SILMARGIN', 40, 'Number of extra frames included before and after start and end of speech marks from the speech/silence detector'),
    ('SILSEQCOUNT', 100, 'Number of frames classified as silence needed to mark end of utterance'),
    ('SIMPLEDIFFS', False, 'Use simple differences for delta calculations'),
    ('SOURCEFORMAT', None, 'HTK File format of source'),
    ('SOURCEKIND', None, 'Parameter kind of source'),
    ('SOURCERATE', 0.0, 'Sample rate of source in 100ns units'),
    ('SPCGLCHCOUNT', 0, 'Maximum number of frames marked as silence in window which is classified as speech whilst expecting start of speech'),
    ('SPCSEQCOUNT', 10, 'Window over which speech/silence decision reached'),
    ('SPEAKEROUT', False, 'Enable audio output to machine internal speaker'),
    ('SPEECHTHRESH', 9.0, 'Threshold for speech above silence level (in dB)'),
    ('STEREOMODE', None, 'Select channel: RIGHT or LEFT'),
    ('TARGETFORMAT', None, 'HTK File format of target'),
    ('TARGETKIND', None, 'Parameter kind of target'),
    ('TARGETRATE', 0.0, 'Sample rate of target in 100ns units'),
    ('TRACE', 0, 'Trace setting'),
    ('USEHAMMING', True, 'Use a Hamming window'),
    ('USEPOWER', False, 'Use power not magnitude in fbank analysis'),
    ('USESILDET', False, 'Enable speech/silence detector'),
    ('V1COMPAT', False, 'HTK V1 compatibility setting'),
    ('VARSCALEDIR', None, 'Directory to find cepstral variance vectors'),
    ('VARSCALEDIR', None, 'string'),
    ('VARSCALEFN', None, 'Filename of global variance scaling vector'),
    ('VARSCALEMASK', None, 'Filename mask for cepstral variance vectors'),
    ('VARSCALEPATHMASK', None, 'Path name mask for cepstral variance vectors, the matched string is used to extend'),
    ('VQTABLE', None, 'Name of VQ table'),
    ('WARPFREQ', 1.0, 'Frequency warping factor'),
    ('WARPLCUTOFF', None, 'Lower frequency threshold for non-linear warping'),
    ('WINDOWSIZE', 256000.0, 'Analysis window size in 100ns units'),
    ('ZMEANSOURCE', False, 'Zero mean source waveform before analysis'),
    )

# set of known legal basic TARGETKINDs
HTK_BASIC_TARGET_SET = frozenset((
    'ANON',       #  -  use target format found elsewhere
    'DISCRETE',   # 10  vector quantised data
    'FBANK',      #  7  log mel-filter bank channel outputs
    'IREFC',      #  5  LPC reflection coef in 16 bit integer format
    'LPC',        #  1  linear prediction filter coefficients
    'LPCEPSTRA',  #  3  LPC cepstral coefficients
    'LPDELCEP',   #  4  LPC cepstra plus delta coefficients
    'LPREFC',     #  2  linear prediction reflection coefficients
    'MELSPEC',    #  8  linear mel-filter bank channel outputs
    'MFCC',       #  6  mel-frequency cepstral coefficients
    'PLP',        # 11  PLP cepstral coefficients 
    'USER',       #  9  user defined sample kind
    'WAVEFORM',   #  0  sampled waveform
))

# set of known legal TARGETKIND modifiers
HTK_TARGET_MODIFIER_SET = frozenset((
    '0',  #  020000  has 0th cepstral coef.
    'A',  #  001000  has acceleration coefficients
    'C',  #  002000  is compressed
    'D',  #  000400  has delta coefficients
    'E',  #  000100  has energy
    'K',  #  010000  has CRC checksum
    'N',  #  000200  absolute energy suppressed
    'T',  #  100000  has third differential coef. 
    'V',  #  040000  has VQ data
    'Z',  #  004000  has zero mean static coef.
))

assert not (HTK_BASIC_TARGET_SET & HTK_TARGET_MODIFIER_SET)
HTK_TARGET_SET = HTK_BASIC_TARGET_SET | HTK_TARGET_MODIFIER_SET

HTK_TARGET_SET_USED = frozenset((
    '0',
    'MFCC',
    ))

# add parameter names as attributs in this module
# XXX they should really be factories...
this_module = globals()
for name, default, doc in HTK_PARAMS:
    this_module[name] = name
# just checking
ZMEANSOURCE

HTK_PARAM_MAP = frozendict((name, (name, default, doc)) for name, default, doc in HTK_PARAMS)
HTK_PARAM_SET = frozenset(HTK_PARAM_MAP)
# parameters that client can set (and may be required to set to a particular value)
HTK_PARAM_SET_USED = frozenset((
    'ADDDITHER',
    'CEPLIFTER',
    'DOUBLEFFT',
    'ENORMALISE',
    'ESCALE', # ignored because ENORMALISE must be False
    'HIFREQ',
    'LOFREQ',
    'NUMCEPS',
    'NUMCHANS',
    'PREEMCOEF',
    'SAVEWITHCRC',
    'SOURCEKIND',
    'SOURCERATE',
    'STEREOMODE',
    'TARGETKIND',
    'TARGETRATE',
    'USEHAMMING',
    'USEPOWER',
    'WINDOWSIZE',
    'ZMEANSOURCE',
    ))

# required values; subset of HTK_PARAM_SET_USED
HTK_MANDATORY = dict((
    ('ENORMALISE', False),
    ('SOURCEKIND', 'WAVEFORM'),
    ('SAVEWITHCRC', False),
    ))

_specs0_str = '''
CEPLIFTER      = 22
ENORMALISE     = FALSE
#ESCALE         = 1.0
HIFREQ         = 3800
LOFREQ         = 125
NUMCEPS        = 12
NUMCHANS       = 24
PREEMCOEF      = 0.97
SOURCEKIND     = WAVEFORM
# TARGETKIND     = MFCC_E_D_A
TARGETKIND     = MFCC_0
#TARGETKIND     = MFCC
TARGETRATE     = 100000
USEHAMMING     = TRUE
USEPOWER       = TRUE
WINDOWSIZE     = 250000.0
ZMEANSOURCE    = TRUE
SAVECOMPRESSED = FALSE
SAVEWITHCRC    = FALSE
'''

def parse_htk_specs(iterable, comment_prefix='#'):
    """
    Parse simple text specs.  This does not attempt to follow HTK's syntax
    conventions very far.

    >>> specs = parse_htk_specs(StringIO(_specs0_str))
    >>> for key in sorted(specs.keys()): print key, '=', specs[key]
    CEPLIFTER = 22
    ENORMALISE = False
    HIFREQ = 3800
    LOFREQ = 125
    NUMCEPS = 12
    NUMCHANS = 24
    PREEMCOEF = 0.97
    SAVECOMPRESSED = False
    SAVEWITHCRC = False
    SOURCEKIND = WAVEFORM
    TARGETKIND = MFCC_0
    TARGETRATE = 100000
    USEHAMMING = True
    USEPOWER = True
    WINDOWSIZE = 250000.0
    ZMEANSOURCE = True
    """
    specs = dict()
    for line in iterable:
        line = line.strip()
        parts = line.split()
        if not parts or parts[0].startswith(comment_prefix):
            continue
        if len(parts) >= 2 and parts[1] == '=':
            del parts[1]
        if len(parts) < 2:
            raise ValueError("expected a name and a value field, got %r" % (line,))
        if len(parts) > 2:
            raise ValueError("expected only a name and a value field, got %r" % (line,))

        # simple constants, or numbers, else string
        name, value = parts
        valuel = value.lower()
        if valuel == 'none' or valuel == 'null':
            specs[name] = None
            continue
        if valuel == 'true' or valuel == 't':
            specs[name] = True
            continue
        if valuel == 'false' or valuel == 'f':
            specs[name] = False
            continue
        if valuel[0] in '0123456789':
            # XXX this is way too powerful
            # note: using value, not valuel
            specs[name] = eval(value)
            continue
        specs[name] = value

    return specs

if __name__ == '__main__':
    _specs0 = parse_htk_specs(StringIO(_specs0_str))

_specs1 = frozendict((    
    ('CEPLIFTER', 22),
    ('ENORMALISE', False),
    # ('ESCALE', 1.0),
    ('HIFREQ', 3800),
    ('LOFREQ', 125),
    ('NUMCEPS', 12),
    ('NUMCHANS', 24),
    ('PREEMCOEF', 0.97),
    ('SAVECOMPRESSED', False),
    ('SAVEWITHCRC', False),
    ('SOURCEKIND', 'WAVEFORM'),
    ('SOURCERATE', 625.0),
    # ('TARGETKIND', 'PLP_E_D_A_Z'),
    ('TARGETKIND', 'PLP'),
    ('TARGETRATE', 100000),
    ('USEHAMMING', True),
    ('USEPOWER', True),
    ('WINDOWSIZE', 250000.0),
    ('ZMEANSOURCE', True),
    ))

class HtkChain(object):
    """
    An HTK signal processor.

    >>> specs = dict((('ENORMALISE', False), ('SOURCEKIND', 'WAVEFORM'), ('STEREOMODE', 'LEFT'), ('SOURCERATE', 226.757), ('SAVEWITHCRC', False), ('TARGETRATE', 100000.0), ('TARGETKIND', 'MFCC'), ('LOFREQ', 250), ('HIFREQ', 3500), ('ADDDITHER', 0.0001)))
    >>> chain = HtkChain(specs)  #doctest: +ELLIPSIS
    nsec_per_source_sample 22676  nsec_per_target_sample 10000000  nsec_per_window 25600000
    samples_per_sec 44100  slide_samples 441  window_samples 1129
    fft_length 2048
    numchans 20  lofreq 250  hifreq 3500
    len(chain) 10
    sample_count 100000  sample_count / slide_samples 226
    len(res) 225
    len(res[0]) 12  len(res[-1]) 12
    res[0].min() -11.399...  res[-1].min() -12.468...
    res[0].max() 3.5308...  res[-1].max() 5.30439...
    elapsed time 0... seconds

    >>> specs0 = parse_htk_specs(StringIO(_specs0_str))
    >>> specs0['SOURCERATE'] = 226.757  # 44100 Hz
    >>> chain0 = HtkChain(specs0)  #doctest: +ELLIPSIS
    nsec_per_source_sample 22676  nsec_per_target_sample 10000000  nsec_per_window 25000000
    samples_per_sec 44100  slide_samples 441  window_samples 1102
    fft_length 2048
    numchans 24  lofreq 125  hifreq 3800
    len(chain) 11
    sample_count 100000  sample_count / slide_samples 226
    len(res) 225
    len(res[0]) 12  len(res[-1]) 12
    res[0].min() -23.430...  res[-1].min() -24.065...
    res[0].max() 345.627...  res[-1].max() 328.4...
    elapsed time 0... seconds

    >>> # chain0._mfcc_processor.graph.dot_display(globals=('rankdir=LR',))
    
    >>> sp1 = HtkChain(_specs1)
    Traceback (most recent call last):
       ...
    ValueError: unimplemented TARGETKIND value: PLP

    >>> HtkChain(dict((('ZMEANSOURCE', True), ('Foobar', None))))
    Traceback (most recent call last):
       ...
    ValueError: unexpected configuration parameter: Foobar

    >>> HtkChain(dict((('ZMEANSOURCE', True), ('Foobar', None), ('Baz', True))))
    Traceback (most recent call last):
       ...
    ValueError: unexpected configuration parameters: Baz Foobar

    >>> HtkChain(dict((('ENORMALISE', False), ('SOURCEKIND', 'WAVEFORM'), ('SAVEWITHCRC', False),)))
    Traceback (most recent call last):
       ...
    ValueError: unexpected empty value for TARGETKIND parameter: None

    >>> HtkChain(dict((('ENORMALISE', False), ('SOURCEKIND', 'WAVEFORM'), ('SAVEWITHCRC', False), ('TARGETKIND', 'MFCC_D_S'))))
    Traceback (most recent call last):
       ...
    ValueError: unexpected TARGETKIND value: S

    >>> HtkChain(dict((('ENORMALISE', False), ('SOURCEKIND', 'WAVEFORM'), ('SAVEWITHCRC', False), ('TARGETKIND', 'MFCC_D_S_YY'))))
    Traceback (most recent call last):
       ...
    ValueError: unexpected TARGETKIND values: S YY
    """
    def __init__(self, user_specs):
        unknown_parameters = frozenset(user_specs) - HTK_PARAM_SET
        if unknown_parameters:
            raise ValueError("unexpected configuration parameter%s: %s" % ('s' if len(unknown_parameters) >= 2 else '', ' '.join(sorted(unknown_parameters))))

        unimplemented_parameters = frozenset(user_specs) - HTK_PARAM_SET_USED
        non_default = set()
        for parameter in unimplemented_parameters:
            if user_specs[parameter] != HTK_PARAM_MAP[parameter][1]:
                non_default.add(parameter)
        if non_default:
            raise ValueError("unimplemented configuration parameter%s assigned away from default (don't assign): %s" % ('s' if len(non_default) >= 2 else '', ' '.join(sorted(non_default))))
        
        for key in HTK_MANDATORY:
            if key not in user_specs:
                raise ValueError("parameter %s must be provided, and it must be set as follows: %s = %s" % (key, key, HTK_MANDATORY[key]))
            if user_specs[key] != HTK_MANDATORY[key]:
                raise ValueError("parameter %s must be set as follows: %s = %s" % (key, key, HTK_MANDATORY[key]))

        specs = dict((name, default) for name, (name, default, doc) in HTK_PARAM_MAP.iteritems())
        specs.update(user_specs)

        targetkind = specs['TARGETKIND']
        if not targetkind:
            raise ValueError("unexpected empty value for TARGETKIND parameter: %r" % (targetkind,))
        target = frozenset(targetkind.split('_'))

        unknown_targets = target - HTK_TARGET_SET
        if unknown_targets:
            raise ValueError("unexpected TARGETKIND value%s: %s" % ('s' if len(unknown_targets) >= 2 else '', ' '.join(sorted(unknown_targets))))

        unimplemented_targets = target - HTK_TARGET_SET_USED
        if unimplemented_targets:
            raise ValueError("unimplemented TARGETKIND value%s: %s" % ('s' if len(unimplemented_targets) >= 2 else '', ' '.join(sorted(unimplemented_targets))))

        # straight chain for calculating MFCC coeffs, but not deltas
        chain = list()

        # STEREOMODE
        # data comes in as a sequence of buffers
        stereomode = specs['STEREOMODE']
        if stereomode is None:
            steroindex = 0
            max_chans = 1
        else:
            stereomodel = stereomode.lower()
            assert stereomodel in ('left', 'right'), str(stereomode)
            steroindex = 0 if stereomodel == 'left' else 1
            max_chans = 2
        def stereoselect(data):
            assert 1 <= len(data) <= max_chans, str(len(data))
            return data[steroindex]
        chain.append(FunctionProcessor(stereoselect, label='Chan Select'))
            
        # ADDDITHER precedes preemphasis
        dither = specs['ADDDITHER']
        if dither != 0:
            chain.append(FunctionProcessor(Dither(dither), label='Dither'))

        # PREEMCOEF
        preemcoef = specs['PREEMCOEF']
        if preemcoef > 0:
            chain.append(FunctionProcessor(Preemphasis(preemcoef), label='PreEmphasis'))

        # we work in nano seconds (nsec)

        # conversion from HTK's 100_nsec values
        NSEC_PER_100_NANOS = 100

        # SOURCERATE
        # TARGETRATE
        # WINDOWSIZE
        for param in 'SOURCERATE', 'TARGETRATE', 'WINDOWSIZE':
            if specs[param] < 1:
                raise ValueError("expected positive %s, got %r" % (param, specs[param]))
        nsec_per_source_sample = int(specs['SOURCERATE'] * NSEC_PER_100_NANOS + 0.5)
        nsec_per_target_sample = int(specs['TARGETRATE'] * NSEC_PER_100_NANOS + 0.5)
        nsec_per_window = int(specs['WINDOWSIZE'] * NSEC_PER_100_NANOS + 0.5)

        slide_samples = int(nsec_per_target_sample / nsec_per_source_sample + 0.5)
        window_samples = int(nsec_per_window / nsec_per_source_sample + 0.5)

        fft_length = get_power_of_two_roundup(window_samples)
        if specs['DOUBLEFFT']:
            fft_length *= 2

        # grrrr, try to come up with what the user will be expecting....
        samples_per_sec = int(100000000 / nsec_per_source_sample + 0.5) * 10
        
        print 'nsec_per_source_sample', nsec_per_source_sample, ' nsec_per_target_sample', nsec_per_target_sample, ' nsec_per_window', nsec_per_window
        print 'samples_per_sec', samples_per_sec, ' slide_samples', slide_samples, ' window_samples', window_samples
        assert int(slide_samples) == slide_samples
        assert int(window_samples) == window_samples
        chain.append(SequenceFunctionProcessor(SlidingWindow(window_samples, slide_samples), label='Window'))

        # ZMEANSOURCE
        if specs['ZMEANSOURCE']:
            chain.append(FunctionProcessor(remove_dc, label='Remove DC'))

        # USEHAMMING
        if specs['USEHAMMING']:
            chain.append(FunctionProcessor(hamming, label='Hamming'))

        print 'fft_length', fft_length

        # fft and abs work
        @nd_io
        def fft_abs(data):
            # reuse the data object for the absolute value result
            fft_z = N.fft.rfft(data, fft_length, axis=-1)
            # note: refcheck=False, but not sure where the reference is
            # held, likely in the interpreter...
            data.resize(fft_z.shape, refcheck=False)
            return N.absolute(fft_z, data)
        chain.append(FunctionProcessor(fft_abs, label='Mag'))

        # USEPOWER
        if specs['USEPOWER']:
            def power(data):
                data *= data
                return data
            chain.append(FunctionProcessor(power, label='Power'))

        # MEL
        numchans = specs['NUMCHANS']
        lofreq = specs['LOFREQ']
        hifreq = specs['HIFREQ']
        numceps = specs['NUMCEPS']
        if 'MFCC' in target:
            print 'numchans', numchans, ' lofreq', lofreq, ' hifreq', hifreq
            mel_filter_bank = MelFilterBank(numchans, lofreq, hifreq, samples_per_sec, fft_length)
            chain.append(FunctionProcessor(mel_filter_bank, label='Mel FB'))

            # log
            chain.append(FunctionProcessor(lambda data: N.log(data, data), label='Log'))

            # DCT
            full_dct = dct[numchans]
            if '0' in target:
                # move C0 to the end
                ceps0 = N.roll(full_dct[:numceps, :], -1, axis=0)
            else:
                # skip C0
                ceps0 = full_dct[1:numceps+1, :]
            ceps = N.transpose(ceps0)
            assert ceps.shape[1] == numceps
            chain.append(FunctionProcessor(lambda data: N.dot(data, ceps), label='DCT'))
        else:
            raise ValueError("expected MFCC in TARGETKIND, got %s" % (' '.join(sorted(target))))

        # liftering
        ceplifter = specs['CEPLIFTER']
        if ceplifter > 0:
            lifts = lifter[numceps, ceplifter]
            def lift(data):
                data *= lifts
                return data
            chain.append(FunctionProcessor(lift, label='Lifter'))
        


        print 'len(chain)', len(chain)
        mfcc_processor = self._mfcc_processor = ChainProcessor(chain)
        
        res = self.res = list()
        mfcc_processor.set_sendee(res.append)

        bufsize = 1000
        count = 100

        N.random.set_state(_random_state)
        normal = functools.partial(N.random.normal, size=bufsize)
        process = mfcc_processor.process

        start_time = time.time()
        for i in xrange(count):
            process((normal(),))
        end_time = time.time()

        sample_count = bufsize * count

        print 'sample_count', sample_count, ' sample_count / slide_samples', sample_count // slide_samples
        print 'len(res)', len(res)
        if len(res) > 0:
            print 'len(res[0])', len(res[0]), ' len(res[-1])', len(res[-1])
            print 'res[0].min()', res[0].min(), ' res[-1].min()', res[-1].min()
            print 'res[0].max()', res[0].max(), ' res[-1].max()', res[-1].max()
        print 'elapsed time', time_usec_sec_str(end_time - start_time)[-1], 'seconds'
        #print tuple(len(x) for x in res)
        #print tuple(type(x).__name__ for x in res)
        #print 'len(watcher0)', len(watcher0),'len(watcher1)', len(watcher1)
        #print tuple(type(x).__name__ for x in watcher)

def make_fft_abs_processor(sample_nsec, frame_nsec, window_nsec,
                           max_channels=1, channel_index=0,
                           dither=0, preemcoef=0,
                           zmeansource=False,
                           usehamming=False,
                           doublefft=False):
    """
    Builds a processor for the windowing and FFT part of a typical speech front
    end.  Returns a tuple of the sample_per_sec, fft_size, and the processor.

    Numpy ndarrays of real wave data should be fed in: shape = (numchannels,
    num_samples), where num_samples >= 2, but is otherwise unconstrained.  The
    channel_index arguement selects which channel is used.  It will push
    ndarrays (of length fft_size//2+1) of the magnitudes of the spectrum (from 0
    Hz to sample_frequency/2 inclusive).  The bin_to_hertz factor is simply
    sample_per_sec / fft_size.

    >>> sample_nsec =    22676  # 44100 Hz
    >>> frame_nsec =  10000000  # 10 msec
    >>> window_nsec = 25600000  # 25.6 msec
    >>> with DebugPrint('frontend'):
    ...   samples_per_sec, fft_size, processor = make_fft_abs_processor(sample_nsec, frame_nsec, window_nsec,
    ...                                                                 zmeansource=True, usehamming=True)
    frontend: sample_nsec 22676  frame_nsec 10000000  window_nsec 25600000
    frontend: samples_per_sec 44100  slide_samples 441  window_samples 1129
    frontend: fft_length 2048

    >>> #processor.graph.dot_display(globals=('rankdir=LR',))

    >>> samples_per_sec, fft_size
    (44100, 2048)
    """

    chain = list()

    def stereoselect(data):
        assert 1 <= len(data) <= max_channels, str(len(data)) + ' ' + str(max_channels)
        return data[channel_index]
    chain.append(FunctionProcessor(stereoselect, label='Chan Select'))

    if dither != 0:
        chain.append(FunctionProcessor(Dither(dither), label='Dither'))

    if preemcoef > 0:
        chain.append(FunctionProcessor(Preemphasis(preemcoef), label='PreEmphasis'))

    slide_samples = int(frame_nsec / sample_nsec + 0.5)
    window_samples = int(window_nsec / sample_nsec + 0.5)
    chain.append(SequenceFunctionProcessor(SlidingWindow(window_samples, slide_samples), label='Window'))

    if zmeansource:
        chain.append(FunctionProcessor(remove_dc, label='Remove DC'))

    if usehamming:
        chain.append(FunctionProcessor(hamming, label='Hamming'))


    fft_length = get_power_of_two_roundup(window_samples)
    if doublefft:
        fft_length *= 2
    @nd_io
    def fft_abs(data):
        # reuse the data object for the absolute value result
        fft_z = N.fft.rfft(data, fft_length, axis=-1)
        # note: refcheck=False, but not sure where the reference is
        # held, likely in the interpreter...
        data.resize(fft_z.shape, refcheck=False)
        return N.absolute(fft_z, data)
    chain.append(FunctionProcessor(fft_abs, label='FFT Mag'))

    processor = ChainProcessor(chain)

    # grrrr, try to come up with what the user will be expecting....
    samples_per_sec = int(100000000 / sample_nsec + 0.5) * 10

    dc = dcheck('frontend')
    if dc:
        dc('sample_nsec', sample_nsec, ' frame_nsec', frame_nsec, ' window_nsec', window_nsec)
        dc('samples_per_sec', samples_per_sec, ' slide_samples', slide_samples, ' window_samples', window_samples)
        dc('fft_length', fft_length)

    return samples_per_sec, fft_length, processor

def make_melcepstral_processor(samples_per_sec, fft_length, numchans, low_cutoff, high_cutoff, numceps, c0=False, ceplifter=0):
    """
    Make and return a Mel-Cepstral processor.

    >>> melcep = make_melcepstral_processor(samples_per_sec=44100, fft_length=2048, numchans=24, low_cutoff=300, high_cutoff=3500, numceps=12, c0=True, ceplifter=22)
    >>> #melcep.graph.dot_display(globals=('rankdir=LR',))
    """
    chain = list()

    # MEL
    mel_filter_bank = MelFilterBank(numchans, low_cutoff, high_cutoff, samples_per_sec, fft_length)
    chain.append(FunctionProcessor(mel_filter_bank, label='Mel FB'))

    # Log
    chain.append(FunctionProcessor(lambda data: N.log(data, data), label='Log'))

    # DCT
    full_dct = dct[numchans]
    if c0:
        # move C0 to the end
        ceps0 = N.roll(full_dct[:numceps, :], -1, axis=0)
    else:
        # skip C0
        ceps0 = full_dct[1:numceps+1, :]
    ceps = N.transpose(ceps0)
    assert ceps.shape[1] == numceps
    chain.append(FunctionProcessor(lambda data: N.dot(data, ceps), label='DCT' + ' C0' if c0 else ''))

    # Liftering
    if ceplifter > 0:
        lifts = lifter[numceps, ceplifter]
        def lift(data):
            data *= lifts
            return data
        chain.append(FunctionProcessor(lift, label='Lifter'))

    processor = ChainProcessor(chain)

    return processor


# Equal loudness code etc from HTK version ???
## /* EXPORT->InitPLP: Initialise equal-loudness curve & IDT cosine matrix */
## void InitPLP (FBankInfo info, int lpcOrder, Vector eql, DMatrix cm)
## {
##   int i,j;
##   double baseAngle;
##   float f_hz_mid, fsub, fsq;
##   int  nAuto, nFreq;

##   /* Create the equal-loudness curve */
##   for (i=1; i<=info.numChans; i++) {
##     f_hz_mid = 700*(exp(info.cf[i]/1127)-1); /* Mel to Hz conversion */
##     fsq = (f_hz_mid * f_hz_mid);
##     fsub = fsq / (fsq + 1.6e5);
##     eql[i] = fsub * fsub * ((fsq + 1.44e6)  /(fsq + 9.61e6));
##   }

##   /* Builds up matrix of cosines for IDFT */
##   nAuto = lpcOrder+1;
##   nFreq = info.numChans+2;
##   baseAngle =  PI / (double)(nFreq - 1);
##   for (i=0; i<nAuto; i++) {
##     cm[i+1][1] = 1.0;
##     for (j=1; j<(nFreq-1); j++)
##       cm[i+1][j+1] = 2.0 * cos(baseAngle * (double)i * (double)j);

##     cm[i+1][nFreq] = cos(baseAngle * (double)i * (double)(nFreq-1));
##   }
## }

        
# Note: changing nd_force_copy_and_overwrite to True should not affect any
# results; if results do change, it indicates either that some client is
# incorrectly reusing data that's been passed into one of the DSP functions, or
# that one of the DSP functions is incorrectly using data that it's passed back
# to the client, or that data types aren't being well managed.  If this feature
# proves useful, changing it to two flags will separate the cases.
nd_force_copy_and_overwrite = False

def _input_work(data):
    # input data preparation
    new_data = N.array(data, dtype=N.float32, copy=nd_force_copy_and_overwrite)
    if nd_force_copy_and_overwrite:
        data[:] = 0
    del data
    return new_data
    
def _return_work(result):
    # return data preparation
    if isinstance(result, N.ndarray):
        new_result = N.array(result, dtype=N.float32, copy=nd_force_copy_and_overwrite)
        if nd_force_copy_and_overwrite:
            result[:] = 0
    else:
        new_result = result
    del result
    return new_result
    
def nd_io(f):
    """
    A function wrapper to apply our semantics about ndarray types and
    overwriting.  These semantics are that all DSP data is float32, that a
    called function takes ownership of the arguments, that a called function has
    no ownership of the value it returns.  Only does the work on the first
    argument; the rest are just handed in to f.
    """
    @functools.wraps(f)
    def wrapper(arg0, *rest):
        return _return_work(f(_input_work(arg0), *rest))
    return wrapper

def nd_io2(f):
    """
    A method wrapper (has a self argument) to apply our semantics about ndarray
    types and overwriting.  These semantics are that all DSP data is float32,
    that a called function takes ownership of the arguments, that a called
    function has no ownership of the value it returns.
    """
    @functools.wraps(f)
    def wrapper2(self, arg0, *rest):
        return _return_work(f(self, _input_work(arg0), *rest))
    return wrapper2

# XXX this is a general utility class
class CachingDude(object):
    """
    This object caches values it generates via the factory argument passed to
    the constructor.  An instance is used via indexing syntax, instance[key],
    where key must be immutable.  If the instance has seen key before, it
    returns the prior value that it cached for key.  If it hasn't seen key
    before it calls factory with key, and caches and returns the value that
    factory returns.
    """
    def __init__(self, factory):
        class factory_dict(dict):
            def __missing__(self, key):
                value = self[key] = factory(key)
                return value
        self.cache = factory_dict()
    def __getitem__(self, key):
        return self.cache[key]

class Dither(object):
    """
    Adds randomness to data samples.  Constructor takes a scale argument.
    Instance is callable, with a numpy data array.  It then generates samples
    from the uniform interval [-scale, scale) and adds them inplace to the data
    array.  It returns the modified data array.

    >>> data = N.arange(24, dtype=N.float32).reshape((2,3,4))
    >>> N.random.set_state(_random_state)
    >>> dither = Dither(1)
    >>> dither(data)
    array([[[  0.41260675,   0.14370926,   1.50725913,   2.02778625],
            [  3.02689767,   4.00404644,   6.81668568,   6.23421478],
            [  8.04344654,   9.33064175,  10.11111069,  10.58701611]],
    <BLANKLINE>
           [[ 12.54462719,  12.98636055,  13.78572559,  14.56867504],
            [ 16.87078667,  17.28536415,  18.18152237,  18.09235191],
            [ 20.91135406,  21.6317749 ,  21.48546982,  23.6743679 ]]], dtype=float32)
    """
    def __init__(self, scale):
        self.func = functools.partial(N.random.uniform, -scale, scale)
    @nd_io2
    def __call__(self, data):
        data += self.func(data.shape)
        return data

class Preemphasis(object):
    """
    A stateful first differences preemphasis object using the alpha value
    provided to the constructor.  The instance is callable and expects a Numpy
    ndarray as its argument.  Implements the following along the first dimension
    of data, where data can be any shape, with the restriction that the first
    dimension needs to be two or larger on the first call following a reset():

        output[n] = data[n] - alpha * data[n-1]

    The data argument may be overwritten, so use data.copy() as the argument if
    you need to hang on to the original data.  The output of each call is the
    same shape as each input and always dtype=numpy.float32.  (To implement this
    shape feature, the very first item in the output following a call to reset()
    is faked to be the same as the second item in that output.)  Note that you
    can provide an array with a changed size of the first dimension of the data
    on each call to the instance; you can only change other dimensions or the
    dimensionality on the first call following a call to reset().

    Examples:

    Construct a preemphasis filter with an alpha with only 5 bits of precision
    that's close to 0.97, a typical preemphasis alpha.  As such we get easily
    reproduced floating point results on these tests.

    >>> pre = Preemphasis(0x1f / (1 << 5))
    >>> pre.alpha
    0.96875

    First example calls the instance with a shape (18,) ndarray:

    >>> x = tuple(pre(N.arange(18)))
    >>> x
    (1.0, 1.0, 1.03125, 1.0625, 1.09375, 1.125, 1.15625, 1.1875, 1.21875, 1.25, 1.28125, 1.3125, 1.34375, 1.375, 1.40625, 1.4375, 1.46875, 1.5)

    Second example resets the instance then calls it with three shape (6,)
    ndarrays, which, if catenated would be the same as the array in the first
    example.  The outputs are the same, showing the stateful behavior:

    >>> pre.reset()
    >>> y = tuple(x for i in xrange(3) for x in pre(N.arange(6*i, 6*(i+1))))
    >>> y == x
    True

    A multidimensional example:

    >>> z1 = N.arange(6)
    >>> z2 = N.column_stack((z1, z1 * -1))
    >>> z2.shape
    (6, 2)
    >>> pre.reset()
    >>> pre(z2)
    array([[ 1.     , -1.     ],
           [ 1.     , -1.     ],
           [ 1.03125, -1.03125],
           [ 1.0625 , -1.0625 ],
           [ 1.09375, -1.09375],
           [ 1.125  , -1.125  ]], dtype=float32)

    XXX example using 1-d alpha array

    A brief aside, hinting at floating point's subtleties and pitfalls:

    Our alpha has 5 bits (the highest order bit being implicit in the binary
    representation).

    >>> floatutils.float_to_readable_string(pre.alpha)
    '+(-0001)0xf000000000000'

    Here's, 8 bits very close to 0.98, another common alpha value for
    preemphasis:

    >>> x = 0xfb / (1 << 8)
    >>> x
    0.98046875
    >>> floatutils.float_to_readable_string(x)
    '+(-0001)0xf600000000000'

    Note that 0.97 and 0.98 are poor choices for reproducility because their
    floating-point approximations have full precision:

    >>> floatutils.float_to_readable_string(0.97)
    '+(-0001)0xf0a3d70a3d70a'
    >>> floatutils.float_to_readable_string(0.98)
    '+(-0001)0xf5c28f5c28f5c'
    """

    def __init__(self, alpha):
        self._alpha = alpha
        self.reset()
        
    @property
    def alpha(self):
        return self._alpha

    def reset(self):
        self._temp1 = None
        
    @nd_io2
    def __call__(self, data):
        temp1 = self._temp1
        if temp1 is None:
            temp1 = self._temp1 = N.empty(data.shape, dtype=N.float32)
            prev_end_scaled = data[0] - (data[1] - data[0] * self.alpha)
        else:
            prev_end_scaled = temp1[-1]
            temp1.resize(data.shape)
        N.multiply(data, self.alpha, temp1)
        data[0] -= prev_end_scaled
        # subtle: can't do the the offset work in-place as that would rely on
        # the order-of-work used by Numpy
        data[1:] -= temp1[:-1]
        return data

class SlidingWindow(object):
    """
    Each instance buffers up input arrays and when it has enough data it returns
    an output tuple, which may be empty.  Instance is callable with Numpy
    ndarray.  All such arrays must have the same shape except for the final
    dimension.  It is along this dimension that the windowing and sliding
    happen.

    >>> slider = SlidingWindow(5, 3)
    >>> slider.window_length
    5
    >>> slider.slide_length
    3

    >>> slider(N.arange(4))
    ()
    >>> slider(N.arange(7))
    (array([ 0.,  1.,  2.,  3.,  0.], dtype=float32), array([ 3.,  0.,  1.,  2.,  3.], dtype=float32), array([ 2.,  3.,  4.,  5.,  6.], dtype=float32))
    >>> slider(N.arange(6))
    (array([ 5.,  6.,  0.,  1.,  2.], dtype=float32), array([ 1.,  2.,  3.,  4.,  5.], dtype=float32))
    >>> slider(N.array([1,2]))
    ()
    >>> slider(N.array([10,20]))
    (array([  4.,   5.,   1.,   2.,  10.], dtype=float32),)
    """

    def __init__(self, window_length, slide_length):
        # XXX should take a dtype argument
        assert window_length >= slide_length > 0, str(window_length) + ' ' + str(slide_length)
        self._window_length = window_length
        self._slide_length = slide_length
        self._outbuf = None
        self._outbuf0 = None

    @property
    def window_length(self):
        return self._window_length

    @property
    def slide_length(self):
        return self._slide_length
    
    @nd_io2
    def __call__(self, data):
        window_length = self._window_length
        slide_length = self._slide_length
        overlap = window_length - slide_length
        assert 0 <= overlap < window_length

        outbuf = self._outbuf
        if outbuf is None:
            outbuf = N.empty(shape=data.shape[:-1] + (window_length,), dtype=N.float32)
            outbuf0 = 0
        else:
            outbuf0 = self._outbuf0
        outshape = outbuf.shape
        assert outshape[-1] == window_length

        assert data.shape[:-1] == outshape[:-1], 'incompatible shapes: ' + str(outshape) + ' ' + str(data.shape)

        results = list()
        data_length = data.shape[-1]
        data0 = 0
        while data0 < data_length:
            ncopy = min(window_length - outbuf0, data_length - data0)
            assert 0 < ncopy <= window_length
            outbuf[..., outbuf0:outbuf0+ncopy] = data[..., data0:data0+ncopy]
            outbuf0 += ncopy
            data0 += ncopy

            assert 0 < outbuf0 <= window_length
            assert 0 < data0 <= data_length
            assert outbuf0 == window_length or data0 == data_length

            if outbuf0 == window_length:
                results.append(outbuf)
                outbuf = N.empty(shape=outshape, dtype=N.float32)
                outbuf[..., :overlap] = results[-1][..., slide_length:]
                outbuf0 = overlap

        assert data0 == data_length
        self._outbuf = outbuf
        self._outbuf0 = outbuf0
        return tuple(results)

@nd_io
def remove_dc(data):
    """
    Removes the mean (DC) component of data where the mean is calculated across
    the last axis.  Usually modifies the data in-place.  Returns the modified
    data object.

    >>> remove_dc(N.arange(0,20,2))
    array([-9., -7., -5., -3., -1.,  1.,  3.,  5.,  7.,  9.], dtype=float32)

    >>> x = N.arange(20)
    >>> x.shape = 2, 10
    >>> remove_dc(x)
    array([[-4.5, -3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5],
           [-4.5, -3.5, -2.5, -1.5, -0.5,  0.5,  1.5,  2.5,  3.5,  4.5]], dtype=float32)
    """
    # grrr, N.mean doesn't return an array, though documented to do so, so we
    # have to do the work explicitly
    means = N.sum(data, axis=-1, dtype=N.float32)
    means /= data.shape[-1]
    # this step is necessary for Windows, where means might (still) be a Numpy scalar....
    means = N.asanyarray(means,  dtype=N.float32)
    data -= means[..., N.newaxis]
    return data


class Kaiser(object):
    """
    An instance of this class can be used to do Kaiser window weighting.  The
    instance is callable with a Numpy array which it will (usually) modify in
    place.  It performs the Kaiser weighting along the final dimension and
    returns the resulting ndarray.  The constructor is called with one argument,
    beta, which is the value of beta to use for the discrete Kaiser window.

    The instance caches the necessary Kaiser data it uses, so there is no
    penalty for using the same instance to perform the weighting on differently
    shaped data.

    The module attribute, kaiser58, is an instance of Kaiser with beta=8.  This
    beta gives first sidelobes that are -58 dB below the main lobe.  This is
    comparable attenuation to that of the Blackman window, but with a main lobe
    that is approximately 10% narrower than that of the Blackman window.

    >>> x = kaiser58(N.ones(10))
    >>> x
    array([ 0.00233883,  0.0653422 ,  0.28589031,  0.65247869,  0.9547388 ,
            0.9547388 ,  0.65247869,  0.28589031,  0.0653422 ,  0.00233883], dtype=float32)

    >>> kaiser58(N.array((N.ones(10), N.ones(10) * -2)))
    array([[ 0.00233883,  0.0653422 ,  0.28589031,  0.65247869,  0.9547388 ,
             0.9547388 ,  0.65247869,  0.28589031,  0.0653422 ,  0.00233883],
           [-0.00467766, -0.13068441, -0.57178062, -1.30495739, -1.90947759,
            -1.90947759, -1.30495739, -0.57178062, -0.13068441, -0.00467766]], dtype=float32)
    >>> (kaiser58(N.ones(10)) == x).all()
    True
    """
    def __init__(self, beta):
        self._cache = CachingDude(lambda key: N.kaiser(key, beta))
    @nd_io2
    def __call__(self, data):
        window_size = data.shape[-1]
        data *= self._cache[window_size]
        return data
kaiser58 = Kaiser(8.0)

class Hamming(object):
    """
    An instance of this class can be used to do Hamming window weighting.  The
    instance is callable with a Numpy array which it will (usually) modify in
    place.  It performs the Hamming weighting along the final dimension and
    returns the resulting ndarray.

    The instance caches the necessary Hamming data it uses, so there is no
    penalty for using the same instance to perform the weighting on differently
    shaped data.  The module attribute, hamming, is an instance of Hamming and
    can be used for all Hamming-windowing work.

    While Hamming window is popular, consider using the module's kaiser58.  The
    Hamming window does have a narrow main lobe, but it has a largest sidelobe
    attenuation of only 44 dB.  It doesn't achieve the 58 dB attenuation of the
    kaiser58 window until 18 to 20 times its main lobe width.  The kaiser58
    window has a main lobe that is 1.3 times the width of Hamming window's main
    lobe, but its attentuation is 58 dB or better everywhere outside the main
    lobe.

    >>> x = hamming(N.ones(10))
    >>> x
    array([ 0.08      ,  0.18761955,  0.46012184,  0.76999998,  0.97225863,
            0.97225863,  0.76999998,  0.46012184,  0.18761955,  0.08      ], dtype=float32)

    >>> hamming(N.array((N.ones(10), N.ones(10) * -2)))
    array([[ 0.08      ,  0.18761955,  0.46012184,  0.76999998,  0.97225863,
             0.97225863,  0.76999998,  0.46012184,  0.18761955,  0.08      ],
           [-0.16      , -0.3752391 , -0.92024368, -1.53999996, -1.94451725,
            -1.94451725, -1.53999996, -0.92024368, -0.3752391 , -0.16      ]], dtype=float32)
    >>> (hamming(N.ones(10)) == x).all()
    True
    """
    def __init__(self):
        self._cache = CachingDude(lambda key: N.hamming(key))
    @nd_io2
    def __call__(self, data):
        window_size = data.shape[-1]
        data *= self._cache[window_size]
        return data
hamming = Hamming()

def make_triangle_function(left, mid, right):
    """
    Returns a function that calculates values of a rising-then-falling
    trianglular function.  The rising line of the triangle goes through points
    (left, 0) and (mid, 1), and the falling line of the triangle goes through
    points (mid, 1) and (right, 0).  Within the support domain (left, right) the
    function is non-negative, and outside the support domain (left, right) the
    function is negative.  This formulation is useful for buildihng some simple
    discrete weighting functions, e.g. overlapping triangular Mel filters.

    >>> f = make_triangle_function(-2, 0, 2)
    >>> tuple(f(x) for x in xrange(-3, 4)) #doctest: +ELLIPSIS
    (-0.5, 0.0, 0.5, 1.0, 0.5, ...0.0, -0.5)

    >>> f = make_triangle_function(100.25, 141.8, 200.5)
    >>> tuple((x, int(100*f(x))) for x in xrange(90, 220, 10))
    ((90, -24), (100, 0), (110, 23), (120, 47), (130, 71), (140, 95), (150, 86), (160, 68), (170, 51), (180, 34), (190, 17), (200, 0), (210, -16))
    """
    left_slope = 1 / (mid - left)
    right_slope = 1 / (mid - right)
    def f(x):
        val = (x - left) * left_slope if x <= mid else (x - right) * right_slope
        assert val <= 1
        return val
    return f
        
def HzToMel(hz):
    """
    Return the so-called Mel frequency corresponding to the give Hertz
    frequency.

    >>> int(HzToMel(100) + 0.5)
    150
    >>> HzToMel(6300)
    2595.0
    """
    return 2595 * math.log10(1 + hz/700)

def MelToHz(mel):
    """
    Given the Mel frequency, mel, return the corresponding Hertz frequency.

    >>> int(MelToHz(150) + 0.5)
    100
    >>> MelToHz(2595)
    6300.0
    """
    return 700 * (math.pow(10, mel/2595) - 1)
    
def make_mel_specs(num_filters, low_hz, high_hz, round_to_int=True):
    """
    Return a sequence of the break-point frequencies (in Hertz) for a Mel
    filterbank.  Each filter is specified by three parameters, two of which are
    shared with the parameters for the neighboring filters, so num_filters + 2
    values are returned.  The filters overlap 50%, such that the low-cutoff of
    one filter is the center of the preceding filter, and similarly for the
    high-cutoff.  The filters are spaced evenly on the Mel scale, so they are
    non-linearly spaced on the Hz scale.  Note that responses are intended to be
    zero at both low_hz and high_hz.  If optional round_to_int (default True) is
    True, then the returned break-point frequency values are rounded to
    integers; otherwise floats are returned.

    Specs for a typical set of 31 filters.

    >>> specs = make_mel_specs(31, 200, 3500)
    >>> len(specs)
    33
    >>> specs
    (200, 244, 291, 340, 391, 445, 501, 561, 623, 688, 756, 828, 904, 983, 1066, 1153, 1244, 1340, 1441, 1546, 1657, 1773, 1895, 2023, 2158, 2299, 2446, 2602, 2764, 2935, 3114, 3303, 3500)

    Show that the bandwidth is uniform in the Mel space.

    >>> spec_hzs = make_mel_specs(3, 200, 3500)
    >>> mels = tuple(int(HzToMel(spec_hz)+0.5) for spec_hz in spec_hzs)
    >>> mels
    (283, 717, 1151, 1585, 2019)
    >>> mels[2]-mels[0] == mels[3]-mels[1] == mels[4]-mels[2]
    True
    """
    low_mel = HzToMel(low_hz)
    high_mel = HzToMel(high_hz)
    inc_mel = (high_mel - low_mel) / (num_filters + 1)
    first_center_mel = low_mel + inc_mel

    freqs_hz = list()
    freqs_hz.append(low_hz)
    for index in xrange(num_filters):
        freq_mel = first_center_mel + inc_mel * index
        freqs_hz.append(MelToHz(freq_mel))
    freqs_hz.append(high_hz)

    if round_to_int:
        return tuple(int(f+0.5) for f in freqs_hz)
    else:
        return tuple(freqs_hz)

# XXX this is a general utility class
class Selector(object):
    """
    Single instance, selector, is used to return the object used as an index.
    Helpful for capturing fancy indexing constants.

    The module attribute, selector, is an instance of Selector, intended to be
    used for all such selecting work.

    >>> selector[3:4]
    slice(3, 4, None)
    >>> selector[2, ..., 3:4, N.newaxis, :, :-1, ::]
    (2, Ellipsis, slice(3, 4, None), None, slice(None, None, None), slice(None, -1, None), slice(None, None, None))
    """
    def __getitem__(self, index):
        return index
selector = Selector()

class MelFilter(object):
    """
    A mel filter.  The low, mid, and high breakpoints are given in terms of
    integer indicies, but these can be floats with fractional values.  The Mel
    filter is piecewise linear and (conceptually) goes through the points (low,
    0), (mid, 1), (high, 0).

    The Mel filter provides the positive values along those two linear segments
    at integer indices between low and high.  These are the indicies into the
    vector of amplitude or energy values to which the filter will be applied.
    If optional constructor argument normalize is True (default False), the
    weights will be scaled so that their sum is one.

    When an instance is called with a vector of values, it applies its weights
    to the part of the vector corresponding to the integer indices of the filter
    with positive weights, and then returns the summed value.

    Here's a simple example using integer breakpoints and an asymetrical filter

    >>> m = MelFilter(10, 14, 22)
    >>> m(N.ones(24))
    6.0

    A look inside shows the weights

    >>> m._filter
    array([ 0.25 ,  0.5  ,  0.75 ,  1.   ,  0.875,  0.75 ,  0.625,  0.5  ,
            0.375,  0.25 ,  0.125], dtype=float32)

    And the range of indices these weights are applied to

    >>> m._select
    (Ellipsis, slice(11, 22, None))

    Note that the resulting filter can be used on multichannel (or
    multi-dimensional) data.  Filtering is done on the right-most dimension.
    The returned data has one fewer dimensions than the input data by dropping
    the rightmost dimension; i.e. output.shape = input.shape[:-1].  This has
    implications when multiple filters are being used and you want the result of
    combining the outputs of these multiple filters to appear as the rightmost
    dimension, e.g. see the implementation of MelFilterBank.__call__() below.

    >>> input = N.array([N.ones(24), 2 * N.ones(24)], dtype=N.float32)
    >>> input.shape
    (2, 24)
    >>> output = m(input)
    >>> output.shape
    (2,)
    >>> output
    array([  6.,  12.], dtype=float32)

    >>> input2 = input[N.newaxis, : ,N.newaxis, :]
    >>> input2.shape
    (1, 2, 1, 24)
    >>> output2 = m(input2)
    >>> output2.shape
    (1, 2, 1)

    In general, the supplied indices will have fractional parts due to both the
    temporal sampling rate and spectral sampling interval.  For example,
    consider a 44,100 Hz sampled signal.  Frequency components are calculated
    for 1,024 point buffer of real wave data (23 msec) using the FFT.  The FFT
    returns 1,024 (complex) values in an integer-indexed array.  The frequency
    corresponding to an index is given by (index * 44100 / 1024).  If you want a
    Mel filter with breakpoint frequencies of 500, 560, and 625, you would use
    indices of 11.6, 13, and 14.5 to specify the Mel filter, (calculated simply
    by multiplying the frequency by the buffer-length to sample-frequency ratio,
    1024/44100).

    >>> m = MelFilter(11.6, 13, 14.5)

    Again, looking inside we have only three weights

    >>> m._filter
    array([ 0.2857143 ,  1.        ,  0.33333334], dtype=float32)

    And the integer indices(into the 1,024-point set of FFT outputs)
    corresponding to those weights, 12, 13, and 14.

    >>> m._select
    (Ellipsis, slice(12, 15, None))

    >>> m(N.ones(15)) #doctest: +ELLIPSIS
    1.6190476...

    A similar example, but using normalization

    >>> MelFilter(11.6, 13, 14.5, normalize=True)(N.ones(15))
    1.0
    """
    def __init__(self, low, mid, high, normalize=False):
        # the function
        func = make_triangle_function(low, mid, high)
        assert func(low) <= 0
        assert func(high) <= 0

        # range of positive values
        lower = int(math.floor(low) + 1)
        assert  func(lower-1) <= 0 < func(lower)
        higher = int(math.ceil(high) - 1)
        upper = higher + 1
        assert func(higher) > 0 >= func(upper)

        self._select = selector[..., lower:upper]
        filter = self._filter = N.fromiter((func(index) for index in xrange(lower, upper)), dtype=N.float32)
        assert filter.min() > 0
        assert filter.max() <= 1
        if normalize:
            filter /= filter.sum()
        assert 0 < filter.min()
        assert filter.max() <= 1
    # note: no @nd_io2 decorator since MelFilterBank really wants to reuse the data
    def __call__(self, data):
        return N.inner(data[self._select], self._filter)

class MelFilterBank(object):
    """
    A simple filterbanking object.

    Example shows that the higher frequency filters in this 'telephone' set of
    filters cover 4.5 times as much spectrum as the lower frequency filters

    >>> fb = MelFilterBank(30, 200, 3500, 44100, 1024)
    >>> res1 = fb(N.ones(513))
    >>> ref = N.array([ 1.07500422,  1.12112248,  1.33344889,  1.22044253,  1.24887204,
    ...                 1.50392032,  1.38542223,  1.61813593,  1.54067957,  1.76965499,
    ...                 1.8112514 ,  1.86211777,  1.97671068,  2.07826614,  2.16557503,
    ...                 2.37578273,  2.34643364,  2.5892086 ,  2.62946463,  2.79979753,
    ...                 2.96687913,  3.11538792,  3.2500031 ,  3.40087175,  3.62335014,
    ...                 3.78696251,  3.95911884,  4.16730022,  4.42668438,  4.58136463], dtype=N.float32)
    >>> N.allclose(ref, res1)
    True

    Smaller filterbank, show that normalizing works

    >>> fb = MelFilterBank(10, 100, 2000, 44100, 1024, normalize=True)
    >>> res2 = fb(N.ones(513))
    >>> ref = N.array([ 0.99999994,  1.        ,  1.        ,  0.99999988,  1.        ,
    ...                 1.        ,  1.        ,  1.        ,  1.        ,  0.99999988], dtype=N.float32)
    >>> N.allclose(ref, res2)
    True

    Use on multichannel data

    >>> res3 = fb(N.array([N.ones(513), 2 * N.ones(513)]))
    >>> ref = N.array([[ 1.        ,  1.        ,  0.99999994,  0.99999994,  1.        ,
    ...                  1.        ,  1.        ,  1.00000012,  1.        ,  0.99999988],
    ...                [ 2.        ,  2.        ,  1.99999988,  1.99999988,  2.        ,
    ...                  2.        ,  2.        ,  2.00000024,  2.        ,  1.99999976]], dtype=N.float32)
    >>> N.allclose(ref, res3)
    True
    """

    # Optimization possibilities: For filterbanking we do an inner product for
    # each of the N filters, where this loop is in Python (see the use of
    # inner() in MelFilter.__call__(), and the generator loop in
    # MelFilterBank.__call__()).  We could do a single full N by M numpy.dot(),
    # where M is the data length, but that seems like not a win because so much
    # of M is zeros, and it requires that all the filters live in one ndarray.
    # Or, with integrally overlapped filters, where L is the overlap number,
    # (L==2 for 50%), it's conceivable to have L sets of weights and do L numpy
    # multiplies of the data, giving L sets of weighted data, and then do the N
    # sums with fancy indexing....

    def __init__(self, num_filters, low_hz, high_hz, sample_rate_hz, num_unit_circle_samples, normalize=False):
        sample_rate_recip = 1 / sample_rate_hz
        breakpoints = tuple(hertz * num_unit_circle_samples * sample_rate_recip for hertz in make_mel_specs(num_filters, low_hz, high_hz))        
        bp1, bp2, bp3 = tee(iter(breakpoints), 3)
        bp2.next()
        bp3.next(); bp3.next()
        self._fb = tuple(MelFilter(f1, f2, f3, normalize) for f1, f2, f3 in zip(bp1, bp2, bp3))
        assert len(self._fb) == num_filters
    @nd_io2
    def __call__(self, data):
        gen = (f(data) for f in self._fb)
        if data.ndim == 1:
            return N.fromiter(gen, dtype=N.float32)
        else:
            return N.swapaxes(N.array(tuple(gen), dtype=N.float32), -1, -2)

def make_dct(length):
    """
    Return a 2-D ndarray for calculating Discrete Cosine Transform coefficients
    on vectors with length elements.

    The module attribute, dct, caches the ndarrays it's seen, and is suitable
    for all DCT work.

    >>> make_dct(1)
    array([[ 1.41421354]], dtype=float32)
    >>> res4 = make_dct(4)
    >>> ref = N.array([[ 0.70710677,  0.70710677,  0.70710677,  0.70710677],
    ...                [ 0.65328145,  0.27059805, -0.27059811, -0.65328151],
    ...                [ 0.49999997, -0.49999997, -0.49999991,  0.50000018],
    ...                [ 0.27059805, -0.65328145,  0.65328151, -0.27059838]], dtype=N.float32)
    >>> N.allclose(ref, res4)
    True

    Using the module's dct attribute

    >>> dct[1]
    array([[ 1.41421354]], dtype=float32)
    >>> dct[1] is dct[1]
    True
    >>> dct[12] is dct[12]
    True
    """
    def dct_func(i, j):
        recip_length = 1 / length
        # note: (j + 0.5) is derived from ((j+1) - 0.5) for a 1-based j-counter equation
        return math.sqrt(2 * recip_length) * N.cos(math.pi * recip_length * i * (j + 0.5))
    return N.fromfunction(dct_func, (length, length), dtype=N.float32)
dct = CachingDude(make_dct)


def make_lifter(numchans, lifter):
    """
    Return an array of liftering factors for vectors with numchans items using
    lifter as the coefficient.  The value of lifter can be 1.0, in which case
    the vector of factors is essentially all ones.  Otherwise lifter is
    typically between 1.5 and 2.0 times numchans.

    The indexable module attribute, lifter, caches the vectors.

    >>> N.allclose(make_lifter(10, 1), N.ones(10))
    True

    >>> ref = N.array([  2.5654633 ,   4.09905815,   5.5695653 ,   6.94704914,
    ...                  8.20346832,   9.31324577,  10.25378895,  11.00595188,
    ...                 11.55442238,  11.88803577,  12.        ,  11.88803577], dtype=N.float32)

    >>> N.allclose(make_lifter(12, 22), ref)
    True

    >>> N.allclose(lifter[10, 1], N.ones(10))
    True
    >>> N.allclose(lifter[12, 22], ref)
    True
    """
    assert lifter >= 1, str(lifter)
    assert int(lifter) == lifter, str(lifter)
    def make_lift(n):
        return 1 + lifter / 2 * N.sin(N.pi * (n+1) / lifter)
    return N.fromfunction(make_lift, (numchans,),  dtype=N.float32)
lifter = CachingDude(lambda key: make_lifter(*key))


def get_power_of_two_roundup(x):
    """
    Round up to smallest power of two that is equal to or greater than x, where
    1 is considered the smallest possible power of two.

    The indexable module attribute, power_of_two_roundup, caches the values.

    >>> get_power_of_two_roundup(1023)
    1024

    >>> power_of_two_roundup[1024]
    1024
    >>> power_of_two_roundup[0]
    1

    >>> tuple(power_of_two_roundup[x] for x in xrange(34))
    (1, 1, 2, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 64)
    >>> tuple(power_of_two_roundup[x] for x in (31, 32, 33))
    (32, 32, 64)
    >>> tuple(power_of_two_roundup[x/4] for x in xrange(18))
    (1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 8)
    """
    if x <= 1:
        return 1
    pot = 2
    while pot < x:
        pot <<= 1
    return pot
    # it's likely that this float based implementation would have incorrect values
    #return 1 << int(math.ceil(math.log(x)/math.log(2)))
power_of_two_roundup = CachingDude(get_power_of_two_roundup)

    
class HtkDelta(object):
    """
    HTK-style delta processing.

    >>> d1 = HtkDelta(1)
    >>> d0 = HtkDelta(0)
    >>> d3 = HtkDelta(3)
    >>> data = N.array((N.arange(80), N.arange(0, 160, 2)))
    >>> data.shape
    (2, 80)
    >>> data.shape = 2, 16, 5
    >>> data = N.transpose(data, (1,0,2))
    >>> for buf in data: print 'b', buf; print 'd', d3(buf)
    b [[0 1 2 3 4]
     [0 2 4 6 8]]
    d ()
    b [[ 5  6  7  8  9]
     [10 12 14 16 18]]
    d ()
    b [[10 11 12 13 14]
     [20 22 24 26 28]]
    d ()
    b [[15 16 17 18 19]
     [30 32 34 36 38]]
    d (array([[-40., -40., -40., -40., -40.],
           [-80., -80., -80., -80., -80.]]),)
    b [[20 21 22 23 24]
     [40 42 44 46 48]]
    d (array([[ -55.,  -55.,  -55.,  -55.,  -55.],
           [-110., -110., -110., -110., -110.]]),)
    b [[25 26 27 28 29]
     [50 52 54 56 58]]
    d (array([[ -70.,  -70.,  -70.,  -70.,  -70.],
           [-140., -140., -140., -140., -140.]]),)
    b [[30 31 32 33 34]
     [60 62 64 66 68]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[35 36 37 38 39]
     [70 72 74 76 78]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[40 41 42 43 44]
     [80 82 84 86 88]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[45 46 47 48 49]
     [90 92 94 96 98]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[ 50  51  52  53  54]
     [100 102 104 106 108]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[ 55  56  57  58  59]
     [110 112 114 116 118]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[ 60  61  62  63  64]
     [120 122 124 126 128]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[ 65  66  67  68  69]
     [130 132 134 136 138]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[ 70  71  72  73  74]
     [140 142 144 146 148]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)
    b [[ 75  76  77  78  79]
     [150 152 154 156 158]]
    d (array([[ -80.,  -80.,  -80.,  -80.,  -80.],
           [-160., -160., -160., -160., -160.]]),)

    For now, we use None as the gap or EOF signifier.  First None flushes the
    delta queue.  Any others do nothing.

    >>> d3(None)
    (array([[ -70.,  -70.,  -70.,  -70.,  -70.],
           [-140., -140., -140., -140., -140.]]), array([[ -55.,  -55.,  -55.,  -55.,  -55.],
           [-110., -110., -110., -110., -110.]]), array([[-40., -40., -40., -40., -40.],
           [-80., -80., -80., -80., -80.]]))
    >>> d3(None)
    ()
    """
    def __init__(self, window):
        """
        Window is the one-sided length of the window for HTL-style delta processing.
        """
        checkutils.check_instance(int, window)
        checkutils.check_nonnegative(window)

        # XXX instead of a deque, implement this using numpy arrays and axis rotation....

        denom = 2 * sum(x*x for x in xrange(1, window + 1))
        recip = 1 / denom if window > 0 else None

        # XXX we really want the Python 2.6 maxlen argument, i.e. deque((), 2 * window + 1)
        self._delta_data = window, 2 * window + 1, deque(), recip

    def __call__(self, data):
        return self._do_gap() if data is None else self._do_data(data)

    @staticmethod
    def _do_delta(delta_data):
        # implement HTK's FIR weighting for deltas
        window, queue_size, queue, recip = delta_data
        assert queue_size == 2 * window + 1
        assert len(queue) == queue_size
        shape = queue[0].shape
        temp = N.empty(shape)
        res = N.zeros(shape)
        for scale, left, right in izip(xrange(window-1,0,-1), queue, reversed(queue)):
            N.subtract(right, left, temp)
            temp *= scale
            res += temp
        if recip is not None:
            res *= 1
        return res

    def _do_data(self, data):
        window, queue_size, queue, recip = delta_data = self._delta_data
        # newest is at queue[0], oldest at queue[-1]
        queue.appendleft(data)

        if len(queue) <= window:
            # not enough in queue
            return ()

        num_left_padding = queue_size - len(queue)
        assert 0 <= num_left_padding <= window
        for i in xrange(num_left_padding):
            # padding for partial queue
            queue.appendleft(queue[0])

        res = self._do_delta(delta_data)

        for i in xrange(num_left_padding):
            queue.popleft()
        if len(queue) >= queue_size:
            # XXX Python 2.6 maxlen deque will eliminate the need for this
            # conditional block
            assert len(queue) == queue_size
            queue.pop()

        return (res,)

    def _do_gap(self):
        window, queue_size, queue, recip = delta_data = self._delta_data
        # newest is at queue[0], oldest at queue[-1]
        result = list()
        while len(queue) > window:
            
            num_right_padding = queue_size - len(queue)
            assert 0 <= num_right_padding <= window
            for i in xrange(num_right_padding):
                # padding for partial queue
                queue.append(queue[-1])

            result.append(self._do_delta(delta_data))

            for i in xrange(num_right_padding):
                queue.pop()

            queue.pop()

        queue.clear()
        return tuple(result)

        
# ugggg: Numpy has a complex random state; needed for reprodicibility in tests
_random_state = ('MT19937',
                 N.array([2770335632, 1417304791, 3182359249, 1138507138, 2700043002,
                           441196414, 4076735196, 1138740068, 3244280105, 3488860226,
                          2640221119, 3545913321,  495651649, 3338852954, 3066747729,
                          3629501752, 2236333236, 4145368274, 3796235355, 3447710693,
                          2555329548, 2948488091,  491322256, 2374403289,  925203275,
                           645105436, 1453593683, 3502535010,  731124609, 3327375089,
                          3445564617, 3944583142, 3822547089, 3871800972, 1212137124,
                          1011577396, 3340333056,  169208026,  490203866, 2787313650,
                          1834303699, 1849557074, 2146105048,  699432549, 1767719151,
                          2897079596, 1675701228, 4269022600,  910048346,  543951375,
                          3910610813, 1003235651, 2244960448, 3772798982, 1136707813,
                          4094310014, 2449882652, 3114687113, 2435341987, 4239100605,
                          4057033375, 2665318525, 1173412516, 1520036141, 1405200740,
                          3347741539, 2500541244, 3871505038,  734728076,  374699041,
                          2564841366, 1771636950, 3291831478,  482257538, 2687358798,
                          3432490183, 3127135751, 2204130462, 3321750136, 4038835286,
                          1534636097,  690904341, 2249038175, 3657730137, 1590604645,
                          2618635707,  839933607, 3676028732, 3386613512,  496869948,
                          2620712593, 3529543508,  593752856, 1788612636, 3055799606,
                          2749262418, 1292101052, 3072636780, 1525351616, 3235126814,
                          3468415355, 2362019266, 4032723526, 1257943251, 3283422780,
                           771312686, 2170216528, 4045212801, 3357522716, 2644068892,
                           416613637,  255839653, 1199977008,   24063960, 1369575798,
                          3159102505, 1220911531, 2201299380, 1049862976,  588712948,
                          2273861722, 1443992066, 3989619314,  231827209, 2644481021,
                          1790537288, 1463105418, 1817516104, 3878512087, 3487883160,
                          4279865234, 3823284882,  981787503,  316342931, 1284478668,
                          2558333475, 1865813549, 2724504524, 3275179652,  891324149,
                          3192394304, 4065196456, 3090363252, 4178198090,  503998245,
                          4040528239, 1605625218, 2163418222, 4229879376, 2452707856,
                           554210378, 1913080345,   64881835, 1920170065, 1615377894,
                           731666719,  485312738,  768187274,   62828476, 3010180622,
                          1024663420, 3085202578, 1653657216, 1250878132, 3754241753,
                          2026547849, 1192103606,  546245170, 1050627520, 1903014338,
                          2876802155, 1741970320, 3588718709, 2719970145,  698389350,
                           473400895, 1565363635,  116939369, 3827181020, 4115705850,
                           486685026, 3225994133, 1021764978, 3037109969,  116476169,
                          2786495441,  643143935,  205502367, 2831313882,  997411997,
                           394239235,  892373348, 3191676145,  215825010, 1919195277,
                          3168785173,  737532090, 1028229618, 4138028162, 3339691480,
                          2522531701,  671932148,  181541498, 4257095578,  462240177,
                          2156117405,  805491909,  234543881, 3798129095,  178588125,
                          2672106405, 1153446101, 1238453915, 2074797740, 1390389730,
                           276124274,  903284632, 1192412094, 1816742015, 4093485352,
                          2073080075, 3054730165, 2208241560,   94869923, 2267630952,
                          2645039524, 4233541066, 1476894700, 3491386423, 3984431113,
                          2872276009, 1279843189, 3827592009, 2060139448, 1087561202,
                          4234061692, 2314235742, 1710682261, 2498460161, 3538431389,
                          3642907364,  475523615,  512119828,  744916169,  753424779,
                          4042082991,  351971570, 2400623650, 3862202715,  168275224,
                           769403487, 2599409771,  638811384,  344835567, 3598720852,
                          3113591386, 2347738205, 2154181951, 3512221718, 3000638705,
                           188414312, 1139478414, 3105915571, 2665198346, 2576457266,
                           246005590, 2452562458,  733294697, 3834323369, 3665800424,
                           598111097,  918247755, 3609711969, 2994706583, 4111693245,
                          1423852052,  402492677,  678865684, 3710552038, 3982161740,
                          3676337018, 1724635914, 3890073372,  601097689, 3022277571,
                           857031174, 3056599732,  622356976, 2142974756, 2946878371,
                           136830551, 2049039192, 3329387879,  672391255, 2439882187,
                          4100486591, 3810053849, 3653136506, 1250741258, 1362353440,
                          3721011761, 3373467683, 2305269421,  712642809, 1889511665,
                          2819234283,  910458112,  576414576, 2160594508, 1736124796,
                          2893128159, 4070437541, 1911594333,  658579079, 1203015978,
                          4097079614, 2793325554, 3702811989,  795750342, 3849570880,
                          2211058022, 2857899332, 3902276976, 3886071533, 1544700592,
                          1482894956, 2183853234, 2090880982,  502997728, 3181255119,
                          3096911737, 2280222526, 2527997573, 3896210888, 1804948575,
                          3614679862, 3831756855, 2789989157,   16507859, 1838029344,
                          3693951744, 1978504574, 3972866267, 3737883633,  465917353,
                          3603838926, 1090707289,   88740394,  493428222, 1302624367,
                           760123814, 4054516399, 1339726349,  329101825, 1159214121,
                          3108978349,  618844669, 4032056969,  121895962, 2874596872,
                           978220409, 2275423384,  854929125, 3490795909, 3245823066,
                          2202856359,  751077831, 4177947692, 1307720093, 2160045784,
                          1355444148,  743643029,  663648954, 3094162911, 2922783475,
                          1681761653, 3835666201, 2375161707, 2873472365,   65214074,
                           668540036,  323234808, 3775992379,  403156638, 2582457671,
                          3833555556,  571604603, 4024774502, 3950144951, 3495104191,
                           415374230, 2603491141, 2666762583, 2566178356, 3448790280,
                           701731398, 2535013981, 2991797990, 2235688371, 3121389334,
                          1676887099,  972254627, 3357642789,  729426351,  236957799,
                           660506200, 3173687185, 3528654395, 1036814057, 1217177237,
                           427984576, 1234586145, 1964568910, 1848384520, 3563498343,
                          4198741626, 4273815993,  727750389, 3945556901, 4020996641,
                          3088985867, 2763591714,  633012600, 4214044100, 3761954420,
                          1505046950,  526437187, 3122593516, 2081278885, 1040227320,
                          4246684598,  340271318, 4158610105,  323338840,  184250330,
                          2366744907,  816683881, 3492092870, 1952066878, 1582374427,
                          3030259854, 2870610146, 3448064998, 1528418013, 2972091399,
                          2514980394, 3140242874, 1194335613, 3101402306,   35892769,
                          1966958493,  906804757, 4198442021,  833491059,  451865668,
                          3988020085,  222379882, 3508569541, 1879900294, 3692740479,
                          1562910317, 2482145328, 1930916694,  809633507, 3776421365,
                          3672189393, 1154228994, 1554953800, 3877246602,  245457927,
                           824411402, 1978908250, 1834400389,  195872032, 2984656773,
                          3304019961, 3849714477, 3039329658, 1552803869, 1235484245,
                          1610733627, 1246753108, 1314357751, 2380369860, 4170492246,
                          2812967524, 3462366867, 2291872135, 1808486876,   74952685,
                          3226502428,  717376779, 1234287716, 3457376890, 4292588539,
                          2467073506, 4227949313,  688750401, 1119179229, 4094365791,
                           746816761, 3938892715, 1027751130, 3896490781, 1909809665,
                          3471008184, 3717046409, 1889416332, 1778122817, 3686414626,
                          2532388950, 2258145247,  972734315, 2384792285, 3940140666,
                          1346726443, 1946203637,  638381825, 2142447240,  772896496,
                          4239054381, 3338819704, 1897089574, 2742074637, 1315471170,
                          1475611408, 1969348647, 1467152778,  126752237, 3620620347,
                          4074263165, 3438039998, 1185171645, 3028021708, 2918361436,
                          1361220119,  501332669,  447682828, 3176889890,  815318862,
                          1056856920, 3885271175, 1437561698, 2776257531, 2873609049,
                          1467233587, 3530714325, 3678590337, 1543230735, 1016777674,
                          3552321917, 1518041740, 4044672846, 1849474171, 3665477077,
                          1918396998, 4185423880, 1086333199,  778551361, 1639232613,
                          4253227569, 1576463010, 2548331041, 1519915977, 3656507138,
                          3369650165, 1006363035, 4134597848, 4154120711, 2095276661,
                          1489751958, 2557001596, 1667278519,  480159027, 3317431121,
                          3137940012,  346038423, 3208654025, 1233827925, 2970626911,
                            84615891, 2636533007, 2907132711, 2559467646,  789568618,
                          3353154235, 1676155826, 2491533796, 3152241265, 3482870591,
                          2859246240,  270913512,    9014317, 2628326584, 1421850299,
                          1762129532, 1153738616, 3515423900, 1848155833, 3879922542,
                          1510208438,  213710759,  577353874, 3986492784, 1649149295,
                          3883798969, 3655746124, 2003486518, 1586821206,  520736505,
                          2793192641, 4159104667, 1006897266,   43987428, 4234488498,
                          1007967132, 3323654702, 1745589406,  931771199, 2045826751,
                          1999778505, 2784666635, 2852451194, 2241250709], dtype=N.uint32),
                 # arrrrg: the state on Windows is different; it turns out, as
                 # of 2008-01-06, that using the first three components of the
                 # longer state "just works"
                 20) + ((0, 0.10458831092175595) if len(N.random.get_state()) > 3 else ())

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
