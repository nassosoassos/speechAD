###########################################################################
#
# File:         htkaudio.py (directory: ./py/onyx/htkfiles)
# Date:         Wed 29 Oct 2008 14:52
# Author:       Ken Basye
# Description:  Read HTK audio files into arrays
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
    Read HTK audio files into arrays

    >>> module_dir, module_name = os.path.split(__file__)

    >>> fname0 = os.path.join(module_dir, "gaw0_7_st0451.mfc")

    NOTE: Make sure you open the file in binary mode (use 'rb' as the mode).
    >>> with open(fname0, 'rb') as f:
    ...    (data0, (kind, qualifiers), samp_period) = read_htk_audio_file(f)
    
    >>> kind
    (6, 'MFCC', 'mel-frequency cepstral coefficients')
    >>> qualifiers
    (('C', 'is compressed'), ('E', 'has energy'), ('K', 'has CRC checksum'))
    
    # The shape of the data is (number of vectors, number of features per vector)
    >>> data0.shape
    (251, 13)
    
    # Verify that we've correctly read the data by comparing against HTK 'HList -r' output
    >>> refname = os.path.join(module_dir, "gaw0_7_st0451.raw")
    >>> with open(refname) as f:
    ...     tempdata = []
    ...     for line in f:
    ...         tokens = line.split()
    ...         vals = [float(t) for t in tokens]
    ...         tempdata.append(vals)
    >>> refdata = array(tempdata)
    
    # Here's a different version of the same file without compression
    >>> fname1 = os.path.join(module_dir, "gaw0_7_st0451_uncomp.mfc")
    >>> with open(fname1, 'rb') as f:
    ...    (data1, (kind, qualifiers), samp_period) = read_htk_audio_file(f)
    
    >>> kind
    (6, 'MFCC', 'mel-frequency cepstral coefficients')
    >>> qualifiers
    (('E', 'has energy'),)
    
    >>> numpy.allclose(data0, data1)
    True
    >>> numpy.allclose(data0, refdata)
    True
    
    # Here's a version of the file where deltas and delta-deltas have been added
    >>> fname2 = os.path.join(module_dir, "gaw0_7_st0451_39feats.mfc")
    >>> with open(fname2, 'rb') as f:
    ...    (data2, (kind, qualifiers), samp_period) = read_htk_audio_file(f)
    >>> kind
    (6, 'MFCC', 'mel-frequency cepstral coefficients')
    >>> qualifiers
    (('A', 'has acceleration coefficients'), ('E', 'has energy'), ('D', 'has delta coefficients'))
    
    >>> data2.shape
    (251, 39)


"""
from __future__ import with_statement
import os
import numpy
from cStringIO import StringIO
from itertools import izip, count
from numpy import array, zeros, fromiter
from struct import unpack


def read_htk_audio_file(file):
    """
    Read an HTK audio file into an array.

    NOTE: Make sure you open the file in binary mode (use 'rb' as the mode).
    """
    header = file.read(12)
    if len(header) != 12:
        raise IOError("not enough bytes in file for a complete header (make sure you open in binary mode?)")
    n_samples, sample_period, sample_size, sample_kind = unpack(">IIHH", header)

    ##     From the HTK Book:
    ##  The basic parameter kind codes are:
    kind_dict = {
        0: ('WAVEFORM', 'sampled waveform'),
        1: ('LPC', 'linear prediction filter coefficients'),
        2: ('LPREFC', 'linear prediction reflection coefficients'),
        3: ('LPCEPSTRA', 'LPC cepstral coefficients'),
        4: ('LPDELCEP', 'LPC cepstra plus delta coefficients'),
        5: ('IREFC', 'LPC reflection coef in 16 bit integer format'),
        6: ('MFCC', 'mel-frequency cepstral coefficients'),
        7: ('FBANK', 'log mel-filter bank channel outputs'),
        8: ('MELSPEC', 'linear mel-filter bank channel outputs'),
        9: ('USER', 'user defined sample kind'),
        10: ('DISCRETE', 'vector quantised data'),
        }
    ##  and the bit-encoding for the qualifiers (in octal) is
    qualifiers_dict = {
        'E': (000100, 'has energy'),
        'N': (000200, 'absolute energy suppressed'),
        'D': (000400, 'has delta coefficients'),
        'A': (001000, 'has acceleration coefficients'),
        'C': (002000, 'is compressed'),
        'Z': (004000, 'has zero mean static coef.'),
        'K': (010000, 'has CRC checksum'),
        'O': (020000, 'has 0'),
        }

    kind = sample_kind & 0x03f
    if kind not in kind_dict.keys():
        raise IOError("unknown sample kind - expected a value in %s but got %d - could be endian issues" %
                      (kind_dict.keys(), kind))
    kind_tuple = (kind, kind_dict[kind][0], kind_dict[kind][1])

    qualifiers = []
    for (code, (mask, desc)) in qualifiers_dict.items():
        if sample_kind & mask:
            qualifiers.append((code, desc))
    qualifiers = tuple(qualifiers)

    # The remainder of the file is data, but we have to deal with the compressed
    # case very specially, as it's quite clever :-<.
    compress_mask = qualifiers_dict['C'][0]
    if sample_kind & compress_mask:
        # In the compressed case, the header is lying about the number of
        # samples; there are 4 less than it says there are.  Why 4 less?
        # Because there are two vectors of floats which are needed for
        # decompression and the data has been compressed into 2-byte features.
        # So it takes 4 samples of n_feats * 2 bytes to make the 2 * n_feats
        # vectors of floats (which are stored at the beginning of the data).
        n_samples -= 4
        n_feats_per_sample = sample_size / 2
        fmt = '>%df' % n_feats_per_sample
        # read the float vectors for decompression
        float_vec_size = n_feats_per_sample * 4
        raw = file.read(float_vec_size)
        data = unpack(fmt, raw)
        A = fromiter(data, dtype = float)
        raw = file.read(float_vec_size)
        data = unpack(fmt, raw)
        B = fromiter(data, dtype = float)
        assert A.shape == B.shape == (n_feats_per_sample,)

        ret = zeros((n_samples, n_feats_per_sample), dtype = float)

        fmt = '>%dh' % n_feats_per_sample
        for samp in xrange(n_samples):
            raw = file.read(sample_size)
            data = unpack(fmt, raw)
            assert len(data) == n_feats_per_sample
            assert all((type(d) == int for d in data))
            compressed = fromiter(data, dtype = int)
            ret[samp] = (compressed + B) / A
        
    else:  # not compressed, so data is 4-byte floats
        n_feats_per_sample = sample_size / 4
        ret = zeros((n_samples, n_feats_per_sample), dtype = float)
        fmt = '>%df' % n_feats_per_sample
        for samp in xrange(n_samples):
            raw = file.read(sample_size)
            data = unpack(fmt, raw)
            assert len(data) == n_feats_per_sample
            assert all((type(d) == float for d in data))
            ret[samp] = data

    # XXX At this point we've read all the data.  There's likely still a CRC
    # checksum at the end of the file, but I'm not going to try to deal with
    # that now.
    return (ret, (kind_tuple, qualifiers), sample_period)
    

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



