###########################################################################
#
# File:         audiodata.py
# Date:         02-Apr-2009
# Author:       Hugh Secker-Walker
# Description:  Interface for getting audio data from files or strings
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
Interfaces for getting audio data from files

"""

from __future__ import with_statement
from __future__ import division
import os
from ctypes import cdll, c_char_p, c_void_p, c_short, c_int, c_float, c_double, POINTER, cast, byref
from itertools import izip, islice
import contextlib
import cStringIO
from distutils import sysconfig
import numpy

import onyx
from onyx import builtin
from onyx.util import process


# Exceptions:
#
# note: we make them all appear in the module's globals() so that the logic in
# pp_wrapper is simple
ValueError = ValueError
class AudioException(onyx.OnyxException):
    """
    Baseclass for exceptions occuring while processing audio from external
    sources.

    >>> raise AudioException("something is wrong with the audio data")
    Traceback (most recent call last):
      ...
    AudioException: something is wrong with the audio data
    """

class AudioTypeError(AudioException, onyx.DataTypeError):
    """
    Exception raised when the type of the audio cannot be determined.

    >>> raise AudioTypeError("unknown audio type in file 'foobar.ima'")
    Traceback (most recent call last):
      ...
    AudioTypeError: unknown audio type in file 'foobar.ima'
    """

sndfilewrap_dl_name = onyx.pylib_build_path + os.sep + 'sndfilewrap' + sysconfig.get_config_var('SO')
#print 'sndfilewrap_dl_name:', repr(sndfilewrap_dl_name)
sndfilewrap = cdll.LoadLibrary(sndfilewrap_dl_name)

mpg123wrap_dl_name = onyx.pylib_build_path + os.sep + 'mpg123wrap' + sysconfig.get_config_var('SO')
mpg123wrap = cdll.LoadLibrary(mpg123wrap_dl_name)

c_int_p = POINTER(c_int)
c_int_pp = POINTER(c_int_p)
c_char_pp = POINTER(c_char_p)
c_void_pp = POINTER(c_void_p)
def int_pp():
    return c_int_pp(c_int_p())
def char_pp():
    return c_char_pp(c_char_p())
def void_pp():
    return c_void_pp(c_void_p())

def ctypes_pp_wrapper(free_func, *pp_args):
    """
    Returns a decorator to be used on a function that calls a C function in an
    Onyx-style shared library via Python's ctypes module.  The decorator
    itself returns a Context Manager that manages the output arguments to and
    from the C function.

    Such C functions must adhere to the signature and allocation idioms of an
    Onyx-style shared library.  These functions take input parameters that are
    strings, integers, pointers to arrays, etc, the C types of which the user
    has set up using the C function's argtypes attribute in the ctypes module.

    In the C function's signature the input parameters are followed by output
    parameters.  The output arguments are always pointers to pointers.  The
    returned Context Manager takes care of setting up the pointers to pointer
    for the C function.  The C function itself does the pointed-to pointer
    allocation.  The contents of the pointer(s) to pointer are made available as
    the target of the with statement.  When the with statement suite exits, the
    Context Manager takes care of calling a function in the shared library to
    free the memory that was allocated by the C function.

    The ctypes_pp_wrapper function takes one or more arguments.  The required
    first argument is a memory-freeing function in the shared library.  This
    function is will be used by the Context Manager to free memory that is
    allocated for the return arguments of the wrapped C function.  Each of the
    remaining arguments is a ctypes pointer-to-pointer type for the output type
    of the C function.

    When the decorated user function is called, the outer pointers will be
    instantiated, and the user's function will be called with an argument list
    that is the catenation of arguments supplied by the user and the pointer
    instances.  The C function will use the input values, and it will allocate
    the inner pointers of the output arguments, and it will fill in the output
    data.  If no errors occur it will return a NULL string pointer.

    In this case, the target of the with statement will be assigned the sequence
    of the contents of the ctypes pointer-to-pointer output arguments.  These
    ctypes are used in the body of the with statement.  Immutable output types
    are available as the value attribute of the corresponding target element;
    these values will persist after the with statement is exited.  The values in
    (mutable) output arrays must be used, e.g. copied, because the underlying
    data will be deallocated when the with statement is exited.

    The Onyx-style shared-library idiom is that if an error occured in the C
    function, the returned pointer will be a non-NULL string pointer.  The error
    string will be such that the first whitespace-separated token is the name of
    a Python exception, and the remainder of the string is an error message.  If
    the return value is non-NULL, the decorator will parse the return value, and
    it will construct and raise the appropriate Python error.  It will also free
    any memory that the C function allocated in the output pointer to points.

    Some usage examples should clarify the robust simplicity of using the
    generator returned by this function, as well as the idioms in the C shared
    library.
    """
    @contextlib.contextmanager
    def ptrs():
        pps = tuple(pp() for pp in pp_args)
        try:
            yield pps
        finally:
            for pp in pps:
                res = free_func(pp)
                assert res is None, str(res)
    def pp_wrapper(func):
        @contextlib.contextmanager
        def wrapper(*w_args):
            with ptrs() as pps:
                args = w_args + pps
                err = func(*args)
                if err is not None:
                    err = err.replace('\n', '  ').replace('\r', '  ')
                    exception_name, msg = err.split(None, 1)
                    exception = globals().get(exception_name)
                    raise exception(msg) if exception is not None else AudioException(err)
                yield tuple(pp.contents for pp in pps)
        return wrapper
    return pp_wrapper

def setup(sndfilewrap):

    sndfilewrap.free_ptr.restype = c_char_p
    # note: we're lying about the argtype; it should be c_void_pp, but ctypes
    # (correctly, I believe) won't automatically cast c_char_pp to c_void_pp; it
    # will cast to c_void_p, and the target C function correctly casts to void**
    sndfilewrap.free_ptr.argtypes = [c_void_p]

    sndfilewrap.get_api_info.restype = c_char_p
    sndfilewrap.get_api_info.argtypes = [c_char_pp, c_char_pp, c_char_pp]

    sndfilewrap.get_audio.restype = c_char_p
    sndfilewrap.get_audio.argtypes = [c_int, c_char_p, c_char_p, c_char_pp, c_char_pp, c_void_pp]    


    mpg123wrap.free_ptr.restype = c_char_p
    mpg123wrap.free_ptr.argtypes = [c_void_p]

    mpg123wrap.mpg123_init_wrap.restype = c_char_p
    mpg123wrap.mpg123_init_wrap.argtypes = []

    mpg123wrap.mpg123_exit_wrap.restype = c_char_p
    mpg123wrap.mpg123_exit_wrap.argtypes = []
    
    mpg123wrap.get_audio.restype = c_char_p
    mpg123wrap.get_audio.argtypes = [c_int, c_char_p, c_char_p, c_char_pp, c_char_pp, c_void_pp]    


    @ctypes_pp_wrapper(sndfilewrap.free_ptr, char_pp, char_pp, char_pp)
    def sndfilewrap_get_api_info(name, version, info):
        return sndfilewrap.get_api_info(name, version, info)
    with sndfilewrap_get_api_info() as ptrs:
        # get the returned strings
        name, version, info = tuple(ptr.value for ptr in ptrs)
    version = tuple(int(x) for x in version.split())
    return name, version, info

api_name, api_version, api_info = setup(sndfilewrap)
def get_api_info():
    """
    Get information about the audiodata interface and the underlying
    libsndfile library.

    Returns a triple, (api_name, version_tuple, info), where api_name is a
    string giving the name of the api, version_tuple is a pair of integers,
    (major, minor), giving the version number of the api, and info is a
    descriptive string that includes the api_name and version as well as a
    description from the underlying libsndfile api.

    >>> get_api_info() #doctest: +ELLIPSIS
    ('sndfilewrap', (0, 0), 'sndfilewrap-0.0 (using libsndfile-1.0...)')
    """
    return api_name, api_version, api_info


@ctypes_pp_wrapper(mpg123wrap.free_ptr)
def mpg123wrap_init():
    return mpg123wrap.mpg123_init_wrap()
with mpg123wrap_init() as ptrs:
    #print 'mpg123wrap_init: len(ptrs)', len(ptrs)
    pass

def parse_key_value(info_str):
    infoparts = info_str.split()
    for key, value in izip(islice(infoparts, 0, None, 2), islice(infoparts, 1, None, 2)):
        for typ in int, float:
            try:
                value = typ(value)
            except ValueError:
                continue
            else:
                break
        yield key, value
    
wave_c_formats = dict((
    # the set of items is dictated by the libsndfile read* API
    ('int16', c_short),
    ('int32', c_int),
    ('float32', c_float),
    ('float64', c_double),
    ))
wave_numpy_formats = dict((
    # numpy dtypes we will return
    ('int16', numpy.int16),
    ('int32', numpy.int32),
    ('float32', numpy.float32),
    ('float64', numpy.float64),
    ))
assert set(wave_numpy_formats) == set(wave_c_formats)

BYTE_BITS = 8
def bitsplit(bytes, bits):
    r"""
    Split a sequence of bytes into a sequence of non-negative integers.  The
    bits argument is a sequence of non-negative integers specifying the bit
    masks into which the bytes are split.  Conceptually, the bytes are arranged
    left to right, and then items composed from each successive set of the
    bytes' bits from left to right are placed into the result, where the number
    of bits in each set is given by the corresponding value in the bits
    argument.

    For instance, an MPEG version 1, Layer III audio frame has a four byte
    header.  The header has 13 fields of bits of lengths 11, 2, 2, 1, 4, 2, 1,
    1, 2, 2, 1, 1, 2.  To get the numerical values of these 13 fields, you would
    do the following:
    
    >>> mp3_header_bits = 11, 2, 2, 1, 4, 2, 1, 1, 2, 2, 1, 1, 2
    >>> assert sum(mp3_header_bits) == 32
    >>> mp3_header_bytes = sum(mp3_header_bits) // 8
    >>> assert mp3_header_bytes == 4
    >>> header = '\xff\xfb\x90d'  # a valid header, e.g. via open('audio.mp3', 'rb').read(4)
    >>> assert len(header) == mp3_header_bytes
    >>> fields = bitsplit(header, mp3_header_bits)
    >>> fields
    [2047, 3, 1, 1, 9, 0, 0, 0, 1, 2, 0, 1, 0]


    Now lets do some checks; in production code, failure of the checks in the
    asserts would mean it isn't a valid mpeg audio header.  See
    http://www.mpgedit.org/mpgedit/mpeg_format/mpeghdr.htm

    >>> sync, mpeg_id, layer, protection, bit_rate_index, sample_rate_index, has_padding, private, channel_mode, mode_extension, is_copyright, is_original, emphasis_index = fields
    >>> assert sync == 2047
    >>> assert mpeg_id != 1
    >>> 'mpeg_id', ('2.5', None, '2', '1')[mpeg_id]
    ('mpeg_id', '1')
    >>> assert layer != 0
    >>> 'layer', (None, 'III', 'II', 'I')[layer]
    ('layer', 'III')
    >>> 'protection', (True, False)[protection]
    ('protection', False)

    The following tables are only valid for MPEG version 1, Layer III

    >>> assert mpeg_id == 3 and layer == 1 # i.e. Version 1, Layer III
    >>> V1LIIIbitrate = None, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, None
    >>> V1LIIIsample_rate = 44100, 48000, 32000, None

    >>> assert bit_rate_index != 0x00 and bit_rate_index != 0xff
    >>> 'bit_rate', V1LIIIbitrate[bit_rate_index]
    ('bit_rate', 128)
    >>> assert sample_rate_index != 3
    >>> 'sample_rate', V1LIIIsample_rate[sample_rate_index]
    ('sample_rate', 44100)
    >>> 'padding', bool(has_padding)
    ('padding', False)

    >>> 'frame_bytes', 144 * V1LIIIbitrate[bit_rate_index] * 1000 // V1LIIIsample_rate[sample_rate_index] + has_padding 
    ('frame_bytes', 417)

    >>> 'private', private
    ('private', 0)
    >>> 'channel_mode', ('Stereo', 'JointStereo', 'DualChannel', 'SingleChannel')[channel_mode]
    ('channel_mode', 'JointStereo')
    >>> 'mode_extension', hex(mode_extension)
    ('mode_extension', '0x2')

    >>> 'copyright', bool(is_copyright)
    ('copyright', False)
    >>> 'original', bool(is_original)
    ('original', True)

    >>> assert emphasis_index != 2
    >>> 'emphasis', ('none', '50/15 msec', None, 'CCIT J.17')[emphasis_index]
    ('emphasis', 'none')
    """

    bits = tuple(bits)

    bad_bits = set(bit for bit in bits if not isinstance(bit, int) or bit < 0)
    if bad_bits:
        raise ValueError("expected each of bits to be non-negative int, but also got these: %s" % (' '.join(repr(bit) for bit in sorted(bad_bits)),))
    num_bits = len(bytes) * BYTE_BITS
    if sum(bits) != num_bits:
        raise ValueError("expected bits to sum to %d, got %d" % (num_bits, sum(bits)))

    all_bytes = 0L
    for byte in bytes:
        all_bytes = (all_bytes << BYTE_BITS) | ord(byte)

##     print hex(all_bytes)
##     print '  '.join(repr(bit) for bit in bits)
##     print '  '.join('1'*bit if bit else '-' for bit in bits)
##     print '  '.join(hex((1<<bit)-1) for bit in bits)

    res = list()
    for bit in reversed(bits):
        item = all_bytes & ((1 << bit) - 1)
        if int(item) == item:
            item = int(item)
        res.append(item)
        all_bytes >>= bit

    res.reverse()
    return res

MP3_HEADER_BITS = 11, 2, 2, 1, 4, 2, 1, 1, 2, 2, 1, 1, 2
assert sum(MP3_HEADER_BITS) == 32
MP3_HEADER_NUM_BYTES = sum(MP3_HEADER_BITS) // BYTE_BITS
assert MP3_HEADER_NUM_BYTES == 4
def mpeg_header(header):
    r"""
    Given a 4-byte header, return the sequence of 13 MPEG header fields if the
    four bytes appears to be a valid MPEG audio frame header, otherwise return
    False.

    See http://www.mpgedit.org/mpgedit/mpeg_format/mpeghdr.htm

    >>> mpeg_header('\xf1\xfa\x90d')
    False
    >>> mpeg_header('\xff\xfa\x90d')
    [2047, 3, 1, 0, 9, 0, 0, 0, 1, 2, 0, 1, 0]
    """
    if len(header) != MP3_HEADER_NUM_BYTES:
        raise ValueError("expected header to be %d bytes, got %d" % (MP3_HEADER_NUM_BYTES, len(header)))
    fields = bitsplit(header, MP3_HEADER_BITS)
    sync, mpeg_id, layer_id, protection, bit_rate_index, sample_rate_index, has_padding, private, channel_mode, mode_extension, is_copyright, is_original, emphasis_index = fields
    return (True
            and sync == 0x7ff # 2047
            and mpeg_id != 0 # excludes Version 2.5
            and mpeg_id != 1
            and layer_id != 0
            and bit_rate_index != 0x0 # excludes 'free' sample_rate_index
            and bit_rate_index != 0xf
            and sample_rate_index != 3
            and emphasis_index != 2
            and fields
            )

def get_file_info(filename):
    """
    Get information about the audio data in a file.

    Returns an attrdict with the following attributes:
        ======================== ====  ============================ =============
        Attribute Name           Type  Description                  Example Value
        ======================== ====  ============================ =============
        file_item_bytes          int   size of each item in bytes   2
        file_item_coding         str   coding of each item in file  'int16'
        file_name                str   name given to access file    'zero.wav'
        file_name_full           str   full name of file            '/Users/johndoe/azure/py/onyx/audio/zero.wav'
        file_num_bytes           int   size of file in bytes        512314
        file_num_channels        int   number of audio channels     2
        file_num_items           int   number of items in file      256128
        file_num_samples         int   number of samples in file    128064
        file_sample_rate         int   samples per second           44100
        file_sndfile_extension   str   nominal extension            'wav'
        file_sndfile_format      str   libsndfile's format          '0x10002'
        file_sndfile_type        str   libsndfile's type info       'Signed 16 bit PCM WAV (Microsoft)'
        ======================== ====  ============================ =============

    >>> module_dir, module_name = os.path.split(__file__)

    >>> info1 = get_file_info(os.path.join(module_dir, 'zero.wav'))
    >>> for key in sorted(info1): print "%-24s  %r" % (key, info1[key]) #doctest: +ELLIPSIS
    file_item_bytes           2
    file_item_coding          'int16'
    file_name                 '...zero.wav'
    file_name_full            '/.../py/onyx/audio/zero.wav'
    file_num_bytes            512314
    file_num_channels         2
    file_num_items            256128
    file_num_samples          128064
    file_sample_rate          44100
    file_sndfile_extension    'wav'
    file_sndfile_format       '0x10002'
    file_sndfile_type         'Signed 16 bit PCM WAV (Microsoft)'

    >>> info2 = get_file_info(os.path.join(module_dir, 'zero.sph'))
    >>> for key in sorted(info2): print "%-24s  %r" % (key, info2[key]) #doctest: +ELLIPSIS
    file_item_bytes           2
    file_item_coding          'int16'
    file_name                 '...zero.sph'
    file_name_full            '/.../py/onyx/audio/zero.sph'
    file_num_bytes            513280
    file_num_channels         2
    file_num_items            256128
    file_num_samples          128064
    file_sample_rate          44100
    file_sndfile_extension    'sph'
    file_sndfile_format       ''
    file_sndfile_type         'pcm SPH (NIST Sphere)'

    
    This is an incorrect file in that its header doesn't match the data.
    Unfortunately, sph2pipe doesn't notice, so we impose a blunt check.
    See below where libsndfile does notice the problem.

    >>> info3 = get_file_info(os.path.join(module_dir, 'problem.sph')) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    AudioTypeError: unhandled sample_n_bytes, 4, in NIST Sphere file '/.../problem.sph'


    Test mp3 info since we know it's using a separate library under the covers

    >>> info5 = get_file_info(os.path.join(module_dir, 'seven.mp3'))
    >>> for key in sorted(info5): print "%-24s  %r" % (key, info5[key]) #doctest: +ELLIPSIS
    file_item_bytes           -1
    file_item_coding          'int16'
    file_name                 '...seven.mp3'
    file_name_full            '/.../py/onyx/audio/seven.mp3'
    file_num_bytes            14192
    file_num_channels         2
    file_num_items            56542
    file_num_samples          28271
    file_sample_rate          44100
    file_sndfile_extension    'mp3'
    file_sndfile_format       '0xd0'
    file_sndfile_type         'MPEG_Version 1.0  Audio_Layer 3  Channel_Mode joint_stereo  BitrateKbps 128'

    Grrrr, with mpg123 version 1.7.0, problem.mp3 would silently give a bogus
    sample rate.  As of mpg123 version 1.9.0, mpg123 logs an error message to
    stderr, but returns a non-zero sample rate and a somewhat bogus info... so
    we skip both the old test and the new listing of the bogus info.

    >>> info6 = get_file_info(os.path.join(module_dir, 'problem.mp3')) #doctest: +ELLIPSIS +SKIP
    Traceback (most recent call last):
      ...
    AudioTypeError: .../cpp/liveaudio/mpg123wrap.c:... get_audio(): invalid sample rate 0 for file '...problem.mp3'

    >>> for key in sorted(info6): print "%-24s  %r" % (key, info6[key]) #doctest: +ELLIPSIS +SKIP
    file_item_bytes           -1
    file_item_coding          'int16'
    file_name                 '...problem.mp3'
    file_name_full            '/.../py/onyx/audio/problem.mp3'
    file_num_bytes            512318
    file_num_channels         2
    file_num_items            1536
    file_num_samples          768
    file_sample_rate          44100
    file_sndfile_extension    'mp3'
    file_sndfile_format       '0xd0'
    file_sndfile_type         'MPEG_Version 1.0  Audio_Layer 1  Channel_Mode stereo  BitrateKbps 0'

    """
    file_info, audio_info, wave = _get_file_audio(filename, None)
    assert wave is None
    return file_info
    
def get_file_audio(filename, wave_format):
    """
    Get audio data from a file.  Reads all the audio data from filename and
    converts it to the requested wave_format which must be one of 'int16',
    'int32', 'float32', or 'float64'.

    XXX needs more about the conversion and normalization issues....

    Returns a triple, (file_info, audio_info, wave_data), where file_info is an
    attrdict with information about the data as it appears in the file,
    audio_info is an attrdict with information about the data as it appears in
    the wave_data, and wave_data is a two-dimensional numpy array containing the
    one or more channels of wave samples from the file.

    The file_info will be equal to the attrdict that would be returned by get_file_info().

    >>> module_dir, module_name = os.path.split(__file__)
    >>> zero_wav = os.path.join(module_dir, 'zero.wav')
    >>> zero_sph = os.path.join(module_dir, 'zero.sph')

    Get the audio from a file

    >>> file_info1, audio_info1, wave_data1 = get_file_audio(zero_wav, 'int32')

    Look at the file info

    >>> for key in sorted(file_info1): print "%-24s  %r" % (key, file_info1[key]) #doctest: +ELLIPSIS
    file_item_bytes           2
    file_item_coding          'int16'
    file_name                 '...zero.wav'
    file_name_full            '/.../py/onyx/audio/zero.wav'
    file_num_bytes            512314
    file_num_channels         2
    file_num_items            256128
    file_num_samples          128064
    file_sample_rate          44100
    file_sndfile_extension    'wav'
    file_sndfile_format       '0x10002'
    file_sndfile_type         'Signed 16 bit PCM WAV (Microsoft)'

    It's the same as what get_file_info() returns

    >>> file_info1 == get_file_info(zero_wav)
    True

    Look at the audio info

    >>> for key in sorted(audio_info1): print "%-24s  %r" % (key, audio_info1[key])
    audio_item_bytes          4
    audio_item_coding         'int32'
    audio_num_bytes           1024512
    audio_num_channels        2
    audio_num_items           256128
    audio_num_samples         128064
    audio_sample_rate         44100

    Show the sets of keys in file_info and audio_info that differ only in their
    suffix and have the same values.

    >>> all(file_info1['file_' + suffix] == audio_info1['audio_' + suffix] for suffix in ('num_channels', 'num_items', 'num_samples', 'sample_rate'))
    True

    Look at the int32 data

    >>> wave_data1.shape
    (2, 128064)
    >>> wave_data1.itemsize
    4
    >>> wave_data1.min(), wave_data1.max()
    (-645136384, 673447936)
    >>> print str(wave_data1)
    [[       0   -65536  -196608 ..., 13303808 13828096 13041664]
     [       0        0  -196608 ..., 14745600 14745600 14483456]]

    Look at int16 version of the data

    >>> _, _, wave_data1a = get_file_audio(zero_wav, 'int16')
    >>> wave_data1a.shape
    (2, 128064)
    >>> wave_data1a.itemsize
    2
    >>> wave_data1a.min(), wave_data1a.max()
    (-9844, 10276)
    >>> wave_data1a
    array([[  0,  -1,  -3, ..., 203, 211, 199],
           [  0,   0,  -3, ..., 225, 225, 221]], dtype=int16)
    

    Look at another file that has a different internal format, but the same audio_info and wave_data

    >>> file_info2, audio_info2, wave_data2 = get_file_audio(zero_sph, 'int32')
    >>> file_info2 == file_info1
    False
    >>> audio_info2 == audio_info1
    True
    >>> (wave_data2 == wave_data1).all()
    True

    Show how wave_format 'int16' and 'int32' return differently scaled data

    >>> file_info3, audio_info3, wave_data3 = get_file_audio(zero_sph, 'int16')
    >>> wave_data3.sum() << 16 == wave_data2.sum()
    True
    
    >>> (numpy.array(wave_data3, dtype=numpy.int32) << 16 == wave_data2).all()
    True

    Show that wave_format 'float32' and 'float64' return equal data

    >>> file_info4, audio_info4, wave_data4 = get_file_audio(zero_wav, 'float32')
    >>> file_info5, audio_info5, wave_data5 = get_file_audio(zero_sph, 'float64')
    >>> (wave_data5 == wave_data4).all()
    True

    Test mp3 reading since we know it's using a separate library under the covers
    
    >>> seven_mp3 = os.path.join(module_dir, 'seven.mp3')
    >>> file_info6, audio_info6, wave_data6 = get_file_audio(seven_mp3, 'int16')

    >>> for key in sorted(file_info6): print "%-24s  %r" % (key, file_info6[key]) #doctest: +ELLIPSIS
    file_item_bytes           -1
    file_item_coding          'int16'
    file_name                 '...seven.mp3'
    file_name_full            '/.../py/onyx/audio/seven.mp3'
    file_num_bytes            14192
    file_num_channels         2
    file_num_items            56542
    file_num_samples          28271
    file_sample_rate          44100
    file_sndfile_extension    'mp3'
    file_sndfile_format       '0xd0'
    file_sndfile_type         'MPEG_Version 1.0  Audio_Layer 3  Channel_Mode joint_stereo  BitrateKbps 128'

    >>> file_info6 == get_file_info(seven_mp3)
    True

    >>> for key in sorted(audio_info6): print "%-24s  %r" % (key, audio_info6[key])
    audio_item_bytes          2
    audio_item_coding         'int16'
    audio_num_bytes           113084
    audio_num_channels        2
    audio_num_items           56542
    audio_num_samples         28271
    audio_sample_rate         44100

    >>> wave_data6.shape
    (2, 28271)
    >>> wave_data6.itemsize
    2
    >>> wave_data6.min(), wave_data6.max() #doctest: +ELLIPSIS
    (-306..., 250...)
    >>> wave_data6
    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0]], dtype=int16)

    Note that for mp3 we can't assert that the int32 data is just a 16 bit shift
    from the int16 data.

    >>> file_info7, audio_info7, wave_data7 = get_file_audio(seven_mp3, 'int32')
    >>> file_info7 == file_info6
    True
    >>> for key in sorted(audio_info7): print "%-24s  %r" % (key, audio_info7[key])
    audio_item_bytes          4
    audio_item_coding         'int32'
    audio_num_bytes           226168
    audio_num_channels        2
    audio_num_items           56542
    audio_num_samples         28271
    audio_sample_rate         44100

    >>> wave_data7.shape
    (2, 28271)
    >>> wave_data7.itemsize
    4
    >>> wave_data7.min(), wave_data7.max() #doctest: +ELLIPSIS
    (-2007331..., 16422...)
    >>> print str(wave_data7)  #doctest: +ELLIPSIS
    [[   0    0    0 ...,  48...  574  278]
     [   0    0    0 ..., -48... -574 -278]]

    >>> file_info8, audio_info8, wave_data8 = get_file_audio(seven_mp3, 'float32')
    >>> file_info8 == file_info6
    True
    >>> for key in sorted(audio_info8): print "%-24s  %r" % (key, audio_info8[key])
    audio_item_bytes          4
    audio_item_coding         'float32'
    audio_num_bytes           226168
    audio_num_channels        2
    audio_num_items           56542
    audio_num_samples         28271
    audio_sample_rate         44100

    >>> wave_data8.shape
    (2, 28271)
    >>> wave_data8.itemsize
    4
    >>> wave_data8.min(), wave_data8.max() #doctest: +ELLIPSIS
    (-0.93473..., 0.7647...)
    >>> wave_data8 #doctest: +ELLIPSIS
    array([[  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
              2.25688...e-07,   2.67382...e-07,   1.29675...e-07],
           [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00, ...,
             -2.25688...e-07,  -2.67382...e-07,  -1.29675...e-07]], dtype=float32)

    
    Sometimes you lose
    
    >>> file_info9, audio_info9, wave_data9 = get_file_audio(seven_mp3, 'float64') #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: .../cpp/liveaudio/mpg123wrap.c:... get_audio(): unsupported format 'float64' for file '...seven.mp3'


    Again, we skip fiddling around with the problem file because different
    versions of mpg123 handle it differently.

    >>> file_info10, audio_info10, wave_data10 = get_file_audio(os.path.join(module_dir, 'problem.mp3'), 'int16') #doctest: +SKIP
    >>> for key in sorted(file_info10): print "%-24s  %r" % (key, file_info10[key]) #doctest: +ELLIPSIS +SKIP
    file_item_bytes           -1
    file_item_coding          'int16'
    file_name                 'problem.mp3'
    file_name_full            '/.../py/onyx/audio/problem.mp3'
    file_num_bytes            512318
    file_num_channels         2
    file_num_items            1536
    file_num_samples          768
    file_sample_rate          44100
    file_sndfile_extension    'mp3'
    file_sndfile_format       '0xd0'
    file_sndfile_type         'MPEG_Version 1.0  Audio_Layer 1  Channel_Mode stereo  BitrateKbps 0'
    >>> for key in sorted(audio_info10): print "%-24s  %r" % (key, audio_info10[key]) #doctest: +SKIP
    audio_item_bytes          2
    audio_item_coding         'int16'
    audio_num_bytes           3072
    audio_num_channels        2
    audio_num_items           1536
    audio_num_samples         768
    audio_sample_rate         44100

    """
    file_info, audio_info, wave = audio = _get_file_audio(filename, wave_format)
    assert wave is not None
    return audio

def get_str_info(str_data, name):
    """
    BROKEN -- libsndfile 1.0.19 doesn't handle pipes correctly, mpg123 will be tricky to finesse on a pipe

    Get information about the audio data in a string, where the string is the
    contents of an audio file.

    Returns an attrdict of information about the data contained in the string.
    See get_file_info() for a detailed description of what is returned.

    >>> module_dir, module_name = os.path.split(__file__)
    >>> filename = 'zero.sph'
    >>> with open(os.path.join(module_dir, filename), 'rb') as infile:
    ...   str_data = infile.read()

    >>> info = get_str_info(str_data, filename)
    Traceback (most recent call last):
      ...
    NotImplementedError: string-based audio is broken

    >>> for key in sorted(info): print "%-24s  %r" % (key, info[key]) #doctest: +SKIP
    file_item_bytes           2
    file_item_coding          'int16'
    file_name                 'zero.sph'
    file_name_full            '<from_string zero.sph>'
    file_num_bytes            513280
    file_num_channels         2
    file_num_items            -514
    file_num_samples          -257
    file_sample_rate          44100
    file_sndfile_extension    'wav'
    file_sndfile_format       '0x10070002'
    file_sndfile_type         'Signed 16 bit PCM WAV (NIST Sphere)'
    """
    raise NotImplementedError("string-based audio is broken")

    with str_as_fd(str_data) as pread:
        file_info, audio_info, wave = _get_audio_fd(pread, len(str_data), name, None, "<from_string %s>" % (name,))
    assert wave is None
    return file_info

def get_str_audio(str_data, name, wave_format):
    """
    BROKEN -- libsndfile 1.0.19 doesn't handle pipes correctly, mpg123 will be tricky to finesse on a pipe

    Get information about the audio data in a string, where the string is the
    contents of an audio file.

    Returns an attrdict of information about the data contained in the string.
    See get_file_audio() for a detailed description of what is returned.

    >>> module_dir, module_name = os.path.split(__file__)
    >>> filename = 'zero.sph'
    >>> with open(os.path.join(module_dir, filename), 'rb') as infile:
    ...   str_data = infile.read()

    >>> file_info, audio_info, wave_data = get_str_audio(str_data, filename, 'int16')
    Traceback (most recent call last):
      ...
    NotImplementedError: string-based audio is broken

    >>> for key in sorted(file_info): print "%-24s  %r" % (key, file_info[key]) #doctest: +SKIP

    >>> for key in sorted(audio_info): print "%-24s  %r" % (key, audio_info[key]) #doctest: +SKIP

    >>> wave_data #doctest: +SKIP
    """
    raise NotImplementedError("string-based audio is broken")

    with str_as_fd(str_data) as pread:
        file_info, audio_info, wave = audio = _get_audio_fd(pread, len(str_data), name, wave_format, "<from_string %s>" % (name,))
    assert wave is not None
    return audio


def read_fd_strict(fd, nbytes, name):
    """
    Read nbytes from file descriptor.  Raises EOFError if nbytes cannot be
    obtained, using name in the error message.
    """
    got = os.read(fd, nbytes)
    if len(got) == nbytes:
        return got
    gotten = [got]
    ngotten = len(got)
    while ngotten < nbytes:
        got = os.read(fd, nbytes - ngotten)
        if len(got) == 0:
            raise EOFError("unexpected EOF on file '%s'" % (name,))
        gotten.append(got)
        ngotten += len(got)
    ret = ''.join(gotten)
    assert len(ret) == nbytes
    return ret


def _get_file_audio(filename, wave_format=None):
    """
    Read audio info from the file at filename using libsndfile.  Optional
    wave_format string gives the Numpy dtype format for the returned wave data.

    If wave_format is given it must be one of 'int16', 'int32', 'float32', or
    'float64' and the audio data is read from the file.
    
    Returns a pair, (info, wave_data), where info is an attrdict containing
    information about the audio data.  If wave_format was not given then
    wave_data is None.  If wave_format was given then the returned wave_data
    will be a Numpy array of the wave_format dtype with shape
    (info.audio_num_channels, info.audio_num_samples).

    The info attrdict contains the following items (those marked with a asterisk
    '*' are only present if there is wave data, that is if wave_format, and thus
    wave_data, are not None; those marked with a plus-sign '+' are not present
    when get_audio_fd() is called):

       Attribute                    Type  Description
       info.audio_sample_rate        int   audio frame rate, e.g. 8000
       info.audio_num_channels      int   number of audio channels, e.g. 2
       info.audio_num_samples        int   number of audio frames, e.g. 4867872
       info.audio_num_items         int   number of audio items, e.g. 9735744
     * info.audio_item_bytes        int   size of each item in wave_data, e.g. 2
     * info.audio_num_bytes         int   number of bytes in wave_data object, e.g. 19471488
     * info.audio_item_coding       str   Numpy dtype (wave_format) of items in wave_data, e.g. 'int16'
       info.file_item_bytes         int   size of items in file, -1 for variable, -2 for unknown, e.g. 1
     + info.file_num_bytes          int   number of bytes in the file, e.g. 9736768
       info.file_item_coding        str   coding used in the file, e.g. 'ulaw'
       info.file_sndfile_extension  str   nominal filename extension for the audio type, e.g. 'wav'
       info.file_sndfile_format     str   hex represenation of libsndfile's format, e.g. '0x10070010'
       info.file_sndfile_type       str   nominal encoding and file type, e.g. 'U-Law WAV (NIST Sphere)'

    Raises ValueError if there are problems with wave_format, the file, its
    contents, memory allocation, etc.
    """

    if not (wave_format is None or wave_format in wave_c_formats):
        raise ValueError("expected wave_format to be one of (None %s), but got %r"
                         % (' '.join(repr(x) for x in sorted(wave_c_formats)), wave_format))

    filename_abs = os.path.abspath(filename)
    with open(filename_abs, 'rb') as audio_file:
        file_num_bytes = os.path.getsize(filename_abs)
        if int(file_num_bytes) == file_num_bytes:
            file_num_bytes = int(file_num_bytes)

        fileno = audio_file.fileno()
        header_size = 4
        header = read_fd_strict(fileno, header_size, filename_abs)
        os.lseek(fileno, 0, 0)

        if len(header) < header_size:
            raise ValueError("file too short to determine its audio type, '%s'" % (filename_abs,))
        assert len(header) == header_size

        if header == 'NIST':
            file_info, audio_info, wave = _get_sphere_fd(fileno, file_num_bytes, filename, wave_format, filename_abs)
        else:
            is_mpeg = bool(mpeg_header(header))
            file_info, audio_info, wave = _get_audio_fd(fileno, file_num_bytes, filename, wave_format, filename_abs, is_mpeg)

        audio_info.audio_num_items = audio_info.audio_num_samples * audio_info.audio_num_channels

##         #print 'file_info:', ' '.join(sorted(file_info))
        file_info.file_num_bytes = file_num_bytes
        file_info.file_name = filename
        file_info.file_name_full = filename_abs
        file_info.update((key.replace('audio', 'file'), value) for key, value in audio_info.iteritems())

        if wave is not None:
            assert wave.size == audio_info.audio_num_items
            audio_info.audio_num_bytes = wave.nbytes
            audio_info.audio_item_bytes = wave.itemsize
            audio_info.audio_item_coding = wave.dtype.name

        return file_info, audio_info, wave

def _get_sphere_fd(fileno, file_num_bytes, filename, wave_format, filename_abs):
    """
    Low-level reading of NIST Sphere audio data.
    
    >>> module_dir, module_name = os.path.split(__file__)
    >>> zero_sph = os.path.join(module_dir, 'zero.sph')
    >>> shorten_sph = os.path.join(module_dir, 'shorten.sph')

    >>> with open(zero_sph, 'rb') as audio_file:
    ...   file_info, audio_info, wave = _get_sphere_fd(audio_file.fileno(), os.path.getsize(zero_sph), zero_sph, 'int16', os.path.abspath(zero_sph))

    >>> for key in sorted(file_info): print "%-24s  %r" % (key, file_info[key])
    file_item_bytes           2
    file_item_coding          'int16'
    file_sndfile_extension    'sph'
    file_sndfile_format       ''
    file_sndfile_type         'pcm SPH (NIST Sphere)'

    >>> for key in sorted(audio_info): print "%-24s  %r" % (key, audio_info[key])
    audio_num_channels        2
    audio_num_samples         128064
    audio_sample_rate         44100

    >>> print str(wave)
    [[  0  -1  -3 ..., 203 211 199]
     [  0   0  -3 ..., 225 225 221]]


    >>> with open(shorten_sph, 'rb') as audio_file:
    ...   file_info2, audio_info2, wave2 = _get_sphere_fd(audio_file.fileno(), os.path.getsize(shorten_sph), shorten_sph, 'int16', os.path.abspath(shorten_sph))

    >>> for key in sorted(file_info2): print "%-24s  %r" % (key, file_info2[key])
    file_item_bytes           2
    file_item_coding          'int16'
    file_sndfile_extension    'sph'
    file_sndfile_format       ''
    file_sndfile_type         'pcm,embedded-shorten-v2.00 SPH (NIST Sphere)'

    >>> for key in sorted(audio_info2): print "%-24s  %r" % (key, audio_info2[key])
    audio_num_channels        1
    audio_num_samples         37120
    audio_sample_rate         20000

    >>> print str(wave2)
    [[-1  1  1 ..., -4 -8 -5]]
    """

    assert wave_format is None or wave_format in wave_c_formats

    # As of 2009-04-20 see the following for NIST's underspecified
    # format description:
    #   http://ftp.cwi.nl/audio/NIST-SPHERE
    #   http://www.ldc.upenn.edu/Catalog/docs/LDC93S5/WAV_SPEC.TXT

    nist_1a = 'NIST_1A'
    header1 = read_fd_strict(fileno, 128, filename_abs)
    if not header1.startswith(nist_1a):
        raise AudioTypeError("did not find %r in header of purported NIST Sphere file %r" % (nist_1a, filename_abs))
    nist, header_size, rest = header1.split(None, 2)
    assert nist == nist_1a

    header_size = int(header_size)
    rest += read_fd_strict(fileno, header_size - len(header1), filename_abs)

    # For now, we require the following fields:
    #   sample_count -i 128064
    #   sample_n_bytes -i 2
    #   channel_count -i 2
    #   sample_byte_format -s2 01
    #   sample_rate -i 44100
    #   sample_coding -s3 pcm
    info = builtin.attrdict()
    for line in cStringIO.StringIO(rest):
        parts = line.split()
        if not parts or parts[0][0] == ';': continue
        if parts[0] == 'end_head': break
        if len(parts) < 3:
            raise AudioTypeError("expected at least three white-space-separated fields in NIST header line %r in file %r" % (line.strip(), filename_abs))
        field_name, field_type, field_value = line.split(None, 3)
        #print field_name, field_type, field_value
        if field_type in ('-i', '-r'):
            field_value, _, _ = field_value.partition(';')
            info[field_name] = (int if field_type == '-i' else float)(field_value)
        elif field_type.startswith('-s'):
            # here we do a stricter interpretation of the spec
            prefix_len = len(field_name + ' ' + field_type + ' ')
            str_len = int(field_type[2:])
            info[field_name] = line[prefix_len:prefix_len+str_len]
        else:
            raise (AudioTypeError, "unhandled field_type %r for field_name %r" % (field_type, field_name))

    missing = set(('sample_count', 'sample_n_bytes', 'channel_count', 'sample_byte_format', 'sample_rate', 'sample_coding')) - set(info)
    if missing:
        raise AudioTypeError("missing required header fields (%s) in NIST Sphere file %r" % (', '.join(sorted(missing)), filename_abs))

    # this is a blunt check against bogus data
    if info.sample_n_bytes > 2:
        raise AudioTypeError("unhandled sample_n_bytes, %d, in NIST Sphere file %r" % (info.sample_n_bytes, filename_abs))

    audio_info = builtin.attrdict((('audio_num_channels', info.channel_count),
                                   ('audio_num_samples', info.sample_count),
                                   ('audio_sample_rate', info.sample_rate)))
    check_positive_ints(audio_info, filename_abs)
    file_info = builtin.attrdict((('file_item_bytes', info.sample_n_bytes),
                                  ('file_item_coding', ('int' + str(8*info.sample_n_bytes)) if info.sample_coding.lower().startswith('pcm') else info.sample_coding[:4].lower()),
                                  ('file_sndfile_extension', 'sph'),
                                  ('file_sndfile_format', ''),
                                  ('file_sndfile_type', info.sample_coding + ' SPH (NIST Sphere)'),
                                  ))
    check_positive_ints(file_info, filename_abs)

    if wave_format is not None:
        args = 'sph2pipe', '-p', '-f', 'raw', filename_abs
        stdout, stderr, cmd = process.subprocess(args)
        if stderr:
            raise onyx.ExternalError("unexpected stderr from command '%s': '%s'" % (cmd, stderr.strip()))
        num_audio_bytes = audio_info.audio_num_channels * audio_info.audio_num_samples * numpy.int16().itemsize
        if len(stdout) != num_audio_bytes:
            raise onyx.DataFormatError("expected %d bytes of audio data, got %d" % (num_audio_bytes, len(stdout)))             
        wave = numpy.fromstring(stdout, dtype=numpy.int16)
        assert wave.shape == (audio_info.audio_num_channels * audio_info.audio_num_samples,)
        # reshape it, etc; then construct a new ndarray
        wave = wave.reshape((-1, audio_info.audio_num_channels)).transpose()
        wave = numpy.array(wave, dtype=wave_numpy_formats[wave_format])
        # do scaling the same way libsndfile does
        if wave_format == 'int32':
            wave <<= 16
        elif wave_format in ('float32', 'float64'):
            wave *= (1 / (1 << 15))
        assert wave.shape == (audio_info.audio_num_channels, audio_info.audio_num_samples)
    else:
        wave = None

    return file_info, audio_info, wave
        

@contextlib.contextmanager
def str_as_fd(str_data):
    """
    Takes a string and returns a context manager.  The target of the context
    manager is a file descriptor (readable end of a pipe) from which the string
    can be read.  The context manager closes the file descriptor as the context
    is exited, so the user should not close it.
    """
    if not isinstance(str_data, str):
        raise TypeError("expected a %s, got a %s" % (str.__name__, type(str_data).__name__))
    import os, threading, errno
    pread, pwrite = os.pipe()
    def str_as_fd_writer():
        try:
            os.write(pwrite, str_data)
        except OSError, err:
            # broken pipe (EPIPE) happens if user didn't consume everything,
            # which is OK; anything else gets re-raised
            if err.errno != errno.EPIPE:
                raise
        finally:
            os.close(pwrite)
    thread = threading.Thread(target=str_as_fd_writer)
    thread.setDaemon(True)
    thread.start()
    yield pread
    os.close(pread)
    thread.join()

def _test_str_as_fd():
    """
    Test str_as_fd() on its own doc string, reading it in 12 byte chunks.
    
    >>> res = list()
    >>> with str_as_fd(str_as_fd.__doc__) as fd:
    ...   while True:
    ...     x = os.read(fd, 12)
    ...     if not x: break
    ...     res.append(x)
    >>> len(res) > 20
    True
    >>> max(len(x) for x in res) == 12
    True
    >>> ''.join(res) == str_as_fd.__doc__
    True
    """

def get_audio_str(str_data, name, wave_format=None):
    with str_as_fd(str_data) as pread:
        return get_audio_fd(pread, len(str_data), name, wave_format)

def check_positive_ints(mapping, name, exclusions=()):
    for key, value in mapping.iteritems():
        if isinstance(value, int) and value <= 0 and key not in exclusions:
            raise ValueError("expected a positive value for %s from '%s', got %d" % (key, name, value))
    
@ctypes_pp_wrapper(sndfilewrap.free_ptr, char_pp, char_pp, void_pp)
def _sndfilewrap_get_audio_fd(fd, name, wave_format, info, casual, wave):
    return sndfilewrap.get_audio(fd, name, wave_format, info, casual, wave)
    
@ctypes_pp_wrapper(mpg123wrap.free_ptr, char_pp, char_pp, void_pp)
def _mpg123wrap_get_audio_fd(fd, name, wave_format, info, casual, wave):
    return mpg123wrap.get_audio(fd, name, wave_format, info, casual, wave)

def _get_audio_fd(fd, file_num_bytes, name, wave_format, full_name=None, is_mpeg=None):
    """
    Low-level reading of audio from libsndfile and mpg123 audio libraries.
    
    Test a correct error from libsndfile that sph2pipe does not catch; this is a
    small exercise of the error handling chain
    
    >>> module_dir, module_name = os.path.split(__file__)

    >>> problem_sph = os.path.join(module_dir, 'problem.sph')
    >>> with open(problem_sph, 'rb') as infile:
    ...   info3 = _get_audio_fd(infile.fileno(), os.path.getsize(problem_sph), problem_sph, None, problem_sph, False) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    AudioTypeError: .../cpp/liveaudio/sndfilewrap.c:... get_sndfile(): unexpected NULL from sf_open_fd: maybe not a valid sound file: '...problem.sph': SFC_GET_LOG_INFO: Length : 1104  psf->bytewidth (4) != bytes (2)  

    Also test that libsndfile notices that it cannot deal with shorten encoding;
    also a small exercise of the error handling chain

    >>> shorten_sph = os.path.join(module_dir, 'shorten.sph')
    >>> with open(shorten_sph, 'rb') as infile:
    ...   info3 = _get_audio_fd(infile.fileno(), os.path.getsize(shorten_sph), shorten_sph, None, shorten_sph, False) #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    AudioTypeError: .../cpp/liveaudio/sndfilewrap.c:... get_sndfile(): unexpected NULL from sf_open_fd: maybe not a valid sound file: '...shorten.sph': SFC_GET_LOG_INFO: Length : 28357  *** Unknown encoding : pcm,embedded-shorten-v2.00  
    """
    assert fd >= 0, str(fd)
    assert wave_format is None or wave_format in wave_c_formats

    getter = _sndfilewrap_get_audio_fd if not is_mpeg else _mpg123wrap_get_audio_fd
    with getter(fd, name, wave_format) as ptrs:

        # get the returned strings
        info_str, casual, _ = tuple(ptr.value for ptr in ptrs)

        # populate info
        audio_info = builtin.attrdict((key, value) for key, value in parse_key_value(info_str) if key.startswith('audio'))
        check_positive_ints(audio_info, name)

        file_info = builtin.attrdict((key, value) for key, value in parse_key_value(info_str) if key.startswith('file'))
        file_info.file_sndfile_type = casual
        check_positive_ints(file_info, name, exclusions=('file_item_bytes',))

        # deal with the wave data at ptrs[-1], if any
        wave_ptr = ptrs[-1]
        assert (wave_ptr.value is None) == (wave_format is None)
        if wave_format is not None:
            # cast the void** data in wave_ptr to a C-array of the correct
            # length; then have Numpy use that data in a ctypeslib.as_array; get
            # a reshaped and transposed view (cheap); then copy to a bona-fide
            # Numpy array
            from numpy import ctypeslib
            # create a C type for the returned data
            c_array_t = wave_c_formats[wave_format] * audio_info.audio_num_samples * audio_info.audio_num_channels
            # use the returned memory
            wave = ctypeslib.as_array(cast(wave_ptr, POINTER(c_array_t)).contents)
            # reshape it, etc; then copy to a new ndarray
            wave = wave.reshape((-1, audio_info.audio_num_channels)).transpose().copy()
            # at this point we're done with the ptrs
            assert wave.shape == (audio_info.audio_num_channels, audio_info.audio_num_samples)
        else:
            wave = None

    return file_info, audio_info, wave


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
