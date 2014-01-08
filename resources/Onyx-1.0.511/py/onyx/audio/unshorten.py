###########################################################################
#
# File:         unshorten.py
# Date:         21-Apr-2009
# Author:       Hugh Secker-Walker
# Description:  Reference implementation of the unshorten decoder
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
Code for parsing the bitstream of a shorten encoded file.

This is a start on a reference for a replacement for Tony Robinson's
license-restricted code for unshortening shorten files.  To date, it appears to
be able to decode the bit-stream and generate the arrays of numbers involved,
but it doesn't yet implement the numerical work needed to reconstruct the
signals.  Once that is implemented, we could work with Erik de Castro Lopo of
libsndfile to make a C version for use in libsndfile.

See shorten_x.c which appears in the sph2pipe package from LDC.
"""
from __future__ import with_statement
import os
import cStringIO

CHAR_BIT = 8
char_bit_masks = tuple((1 << (7 - i)) for i in xrange(CHAR_BIT))

def file_bit_gen(infile):
    """
    Return a generator that yields the bits, 0 or 1, from the contents of each
    byte of infile, from MSB to LSB.

    >>> tuple(file_bit_gen(cStringIO.StringIO(' ab' + chr(255) + chr(1)))) # using chr() because Sphinx isn't happy with non-ascii chars
    (0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1)
    """
    while True:
        byte = infile.read(1)
        if not byte:
            return
        for mask in char_bit_masks:
            yield int(bool(ord(byte) & mask))

# Rice decoding scheme
def uvar_get(bitstream, nbit):
    # count zeros to get excess bits
    result = 0
    while bitstream.next() == 0:
        result += 1
    # shift in the nbits of low bits
    for i in xrange(nbit):
        result = (result << 1) | bitstream.next()
    return result

ULONGSIZE = 2
def ulong_get(bitstream):
    nbit = uvar_get(bitstream, ULONGSIZE)
    ulong = uvar_get(bitstream, nbit)
    return ulong

def var_get(bitstream, nbin):
    uvar = uvar_get(bitstream, nbin + 1)
    return ~(uvar >> 1) if (uvar & 1) else (uvar >> 1)

MAGIC = 'ajkg'

XBYTESIZE = 7
ENERGYSIZE = 3
BITSHIFTSIZE = 2
LPCQSIZE = 2
LPCQUANT = 5

# commands
FNSIZE = 2
FN_DIFF0 = 0
FN_DIFF1 = 1
FN_DIFF2 = 2
FN_DIFF3 = 3
FN_QUIT = 4
FN_BLOCKSIZE = 5
FN_BITSHIFT = 6
FN_QLPC = 7
FN_ZERO = 8

block_set = frozenset((
    FN_ZERO,
    FN_DIFF0,
    FN_DIFF1,
    FN_DIFF2,
    FN_DIFF3,
    FN_QLPC,
    ))

def parse_shorten_bitstream(filename):
    """
    >>> module_dir, module_name = os.path.split(__file__)
    >>> shorten_shn = os.path.join(module_dir, 'shorten.shn')

    >>> import itertools
    >>> with open(shorten_shn, 'rb') as infile:
    ...   h = infile.read(4)
    ...   v = ord(infile.read(1))
    ...   y = tuple(itertools.islice(file_bit_gen(infile), 32))
    >>> h
    'ajkg'
    >>> v
    2
    >>> y
    (1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1)

    >>> parse_shorten_bitstream(shorten_shn)
    ftype 3
    nchan 1
    blocksize 256
    maxnlpc 0
    nmean 4
    nskip 0
    count 146
    """
    outfile = cStringIO.StringIO()
    with open(filename, 'rb') as infile:
        head = infile.read(4)
        assert head == MAGIC, head

        version = ord(infile.read(1))
        assert version == 1 or version == 2

        bitstream = file_bit_gen(infile)

        ftype = ulong_get(bitstream)
        nchan = ulong_get(bitstream)
        blocksize = ulong_get(bitstream)
        maxnlpc = ulong_get(bitstream)
        nmean = ulong_get(bitstream)
        nskip = ulong_get(bitstream)
        for i in xrange(nskip):
            byte = chr(uvar_get(bitstream, XBYTESIZE))
            outfile.write(byte)

        print 'ftype', ftype
        print 'nchan', nchan
        print 'blocksize', blocksize
        print 'maxnlpc', maxnlpc
        print 'nmean', nmean
        print 'nskip', nskip

        bitshift = 0

        count = 0
        while True:
            cmd = uvar_get(bitstream, FNSIZE)
            count += 1

            if cmd in block_set:
                if cmd == FN_ZERO:
                    pass
                else:
                    resn = uvar_get(bitstream, ENERGYSIZE)
                    if cmd == FN_QLPC:
                        nlpc = uvar_get(bitstream, LPCQSIZE)
                        for i in xrange(nlpc):
                            var_get(bitstream, LPCQUANT)
                    for i in xrange(blocksize):
                        var_get(bitstream, resn)

            elif cmd == FN_BLOCKSIZE:
                blocksize = ulong_get(bitstream)
            elif cmd == FN_BITSHIFT:
                bitshift = uvar_get(bitstream, BITSHIFTSIZE)

            elif cmd == FN_QUIT:
                print 'count', count
                break

            else:
                assert False


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
