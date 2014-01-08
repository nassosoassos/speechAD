###########################################################################
#
# File:         bitlm.py (directory: ./projects/ddt)
# Date:         18-Feb-2008
# Author:       Hugh Secker-Walker
# Description:  Generate an auxilliary, language-model-like information source for the bits
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
    Create a crude lm, no discounting, simple backoff.

    >>> True
    True
"""

from __future__ import with_statement
from collections import defaultdict
import cPickle

from onyx.builtin import frozendict

def main(args):
    if not args:
        return

    # last argument is the output file for the pickled lm database
    outfilename = args.pop()

    legal_chars = frozenset(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # accumulator: positives, total
    def initial(): return [0, 0]
    lm = defaultdict(initial)
    num_bits = 8
    bit_backoffs = tuple(initial() for i in xrange(num_bits))
    num_samples = 0

    # open this here so that we error prior to time-consuming gathering of stats
    with open(outfilename, 'wb') as outfile:

        for sample in text_utils.deterministic_labels(filenames_stream(args), legal_chars):
            num_samples += 1

            char_code, prior_codes, bit_index, bit_value = sample

            # unigrams by bit_index
            backoff = bit_backoffs[bit_index]
            backoff[-1] += 1

            # prior is two prior char_codes and the bit index
            key = prior_codes + (bit_index,)
            stats = lm[key]
            stats[-1] += 1

            if bit_value != 0:
                stats[0] += 1
                backoff[0] += 1

        # max likelihood for the bit
        bit_backoffs = tuple(1 if 2 * samples >= total else 0 for samples, total in bit_backoffs)
        lm = frozendict((key, 1 if 2 * samples >= total else 0) for key, (samples, total) in lm.iteritems())

        cPickle.dump(num_samples, outfile, -1)
        cPickle.dump(dict(lm), outfile, -1)


    # verify what we wrote
    with open(outfilename, 'rb') as infile:
        num_samples2 = cPickle.load(infile)
        lm2 = frozendict(cPickle.load(infile))
        
    assert num_samples2 == num_samples
    assert lm2 == lm

    print len(args), num_samples, len(lm)
    print bit_backoffs


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    main(argv[1:])
