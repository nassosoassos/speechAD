###########################################################################
#
# File:         twoclass.py
# Date:         15-Feb-2008
# Author:       Hugh Secker-Walker
# Description:  Utilities for proof-of-concept work on semi-supervised training
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
    Utilities for toy experiments on bit labels of text streams.
"""

from __future__ import with_statement
from functools import partial
from itertools import izip, imap, chain, repeat
from collections import defaultdict, deque
from cStringIO import StringIO
import random

from onyx.builtin import frozendict
from onyx.am.gaussian import SimpleGaussianModel

def filenames_stream(filenames):
    """
    Return a generator that yields the lines of the names files.
    """
    for filename in filenames:
        with open(filename, 'rt') as infile:
            for line in infile:
                yield line

def stream_lines(stream):
    """
    Return a generator that yields the non-blank lines of stream.

    >>> tuple(stream_lines(StringIO(' \\n  a foo  \\n  \\n    bar bazzer  ')))
    ('a foo', 'bar bazzer')
    """
    for line in stream:
        assert type(line) is str
        line = line.strip()
        if line:
            yield line

def line_tokens(line):
    """
    Return a generator that yields the white-space-separated tokens of
    line from left to right.

    >>> tuple(line_tokens(' a foo      bar bazzer  '))
    ('a', 'foo', 'bar', 'bazzer')
    """
    assert type(line) is str
    parts = line.split()
    assert parts
    return iter(parts)
    
def token_chars(token):
    """
    Return a generator that yields the characters of token from left
    to right.

    >>> tuple(token_chars('foo'))
    ('f', 'o', 'o')
    """
    # not handling unicode yet
    assert type(token) is str
    assert token
    return iter(token)

def char_bits(char):
    """
    Return a generator that yields the bits of character from most
    significant to least significant.

    >>> hex(ord('g'))
    '0x67'
    >>> tuple(char_bits('g'))
    (0, 1, 1, 0, 0, 1, 1, 1)
    """
    # not handling unicode yet
    assert type(char) is str
    assert len(char) == 1
    code = ord(char)
    mask = 1 << 7
    while mask:
        yield 1 if code & mask else 0
        mask >>= 1
            
class text_utils(object):
    """
    >>> for key in sorted(text_utils.numbits_by_char.keys()): print repr(key), text_utils.numbits_by_char[key],
    ' ' 1 '#' 3 '(' 2 ')' 3 '*' 3 '+' 4 ',' 3 '.' 4 ':' 4 'a' 3 'b' 3 'c' 4 'd' 3 'e' 4 'f' 4 'g' 5 'h' 3 'i' 4 'j' 4 'k' 5 'l' 4 'm' 5 'n' 5 'o' 6 'p' 3 'q' 4 'r' 4 's' 5 't' 4 'u' 5 'v' 5 'w' 6 'x' 4 'y' 5 'z' 5
    
    >>> for key in sorted(text_utils.charset_by_numbits.keys()): print key, text_utils.charset_by_numbits[key]
    1 frozenset([' '])
    2 frozenset(['('])
    3 frozenset(['a', '#', 'b', 'd', ')', 'h', '*', ',', 'p'])
    4 frozenset(['c', 'e', 'f', 'i', '+', 'j', 'l', '.', 'q', 'r', 't', 'x', ':'])
    5 frozenset(['g', 'k', 'm', 'n', 's', 'u', 'v', 'y', 'z'])
    6 frozenset(['o', 'w'])

    >>> text_utils.num_chars_by_bit_index
    (0, 26, 35, 12, 18, 14, 18, 16)

    >>> for key in sorted(text_utils.charset_by_bit.keys()): print key, text_utils.charset_by_bit[key]
    1 frozenset(['a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'])
    2 frozenset([' ', '#', ')', '(', '+', '*', ',', '.', ':', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i', 'h', 'k', 'j', 'm', 'l', 'o', 'n', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'])
    3 frozenset([':', 'q', 'p', 's', 'r', 'u', 't', 'w', 'v', 'y', 'x', 'z'])
    4 frozenset(['*', 'l', 'n', ':', ')', '(', '+', 'j', 'm', ',', 'o', '.', 'i', 'h', 'y', 'x', 'z', 'k'])
    5 frozenset(['e', 'd', 'g', 'f', 'm', ',', 'o', '.', 'u', 't', 'w', 'v', 'n', 'l'])
    6 frozenset(['j', '#', 'n', 'c', 'b', 'g', 'f', ':', '+', '*', 'o', '.', 's', 'r', 'w', 'v', 'z', 'k'])
    7 frozenset(['c', 'a', '#', 'e', 'g', ')', '+', 'm', 'o', 'q', 'i', 's', 'u', 'w', 'y', 'k'])

    >>> for index, chars in enumerate(text_utils.charseq_by_bit_index): print index, chars
    0 (' ', '#', '(', ')', '*', '+', ',', '.', ':', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    1 ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    2 (' ', '#', '(', ')', '*', '+', ',', '.', ':', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    3 (':', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z')
    4 ('(', ')', '*', '+', ',', '.', ':', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'x', 'y', 'z')
    5 (',', '.', 'd', 'e', 'f', 'g', 'l', 'm', 'n', 'o', 't', 'u', 'v', 'w')
    6 ('#', '*', '+', '.', ':', 'b', 'c', 'f', 'g', 'j', 'k', 'n', 'o', 'r', 's', 'v', 'w', 'z')
    7 ('#', ')', '+', 'a', 'c', 'e', 'g', 'i', 'k', 'm', 'o', 'q', 's', 'u', 'w', 'y')
    """

    legal_chars = frozenset(' abcdefghijklmnopqrstuvwxyz,:()#.*+')
    num_legal_chars = len(legal_chars)
    #assert num_legal_chars == 27

    # set bit count for each char
    numbits_by_char = frozendict((char, sum(char_bits(char))) for char in legal_chars)

    charset_by_numbits = defaultdict(set)
    for char, numbits in numbits_by_char.iteritems():
        charset_by_numbits[numbits].add(char)
    charset_by_numbits = frozendict((numbits, frozenset(chars)) for numbits, chars in charset_by_numbits.iteritems())


    # number of chars with that bit set
    num_chars_by_bit_index = tuple(sum(bits) for bits in izip(*imap(char_bits, legal_chars)))

    # sets of chars with given bit 
    charset_by_bit = defaultdict(set)
    for char in legal_chars:
        for index, bit in enumerate(char_bits(char)):
            if bit:
                charset_by_bit[index].add(char)
    charset_by_bit = frozendict((index, frozenset(chars)) for index, chars in charset_by_bit.iteritems())

    charseq_by_bit_index = list()
    for char in legal_chars:
        for bit_index, bit_value in enumerate(char_bits(char)):
            while len(charseq_by_bit_index) <= bit_index:
                charseq_by_bit_index.append(set())
            if bit_value:
                charseq_by_bit_index[bit_index].add(char)
    # note: an empty set is replaced by the full legal_chars set since a uniform sample is appropriate
    charseq_by_bit_index = tuple(tuple(sorted(charset if charset else legal_chars)) for charset, legal_chars in izip(charseq_by_bit_index, repeat(legal_chars)))

    assert sum(numbits_by_char.values()) == sum(num_chars_by_bit_index)

    @staticmethod
    def deterministic_labels(stream, legal_chars=legal_chars):
        """
        Return a generator that yields event labels from the text stream.

        >>> doc = '''
        ...  foo 
        ...
        ... a b
        ... '''
        >>> from cStringIO import StringIO
        >>> x = tuple(text_utils.deterministic_labels(StringIO(doc)))
        >>> len(x)
        56
        >>> x[0:3], x[-3:]
        (((102, (-1, -1), 0, 0), (102, (-1, -1), 1, 1), (102, (-1, -1), 2, 1)), ((98, (97, 32), 5, 0), (98, (97, 32), 6, 1), (98, (97, 32), 7, 0)))
        """
        numbits_by_char = text_utils.numbits_by_char
        num_chars_by_bit_index = text_utils.num_chars_by_bit_index
        num_priors = 2
        priors = deque(-1 for x in xrange(num_priors))
        sep = ()
        for line in stream_lines(stream):
            for token in line_tokens(line):
                for char in chain(sep, token_chars(token)):
                    if char not in legal_chars:
                        continue
                    sep = (' ',)
                    ord_char = ord(char)
                    tuple_priors = tuple(priors)
                    for bit_index, bit_value in enumerate(char_bits(char)):
                        yield ord_char, tuple_priors, bit_index, bit_value,
                    priors.popleft()
                    priors.append(ord_char)
                    


    @staticmethod
    def damage_labels(permil, labels, seed=None):
        """
        Randomly damage the bit-value labels of a label stream.
        Introduce damage to permil / 1000 of the labels.  If seed is
        not None it is used to reproducibly seed the randomness.

        >>> from cStringIO import StringIO

        No damage
        >>> tuple(text_utils.damage_labels(0, text_utils.deterministic_labels(StringIO('a b cd')), seed=0))[-10:-6]
        ((99, (98, 32), 6, 1), (99, (98, 32), 7, 1), (100, (32, 99), 0, 0), (100, (32, 99), 1, 1))

        Fifty-percent damage
        >>> tuple(text_utils.damage_labels(500, text_utils.deterministic_labels(StringIO('a b cd')), seed=0))[-10:-6]
        ((99, (98, 32), 6, 1), (99, (98, 32), 7, 1), (100, (32, 99), 0, 1), (100, (32, 99), 1, 0))
        """
        mil = 1000
        assert 0 <= permil <= mil
        rand = random.Random()
        rand.seed(seed)
        randint = partial(rand.randint, 0, mil)
        for label in labels:
            if randint() < permil:
                char_code, priors, bit_index, bit_value = label
                # note: 1 - bit_value only works for (0, 1) set of labels
                yield char_code, priors, bit_index, 1 - bit_value
            else:
                yield label

    @staticmethod
    def make_model_samplers(means, vars):
        """
        >>> SimpleGaussianModel.seed(0)
        >>> tuple(int(128 * sampler()) for sampler in text_utils.make_model_samplers((0, 1), (1, 0.5)))
        (-23, 130)
        >>> tuple(int(128 * sampler()) for sampler in text_utils.make_model_samplers((0, 1, 2), (1, 0.5, 2)))
        (89, 119, 511)
        """
        assert len(means) == len(vars)
        num_classes = len(means)
        models = tuple(SimpleGaussianModel(1, SimpleGaussianModel.DIAGONAL_COVARIANCE) for i in xrange(num_classes))
        for model, mean, var in izip(models, means, vars):
            model.set_model((mean,), (var,))
        samplers = tuple(model.sample for model in models)
        assert len(samplers) == num_classes
        return samplers
        
    @staticmethod
    def generate_samples(samplers, labels):
        """
        >>> from cStringIO import StringIO
        >>> random.seed(0)
        >>> SimpleGaussianModel.seed(0)
        >>> tuple(text_utils.generate_samples(text_utils.make_model_samplers((0, 1), (1, 0.5)), text_utils.deterministic_labels(StringIO('a b cd'))))[-10:-6]
        (((99, (98, 32), 6, 1), (-20, 115)), ((99, (98, 32), 7, 1), (51, 111)), ((100, (32, 99), 0, 0), (96, 32)), ((100, (32, 99), 1, 1), (182, 109)))
        """
        charseq_by_bit_index = text_utils.charseq_by_bit_index
        choice = random.choice
        for label in labels:
            char_code, priors, bit_index, bit_value = label
            yield label, (int(128 * samplers[bit_value]()), ord(choice(charseq_by_bit_index[bit_index])))

def main(args):
    if not args:
        return
    with open(args[0], 'rt') as infile:
        SimpleGaussianModel.seed(0)
        for sample in text_utils.generate_samples(text_utils.make_model_samplers((0, 1), (1, 0.75)), text_utils.deterministic_labels(infile)):
            print sample, 'X' if sample[0][0] == sample[-1][-1] else ''
    

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()

    from sys import argv
    main(argv[1:])
