###########################################################################
#
# File:         htklex.py
# Date:         Thu 23 Oct 2008 15:45
# Author:       Ken Basye
# Description:  Read HTK lexicon files into our lexicon structure
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
Read HTK lexicon files into our lexicon structure

>>> line_iter0 = StringIO(_dict0)
>>> lex0 = read_htk_lexicon(line_iter0)
>>> print(lex0)
Lexicon with 19 orthographies and 21 prons

>>> module_dir, module_name = os.path.split(__file__)

>>> htk_lex_file0 = os.path.join(module_dir, "en_lexic.v08")
>>> f = open(htk_lex_file0)
>>> lex1 = read_htk_lexicon(f)
>>> print(lex1)
Lexicon with 90996 orthographies and 99364 prons

>>> lex1.num_phones
44

"""

import os
from itertools import ifilter
from cStringIO import StringIO
from onyx.lexicon.lexicon import LexiconBuilder, FrozenLexicon

def read_htk_lexicon(line_iterable):
    """
    From an iterable over lines in an HTK lexicon file, create a FrozenLexicon

    """
    def not_blank_or_comment(s):
        return not (s.isspace() or s.lstrip()[0] == '#')
    lb = LexiconBuilder()
    lb.add_from_strings(ifilter(not_blank_or_comment, line_iterable))
    return FrozenLexicon(lb)


_dict0 = """
         <s>       sil
         </s>      sil
         <sil>     sil
         <unk>     unk
         <nonspch> nonspch
         <laugh>   laugh
         a         ei
         a's       ei Z
         aachen    a k I n
         aalseth   a l S E T
         aalseth   ae l S E T
         aames     ei m Z
         aancor    ae n k o9wD r
         aaron     ei r I n
         aarons    ei r I n Z
         ababa     ae b @ b @
         abaci     ae b @ k aI
         aback     @ b ae k
         abaco     @ b ae k oU
         abaco     ae b @ k oU
         abacus    ae b @ k @ S
         """


if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



