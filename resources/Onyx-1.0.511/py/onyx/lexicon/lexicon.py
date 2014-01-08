###########################################################################
#
# File:         lexicon.py
# Date:         Fri 24 Oct 2008 11:48
# Author:       Ken Basye
# Description:  
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
A module for maintaining and using word and pronunciation collections.

A LexiconBuilder can be constructed in empty form (the default), from a string,
or from a FrozenLexicon object.  See help(LexiconBuilder) for more details.
>>> lb0 = LexiconBuilder(_dict0)

A FrozenLexicon is a form of FrozenCfg and can be used anywhere a FrozenCfg can
be used.  In addition, FrozenLexicon has a few other properties.
>>> lex0 = FrozenLexicon(lb0)
>>> lex0.num_orthos
19
>>> lex0.num_prons
21

>>> print(lex0)
Lexicon with 19 orthographies and 21 prons

>>> lb1 = LexiconBuilder(lex0)
>>> lb1.add_from_strings(("this_word th i S w e r d", "that_word th ae t w e r d"))
>>> lex1 = FrozenLexicon(lb1)
>>> print(lex1)
Lexicon with 21 orthographies and 23 prons

"""

from cStringIO import StringIO
from onyx.util.singleton import Singleton
from onyx.dataflow.simplecfg import CfgBuilder, FrozenCfg

class LexiconBuilder(CfgBuilder):
    """
    A class for building word and pronunciation collections.  

    A LexiconBuilder can be constructed in empty form (the default), from a
    string, or from a FrozenLexicon object.  In the string case, the string
    should be a collection of lines, with each line consisting of
    space-separated tokens.  Each line represents one word/pron combination; the
    first token of the line is the word, the remaining tokens collectively are
    the pron.

    >>> lb0 = LexiconBuilder(_dict0)
    """
    _W = Singleton('onyx.lexicon.WORD')
    _P = Singleton('onyx.lexicon.PHONE')

    def __init__(self, source = None):
        if source is not None:
            if isinstance(source, str):
                super(LexiconBuilder, self).__init__()
                iterable = StringIO(source)
                self.add_from_strings(iterable)
            elif isinstance(source, FrozenLexicon):
                # XXX would like to just initialize our parent from source directly
                super(LexiconBuilder, self).__init__()
                source.verify()
                for p in source:
                    super(LexiconBuilder, self).add_production(*p)
            else:
                raise TypeError("expected %s or %s, got %s" % (str, FrozenLexicon, type(source)))
        else:
            super(LexiconBuilder, self).__init__()


    def add_from_strings(self, iterable):
        """
        Add word/prons to a lexicon from a string source.
        
        iterable should give strings of tokens tokens separated by spaces.  Each
        string represents one word/pron combination; the first token of the
        string is the word, the remaining tokens collectively are the pron.
        """
        for s in iterable:
            tokens = s.split()
            if len(tokens) < 2:
                raise IOError("Expected at least two tokens in each string, but got %r" % (s,))
            self.add_word_with_pron(tokens[0], tokens[1:])

    def add_word_with_pron(self, word, phones):
        """
        Add word/prons to a lexicon.
        
        word should be a string and phones an iterable of strings.
        """
        def make_word(token):
            return (self._W, token)

        def make_phones(tokens):
            return ((self._P, t) for t in tokens)

        ortho = make_word(word)
        pron = make_phones(phones)
        super(LexiconBuilder, self).add_production(ortho, pron)

    # Rather than expose the tuple-ness of our production tokens or implicitly add them here,
    # we just clobber this function in this class.
    def add_production(self, *dummy):
        raise NotImplementedError("add_production not implemented, use add_word_with_pron() instead")
            
class FrozenLexicon(FrozenCfg):
    """
    A module for using word and pronunciation collections.

    A FrozenLexicon is a form of FrozenCfg and can be used anywhere a FrozenCfg
    can be used.  In addition, FrozenLexicon has a few other properties.

    Make FrozenLexicons from LexiconBuilders:
    >>> lb0 = LexiconBuilder(_dict0)
    >>> lex0 = FrozenLexicon(lb0)

    >>> lex0.num_orthos
    19
    >>> lex0.num_prons
    21

    >>> lex0.num_phones
    22

    >>> print(lex0)
    Lexicon with 19 orthographies and 21 prons
    
    >>> lex0.size
    89

    >>> lex0.num_productions
    21

    >>> len(lex0.terminals)
    22

    >>> len(lex0.non_terminals)
    19

    >>> sorted(lex0.terminals)[0]
    (onyx.util.singleton.Singleton('onyx.lexicon.PHONE'), '@')

    >>> sorted(lex0.non_terminals)[0]
    (onyx.util.singleton.Singleton('onyx.lexicon.WORD'), '</s>')


    # >>> for lhs, rhs in lex0:
    # ...    print('%s ====> %s)' % (lhs, rhs))
    """
    def __init__(self, builder):
        super(FrozenLexicon, self).__init__(builder)
        self.verify()

    def __str__(self):
        ret = ("Lexicon with %d orthographies and %d prons" % (self.num_orthos,
                                                               self.num_prons))
        return ret

    @property
    def num_orthos(self):
        return len(self.non_terminals)

    @property
    def num_prons(self):
        return self.num_productions

    @property
    def num_phones(self):
        return len(self.terminals)

    def verify(self):
        for lhs, rhs in self:
            if not (hasattr(lhs, '__len__') and len(lhs) == 2 and lhs[0] == LexiconBuilder._W):
                raise ValueError("expected all LHSs to be tuples with %s as first item, got %s" %
                                 (LexiconBuilder._W, lhs))
            for phone in rhs:
                if not hasattr(phone, '__len__') or len(phone) != 2 or phone[0] != LexiconBuilder._P:
                    raise ValueError("expected all phones to be tuples with %s as first item, got %s" %
                                     (LexiconBuilder._P, phone))
                
_dict0 = """<s>       sil
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
            abacus    ae b @ k @ S"""
        

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



