###########################################################################
#
# File:         mmf.[g,py] (directory: py/onyx/htkfiles)
# Date:         2008-04-23 Wed 11:39:47
# Author:       Ken Basye
# Description:  Configuration for yapps2 runtime
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

# Note: mmf.g is the yapps2 source; mmf.py is automatically generated from mmf.g
# by the yapps2 tools.  See py/onyx/htkfiles/SConscript.

from onyx.builtin import attrdict
from numpy import array

def make_fp_array(fp_list, shape, name, pos):
#   XXX For some reason I'm unable to track down, sphinx was unhappy about the indentation in the
#   results of the first test in this docstring, so I reworked it to print the two rows of the array
#   separately, and now it's happy again.
    """
    Convert a list of floating point numbers to an array.  Raise a
    Yapps SyntaxError if we don't get the right number of FP values.

    Straigtforward case
    >>> arr = make_fp_array(xrange(10), (2, 5), 'test1', 1)

    >>> print arr[0], arr[1]
    [ 0.  1.  2.  3.  4.] [ 5.  6.  7.  8.  9.]

    Edge case
    >>> make_fp_array(xrange(0), (0,), 'test2', 1)
    array([], dtype=float64)

    Test of Yapp's SyntaxError which is unlike Python's....
    >>> try: make_fp_array(xrange(10), (2, 5, 3), 'test3', 1)
    ... except SyntaxError, e: print e.msg
    expected 30 values for field 'test3', got 10

    Check error on an empty shape
    >>> try: make_fp_array(xrange(10), (), 'test4', 1)
    ... except SyntaxError, e: print e.msg
    expected a non-empty shape sequence for field 'test4', got ()
    """
    if not shape:
        # note: SyntaxError is in yappsrt.py and gets globbed into our
        # namespace, thus shadowing Python's builtin!
        raise SyntaxError(pos, "expected a non-empty shape sequence for field %r, got %r" % (name, shape,))
    from operator import mul
    len_needed = reduce(mul, shape)
    if len_needed != len(fp_list):
        # note: SyntaxError is in yappsrt.py and gets globbed into our
        # namespace, thus shadowing Python's builtin!
        raise SyntaxError(pos, "expected %d values for field %r, got %d" % (len_needed, name, len(fp_list)))
    a = array(fp_list, dtype = float)
    a.shape = shape
    return a


from string import *
import re
from yapps2.yappsrt import *

class MMFScanner(Scanner):
    patterns = [
        ('\\s+', re.compile('\\s+')),
        ('OPTMACRO', re.compile('~o')),
        ('HMMMACRO', re.compile('~h')),
        ('VARMACRO', re.compile('~v')),
        ('STATE', re.compile('<(STATE|State)>')),
        ('MEAN', re.compile('<(MEAN|Mean)>')),
        ('VARIANCE', re.compile('<(VARIANCE|Variance)>')),
        ('TRANSP', re.compile('<(TRANSP|TransP)>')),
        ('BEGINHMM', re.compile('<(BEGINHMM|BeginHMM)>')),
        ('ENDHMM', re.compile('<(ENDHMM|EndHMM)>')),
        ('NUMSTATES', re.compile('<(NUMSTATES|NumStates)>')),
        ('VECSIZE', re.compile('<(VECSIZE|VecSize)>')),
        ('STREAMINFO', re.compile('<(STREAMINFO|StreamInfo)>')),
        ('GCONST', re.compile('<(GCONST|GConst)>')),
        ('NULLD', re.compile('<(NULLD|nullD)>')),
        ('DIAGC', re.compile('<(DIAGC|DiagC)>')),
        ('FULLC', re.compile('<(FULLC|FullC)>')),
        ('OPEN', re.compile('<')),
        ('CLOSE', re.compile('>')),
        ('WAVEFORM', re.compile('(WAVEFORM|Waveform)')),
        ('LPC', re.compile('(LPC|Lpc)')),
        ('LPREFC', re.compile('(LPREFC|Lprefc)')),
        ('LPCEPSTRA', re.compile('(LPCEPSTRA|Lpcepstra)')),
        ('LPDELCEP', re.compile('(LPDELCEP|Lpdelcep)')),
        ('IREFC', re.compile('(IREFC|Irefc)')),
        ('MFCC', re.compile('(MFCC|Mfcc)')),
        ('FBANK', re.compile('(FBANK|Fbank)')),
        ('MELSPEC', re.compile('(MELSPEC|Melspec)')),
        ('USER', re.compile('(USER|User)')),
        ('DISCRETE', re.compile('(DISCRETE|Discrete)')),
        ('KINDQUAL_E', re.compile('_E')),
        ('KINDQUAL_N', re.compile('_N')),
        ('KINDQUAL_D', re.compile('_D')),
        ('KINDQUAL_A', re.compile('_A')),
        ('KINDQUAL_C', re.compile('_C')),
        ('KINDQUAL_Z', re.compile('_Z')),
        ('KINDQUAL_K', re.compile('_K')),
        ('KINDQUAL_O', re.compile('_O')),
        ('MARKER', re.compile('<[A-Z]+>')),
        ('FP', re.compile('(\\+|-)?(([1-9]\\d*|0)?\\.\\d+)|(([1-9]\\d*|0)\\.)')),
        ('FPE1', re.compile('(\\+|-)?([1-9]\\d*)((e|E)(\\+|-)?\\d+)')),
        ('FPE2', re.compile('(\\+|-)?((([1-9]\\d*|0)?\\.\\d+)|(([1-9]\\d*|0)\\.))((e|E)(\\+|-)?\\d+)')),
        ('MACRO', re.compile('~.')),
        ('STR', re.compile('"([^\\\\"]+|\\\\.)*"')),
        ('UINT', re.compile('[0-9]+')),
        ('END', re.compile('$')),
    ]
    def __init__(self, str):
        Scanner.__init__(self,None,['\\s+'],str)

class MMF(Parser):
    def fp_num(self):
        _token_ = self._peek('FP', 'FPE1', 'FPE2')
        if _token_ == 'FP':
            FP = self._scan('FP')
            return atof(FP)
        elif _token_ == 'FPE1':
            FPE1 = self._scan('FPE1')
            return atof(FPE1)
        else:# == 'FPE2'
            FPE2 = self._scan('FPE2')
            return atof(FPE2)

    def top(self):
        result = {'models':[] }
        optdecl = self.optdecl()
        result['options'] = optdecl
        if self._peek('VARMACRO', 'END', 'BEGINHMM', 'HMMMACRO') == 'VARMACRO':
            vardecl = self.vardecl()
            result['vardecl'] = vardecl
        while self._peek('END', 'BEGINHMM', 'HMMMACRO') != 'END':
            hmm = self.hmm()
            result['models'].append(hmm)
        END = self._scan('END')
        return result

    def optdecl(self):
        OPTMACRO = self._scan('OPTMACRO')
        options = {}
        while self._peek('NULLD', 'VECSIZE', 'OPEN', 'STREAMINFO', 'DIAGC', 'FULLC', 'VARMACRO', 'END', 'BEGINHMM', 'HMMMACRO') not in ['VARMACRO', 'END', 'BEGINHMM', 'HMMMACRO']:
            option = self.option()
            options[option[0]] = option[1]
        return options

    def option(self):
        _token_ = self._peek('NULLD', 'VECSIZE', 'OPEN', 'STREAMINFO', 'DIAGC', 'FULLC')
        if _token_ == 'VECSIZE':
            vecsize = self.vecsize()
            return vecsize
        elif _token_ == 'OPEN':
            kind = self.kind()
            return ('feature_kind', kind)
        elif _token_ == 'NULLD':
            NULLD = self._scan('NULLD')
            return ('duration', 'nulld')
        elif _token_ == 'STREAMINFO':
            streaminfo = self.streaminfo()
            return streaminfo
        else:# in ['DIAGC', 'FULLC']
            covar_spec = self.covar_spec()
            return covar_spec

    def kind(self):
        OPEN = self._scan('OPEN')
        ret = []
        kind_base = self.kind_base()
        ret.append(kind_base)
        while self._peek('CLOSE', 'KINDQUAL_E', 'KINDQUAL_N', 'KINDQUAL_D', 'KINDQUAL_A', 'KINDQUAL_C', 'KINDQUAL_Z', 'KINDQUAL_K', 'KINDQUAL_O') != 'CLOSE':
            kind_qual = self.kind_qual()
            ret.append(kind_qual)
        CLOSE = self._scan('CLOSE')
        return ret

    def kind_base(self):
        _token_ = self._peek('LPC', 'LPREFC', 'LPCEPSTRA', 'LPDELCEP', 'IREFC', 'MFCC', 'FBANK', 'MELSPEC', 'USER', 'DISCRETE')
        if _token_ == 'LPC':
            LPC = self._scan('LPC')
            return 'LPC'
        elif _token_ == 'LPREFC':
            LPREFC = self._scan('LPREFC')
            return 'LPREFC'
        elif _token_ == 'LPCEPSTRA':
            LPCEPSTRA = self._scan('LPCEPSTRA')
            return 'LPCEPSTRA'
        elif _token_ == 'LPDELCEP':
            LPDELCEP = self._scan('LPDELCEP')
            return 'LPDELCEP'
        elif _token_ == 'IREFC':
            IREFC = self._scan('IREFC')
            return 'IREFC'
        elif _token_ == 'MFCC':
            MFCC = self._scan('MFCC')
            return 'MFCC'
        elif _token_ == 'FBANK':
            FBANK = self._scan('FBANK')
            return 'FBANK'
        elif _token_ == 'MELSPEC':
            MELSPEC = self._scan('MELSPEC')
            return 'MELSPEC'
        elif _token_ == 'USER':
            USER = self._scan('USER')
            return 'USER'
        else:# == 'DISCRETE'
            DISCRETE = self._scan('DISCRETE')
            return 'DISCRETE'

    def kind_qual(self):
        _token_ = self._peek('KINDQUAL_E', 'KINDQUAL_N', 'KINDQUAL_D', 'KINDQUAL_A', 'KINDQUAL_C', 'KINDQUAL_Z', 'KINDQUAL_K', 'KINDQUAL_O')
        if _token_ == 'KINDQUAL_E':
            KINDQUAL_E = self._scan('KINDQUAL_E')
            return '_E'
        elif _token_ == 'KINDQUAL_N':
            KINDQUAL_N = self._scan('KINDQUAL_N')
            return '_N'
        elif _token_ == 'KINDQUAL_D':
            KINDQUAL_D = self._scan('KINDQUAL_D')
            return '_D'
        elif _token_ == 'KINDQUAL_A':
            KINDQUAL_A = self._scan('KINDQUAL_A')
            return '_A'
        elif _token_ == 'KINDQUAL_C':
            KINDQUAL_C = self._scan('KINDQUAL_C')
            return '_C'
        elif _token_ == 'KINDQUAL_Z':
            KINDQUAL_Z = self._scan('KINDQUAL_Z')
            return '_Z'
        elif _token_ == 'KINDQUAL_K':
            KINDQUAL_K = self._scan('KINDQUAL_K')
            return '_K'
        else:# == 'KINDQUAL_O'
            KINDQUAL_O = self._scan('KINDQUAL_O')
            return '_O'

    def covar_spec(self):
        _token_ = self._peek('DIAGC', 'FULLC')
        if _token_ == 'DIAGC':
            DIAGC = self._scan('DIAGC')
            return ('covar', 'diagc')
        else:# == 'FULLC'
            FULLC = self._scan('FULLC')
            return ('covar', 'fullc')

    def vecsize(self):
        VECSIZE = self._scan('VECSIZE')
        UINT = self._scan('UINT')
        return ('vecsize', atoi(UINT))

    def uint1(self):
        UINT = self._scan('UINT')
        return atoi(UINT)

    def uint2(self):
        UINT = self._scan('UINT')
        return atoi(UINT)

    def streaminfo(self):
        STREAMINFO = self._scan('STREAMINFO')
        uint1 = self.uint1()
        uint2 = self.uint2()
        return ('streaminfo', (uint1, uint2))

    def vardecl(self):
        VARMACRO = self._scan('VARMACRO')
        STR = self._scan('STR')
        var = self.var()
        return (eval(STR), var)

    def hmm(self):
        result = attrdict()
        if self._peek('BEGINHMM', 'HMMMACRO') == 'HMMMACRO':
            hmmdecl = self.hmmdecl()
            result.decl = hmmdecl
        BEGINHMM = self._scan('BEGINHMM')
        numstates = self.numstates()
        result.numstates = numstates
        result.states = []
        while self._peek('STATE', 'TRANSP') == 'STATE':
            state = self.state()
            result.states.append(state)
        transp = self.transp()
        result.transp = transp
        ENDHMM = self._scan('ENDHMM')
        return ('HMM', result)

    def hmmdecl(self):
        HMMMACRO = self._scan('HMMMACRO')
        STR = self._scan('STR')
        return eval(STR)

    def numstates(self):
        NUMSTATES = self._scan('NUMSTATES')
        UINT = self._scan('UINT')
        return atoi(UINT)

    def transp(self):
        values = []
        TRANSP = self._scan('TRANSP')
        UINT = self._scan('UINT')
        while self._peek('FP', 'FPE1', 'FPE2', 'ENDHMM') != 'ENDHMM':
            fp_num = self.fp_num()
            values.append(fp_num)
        return make_fp_array(values, (atoi(UINT), atoi(UINT)), 'TransP', self._scanner.pos)

    def state(self):
        result = attrdict()
        STATE = self._scan('STATE')
        UINT = self._scan('UINT')
        result.statenum = atoi(UINT)
        mean = self.mean()
        result.mean = mean
        var = self.var()
        result.var = var
        if self._peek('GCONST', 'STATE', 'TRANSP') == 'GCONST':
            gconst = self.gconst()
            result.gconst = gconst
        return ('state', result)

    def mean(self):
        values = []
        MEAN = self._scan('MEAN')
        UINT = self._scan('UINT')
        dim = atoi(UINT)
        while self._peek('FP', 'FPE1', 'FPE2', 'VARIANCE') != 'VARIANCE':
            fp_num = self.fp_num()
            values.append(fp_num)
        return make_fp_array(values, (1,dim), 'Means', self._scanner.pos)

    def var(self):
        values = []
        VARIANCE = self._scan('VARIANCE')
        UINT = self._scan('UINT')
        dim = atoi(UINT)
        while self._peek('FP', 'FPE1', 'FPE2', 'GCONST', 'END', 'BEGINHMM', 'HMMMACRO', 'STATE', 'TRANSP') in ['FP', 'FPE1', 'FPE2']:
            fp_num = self.fp_num()
            values.append(fp_num)
        return make_fp_array(values, (1,dim), 'Vars', self._scanner.pos)

    def gconst(self):
        GCONST = self._scan('GCONST')
        fp_num = self.fp_num()
        return fp_num


def parse(rule, text):
    P = MMF(MMFScanner(text))
    return wrap_error_reporter(P, rule)




if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
