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

%%
parser MMF:
    ignore:            r'\s+'
    token OPTMACRO:    r'~o'
    token HMMMACRO:    r'~h'
    token VARMACRO:    r'~v'
    token STATE:       r'<(STATE|State)>'
    token MEAN:        r'<(MEAN|Mean)>'
    token VARIANCE:    r'<(VARIANCE|Variance)>'
    token TRANSP:      r'<(TRANSP|TransP)>'
    token BEGINHMM:    r'<(BEGINHMM|BeginHMM)>'
    token ENDHMM:      r'<(ENDHMM|EndHMM)>'
    token NUMSTATES:   r'<(NUMSTATES|NumStates)>'
    token VECSIZE:     r'<(VECSIZE|VecSize)>'
    token STREAMINFO:  r'<(STREAMINFO|StreamInfo)>'
    token GCONST:      r'<(GCONST|GConst)>'
    token NULLD:       r'<(NULLD|nullD)>'
    token DIAGC:       r'<(DIAGC|DiagC)>'
    token FULLC:       r'<(FULLC|FullC)>'

    token OPEN:        r'<'
    token CLOSE:       r'>'

    token WAVEFORM:    r'(WAVEFORM|Waveform)'
    token LPC:         r'(LPC|Lpc)'
    token LPREFC:      r'(LPREFC|Lprefc)'
    token LPCEPSTRA:   r'(LPCEPSTRA|Lpcepstra)'
    token LPDELCEP:    r'(LPDELCEP|Lpdelcep)'
    token IREFC:       r'(IREFC|Irefc)'
    token MFCC:        r'(MFCC|Mfcc)'
    token FBANK:       r'(FBANK|Fbank)'
    token MELSPEC:     r'(MELSPEC|Melspec)'
    token USER:        r'(USER|User)'
    token DISCRETE:    r'(DISCRETE|Discrete)'

    token KINDQUAL_E:  r'_E'
    token KINDQUAL_N:  r'_N'
    token KINDQUAL_D:  r'_D'
    token KINDQUAL_A:  r'_A'
    token KINDQUAL_C:  r'_C'
    token KINDQUAL_Z:  r'_Z'
    token KINDQUAL_K:  r'_K'
    token KINDQUAL_O:  r'_O'

    token MARKER:      r'<[A-Z]+>'
    token FP:          r'(\+|-)?(([1-9]\d*|0)?\.\d+)|(([1-9]\d*|0)\.)'   # no exponent
    token FPE1:        r'(\+|-)?([1-9]\d*)((e|E)(\+|-)?\d+)'             # exponent, but no decimal point
    token FPE2:        r'(\+|-)?((([1-9]\d*|0)?\.\d+)|(([1-9]\d*|0)\.))((e|E)(\+|-)?\d+)'  # exponent and decimal point
    token MACRO:       r'~.'
    token STR:         r'"([^\\"]+|\\.)*"'
    token UINT:        r'[0-9]+'
    token END:         r'$'


    rule fp_num:       FP              {{ return atof(FP) }}
                     | FPE1            {{ return atof(FPE1) }}
                     | FPE2            {{ return atof(FPE2) }}

    rule top :            {{ result = {'models':[] } }}
                optdecl   {{ result['options'] = optdecl }}
               [vardecl   {{ result['vardecl'] = vardecl }}]
               (
                   hmm    {{ result['models'].append(hmm) }}
               )*  END
                          {{ return result }}


    ################  GLOBAL OPTIONS  ################

    # Note that there is no attempt here to disallow multiple declarations of the
    # same option.  Instead, the last declaration will clobber any previous one.
    # I am not sure how this squares with what HTK does.

    rule optdecl: OPTMACRO {{ options = {} }}
                  ( option      {{ options[option[0]] = option[1] }} )*
                  {{ return options }}

    rule option:   vecsize     {{ return vecsize }}
                 | kind        {{ return ('feature_kind', kind) }}
                 | NULLD       {{ return ('duration', 'nulld') }}
                 | streaminfo  {{ return streaminfo }}
                 | covar_spec  {{ return covar_spec }}
                 
    rule kind:   OPEN        {{ ret = [] }}
                 kind_base   {{ ret.append(kind_base) }}
                (kind_qual   {{ ret.append(kind_qual) }})*
                 CLOSE       {{ return ret }}

    rule kind_base:  
                        LPC          {{ return 'LPC' }}  
                      | LPREFC       {{ return 'LPREFC' }}
                      | LPCEPSTRA    {{ return 'LPCEPSTRA' }}
                      | LPDELCEP     {{ return 'LPDELCEP' }}
                      | IREFC        {{ return 'IREFC' }}
                      | MFCC         {{ return 'MFCC' }}
                      | FBANK        {{ return 'FBANK' }}
                      | MELSPEC      {{ return 'MELSPEC' }}
                      | USER         {{ return 'USER' }}
                      | DISCRETE     {{ return 'DISCRETE' }}

        
    rule kind_qual:     
                        KINDQUAL_E  {{ return '_E' }}
                      | KINDQUAL_N  {{ return '_N' }}
                      | KINDQUAL_D  {{ return '_D' }}
                      | KINDQUAL_A  {{ return '_A' }}
                      | KINDQUAL_C  {{ return '_C' }}
                      | KINDQUAL_Z  {{ return '_Z' }}
                      | KINDQUAL_K  {{ return '_K' }}
                      | KINDQUAL_O  {{ return '_O' }}


    rule covar_spec:     DIAGC  {{ return ('covar', 'diagc') }}
                       | FULLC  {{ return ('covar', 'fullc') }}

    rule vecsize: VECSIZE UINT   {{ return ('vecsize', atoi(UINT)) }}

    rule uint1: UINT {{ return atoi(UINT) }}
    rule uint2: UINT {{ return atoi(UINT) }}
    rule streaminfo: STREAMINFO uint1 uint2  {{ return ('streaminfo', (uint1, uint2)) }}


    rule vardecl:    VARMACRO STR var {{ return (eval(STR), var) }}

    ################  HMM  ################
    rule hmm:            {{ result = attrdict() }}
              [hmmdecl   {{ result.decl = hmmdecl }} ]
              BEGINHMM
              numstates  {{ result.numstates = numstates }}
                         {{ result.states = [] }}
              (state     {{ result.states.append(state) }} )*
              transp     {{ result.transp = transp }}
              ENDHMM     {{ return ('HMM', result) }}

    rule hmmdecl: HMMMACRO STR   {{ return eval(STR) }}
    rule numstates:  NUMSTATES UINT   {{ return atoi(UINT) }}

    rule transp:      {{ values = [] }}
        TRANSP UINT
        (fp_num       {{ values.append(fp_num) }}   )*
                      {{ return make_fp_array(values, (atoi(UINT), atoi(UINT)), 'TransP', self._scanner.pos) }}


    ################  STATE  ################
    rule state:   {{ result = attrdict() }}
                STATE
                UINT {{ result.statenum = atoi(UINT) }}
                mean {{ result.mean = mean }}
                var  {{ result.var = var }}
                [gconst  {{ result.gconst = gconst }} ]
                 {{ return ('state', result) }}

    rule mean:        {{ values = [] }}
                MEAN
                UINT  {{ dim = atoi(UINT) }} 
                (fp_num       {{ values.append(fp_num) }}   )*
                      {{ return make_fp_array(values, (1,dim), 'Means', self._scanner.pos) }}

    rule var:         {{ values = [] }}
                VARIANCE
                UINT  {{ dim = atoi(UINT) }} 
                (fp_num       {{ values.append(fp_num) }}   )*
                      {{ return make_fp_array(values, (1,dim), 'Vars', self._scanner.pos) }}

    rule gconst : GCONST fp_num   {{ return fp_num }}
%%

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
