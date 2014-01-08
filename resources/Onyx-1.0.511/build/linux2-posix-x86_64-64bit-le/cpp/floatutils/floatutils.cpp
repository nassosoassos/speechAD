///////////////////////////////////////////////////////////////////////////
//
// File:         floatutils.cpp
// Date:         Thu 25 Sep 2008 17:37
// Author:       Ken Basye
// Description:  C++ implementation of useful float functions, especially string encodings
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2008 The Johns Hopkins University
//
// Licensed under the Apache License, Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.  See the License for the specific language governing
// permissions and limitations under the License.
//
///////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include<inttypes.h>

#include "floatutils.h"

bool fp_ok(double f)
{
    int c = fpclassify(f);
        return(c == FP_NORMAL || c == FP_ZERO || c == FP_SUBNORMAL);
}

void float_to_readable_string(double f, char* buf)
{
    assert(fp_ok(f));
    uint64_t bits = *((uint64_t*)(&f));
    int exponent = ((bits >> 52) & 0x7ff) - 1023;
    uint64_t mantissa = bits & (uint64_t) 0x000fFfffFfffFfffULL;
    char sign = ((bits >> 63) == 0 ? '+' : '-');
    // We've encountered some problems with this snprintf call, but currently
    // all our platforms get this right.  The symptom is that you'll see only 8
    // of the 13 mantissa values as non-zero, with the leading 5 characters
    // always 0.  The test in testfloatutils should catch this immediately.
    snprintf(buf, 24, "%c(%+05d)0x%013jx", sign, exponent, mantissa);
    assert(strlen(buf) == 23);
}

double readable_string_to_float(const char *s)
{
    if(strlen(s) != 23 ||
       !(s[0] == '+' || s[0] == '-') ||
       !(s[1] == '(' && s[7] == ')') ||
       !(s[8] == '0' && s[9] == 'x'))
    {
        // Note that different systems interpret the argument to nan() in different ways.
        // So far, everyone can deal with the empty string OK.
        const char* dummy = "";
        return nan(dummy);
    }
    char scratch[24];
    strcpy(scratch, s);
    uint64_t sign = ((s[0] == '-') ? 1 : 0);
    scratch[7] = '\0';
    char* endptr;
    int64_t exponent = strtoll(scratch+2, &endptr, 10) + 1023;
    if( *endptr != '\0' || !(0 <= exponent && exponent < 0x7ff))
    {
        const char* dummy = "";
        return nan(dummy);
    }
    // NB: We have to use strtoumax here and we have to include inttypes.h above to get the
    // correct strtoumax on MacOS.
    uint64_t bits = strtoumax(scratch+8, &endptr, 16);
    if( *endptr != '\0')
    {
        const char* dummy = "";
        return nan(dummy);
    }
    bits += (sign << 63) + (exponent << 52);
    double ret = *((double*)(&bits));
    assert(fp_ok(ret));
    return ret;
}


