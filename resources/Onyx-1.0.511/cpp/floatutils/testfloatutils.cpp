///////////////////////////////////////////////////////////////////////////
//
// File:         testfloatutils.cpp
// Date:         Fri 26 Sep 2008 11:39
// Author:       Ken Basye
// Description:  Tests of floatutils library
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

#include<stdio.h>
#include<stdlib.h>
#include<float.h>
#include<math.h>
#include<inttypes.h>
#include "floatutils.h"

void check_float(double f, int verbose)
{
    char out[24];
    float_to_readable_string(f, out);
    double f2 = readable_string_to_float(out);
    if(f != f2)
    {
        printf("CHECK FAILED! ");
        verbose = 1;
    }
    if(verbose)
    {
        printf("Original value = %g; encoding = %s; decoded value = %g\n", f, out, f2);
    }
}


double make_random_double()
{
    // Make a random double by catting some bits together, then
    // throwing out anything that's not a valid double.
    double ret = 0.0;
    do
    {
        unsigned int low = rand();
        uint64_t val = rand();
        // Since rand() gives us 31-bit values, we shift the
        // upper bits by 33 to make sure some sign bits are 1.  Then
        // we shift the low bits by 2 to make sure all the bits are
        // covered except the least significan 2, which will always be
        // 0.  The result is that all the mantissas we generate here
        // are divisible by 4.
        val = (val << 33) + (low << 2);
        ret = *((double*)(&val));
    }
    while(fpclassify(ret) != FP_NORMAL);

    return ret;
}

int main()
{
    double d = -100*3.14159;
    check_float(d, 1);
    check_float(0.0, 0);
    check_float(DBL_MAX, 0);
    check_float(DBL_MIN, 0);
    check_float(-DBL_MAX, 0);
    check_float(-DBL_MIN, 0);
    unsigned int num_runs = 1024;
    unsigned int i = 0;
    for(i = 0; i < num_runs; i++)
    {
        d = make_random_double();
        check_float(d, 0);
    }

        d = -100*3.14159;
    // Test some error conditions; these should all return NaN values,
        // but note that FP_NAN is not always the same int value across platforms!!
    char out[24];
    float_to_readable_string(d, out);

    // Buffer too short
    double d2 = readable_string_to_float(out+1);
    printf("Original value = %g; encoding = %s; fpclassify(decoded value) == FP_NAN is %d\n",
           d, out+1, (fpclassify(d2) == FP_NAN));

    char temp = out[0];
    out[0] = '=';
    d2 = readable_string_to_float(out);
    printf("Original value = %g; encoding = %s; fpclassify(decoded value) == FP_NAN is %d\n",
           d, out+1, (fpclassify(d2) == FP_NAN));
    out[0] = temp;

    temp = out[10];
    out[10] = '=';
    d2 = readable_string_to_float(out);
    printf("Original value = %g; encoding = %s; fpclassify(decoded value) == FP_NAN is %d\n",
           d, out+1, (fpclassify(d2) == FP_NAN));
    out[10] = temp;

    const char* subnorm_readable = "+(-1023)0x0000000000001";
    double subnorm = readable_string_to_float(subnorm_readable);
    printf("subnorm = %g; readable = %s; fpclassify(subnorm) == FP_SUBNORMAL is <platform_specific>\n",
           subnorm, subnorm_readable);
}

