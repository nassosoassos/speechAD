///////////////////////////////////////////////////////////////////////////
//
// File:         floatutils.h
// Date:         Thu 25 Sep 2008 17:37
// Author:       Ken Basye
// Description:  Declaration of useful float functions, especially string encodings
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

#ifndef ONYX_GUARD_FLOATUTILS_H
#define ONYX_GUARD_FLOATUTILS_H

// Length of a readable float in chars, including null terminator
#define FLOATUTILS_READABLE_LEN 24
void float_to_readable_string(double f, char* buf);
double readable_string_to_float(const char *s);

// For MSVC, we provide fpclassify() and the manifest constants it returns.
// Other platforms can get these from math.h.  The Python wrapper includes
// fpclassify() and these constants.
#ifdef _MSC_VER
int fpclassify(double f);
#define FP_NAN          1
#define FP_INFINITE     2
#define FP_ZERO         3
#define FP_NORMAL       4
#define FP_SUBNORMAL    5
#define FP_SUPERNORMAL  6
#endif // _MSC_VER


#endif  // ONYX_GUARD_FLOATUTILS_H

