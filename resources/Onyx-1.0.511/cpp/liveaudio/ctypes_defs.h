///////////////////////////////////////////////////////////////////////////
//
// File:         ctypes_defs.h
// Date:         16-Mar-2009
// Author:       Hugh Secker-Walker
// Description:  Defines to help with an error interface to Python's ctypes functionality
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2009 The Johns Hopkins University
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

#ifndef ONYX_GUARD_CTYPES_DEFS_H
#define ONYX_GUARD_CTYPES_DEFS_H

#include <stdbool.h>

//  for these macros to work you must:
//    define 'static char error_msg[1024];' in your translation unit
//    define FUNC as the function_name symbol (no quotation marks) for each function

#define ValueError "ValueError"
#define AudioTypeError "AudioTypeError"

#define QUOTED_HELPER(X) #X
#define QUOTED(X) QUOTED_HELPER(X)

#define ERR_IF_CLEANUP_RETURN(COND, CLEANUP, ERR_TYPE, FORMAT, ...) do { if( COND ) { snprintf(error_msg, sizeof(error_msg), ERR_TYPE " %s:%d %s(): " FORMAT, __FILE__, __LINE__, QUOTED(FUNC), __VA_ARGS__); { CLEANUP; } return error_msg; } } while( false )
#define ERR_IF(COND, ERR_TYPE, FORMAT, ...) ERR_IF_CLEANUP_RETURN(COND, (void)0, ERR_TYPE, FORMAT, __VA_ARGS__)
#define ERR_RETURN(ERR_TYPE, FORMAT, ...) ERR_IF_CLEANUP_RETURN(true, (void)0, ERR_TYPE, FORMAT, __VA_ARGS__)

#define ERR_NULL_CLEANUP(PTR, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (PTR) == NULL, (CLEANUP), ValueError, FORMAT, __VA_ARGS__)
#define ERR_NON_ZERO_CLEANUP(ERR, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (ERR) != 0, (CLEANUP), ValueError, FORMAT, __VA_ARGS__)
#define ERR_ZERO_CLEANUP(ERR, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (ERR) == 0, (CLEANUP), ValueError, FORMAT, __VA_ARGS__)
#define ERR_NON_POSITIVE_CLEANUP(ERR, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (ERR) <= 0, (CLEANUP), ValueError, FORMAT, __VA_ARGS__)
#define ERR_NEGATIVE_CLEANUP(ERR, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (ERR) < 0, (CLEANUP), ValueError, FORMAT, __VA_ARGS__)
#define ERR_EQUAL_CLEANUP(ERR1, ERR2, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (ERR1) == (ERR2), (CLEANUP), ValueError, FORMAT, __VA_ARGS__)
#define ERR_NOT_EQUAL_CLEANUP(ERR1, ERR2, CLEANUP, FORMAT, ...) ERR_IF_CLEANUP_RETURN( (ERR1) != (ERR2), (CLEANUP), ValueError, FORMAT, __VA_ARGS__)

#define ERR_NULL(PTR, FORMAT, ...) ERR_NULL_CLEANUP(PTR, (void)0, FORMAT, __VA_ARGS__)
#define ERR_NON_ZERO(ERR, FORMAT, ...) ERR_NON_ZERO_CLEANUP(ERR, (void)0, FORMAT, __VA_ARGS__)
#define ERR_ZERO(ERR, FORMAT, ...) ERR_ZERO_CLEANUP(ERR, (void)0, FORMAT, __VA_ARGS__)
#define ERR_NON_POSITIVE(ERR, FORMAT, ...) ERR_NON_POSITIVE_CLEANUP(ERR, (void)0, FORMAT, __VA_ARGS__)
#define ERR_NEGATIVE(ERR, FORMAT, ...) ERR_NEGATIVE_CLEANUP(ERR, (void)0, FORMAT, __VA_ARGS__)
#define ERR_EQUAL(ERR1, ERR2, FORMAT, ...) ERR_EQUAL_CLEANUP(ERR1, ERR2, (void)0, FORMAT, __VA_ARGS__)
#define ERR_NOT_EQUAL(ERR1, ERR2, FORMAT, ...) ERR_NOT_EQUAL_CLEANUP(ERR1, ERR2, (void)0, FORMAT, __VA_ARGS__)

#endif  // ONYX_GUARD_CTYPES_DEFS_H
