///////////////////////////////////////////////////////////////////////////
//
// File:         onyx_defs.h
// Date:         5-Nov-2007
// Author:       Hugh Secker-Walker
// Description:  C preprocessor defines for Onyx
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2007 The Johns Hopkins University
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

#ifndef ONYX_GUARD_ONYX_DEFS_H
#define ONYX_GUARD_ONYX_DEFS_H


#include <stdio.h>
#include <stdlib.h>

// add 'x' to the end of the line for verbose live logging
#define VERBOSE(x)

#define VERIFY_NULL(expr) do { if(expr) { fprintf(stdout, "%s(%d) : expression is not false : %s\n", __FILE__, __LINE__, #expr); exit(1); } } while( false )
#define VERIFY_TRUE(expr) do { if(!(expr)) { fprintf(stdout, "%s(%d) : expression is not true : %s\n", __FILE__, __LINE__, #expr); exit(1); } } while( false )
#define ASSERT(expr) do { if(!(expr)) { fprintf(stdout, "%s(%d) : assertion failed : %s\n", __FILE__, __LINE__, #expr); exit(1); } } while( false )


// startup code: follow with a function body doing the work
#define ONYX_STARTUP_CODE(NAME) \
namespace ONYX_STARTUP { \
struct STARTUP_##NAME##_CLASS { \
  STARTUP_##NAME##_CLASS(); \
  ~STARTUP_##NAME##_CLASS() {} \
}; \
static const STARTUP_##NAME##_CLASS STARTUP_##NAME##_INSTANCE; \
} \
ONYX_STARTUP::STARTUP_##NAME##_CLASS::STARTUP_##NAME##_CLASS() 


// shutdown code: follow with a function body doing the work
#define ONYX_SHUTDOWN_CODE(NAME) \
namespace ONYX_SHUTDOWN { \
struct SHUTDOWN_##NAME##_CLASS { \
  SHUTDOWN_##NAME##_CLASS() {} \
  ~SHUTDOWN_##NAME##_CLASS(); \
}; \
static const SHUTDOWN_##NAME##_CLASS SHUTDOWN_##NAME##_INSTANCE; \
} \
ONYX_SHUTDOWN::SHUTDOWN_##NAME##_CLASS::~SHUTDOWN_##NAME##_CLASS()



#endif  // ONYX_GUARD_ONYX_DEFS_H
