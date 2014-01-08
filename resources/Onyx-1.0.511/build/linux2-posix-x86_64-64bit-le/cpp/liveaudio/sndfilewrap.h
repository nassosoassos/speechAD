///////////////////////////////////////////////////////////////////////////
//
// File:         sndfilewrap.h
// Date:         24-Feb-2009
// Author:       Hugh Secker-Walker
// Description:  Header for narrow wrapper around sndfile for Python ctypes access
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

#ifndef ONYX_GUARD_SNDFILEWRAP_H
#define ONYX_GUARD_SNDFILEWRAP_H


#ifdef __cplusplus
extern "C" {
#endif

char const * free_ptr(void * const ptr);
char const * get_api_info(char * * const name, char * * const version, char * * const info);
char const * get_audio(int const fd, char const * const name, char const * const wave_format, char * * const info, char * * const casual, void * * const wave);

#ifdef __cplusplus
}
#endif

#endif /* ONYX_GUARD_SNDFILEWRAP_H */
