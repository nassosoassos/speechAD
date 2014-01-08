///////////////////////////////////////////////////////////////////////////
//
// File:         sndfilewrap.c
// Date:         24-Feb-2009
// Author:       Hugh Secker-Walker
// Description:  Narrow wrapper around sndfile for Python ctypes access
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

// for asprintf
#define _GNU_SOURCE
#include <stdio.h>

#include <fcntl.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdbool.h>
#include <errno.h>
#include <string.h>
#include <assert.h>

#include "sndfile.h"

#include "sndfilewrap.h"
#include "ctypes_defs.h"

#define LIBRARY_NAME "sndfilewrap"
static int const LIBRARY_MAJOR_VERSION = 0;
static int const LIBRARY_MINOR_VERSION = 0;

static char const * get_audio_fd(int const fd, char const * const name, char const * const wave_format, char * * const info, char * * const casual, void * * const wave);
static char const * get_sndfile(int const fd, SNDFILE * * const psndfile, SF_INFO * const psf_info, char const * const name);
static char const * get_sndfile_info(SF_INFO const sf_info, char * * const info, char * * const casual, char const * const name);
static char const * get_sndfile_wave(SNDFILE * const sndfile, SF_INFO const sf_info, char const * const wave_format, void * * const wave, char const * const name);

static char error_msg[1024];


// XXX the free_ptr function, and an allocator, need to be in a separate library...

#define FUNC free_ptr
const char * FUNC(void * const in_ptr) {
  void * * const ptr = (void * * const)in_ptr;
  ERR_NULL(ptr, "%s", "unexpected NULL pointer");
  if( *ptr == NULL ) {
    // just means a pointer wasn't used....
    //fprintf(stderr, "%s: free_ptr() *ptr is NULL (at ptr %#x)\n", LIBRARY_NAME, ptr);
  }
  else {
    assert( ptr != NULL && *ptr != NULL );
    //fprintf(stderr, "%s: free_ptr() *ptr %#x (at ptr %#x)\n", LIBRARY_NAME, *ptr, ptr);
    free(*ptr);
    *ptr = NULL;
  }
  return NULL;
}
#undef FUNC

#define FUNC get_api_info
char const * FUNC(char * * const name, char * * const version, char * * const info) {
  ERR_IF(name == NULL || version == NULL || info == NULL, ValueError, "expected non-NULL name, version, and info return pointer-pointers, but they are %8p, %8p, and %8p", name, version, info);
  ERR_IF(name == version || version == info || info == name, ValueError, "expected name, version, and info return pointer-pointers to be distinct, but they are %8p, %8p, and %8p", name, version, info);
  ERR_IF(*name != NULL || *version != NULL || *info != NULL, ValueError, "expected NULL name, version, and info at return pointer-pointers, but they are %8p, %8p, and %8p", *name, *version, *info);

  int err;

  err = asprintf(name, "%s", LIBRARY_NAME);
  ERR_NEGATIVE(err, "asprintf name failed with errno %d: %s", errno, strerror(errno));

  err = asprintf(version, "%d  %d", LIBRARY_MAJOR_VERSION, LIBRARY_MINOR_VERSION);
  ERR_NEGATIVE(err, "asprintf version failed with errno %d: %s", errno, strerror(errno));

  SF_INFO sf_check_info = {0};
  err = sf_format_check(&sf_check_info);
  ERR_NON_ZERO(err, "sndfile error: expected 0 from sf_format_check, got %d", err);

  char sndfile_version[128];
  int const got = sf_command(NULL, SFC_GET_LIB_VERSION, sndfile_version, sizeof(sndfile_version));
  ERR_NON_POSITIVE(got, "expected at least one character for SFC_GET_LIB_VERSION, got %d", got);
  //fprintf(stderr, "%s: got %d chars from sf_command SFC_GET_LIB_VERSION '%s'\n", LIBRARY_NAME, got, sndfile_version);

  err = asprintf(info, "%s-%d.%d (using %s)", LIBRARY_NAME, LIBRARY_MAJOR_VERSION, LIBRARY_MINOR_VERSION, sndfile_version);
  ERR_NEGATIVE(err, "asprintf info failed with errno %d: %s", errno, strerror(errno));

  return NULL;
}
#undef FUNC

#define FUNC get_audio
char const * FUNC(int const fd, char const * const name, char const * const wave_format, char * * const info, char * * const casual, void * * const wave) {
  ERR_NEGATIVE(fd, "expected non-negative file descriptor, got %d", fd);
  ERR_NULL(name, "%s", "expected non-NULL name pointer");
  ERR_IF(info == NULL || casual == NULL || wave == NULL, ValueError, "expected non-NULL info, casual, and wave return pointer-pointers, but they are %8p, %8p, and %8p", info, casual, wave);
  ERR_IF(info == casual || (void**)info == wave || (void**)casual == wave, ValueError, "expected info, casual, and wave return pointer-pointers to be distinct, but they are %8p, %8p, and %8p", info, casual, wave);
  ERR_IF(*info != NULL || *casual != NULL || *wave != NULL, ValueError, "expected NULL info, casual, and wave at the return pointer-pointers, but they are %8p, %8p, and %8p", *info, *casual, *wave);

  return get_audio_fd(fd, name, wave_format, info, casual, wave);
}
#undef FUNC

#define FUNC get_audio_fd
char const * FUNC(int const fd, char const * const name, char const * const wave_format, char * * const info, char * * const casual, void * * const wave) {
  assert( fd >= 0 );
  assert( name != NULL );
  assert( info != NULL && *info == NULL );
  assert( casual != NULL && *casual == NULL );
  assert( wave != NULL && *wave == NULL );

  int err;
  char const * perr;

  SNDFILE * sndfile;
  SF_INFO sf_info;
  perr = get_sndfile(fd, &sndfile, &sf_info, name);
  if( perr != NULL ) return perr;

  perr = get_sndfile_info(sf_info, info, casual, name);
  if( perr != NULL ) {
    sf_close(sndfile);
    return perr;
  }

  if( wave_format != NULL ) {
    perr = get_sndfile_wave(sndfile, sf_info, wave_format, wave, name);
    if( perr != NULL ) {
      sf_close(sndfile);
      return perr;
    }
  }

  err = sf_close(sndfile);
  ERR_NON_ZERO(err, "unexpected value %d from sf_close: file '%s'\n", err, name);

  return NULL;
}
#undef FUNC

#define FUNC get_sndfile
static char const * FUNC(int const fd, SNDFILE * * const psndfile, SF_INFO * const psf_info, char const * const name) {
  assert( fd >= 0 );
  assert( psndfile != NULL );
  assert( psf_info != NULL );
  assert( name != NULL );

  *psndfile = NULL;
  bzero(psf_info, sizeof(*psf_info));
  int err;

  *psndfile = sf_open_fd(fd, SFM_READ, psf_info, false);
  if( *psndfile == NULL ) {
    char  msg[1024] ;
    int const got = sf_command(NULL, SFC_GET_LOG_INFO, msg, sizeof(msg));
    if( got <= 0 ) {
      ERR_RETURN(AudioTypeError, "unexpected NULL from sf_open_fd, maybe not a valid sound file: '%s': also SFC_GET_LOG_INFO failed\n", name);
    }
    ERR_RETURN(AudioTypeError, "unexpected NULL from sf_open_fd: maybe not a valid sound file: '%s': SFC_GET_LOG_INFO: %s", name, msg);
  }
  //  ERR_NULL(*psndfile, "unexpected NULL from sf_open_fd, maybe not a valid sound file: '%s'\n", name);

  err = sf_error(*psndfile);
  if( err == SF_ERR_SYSTEM ) {
    // XXX libsndfile bug: our workaround is to not actually error....
    fprintf(stderr, "unexpected libsndfile error %d in sndfile: %s  file '%s'\n", err, sf_error_number(err), name);
  }
  else {
    ERR_NOT_EQUAL(err, SF_ERR_NO_ERROR, "unexpected libsndfile error %d: %s  file '%s'\n", err, sf_error_number(err), name);
  }

  return NULL;
}
#undef FUNC


/* subtypes (encoding) from sndfile.h version 1.0.19 */
#if 0
        SF_FORMAT_PCM_S8                = 0x0001,               /* Signed 8 bit data */
        SF_FORMAT_PCM_16                = 0x0002,               /* Signed 16 bit data */
        SF_FORMAT_PCM_24                = 0x0003,               /* Signed 24 bit data */
        SF_FORMAT_PCM_32                = 0x0004,               /* Signed 32 bit data */

        SF_FORMAT_PCM_U8                = 0x0005,               /* Unsigned 8 bit data (WAV and RAW only) */

        SF_FORMAT_FLOAT                 = 0x0006,               /* 32 bit float data */
        SF_FORMAT_DOUBLE                = 0x0007,               /* 64 bit float data */

        SF_FORMAT_ULAW                  = 0x0010,               /* U-Law encoded. */
        SF_FORMAT_ALAW                  = 0x0011,               /* A-Law encoded. */
        SF_FORMAT_IMA_ADPCM             = 0x0012,               /* IMA ADPCM. */
        SF_FORMAT_MS_ADPCM              = 0x0013,               /* Microsoft ADPCM. */

        SF_FORMAT_GSM610                = 0x0020,               /* GSM 6.10 encoding. */
        SF_FORMAT_VOX_ADPCM             = 0x0021,               /* OKI / Dialogix ADPCM */

        SF_FORMAT_G721_32               = 0x0030,               /* 32kbs G721 ADPCM encoding. */
        SF_FORMAT_G723_24               = 0x0031,               /* 24kbs G723 ADPCM encoding. */
        SF_FORMAT_G723_40               = 0x0032,               /* 40kbs G723 ADPCM encoding. */

        SF_FORMAT_DWVW_12               = 0x0040,               /* 12 bit Delta Width Variable Word encoding. */
        SF_FORMAT_DWVW_16               = 0x0041,               /* 16 bit Delta Width Variable Word encoding. */
        SF_FORMAT_DWVW_24               = 0x0042,               /* 24 bit Delta Width Variable Word encoding. */
        SF_FORMAT_DWVW_N                = 0x0043,               /* N bit Delta Width Variable Word encoding. */

        SF_FORMAT_DPCM_8                = 0x0050,               /* 8 bit differential PCM (XI only) */
        SF_FORMAT_DPCM_16               = 0x0051,               /* 16 bit differential PCM (XI only) */

        SF_FORMAT_VORBIS                = 0x0060,               /* Xiph Vorbis encoding. */
#endif

static int get_item_bytes(int const subtype) {
  /* return file item bytes, or -1 for variable, or -2 for unknown subtype */
  switch( subtype ) {
    default: return -2;

    case SF_FORMAT_PCM_S8: return 1;
    case SF_FORMAT_PCM_16: return 2;
    case SF_FORMAT_PCM_24: return 3;
    case SF_FORMAT_PCM_32: return 4;
    case SF_FORMAT_PCM_U8: return 1;
    case SF_FORMAT_FLOAT: return 4;
    case SF_FORMAT_DOUBLE: return 8;
    case SF_FORMAT_ULAW: return 1;
    case SF_FORMAT_ALAW: return 1;

    case SF_FORMAT_IMA_ADPCM: return -1;
    case SF_FORMAT_MS_ADPCM: return -1;
    case SF_FORMAT_GSM610: return -1;
    case SF_FORMAT_VOX_ADPCM: return -1;
    case SF_FORMAT_G721_32: return -1;
    case SF_FORMAT_G723_24: return -1;
    case SF_FORMAT_G723_40: return -1;
    case SF_FORMAT_DWVW_12: return -1;
    case SF_FORMAT_DWVW_16: return 2;
    case SF_FORMAT_DWVW_24: return 3;
    case SF_FORMAT_DWVW_N: return -1;
    case SF_FORMAT_DPCM_8: return 1;
    case SF_FORMAT_DPCM_16: return 2;
    case SF_FORMAT_VORBIS: return -1;
  }
}

static char const * get_item_coding(int const subtype) {
  switch( subtype ) {
  default: return "unknown";
  case SF_FORMAT_PCM_S8: return "int8";
  case SF_FORMAT_PCM_16: return "int16";
  case SF_FORMAT_PCM_24: return "int24";
  case SF_FORMAT_PCM_32: return "int32";
  case SF_FORMAT_PCM_U8: return "uns8";
  case SF_FORMAT_FLOAT: return "float32";
  case SF_FORMAT_DOUBLE: return "float64";
  case SF_FORMAT_ULAW: return "ulaw";
  case SF_FORMAT_ALAW: return "alaw";
  case SF_FORMAT_IMA_ADPCM: return "adpcm_ima";
  case SF_FORMAT_MS_ADPCM: return "adpcm_ms";
  case SF_FORMAT_GSM610: return "gsm610";
  case SF_FORMAT_VOX_ADPCM: return "adpcm_vox";
  case SF_FORMAT_G721_32: return "adpcm_g721_32";
  case SF_FORMAT_G723_24: return "adpcm_g723_24";
  case SF_FORMAT_G723_40: return "adpcm_g723_40";
  case SF_FORMAT_DWVW_12: return "dwvw_12";
  case SF_FORMAT_DWVW_16: return "dwvw_16";
  case SF_FORMAT_DWVW_24: return "dwvw_24";
  case SF_FORMAT_DWVW_N: return "dwvw_n";
  case SF_FORMAT_DPCM_8: return "dpcm8";
  case SF_FORMAT_DPCM_16: return "dpcm16";
  case SF_FORMAT_VORBIS: return "vorbis";
  }
}

#define FUNC get_sndfile_info
static char const * FUNC(SF_INFO const sf_info, char * * const info, char * * const casual, char const * const name) {
  assert( info != NULL && *info == NULL );
  assert( casual != NULL && *casual == NULL );
  assert( name != NULL );

  int err;

  /* major type format info */
  SF_FORMAT_INFO format_info_major = {0};
  format_info_major.format = sf_info.format & SF_FORMAT_TYPEMASK;
  err = sf_command(NULL, SFC_GET_FORMAT_INFO, &format_info_major, sizeof(format_info_major));
  ERR_NON_ZERO(err, "unexpected error %d from sf_command SFC_GET_FORMAT_INFO\n", err);

  /* minor subtype format info */
  SF_FORMAT_INFO format_info_subtype = {0};
  format_info_subtype.format = sf_info.format & SF_FORMAT_SUBMASK;
  err = sf_command(NULL, SFC_GET_FORMAT_INFO, &format_info_subtype, sizeof(format_info_subtype));
  ERR_NON_ZERO(err, "unexpected error %d from sf_command SFC_GET_FORMAT_INFO subtype\n", err);

  /* capture some structured info */
  int const audio_sample_rate = sf_info.samplerate;
  int const audio_num_channels = sf_info.channels;
  int const audio_num_samples = sf_info.frames;
  //fprintf(stderr, "%s: audio_num_samples %d\n", LIBRARY_NAME, audio_num_samples);
  int const file_sndfile_format = sf_info.format;
  char const * const file_sndfile_extension = format_info_major.extension;
  int const file_item_bytes = get_item_bytes(sf_info.format & SF_FORMAT_SUBMASK);
  char const * const file_item_coding = get_item_coding(sf_info.format & SF_FORMAT_SUBMASK);
  //  assert( file_item_bytes > 0 );
  assert( file_item_coding != NULL );

  /* create a parseable string */
  err = asprintf(info,
                 "audio_sample_rate %d  audio_num_channels %d  audio_num_samples %d  file_sndfile_format %#x  file_item_bytes %d  file_item_coding %s  file_sndfile_extension %s",
                 audio_sample_rate, audio_num_channels, audio_num_samples, file_sndfile_format, file_item_bytes, file_item_coding, file_sndfile_extension);
  ERR_NEGATIVE(err, "asprintf to info failed with errno %d: %s", errno, strerror(errno));

  /* capture some unstructured info */
  char const * const file_type = format_info_major.name;
  char const * const file_encoding = format_info_subtype.name;
  /* create an informal string */
  err = asprintf(casual, "%s %s", file_encoding, file_type);
  ERR_NEGATIVE(err, "asprintf to informal failed with errno %d: %s", errno, strerror(errno));

  return NULL;
}
#undef FUNC


#define READ_WAVE(TYPE)  do { \
  num_bytes = num_items * sizeof(TYPE); \
  *wave = malloc(num_bytes); \
  ERR_NULL(*wave, "failed to allocate %zu bytes", num_bytes); \
  got = sf_read_##TYPE(sndfile, *wave, num_items); \
  } while( false )

#define FUNC get_sndfile_wave
static char const * FUNC(SNDFILE * const sndfile, SF_INFO const sf_info, char const * const wave_format, void * * const wave, char const * const name) {
  assert( sndfile != NULL );
  assert( wave_format != NULL );
  assert( wave != NULL && *wave == NULL );
  assert( name != NULL );

  int err;

  // count the data
  sf_count_t const num_items = (sf_count_t)sf_info.channels * sf_info.frames;
  //fprintf(stderr, "%s: num_items %lld\n", LIBRARY_NAME, num_items);

  size_t num_bytes;
  sf_count_t got;
  // get the data
  if( !strcmp(wave_format, "int16") ) {
    READ_WAVE(short);
  }
  else if( !strcmp(wave_format, "int32") ) {
    READ_WAVE(int);
  }
  else if( !strcmp(wave_format, "float32") ) {
    READ_WAVE(float);
  }
  else if( !strcmp(wave_format, "float64") ) {
    READ_WAVE(double);
  }
  else {
    ERR_RETURN(ValueError, "unexpected value of wave_format '%s'", wave_format);
  }
  err = sf_error(sndfile);
  ERR_NOT_EQUAL(err, SF_ERR_NO_ERROR, "unexpected libsndfile error %d while reading wave data: %s  file '%s'\n", err, sf_error_number(err), name);

  //fprintf(stderr, "%s: allocated %ld bytes for wave data: *wave %8p (at wave %8p)\n", LIBRARY_NAME, num_bytes, *wave, wave);
  //ERR_NOT_EQUAL(got, num_items, "expected to read %lld items, got %lld: file '%s'\n", num_items, got, name);
  ERR_NOT_EQUAL(got, num_items, "expected to read %jd items, got %jd: file '%s'\n", num_items, got, name);

  /*
  if( err != SF_ERR_NO_ERROR ) {
    // XXX make this an error
    fprintf(stderr, "unexpected libsndfile error %d while reading wave data: %s  file '%s'\n", err, sf_error_number(err), name);
  }
  */
  return NULL;
}
#undef FUNC
