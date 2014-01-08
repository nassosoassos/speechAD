///////////////////////////////////////////////////////////////////////////
//
// File:         mpg123wrap.c
// Date:         15-Mar-2009
// Author:       Hugh Secker-Walker
// Description:  Wrapper for mpg123 MPEG decoder library for ctypes usage
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

#include <assert.h>
#include <stdbool.h>
#include <string.h>
#include <errno.h>
#include <limits.h>
#include <unistd.h>
#include <inttypes.h>

#include <mpg123.h>

#include "mpg123wrap.h"
#include "ctypes_defs.h"

#define LIBRARY_NAME "mgp123wrap"

static char error_msg[1024];


#define FUNC mpg123_init_wrap
char const * FUNC() {
  int const err = mpg123_init();
  ERR_NOT_EQUAL(err, MPG123_OK, "mpg123_init failed with code %d", err);
  return NULL;
}
#undef FUNC

char const * mpg123_exit_wrap() {
  mpg123_exit();
  return NULL;
}


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


static const char * mpg123_encoding_string(int const encoding) {
  switch( encoding ) {
  default:
    return "unknown";
  case MPG123_ENC_SIGNED_16:    /* = (MPG123_ENC_16|MPG123_ENC_SIGNED|0x10)  **<           1101 0000 signed 16 bit   */
    return "int16";
  case MPG123_ENC_UNSIGNED_16:  /* = (MPG123_ENC_16|0x20)                    **<           0110 0000 unsigned 16 bit */
    return "uns16";
  case MPG123_ENC_UNSIGNED_8:   /* = 0x01                                    **<           0000 0001 unsigned 8 bit  */
    return "uns8";
  case MPG123_ENC_SIGNED_8:     /* = (MPG123_ENC_SIGNED|0x02)                **<           1000 0010 signed 8 bit    */
    return "int8";
  case MPG123_ENC_ULAW_8:       /* = 0x04                                    **<           0000 0100 ulaw 8 bit      */
    return "ulaw";
  case MPG123_ENC_ALAW_8:       /* = 0x08                                    **<           0000 1000 alaw 8 bit      */
    return "alaw";
  case MPG123_ENC_SIGNED_32:    /* = MPG123_ENC_32|MPG123_ENC_SIGNED|0x1000  **< 0001 0001 1000 0000 signed 32 bit   */
    return "int32";
  case MPG123_ENC_UNSIGNED_32:  /* = MPG123_ENC_32|0x2000                    **< 0010 0001 0000 0000 unsigned 32 bit */
    return "uns32";
  case MPG123_ENC_FLOAT_32:     /* = 0x200                                   **<      0010 0000 0000 32bit float     */
    return "float32";
  case MPG123_ENC_FLOAT_64:     /* = 0x400                                   **<      0100 0000 0000 64bit float     */
    return "float64";
  }
}

#define MPG_ERR_STR(ERR, MPG) (ERR == MPG123_ERR ? mpg123_strerror(MPG) : mpg123_plain_strerror(ERR))

#define FUNC get_audio
char const * FUNC(int const fd, char const * const name, char const * const wave_format, char * * const info, char * * const casual, void * * const wave) {
  ERR_NEGATIVE(fd, "expected non-negative file descriptor, got %d", fd);
  ERR_NULL(name, "%s", "expected non-NULL name pointer");
  ERR_IF(info == NULL || casual == NULL || wave == NULL, ValueError, "expected non-NULL info, casual, and wave return pointer-pointers, but they are %8p, %8p, and %8p", info, casual, wave);
  ERR_IF(info == casual || (void**)info == wave || (void**)casual == wave, ValueError, "expected info, casual, and wave return pointer-pointers to be distinct, but they are %8p, %8p, and %8p", info, casual, wave);
  ERR_IF(*info != NULL || *casual != NULL || *wave != NULL, ValueError, "expected NULL info, casual, and wave at the return pointer-pointers, but they are %8p, %8p, and %8p", *info, *casual, *wave);
  // note: we make sure that malloc is used for the data we put at these
  // pointers; the caller uses the free_ptr function above to free up the data

  int err;

  // first pass to get the file info
  long rate = 0;
  int channels = 0;
  int file_encoding = 0;
  off_t num_samples = 0;
  struct mpg123_frameinfo frameinfo = {0};
  {
      off_t const off_t_zero = 0;
      off_t const fd_start = lseek(fd, off_t_zero, SEEK_CUR);
      ERR_NEGATIVE(fd_start, "lseek failed for file '%s': errno %d: '%s'", name, errno, strerror(errno));

      mpg123_handle * const mpg0 = mpg123_new(NULL, &err);
      ERR_NOT_EQUAL(err, MPG123_OK, "mpg123_new failed: '%s'", MPG_ERR_STR(err, mpg0));
      ERR_NULL(mpg0, "%s", "unexpected mpg123_handle, NULL, from mpg123_new");

      // verbosity
      err = mpg123_param(mpg0, MPG123_VERBOSE, 1, 0);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg0)),
                            "mpg123_param MPG123_VERBOSE failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg0));
      // more verbosity and error recovery
      //err = mpg123_param(mpg0, MPG123_ADD_FLAGS, MPG123_QUIET|MPG123_NO_RESYNC, 0);
      err = mpg123_param(mpg0, MPG123_ADD_FLAGS, MPG123_NO_RESYNC, 0);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg0)),
                            "mpg123_param MPG123_ADD_FLAGS MPG123_NO_RESYNC failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg0));

      err = mpg123_open_fd(mpg0, fd);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg0)),
                            "mpg123_open_fd failed on fd %d, file '%s': '%s'", fd, name, MPG_ERR_STR(err, mpg0));

      err = mpg123_getformat(mpg0, &rate, &channels, &file_encoding);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_close(mpg0), mpg123_delete(mpg0)),
                            "mpg123_getformat failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg0));

      err = mpg123_info(mpg0, &frameinfo);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_close(mpg0), mpg123_delete(mpg0)),
                            "mpg123_info failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg0));

      err = mpg123_scan(mpg0);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_close(mpg0), mpg123_delete(mpg0)),
                            "mpg123_scan failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg0));

      num_samples = mpg123_length(mpg0);
      ERR_EQUAL_CLEANUP(num_samples, MPG123_ERR,
                        (mpg123_close(mpg0), mpg123_delete(mpg0)),
                        "mpg123_length failed for file '%s': '%s'", name, mpg123_strerror(mpg0));

      err = mpg123_close(mpg0);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg0)),
                            "mpg123_close failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg0));

      mpg123_delete(mpg0);

      if( rate <= 0 ) {
        ERR_RETURN(AudioTypeError, "invalid sample rate %ld for file '%s'", rate, name);
      }

      if( channels <= 0 ) {
        ERR_RETURN(AudioTypeError, "invalid number of channels %d for file '%s'", channels, name);
      }

      if( file_encoding == 0x0 ) {
        ERR_RETURN(AudioTypeError, "unexpected zero encoding for file '%s'", name);
      }

      if( frameinfo.rate != rate ) {
        ERR_RETURN(AudioTypeError,
                   "expected consistent rates from format and from info for file '%s', got %ld and %ld", name, rate, frameinfo.rate);
      }

      off_t const fd_done = lseek(fd, fd_start, SEEK_SET);
      ERR_NEGATIVE(fd_done, "lseek failed for file '%s': errno %d: '%s'", name, errno, strerror(errno));
      // note: this is a recommended way to have a portable format qualifier for off_t....
      ERR_NOT_EQUAL(fd_done, fd_start, "unexpected lseek for file '%s': expected %jd, got %jd", name, (intmax_t)fd_start, (intmax_t)fd_done);
  }
  //fprintf(stderr, "0 rate %ld  channels %d  file_encoding %#x\n", rate, channels, file_encoding);

  if( wave_format != NULL ) {
      // second pass to read all the audio
      size_t const num_items = num_samples * channels;
      size_t num_bytes;
      int audio_encoding;
      if( !strcmp(wave_format, "int16") ) {
        audio_encoding = MPG123_ENC_SIGNED_16;
        assert( sizeof(short) * CHAR_BIT == 16 );
        num_bytes = num_items * sizeof(short);
      }
      else if( !strcmp(wave_format, "int32") ) {
        audio_encoding = MPG123_ENC_SIGNED_32;
        assert( sizeof(int) * CHAR_BIT == 32 );
        num_bytes = num_items * sizeof(int);
      }
      else if( !strcmp(wave_format, "float32") ) {
        audio_encoding = MPG123_ENC_FLOAT_32;
        assert( sizeof(float) * CHAR_BIT == 32 );
        num_bytes = num_items * sizeof(float);
      }
      else if( !strcmp(wave_format, "float64") ) {
        audio_encoding = MPG123_ENC_FLOAT_64;
        assert( sizeof(double) * CHAR_BIT == 64 );
        num_bytes = num_items * sizeof(double);
      }
      else {
        ERR_RETURN(ValueError, "unexpected value of wave_format '%s' for file '%s'", wave_format, name);
      }

      mpg123_handle * const mpg = mpg123_new(NULL, &err);
      ERR_NOT_EQUAL(err, MPG123_OK, "mpg123_new failed: '%s'", MPG_ERR_STR(err, mpg));
      ERR_NULL(mpg, "%s", "unexpected mpg123_handle, NULL, from mpg123_new");

      // verbosity
      err = mpg123_param(mpg, MPG123_VERBOSE, 1, 0);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg)),
                            "mpg123_param MPG123_VERBOSE failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg));
      // more verbosity and error recovery
      err = mpg123_param(mpg, MPG123_ADD_FLAGS, MPG123_NO_RESYNC, 0);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg)),
                            "mpg123_param MPG123_ADD_FLAGS MPG123_NO_RESYNC failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg));

      err = mpg123_format_support(mpg, rate, audio_encoding);
      ERR_ZERO_CLEANUP(err,
                       (mpg123_delete(mpg)),
                       "unsupported format '%s' for file '%s'", wave_format, name);

      err = mpg123_format_none(mpg);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg)),
                            "mpg123_format_none failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg));

      err = mpg123_format(mpg, rate, channels, audio_encoding);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg)),
                            "mpg123_format (%ld, %d, %#x) failed for file '%s': '%s'", rate, channels, audio_encoding, name, MPG_ERR_STR(err, mpg));

      /*
      // we use the generic decoder in a bid for cross-platform reproducibility
      char const * decoder_name = "generic";
      err = mpg123_decoder(mpg, decoder_name);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg)),
                            "mpg123_decoder '%s' failed for file '%s': '%s'", decoder_name, name, MPG_ERR_STR(err, mpg));
      char const * check_decoder_name = mpg123_current_decoder(mpg);
      ERR_NON_ZERO_CLEANUP(strcmp(check_decoder_name, decoder_name),
                           (mpg123_delete(mpg)),
                           "expected decoder name '%s', got '%s'", decoder_name, check_decoder_name);
      */

      err = mpg123_open_fd(mpg, fd);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                           (mpg123_delete(mpg)),
                            "mpg123_open_fd failed on fd %d, file '%s': '%s'", fd, name, MPG_ERR_STR(err, mpg));

      {
        long check_rate;
        int check_channels;
        int check_audio_encoding;
        err = mpg123_getformat(mpg, &check_rate, &check_channels, &check_audio_encoding);
        ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                              (mpg123_close(mpg), mpg123_delete(mpg)),
                              "mpg123_getformat failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg));
        ERR_NOT_EQUAL_CLEANUP(check_rate, rate,
                              (mpg123_close(mpg), mpg123_delete(mpg)),
                              "expected rate to match for file '%s', got %ld and %ld", name, rate, check_rate);
        ERR_NOT_EQUAL_CLEANUP(check_channels, channels,
                              (mpg123_close(mpg), mpg123_delete(mpg)),
                              "expected channels to match for file '%s', got %d and %d", name, channels, check_channels);
        ERR_NOT_EQUAL_CLEANUP(check_audio_encoding, audio_encoding,
                              (mpg123_close(mpg), mpg123_delete(mpg)),
                              "expected audio_encoding to match for file '%s', got %d and %d", name, audio_encoding, check_audio_encoding);
      }

      *wave = malloc(num_bytes);
      ERR_NULL_CLEANUP(*wave,
                       (mpg123_close(mpg), mpg123_delete(mpg)),
                       "failed to allocate %zu bytes", num_bytes);

      //fprintf(stderr, "num_items %zu  num_bytes %zu  *wave %#x\n", num_items, num_bytes, *wave);

      // get the data
      size_t done;
      err = mpg123_read(mpg, *wave, num_bytes, &done);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_close(mpg), mpg123_delete(mpg)),
                            "mpg123_read failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg));
      ERR_NOT_EQUAL_CLEANUP(done, num_bytes,
                            (mpg123_close(mpg), mpg123_delete(mpg)),
                            "expected matching offsets for file '%s': got %zu and %zu", name, num_bytes, done);
      //fprintf(stderr, "done %zu\n", done);

      err = mpg123_close(mpg);
      ERR_NOT_EQUAL_CLEANUP(err, MPG123_OK,
                            (mpg123_delete(mpg)),
                            "mpg123_close failed for file '%s': '%s'", name, MPG_ERR_STR(err, mpg));

      mpg123_delete(mpg);
  }

  int const audio_sample_rate = rate;
  int const audio_num_channels = channels;
  int const audio_num_samples = num_samples;
  int const file_sndfile_format = file_encoding;
  int const file_item_bytes = -1;
  char const * const file_item_coding = mpg123_encoding_string(file_encoding);
  char const * const file_sndfile_extension = "mp3";

  /* create a parseable string */
  err = asprintf(info,
                 "audio_sample_rate %d  audio_num_channels %d  audio_num_samples %d  file_sndfile_format %#x  file_item_bytes %d  file_item_coding %s  file_sndfile_extension %s",
                 audio_sample_rate, audio_num_channels, audio_num_samples, file_sndfile_format, file_item_bytes, file_item_coding, file_sndfile_extension);
  ERR_NEGATIVE(err, "asprintf to info failed with errno %d: %s", errno, strerror(errno));

  /* create an informal string of info */
  char const * mpeg_version;
  switch( frameinfo.version ) {
    default: mpeg_version = "unknown"; break;
    case MPG123_1_0: mpeg_version = "1.0"; break;
    case MPG123_2_0: mpeg_version = "2.0"; break;
    case MPG123_2_5: mpeg_version = "2.5"; break;
  }

  char const * mpeg_mode;
  switch( frameinfo.mode ) {
    default: mpeg_mode = "unknown"; break;
    case MPG123_M_STEREO: mpeg_mode = "stereo"; break;
    case MPG123_M_JOINT: mpeg_mode = "joint_stereo"; break;
    case MPG123_M_DUAL: mpeg_mode = "dual"; break;
    case MPG123_M_MONO: mpeg_mode = "mono"; break;
  }

  err = asprintf(casual, "MPEG_Version %s  Audio_Layer %d  Channel_Mode %s  BitrateKbps %d",
                 mpeg_version, frameinfo.layer, mpeg_mode, frameinfo.bitrate);
  ERR_NEGATIVE(err, "asprintf to informal failed with errno %d: %s", errno, strerror(errno));

  return NULL;
}
#undef FUNC
