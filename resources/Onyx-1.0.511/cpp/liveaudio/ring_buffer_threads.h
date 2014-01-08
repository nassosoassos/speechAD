///////////////////////////////////////////////////////////////////////////
//
// File:         ring_buffer_threads.h (directory: cpp/liveaudio)
// Date:         2007-12-06 Thu 19:34:47
// Author:       Hugh Secker-Walker
// Description:  Header for a thread-safe ring buffer
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2007 - 2009 The Johns Hopkins University
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

#include "onyx_defs.h"

#include <pthread.h>

class ring_dude {
  UInt32 const size;

  bool running;
  volatile UInt32 x0;
  UInt32 x1;
  UInt32 x2;
  volatile UInt32 x3;
  // overflow
  bool reserve_clog;
  // underflow
  bool receive_clog;

  pthread_mutex_t mutex;
  pthread_cond_t cond;

  UInt32 next(UInt32 value) const {
    if( ++value >= size ) {
      ASSERT( value == size );
      value = 0;
    }
    return value;
  }

public:

  ring_dude(UInt32 const size) : size(size) {
    VERIFY_NULL(pthread_mutex_init(&mutex, NULL));
    VERIFY_NULL(pthread_cond_init(&cond, NULL));
    reset();
  }

  ~ring_dude() {
    reset();
    // XXX need to revisit case of a thread waiting in receive()....
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&cond);
  }

  // XXX this object could do with some refactoring; the overflow/underflow
  // conditions should be set when they arise and then used appropriately; we
  // could probably simplifiy things by allowing only one outstanding reserved()
  // index.  We could arrange things so that the OS's audio thread doesn't have
  // to aquire any lock.

  void reset() {
    //stop();
    //fprintf(stdout, "ring_dude: reset get lock\n");
    VERIFY_NULL(pthread_mutex_lock(&mutex));
    //ASSERT( running == false );
    //fprintf(stdout, "ring_dude: reset\n");
    running = false;
    x0 = 0;
    x1 = 0;
    x2 = 0;
    x3 = 0;
    // reserve_clog is true when ring is full: overflow
    reserve_clog = false;
    // receive_clog is true when ring is empty: underflow
    receive_clog = true;
    VERIFY_NULL(pthread_cond_signal(&cond));
    VERIFY_NULL(pthread_mutex_unlock(&mutex));
  }

  void start() {
    // non-blocking, called by supervisory thread
    VERIFY_NULL(pthread_mutex_lock(&mutex));
    running = true;
    //fprintf(stdout, "ring_dude: start\n");
    VERIFY_NULL(pthread_mutex_unlock(&mutex));
  }

  void stop() {
    // non-blocking, called by supervisory thread

    //fprintf(stdout, "ring_dude: stop get lock\n");
    VERIFY_NULL(pthread_mutex_lock(&mutex));
    running = false;
    //fprintf(stdout, "ring_dude: stop signal\n");

    // note: worker thread waiting in receive() can continue after we've
    // returned....
    VERIFY_NULL(pthread_cond_signal(&cond));
    // fprintf(stdout, "ring_dude: stop get unlock\n");
    VERIFY_NULL(pthread_mutex_unlock(&mutex));
    // fprintf(stdout, "ring_dude: stop done\n");
  }

  // reserve() and deliver() are used by a time-senstive thread (e.g. system
  // audio callback) to put data into the ring buffer with non-blocking calls

  int reserve() {
    // non-blocking, called by time-sensitive thread to reserve an index that it
    // can use; returns -1 on buffer overflow
    VERIFY_NULL(pthread_mutex_lock(&mutex));
    ASSERT( running );
    int const ret = (reserve_clog || size == 0u ? -1 : x0);
    if( ret != -1 ) {
      x0 = next(x0);
      if( x0 == x3 ) {
        // XXX shouldn't this be: reserve_clog = (x0 == x3);
        reserve_clog = true;
      }
    }
    VERIFY_NULL(pthread_mutex_unlock(&mutex));
    return ret;
  }

  void deliver(UInt32 const check) {
    // non-blocking, called by time-sensitive thread to announce that it is done
    // with the most recent valid index that it reserved.
    ASSERT( size > 0u );
    VERIFY_NULL(pthread_mutex_lock(&mutex));
    ASSERT( running );
    ASSERT( check == x1 );
    x1 = next(x1);
    receive_clog = false;
    //fprintf(stdout, "ring_dude: deliver x1 %d\n", x1);
    // let the other thread work on this data
    VERIFY_NULL(pthread_cond_signal(&cond));
    VERIFY_NULL(pthread_mutex_unlock(&mutex));
  }


  // receive() and done() are used by the worker thread to pull data out of the
  // ring buffer and deliver it to the client

  int receive() {
    // blocking, waits for other thread to deliver() data

    // XXX needs to handle overflow (clog) condition

    ASSERT( size > 0u );

    VERIFY_NULL(pthread_mutex_lock(&mutex));
    // fprintf(stdout, "ring_dude: receive: lock 0x%08x\n", pthread_self());
    while( running && receive_clog ) {
      //fprintf(stdout, "ring_dude: receive: waiting 0x%08x\n", pthread_self());
      VERIFY_NULL(pthread_cond_wait(&cond, &mutex));
    }
    // fprintf(stdout, "ring_dude: receive: waited\n");

    int const ret = (running ? x2 : -1);
    if( ret != -1 ) {
      ASSERT( receive_clog == false );
      //fprintf(stdout, "ring_dude: receive: x2 %d\n", x2);
      x2 = next(x2);
      if( x2 == x1 ) {
        // XXX shouldn't this be: receive_clog = (x2 == x1);
        receive_clog = true;
      }
    }
    else {
      //fprintf(stdout, "ring_dude: receive stop 0x%08x\n", pthread_self());
    }

    VERIFY_NULL(pthread_mutex_unlock(&mutex));
    //fprintf(stdout, "ring_dude: receive: unlocked ret %d 0x%08x\n", ret, pthread_self());
    return ret;
  }

  void done(UInt32 const check) {
    // non-blocking, called to free up most recently receive()ed index
    ASSERT( size > 0u );
    VERIFY_NULL(pthread_mutex_lock(&mutex));
    ASSERT( check == x3 );
    x3 = next(x3);
    reserve_clog = false;
    VERIFY_NULL(pthread_mutex_unlock(&mutex));
  }

};
