///////////////////////////////////////////////////////////////////////////
//
// File:         main.cpp (directory: cpp/liveaudio)
// Date:         2007-10-31 Wed 13:47:52
// Author:       Hugh Secker-Walker
// Description:  Code for testing live input on the Mac
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

#include <stdio.h>
#include <stdlib.h>

#include "audio.h"
//#include "audio2.h"
//#include "audio3.h"

#include "delay_line.h"


void test_delay() {
  
  typedef onyx::DelayLine<5u, int> int_8_delay;
  int_8_delay mydelay;
  
  fprintf(stdout, "\ndelay line:\n");
  for( int i = 1; i <= 25; ++i ) {
    fprintf(stdout, "%d  %d\n", i, mydelay.update(i));
    if( i == 16 ) {
      mydelay.reset();
    }
  }
}


int main(int argc, char **argv) {
  
  bool const verbose = argc > 1;
  int const seconds = verbose ? atoi(argv[1]) : 1;

  {
    //    Audio aud;
    // aud.runlive(seconds, verbose);
  }

  {
    //    Foo const f;
  }

  int const liveval = runlive(seconds, verbose, true);

  fprintf(stdout, "liveval %d\n", liveval);

  test_delay();

  return 0;
}
