///////////////////////////////////////////////////////////////////////////
//
// File:         audio.cpp
// Date:         28-Oct-2007
// Author:       Hugh Secker-Walker
// Description:  Run live audio for a while
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


#include "audio.h"
#include "LiveAudioInput.h"

#include <unistd.h>


int runlive(int const seconds, bool const verbose, bool const verbose2) {

  LiveAudioInput recorder;
  recorder.SetVerbose(verbose);
  recorder.SetVerbose2(verbose2);
  recorder.ConfigureAU();

  recorder.Start();
  sleep(1);
  recorder.Stop();
  sleep(1);

  recorder.Start();
  // note: negative seconds will cause very long sleeps
  sleep(seconds);
  recorder.Stop();

  return 0;
}

