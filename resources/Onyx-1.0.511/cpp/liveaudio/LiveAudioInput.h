///////////////////////////////////////////////////////////////////////////
//
// File:         LiveAudioInput.h
// Date:         28-Oct-2007
// Author:       Hugh Secker-Walker (based on sample code from Apple)
// Description:  Simple live audio input for Mac
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

#ifndef ONYX_GUARD_LIVEAUDIOINPUT_H
#define ONYX_GUARD_LIVEAUDIOINPUT_H

#include <CoreAudio/CoreAudio.h>
#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>

#include "ring_buffer_threads.h"

#include<vector>

typedef std::vector<AudioBufferList *> buffer_lists;

typedef OSStatus (render_callback)(void*, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32, AudioBufferList*);
typedef void *(*thread_func)(void *);

class LiveAudioInput
{
 public:
  LiveAudioInput();
  virtual ~LiveAudioInput();

  void SetVerbose(bool const verbose);
  void SetVerbose2(bool const verbose);

  AudioBufferList       *AllocateAudioBufferList(UInt32 numChannels, UInt32 size);
  void  DestroyAudioBufferList(AudioBufferList* list);

  OSStatus      ConfigureAU(AudioDeviceID deviceId=0, thread_func worker_func=0, render_callback callback_func=NULL);
  OSStatus      Start();
  OSStatus      Stop();

  void dropped() {
    ++num_dropped;
  }
  UInt32 get_dropped() const {
    return num_dropped;
  }

  bool is_running() const {
    return fRunning;
  }

  bool is_verbose() const {
    return fVerbose;
  }

  AudioBufferList       *fAudioBuffer;
  AudioUnit     fAudioUnit;


  UInt32 const nbuffers;
  buffer_lists buffers;
  ring_dude dude;
  UInt32 num_dropped;
  pthread_t child_thread;
  thread_func fWorkerThreadFunc;

 protected:
  static OSStatus AudioInputProc(void* inRefCon, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList* ioData);

  bool fRunning;
  bool fVerbose;
  bool fVerbose2;

  AudioDeviceID fInputDeviceID;
  UInt32        fAudioChannels, fAudioSamples;
  AudioStreamBasicDescription   fOutputFormat, fDeviceFormat;
  FSRef fOutputDirectory;

};

#endif  // ONYX_GUARD_LIVEAUDIOINPUT_H
