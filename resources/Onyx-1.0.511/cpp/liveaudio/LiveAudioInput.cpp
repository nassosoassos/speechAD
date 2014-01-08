///////////////////////////////////////////////////////////////////////////
//
// File:         LiveAudioInput.cpp
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

#include "LiveAudioInput.h"
#include "onyx_defs.h"

#include <sys/param.h>
#include <math.h>

#include<string>
#include<limits>
#include<cstdlib>


LiveAudioInput::LiveAudioInput()
  : nbuffers(10u), dude(nbuffers), num_dropped(0u), fRunning(false), fVerbose(false)
{
  fVerbose2 = false;
  fInputDeviceID = 0;
  fAudioChannels = fAudioSamples = 0;
  fWorkerThreadFunc = 0;
}

LiveAudioInput::~LiveAudioInput()
{
  if( fInputDeviceID != 0 ) {
    // Stop pulling audio data
    Stop();
  }

  while( !buffers.empty() ) {
    free(buffers.back());
    buffers.pop_back();
  }
}

void LiveAudioInput::SetVerbose(bool const verbose)
{
  fVerbose = verbose;
}

void LiveAudioInput::SetVerbose2(bool const verbose)
{
  fVerbose2 = verbose;
}

// Convenience function to allocate our audio buffers
AudioBufferList *LiveAudioInput::AllocateAudioBufferList(UInt32 numChannels, UInt32 size)
{
  fVerbose && fprintf(stdout, "AllocateAudioBufferList: numChannels %lu  size %lu\n", numChannels, size);

  AudioBufferList*                      list;
  UInt32                                                i;

  list = (AudioBufferList*)calloc(1, sizeof(AudioBufferList) + numChannels * sizeof(AudioBuffer));
  if(list == NULL)

    return NULL;

  list->mNumberBuffers = numChannels;
  for(i = 0; i < numChannels; ++i) {
    list->mBuffers[i].mNumberChannels = 1;
    list->mBuffers[i].mDataByteSize = size;
    list->mBuffers[i].mData = malloc(size);
    if(list->mBuffers[i].mData == NULL) {
      DestroyAudioBufferList(list);
      return NULL;
    }
  }
  return list;
}

// Convenience function to dispose of our audio buffers
void LiveAudioInput::DestroyAudioBufferList(AudioBufferList* list)
{
  UInt32                                                i;

  if(list) {
    for(i = 0; i < list->mNumberBuffers; i++) {
      if(list->mBuffers[i].mData)
        free(list->mBuffers[i].mData);
    }
    free(list);
  }
}


static void printDeviceName(AudioDeviceID const deviceID, char const * const label)
{
  fprintf(stdout, "\n%s\n", label);
  Boolean const Input = 1;
  UInt32 size;
  AudioDeviceGetPropertyInfo(deviceID, 0, Input, kAudioDevicePropertyDeviceName, &size, NULL);
  char *name = (char *)malloc(size);
  AudioDeviceGetProperty(deviceID, 0, Input, kAudioDevicePropertyDeviceName, &size, name);
  fprintf(stdout, "name %s\n", name);
  free(name);
}

static void printAudioFormat(AudioStreamBasicDescription const &format, char const * const label)
{
  fprintf(stdout, "\n%s\n", label);
  fprintf(stdout, "%p\n", &format);

  fprintf(stdout, "mSampleRate %g\n", format.mSampleRate);
  fprintf(stdout, "mFormatID %#lx\n", format.mFormatID);

  fprintf(stdout, "mFormatFlags %#lx  ", format.mFormatFlags);
  if( format.mFormatFlags & kAudioFormatFlagIsFloat) fprintf(stdout, "kAudioFormatFlagIsFloat ");
  if( format.mFormatFlags & kAudioFormatFlagIsPacked) fprintf(stdout, "kAudioFormatFlagIsPacked ");
  if( format.mFormatFlags & kAudioFormatFlagIsNonInterleaved) fprintf(stdout, "kAudioFormatFlagIsNonInterleaved ");
  if( format.mFormatFlags & kLinearPCMFormatFlagIsNonInterleaved) fprintf(stdout, "kLinearPCMFormatFlagIsNonInterleaved ");
  if( format.mFormatFlags & kAudioFormatFlagIsBigEndian) fprintf(stdout, "kAudioFormatFlagIsBigEndian ");
  fprintf(stdout, "\n");

  fprintf(stdout, "mChannelsPerFrame %lu\n", format.mChannelsPerFrame);
  fprintf(stdout, "mBitsPerChannel %lu\n", format.mBitsPerChannel);
  fprintf(stdout, "mBytesPerFrame %lu\n", format.mBytesPerFrame);
  fprintf(stdout, "mFramesPerPacket %lu\n", format.mFramesPerPacket);
  fprintf(stdout, "mBytesPerPacket %lu\n", format.mBytesPerPacket);

  fprintf(stdout, "\n");

}


OSStatus LiveAudioInput::AudioInputProc(void* inRefCon, AudioUnitRenderActionFlags* ioActionFlags, const AudioTimeStamp* inTimeStamp, UInt32 inBusNumber, UInt32 inNumberFrames, AudioBufferList* ioData)
{
  // the callback function that's called by the OS's audio thread; we mustn't
  // get blocked

  OSStatus      err = noErr;

  LiveAudioInput &afr = *static_cast<LiveAudioInput * const>(inRefCon);

  // note: shadows the member variable of the same name
  bool const fVerbose = afr.fVerbose;
  VERBOSE( fVerbose && fprintf(stdout, "\n inNumberFrames %d  inTimeStamp %lld  inTimeStamp %f ", inNumberFrames, inTimeStamp->mHostTime, inTimeStamp->mSampleTime); );
  VERBOSE( fVerbose && fprintf(stdout, "inNumberFrames %d  ioData 0x%x  ioData == afr.fAudioBuffer %d\n", inNumberFrames, ioData, ioData == afr.fAudioBuffer); );

  int const index = afr.dude.reserve();
  VERBOSE( fVerbose && fprintf(stdout, "dude.reserve() %d\n", index); );
  if( index < 0 ) {
    afr.dropped();
    //err = AudioUnitRender(afr.fAudioUnit, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, afr.fAudioBuffer);
    //VERIFY_NULL(err);
    return noErr;
  }

  // Render into audio buffer
  //  err = AudioUnitRender(afr.fAudioUnit, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, afr.fAudioBuffer);
  err = AudioUnitRender(afr.fAudioUnit, ioActionFlags, inTimeStamp, inBusNumber, inNumberFrames, afr.buffers[index]);
  VERIFY_NULL(err);
  if(err) {
    fprintf(stdout, "AudioUnitRender() failed with error %ld\n", err);
    return err;
  }

  afr.dude.deliver(index);
  return noErr;

  // the rest is unreachable code that may be instructive....

  // get hardware device format
  AudioStreamBasicDescription format;
  UInt32 size = sizeof(format);
  err = AudioUnitGetProperty(afr.fAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &format, &size);
  if(err != noErr)
    {
      fprintf(stdout, "AudioInputProc failed to get input device ASBD\n");
      return err;
    }
  VERBOSE( fVerbose && (printAudioFormat(afr.fDeviceFormat, "AudioInputProc fDeviceFormat"), true); )


    {
      //CAStreamBasicDescription DeviceFormat;
      AudioStreamBasicDescription DeviceFormat;
      //Use CAStreamBasicDescriptions instead of 'naked'
      //AudioStreamBasicDescriptions to minimize errors.
      //CAStreamBasicDescription.h can be found in the CoreAudio SDK.

      UInt32 size = sizeof(DeviceFormat);

      //Get the input device format
      AudioUnitGetProperty (afr.fAudioUnit,
                            kAudioUnitProperty_StreamFormat,
                            kAudioUnitScope_Output,
                            0,
                            &DeviceFormat,
                            &size);
      VERBOSE( fVerbose && fprintf(stdout, "AudioInputProc AudioStreamBasicDescription mSampleRate %d\n", DeviceFormat.mSampleRate); )
        }


  // variance calculations (DC normalized RMS amplitude)
  {
    AudioBuffer const &audioBuffer = afr.fAudioBuffer->mBuffers[0];
    typedef float SampleType;
    UInt32 const nSamples = audioBuffer.mDataByteSize / sizeof(SampleType);
    assert( nSamples * sizeof(SampleType) == audioBuffer.mDataByteSize );
    SampleType const * const pSamples = (SampleType const * const)audioBuffer.mData;


    SampleType sum = 0;
    SampleType sumsq = 0;
    int i = nSamples;
    if( --i >= 0 ) {
      SampleType const SCALE = 1 << 15;
      // 8 significant bits, close to 0.98 (0.980469...)
      SampleType const alpha = static_cast<SampleType>(0xfb) / (1 << 8);
      SampleType const alphaScale = 1 / (1 - alpha);
      // equality on floats!
      assert( (1 - alpha) * alphaScale == 1 );
      SampleType wn = pSamples[i];
      while( --i >= 0 ) {

        // plain old preemphasis:
        //   y[n] =  x[n] - alpha * x[n-1]
        SampleType const alphaw = -alpha * wn;
        wn = pSamples[i];
        SampleType const sample = (wn + alphaw) * alphaScale;

        /*
        // bi-linear preemphasis:
        //   y[n] + alpha * y[n-1]  =  x[n] - alpha * x[n-1]
        SampleType const alphaw = -alpha * wn;
        wn = pSamples[i] + alphaw;
        SampleType const sample = wn + alphaw;
        */

        SampleType const scaledSample = sample * SCALE;
        sum += scaledSample;
        sumsq += scaledSample * scaledSample;
      }
    }
    assert( nSamples > 0 );
    SampleType const mean = sum / nSamples;
    SampleType const ms = sumsq / nSamples - mean * mean;
    assert( ms >= 0 );
    // use factor of 10 because we didn't take the sqrt yet
    int const dB10 = 1 + int(10 * log10(ms + 1));

    if( fVerbose ) {
      // std::string const spaces(dB10, ' ');
      // fprintf(stdout, "%s %3d \n", spaces.c_str(), dB10);

      if( true ) {
        std::string vu(dB10, ' ');
        vu += '*';
        vu.resize(100, ' ');
        fprintf(stdout, "\n  %s ", vu.c_str());
      }
      else if( false ) {
        std::string vu(dB10, '=');
        vu.resize(100, ' ');
        fprintf(stdout, "\r  %s\r", vu.c_str());
      }
      else {
        std::string vu(dB10, ' ');
        vu += '*';
        fprintf(stdout, "  %s\n", vu.c_str());
      }

      fflush(stdout);
    }
  }

  for( unsigned int i = 0; i < afr.fAudioBuffer->mNumberBuffers; ++i ) {
    VERBOSE( float *pvalue = (float *)(afr.fAudioBuffer->mBuffers[i].mData); )
    for( int j = 0; j < 10; ++j ) {
      VERBOSE( fVerbose && fprintf(stdout, "  %f", pvalue[j]); )
        }
    VERBOSE( fVerbose && fprintf(stdout, "\n"); )
      }

  return err;
}


OSStatus LiveAudioInput::ConfigureAU(AudioDeviceID deviceId, thread_func worker_func, render_callback callback_func)
{
  fWorkerThreadFunc = worker_func;

  Component                                     component;
  ComponentDescription          description;
  OSStatus      err = noErr;
  UInt32        param;
  AURenderCallbackStruct        callback;

  // Open the AudioOutputUnit
  // There are several different types of Audio Units.
  // Some audio units serve as Outputs, Mixers, or DSP
  // units. See AUComponent.h for listing
  description.componentType = kAudioUnitType_Output;
  description.componentSubType = kAudioUnitSubType_HALOutput;
  description.componentManufacturer = kAudioUnitManufacturer_Apple;
  description.componentFlags = 0;
  description.componentFlagsMask = 0;
  component = FindNextComponent(NULL, &description);
  VERIFY_TRUE(component);
  err = OpenAComponent(component, &fAudioUnit);
  VERIFY_NULL(err);
  if(err != noErr)
    {
      fAudioUnit = NULL;
      return err;
    }

  // Configure the AudioOutputUnit
  // You must enable the Audio Unit (AUHAL) for input and output for the same  device.
  // When using AudioUnitSetProperty the 4th parameter in the method
  // refer to an AudioUnitElement.  When using an AudioOutputUnit
  // for input the element will be '1' and the output element will be '0'.

  // Enable input on the AUHAL
  param = 1;
  err = AudioUnitSetProperty(fAudioUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Input, 1, &param, sizeof(UInt32));
  if(err == noErr)
    {
      // Disable Output on the AUHAL
      param = 0;
      err = AudioUnitSetProperty(fAudioUnit, kAudioOutputUnitProperty_EnableIO, kAudioUnitScope_Output, 0, &param, sizeof(UInt32));
    }

  if( deviceId == 0 ) {
    // Select the default input device
    param = sizeof(AudioDeviceID);
    err = AudioHardwareGetProperty(kAudioHardwarePropertyDefaultInputDevice, &param, &fInputDeviceID);
    if(err != noErr)
      {
        fprintf(stdout, "failed to get default input device\n");
        return err;
      }
  }
  else {
    fInputDeviceID = deviceId;
  }

  // Set the current device to the default input unit.
  err = AudioUnitSetProperty(fAudioUnit, kAudioOutputUnitProperty_CurrentDevice, kAudioUnitScope_Global, 0, &fInputDeviceID, sizeof(AudioDeviceID));
  if(err != noErr)
    {
      fprintf(stdout, "failed to set AU input device\n");
      return err;
    }

  fVerbose && (printDeviceName(fInputDeviceID, "fInputDeviceID"), true);

  // Setup render callback
  callback.inputProcRefCon = this;
  // This will be called when the AUHAL has input data
  callback.inputProc = (callback_func == NULL ? LiveAudioInput::AudioInputProc : callback_func);
  //  err = AudioUnitSetProperty(fAudioUnit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 0, &callback, sizeof(AURenderCallbackStruct));
  err = AudioUnitSetProperty(fAudioUnit, kAudioOutputUnitProperty_SetInputCallback, kAudioUnitScope_Global, 1, &callback, sizeof(AURenderCallbackStruct));

  // get hardware device format
  param = sizeof(AudioStreamBasicDescription);
  err = AudioUnitGetProperty(fAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1, &fDeviceFormat, &param);
  if(err != noErr)
    {
      fprintf(stdout, "failed to get input device ASBD\n");
      return err;
    }

  fVerbose && (printAudioFormat(fDeviceFormat, "fDeviceFormat"), true);

  // Twiddle the format to our liking
  fAudioChannels = MAX(fDeviceFormat.mChannelsPerFrame, 2);
  fOutputFormat.mChannelsPerFrame = fAudioChannels;
  fOutputFormat.mSampleRate = fDeviceFormat.mSampleRate;
  fOutputFormat.mFormatID = kAudioFormatLinearPCM;
  fOutputFormat.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked | kAudioFormatFlagIsNonInterleaved;
  if (fOutputFormat.mFormatID == kAudioFormatLinearPCM && fAudioChannels == 1)
    fOutputFormat.mFormatFlags &= ~kLinearPCMFormatFlagIsNonInterleaved;
#if __BIG_ENDIAN__
  fOutputFormat.mFormatFlags |= kAudioFormatFlagIsBigEndian;
#endif
  fOutputFormat.mBitsPerChannel = sizeof(Float32) * 8;
  fOutputFormat.mBytesPerFrame = fOutputFormat.mBitsPerChannel / 8;
  fOutputFormat.mFramesPerPacket = 1;
  fOutputFormat.mBytesPerPacket = fOutputFormat.mBytesPerFrame;

  fVerbose && (printAudioFormat(fOutputFormat, "fOutputFormat"), true);

  // Set the AudioOutputUnit output data format
  err = AudioUnitSetProperty(fAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &fOutputFormat, sizeof(AudioStreamBasicDescription));
  if(err != noErr)
    {
      fprintf(stdout, "failed to set input device ASBD\n");
      return err;
    }

  // Get the number of frames in the IO buffer(s)
  param = sizeof(UInt32);
  err = AudioUnitGetProperty(fAudioUnit, kAudioDevicePropertyBufferFrameSize, kAudioUnitScope_Global, 0, &fAudioSamples, &param);
  if(err != noErr)
    {
      fprintf(stdout, "failed to get audio sample size\n");
      return err;
    }
  fVerbose && fprintf(stdout, "fAudioSamples %lu\n", fAudioSamples);


  // Allocate our audio buffers
  fAudioBuffer = AllocateAudioBufferList(fOutputFormat.mChannelsPerFrame, fAudioSamples * fOutputFormat.mBytesPerFrame);
  if(fAudioBuffer == NULL)
    {
      fprintf(stdout, "failed to allocate buffers\n");
      return err;
    }

  {
    ASSERT( buffers.empty() );
    for( int i = nbuffers; --i >= 0; ) {
      AudioBufferList *const list = AllocateAudioBufferList(fOutputFormat.mChannelsPerFrame, fAudioSamples * fOutputFormat.mBytesPerFrame);
      if( list == NULL ) {
        fprintf(stdout, "failed to allocate buffers: i %d\n", i);
        return err;
      }
      buffers.push_back(list);
    }
    fVerbose && fprintf(stdout, "buffers %lu\n", nbuffers);
  }


  // Initialize the AU
  err = AudioUnitInitialize(fAudioUnit);
  if(err != noErr)
    {
      fprintf(stdout, "failed to initialize AU\n");
      return err;
    }

  fVerbose2 && fprintf(stdout, "LiveAudioInput: Initialized\n");
  return noErr;
}

template<typename sample_t> std::pair<int, sample_t> dB(sample_t const * const samples, int const num_samples, sample_t ref_sample=1) {
  // ref_sample is the sample value corresponding to 0 dB
  ASSERT( num_samples > 0 );

  sample_t sum = 0;
  sample_t sumsq = 0;
  sample_t min_sample = samples[0];
  sample_t max_sample = samples[0];
  //  sample_t min_sample = std::numeric_limits<sample_t>::max();
  //  sample_t max_sample = std::numeric_limits<sample_t>::min();

  bool const unfiltered = true;
  if( unfiltered ) {
    for( int i = num_samples; --i >= 0; ) {
      sample_t const sample = samples[i];
      // empirically observed from Logitech AK5370 USB mic on Mac HAL on OS X 10.4.10
      // VERIFY_TRUE( -1 <= sample && sample <= 1 );
      sum += sample;
      sumsq += sample * sample;

      if( sample > max_sample ) {
        max_sample = sample;
      }
      else if( sample < min_sample ) {
        min_sample = sample;
      }
    }
  }
  else {
    // do some filtering of the data -- subtle

    // note: we work backwards through the data, but our measurements
    // aren't damaged by the phase reversal
    int i = num_samples;
    if( --i >= 0 ) {
      // 8 significant bits, close to 0.98 (0.980469...)
      sample_t const alpha = static_cast<sample_t>(0xfb) / (1 << 8);
      sample_t const alphaScale = 1 / (1 - alpha);
      // equality on sample_ts!
      ASSERT( (1 - alpha) * alphaScale == 1 );

      sample_t wn = samples[i];
      while( --i >= 0 ) {

        // plain old preemphasis:
        //   y[n] =  x[n] - alpha * x[n-1]
        sample_t const alphaw = -alpha * wn;
        wn = samples[i];
        sample_t const sample = (wn + alphaw) * alphaScale;

        if( wn < min_sample ) {
          min_sample = wn;
        }
        else if( wn > max_sample ) {
          max_sample = wn;
        }

        /*
        // bi-linear preemphasis:
        //   y[n] + alpha * y[n-1]  =  x[n] - alpha * x[n-1]
        sample_t const alphaw = -alpha * wn;
        wn = samples[i] + alphaw;
        sample_t const sample = wn + alphaw;
        */

        sum += sample;
        sumsq += sample * sample;
      }
    }
  }

  // generate the statistics
  sample_t const mean = sum / num_samples;
  sample_t const ms = sumsq / num_samples - mean * mean;
  ASSERT( ms >= 0 );
  sample_t const refsq = ref_sample * ref_sample;
  ASSERT( refsq > 0 );
  // use factor of 10 because we didn't take the sqrt
  int dB10 = int(10 * log10(ms / refsq));
  // sanitize pathalogies, e.g. for digital silence, etc
  if( dB10 < -99 ) {
    dB10 = -99;
  }
  sample_t const range = max_sample - min_sample;
  ASSERT( range >= 0 );
  return std::pair<int, sample_t>(dB10, range);
}

static void log_rms(AudioBufferList const &audio_buffer, bool const verbose) {

  typedef float SampleType;
  typedef std::pair<int, SampleType> dBpair;

  UInt32 const num_channels = 2;
  VERIFY_TRUE( audio_buffer.mNumberBuffers == num_channels );

  char const space = ' ';
  char const dash = '-';
  char const bar = '|';
  //char const colon = ':';
  char const under = '<';
  char const over = '>';
  char const overlap = 'O';
  char const labels[num_channels] = { 'R', 'L' };
  dBpair dbs[num_channels];

  // get the channel measures
  float min_range = std::numeric_limits<float>::max();
  float max_range = std::numeric_limits<float>::min();
  for( UInt32 chan = 0; chan < num_channels; ++chan ) {
    AudioBuffer const &audioBuffer = audio_buffer.mBuffers[chan];
    UInt32 const nSamples = audioBuffer.mDataByteSize / sizeof(SampleType);
    ASSERT( nSamples * sizeof(SampleType) == audioBuffer.mDataByteSize );
    SampleType const * const pSamples = (SampleType const * const)audioBuffer.mData;
    // note: we have observed that raw USB audio samples are in the
    // range (-1, 1), but the built-in mic has a smaller range; we
    // scale to make a sample value of 2^18 be close to 90 dB....
    dbs[chan] = dB(pSamples, nSamples, 1.0f / (1 << 18));
    min_range = std::min(min_range, dbs[chan].second);
    max_range = std::max(max_range, dbs[chan].second);
  }

  // build a chart-plotter string
  UInt32 const offset = 2u;
  int const upper = 91;
  UInt32 const padding = 7u;
  std::string vu(offset + upper + padding, space);
  for( UInt32 chan = 0; chan < num_channels; ++chan ) {
    int const db = dbs[chan].first;
    if( db >= upper ) {
      vu[offset + upper] = over;
    }
    else if( db < 0 ) {
      vu[offset - 1] = under;
    }
    else {
      vu[offset + db] = (vu[offset + db] == space ? labels[chan] : overlap);
    }
  }

  // decade markers
  for( int i = 0; i < 10; ++i ) {
    int const index = 10 * i;
    if( vu[offset + index] == space ) {
      // vu[offset + index] = "0123456789"[i];
      //vu[offset + index] = colon;
      vu[offset + index] = bar;
    }
  }

  // prepend the actual values
  int const right = dbs[0].first;
  int const left = dbs[1].first;
  if( verbose ) {
    VERBOSE( fprintf(stdout, "\n %c  R %2d   L %2d   %5g  %5g ", (abs(right - left) > 1 ? '*' : ' '), right, left, min_range, max_range) );
    fprintf(stdout, "\n %cL%-3d %3dR%c  %s ", (left >= 0 ? space : dash), std::abs(left), std::abs(right), (right >= 0 ? space : dash), vu.c_str());
    fflush(stdout);
  }
}



static void * rms_worker(void * const arg) {
  LiveAudioInput &recorder = *static_cast<LiveAudioInput *>(arg);
  VERBOSE( bool const verbose = recorder.is_verbose(); )

  while( true ) {
    int const index = recorder.dude.receive();
    VERBOSE( verbose && fprintf(stdout, "rms_worker: index %d\n", index) );

    if( index < 0 ) {
      ASSERT( index == -1 );
      // note: pthread_exit doesn't return
      VERBOSE( verbose && fprintf(stdout, "rms_worker: pthread_exit\n") );
      pthread_exit(0);
    }

    // potentially slow work on the data
    log_rms(*recorder.buffers[index], recorder.is_verbose());
    VERBOSE( verbose && fprintf(stdout, "rms_worker: log_rms\n") );

    recorder.dude.done(index);
    VERBOSE( verbose && fprintf(stdout, "rms_worker: dude.done\n") );
  }
}


OSStatus LiveAudioInput::Start()
{
  // start the device

  Stop();

  dude.reset();
  dude.start();

  child_thread = 0x00;
  VERIFY_NULL(pthread_create(&child_thread, NULL, (fWorkerThreadFunc ? fWorkerThreadFunc : rms_worker), reinterpret_cast<void*>(this)));
  fVerbose && fprintf(stdout, "Started: ptr=\"%8p\"\n", child_thread);

  // Start pulling for audio data.  This causes a separate thread to
  // call AudioInputProc() to have samples rendered (see
  // AURenderCallbackStruct.inputProc).
  OSStatus err = AudioOutputUnitStart(fAudioUnit);
  VERIFY_NULL(err);
  if(err != noErr)
    {
      fprintf(stdout, "failed to start AU\n");
      return err;
    }
  fRunning = true;

  // get hardware device format
  AudioStreamBasicDescription format;
  UInt32 size = sizeof(format);
  //  err = AudioUnitGetProperty(fAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 1, &format, &size);
  err = AudioUnitGetProperty(fAudioUnit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 1, &format, &size);
  VERIFY_NULL(err);
  if(err != noErr)
    {
      fprintf(stdout, "Start failed to get input device ASBD\n");
      return err;
    }
  fVerbose && (printAudioFormat(format, "started format"), true);

  fVerbose2 && fprintf(stdout, "LiveAudioInput: Recording started: ptr=\"%8p\"\n", child_thread);
  return err;
}

OSStatus LiveAudioInput::Stop()
{
  // stop the device

  if( fRunning ) {
    fVerbose && fputc('\n', stdout);
    fVerbose && fflush(stdout);

    // Stop pulling audio data
    fRunning = false;

    fVerbose2 && fprintf(stdout, "LiveAudioInput: Stopping: ptr=\"%8p\"\n", child_thread);
    OSStatus err = AudioOutputUnitStop(fAudioUnit);
    VERIFY_NULL(err);
    if(err != noErr)
      {
        fprintf(stdout, "failed to stop AU\n");
        return err;
      }

    dude.stop();

    fVerbose2 && fprintf(stdout, "LiveAudioInput: Joining ptr=\"%8p\"\n", child_thread);
    void * exit_code;
    VERIFY_NULL(pthread_join(child_thread, &exit_code));
    if( exit_code != 0x00 ) {
      fprintf(stdout, "LiveAudioInput: Joined ptr=\"%8p\"  exit_code %p\n", child_thread, exit_code);
    }
    VERIFY_TRUE(exit_code == 0x00);

    fVerbose2 && fprintf(stdout, "LiveAudioInput: Recording stopped: ptr=\"%8p\" dropped %lu\n\n", child_thread, get_dropped());

    return err;
  }

  return noErr;
}
