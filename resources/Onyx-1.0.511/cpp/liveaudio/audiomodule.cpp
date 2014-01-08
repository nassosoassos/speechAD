///////////////////////////////////////////////////////////////////////////
//
// File:         audiomodule.cpp (directory: ./cpp/liveaudio)
// Date:         15-Nov-2007
// Author:       Hugh Secker-Walker
// Description:  Platform independent C++ glue code for live audio into Python
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


#include <Python.h>

#include "LiveAudioInput.h"

// XXX no!
#include <stdio.h>

#include <unistd.h>

#include <limits>

#include <CoreAudio/CoreAudio.h>
#include <IOKit/audio/IOAudioTypes.h>

// from Apple's CAAudioHardwareDevice.h in /Developer/Examples/CoreAudio/PublicUtility/
typedef UInt8   CAAudioHardwareDeviceSectionID;
#define kAudioDeviceSectionInput        ((CAAudioHardwareDeviceSectionID)0x01)
#define kAudioDeviceSectionOutput       ((CAAudioHardwareDeviceSectionID)0x00)
#define kAudioDeviceSectionGlobal       ((CAAudioHardwareDeviceSectionID)0x00)
#define kAudioDeviceSectionWildcard     ((CAAudioHardwareDeviceSectionID)0xFF)


// declare error helper variables
#define ERR_DECL(funcname)  char const * errmsg = "an error occurred in function '" #funcname "'"; int errline = 0;
// create our own error
#define ERR_IF(cond, msg) do { if(cond) { errmsg = (msg); errline = __LINE__; goto err_cleanup; } } while(false);
// bail on an existing Python error
#define ERR_PYOBJECT(cond) do { if(!cond) goto err_cleanup; } while(false);
// label for the cleanup code
#define ERR_CLEANUP() err_cleanup:
// pointer-freeing idiom
#define ERR_FREE(ptr)  do { void * ptrx = (ptr); if(ptrx != NULL) free(ptrx); } while(false);
// maybe set our own error, return NULL
#define ERR_RETURN() do { if( !PyErr_Occurred() ) { PyErr_Format(AudioError, "%s : %s(%d)", errmsg, __FILE__, errline);  }  return NULL; } while(false);

#define ENTRY_POINT_ENTER(name) do { ++entrypointEnterCount; } while(false);
#define ENTRY_POINT_RETURN(name) do { ++entrypointReturnCount; } while(false);


// module globals
// XXX reload support is needed

// start time of this module being loaded
static UInt64 const audioStartTimeNanos = AudioConvertHostTimeToNanos(AudioGetCurrentHostTime());
// counter for number of entrypoint calls
static UInt32 entrypointEnterCount = 0;
// counter for number of successful entrypoint returns
static UInt32 entrypointReturnCount = 0;
// counter for number of devices properties changes callbacks
static UInt32 propertyDevicesCallbackCount = 0;

// counter for number of buffers we process
static UInt32 processbufferCount = 0;

static PyObject *AudioError;


// sandbox functions

static PyObject *
audio_none(PyObject *self, PyObject *args)
{
  Py_INCREF(Py_None);
  return Py_None;
}

static PyObject *
audio_system(PyObject *self, PyObject *args)
{
  const char *command;
  int sts;

  if (!PyArg_ParseTuple(args, "s", &command))
    return NULL;

  sts = system(command);
  return Py_BuildValue("i", sts);
}


// transport names

static PyObject *
transport_name(UInt32 const transportType)
{
  switch(transportType) {
  case 0:
    return Py_BuildValue("s", "Unknown");
  case kIOAudioDeviceTransportTypeBuiltIn:
    return Py_BuildValue("s", "Built-In");
  case kIOAudioDeviceTransportTypePCI:
    return Py_BuildValue("s", "PCI");
  case kIOAudioDeviceTransportTypeUSB:
    return Py_BuildValue("s", "USB");
  case kIOAudioDeviceTransportTypeFireWire:
    return Py_BuildValue("s", "FireWire");
  case kIOAudioDeviceTransportTypeNetwork:
    return Py_BuildValue("s", "Network");
  case kIOAudioDeviceTransportTypeWireless:
    return Py_BuildValue("s", "Wireless");
  case kIOAudioDeviceTransportTypeOther:
    return Py_BuildValue("s", "Other");
  case 'virt':  //      kIOAudioDeviceTransportTypeVirtual
    return Py_BuildValue("s", "Virtual");
  default:
    {
      char const * const the4CC = (char const * const)&transportType;
      char outName[5] = {0};
      // ENDIAN
      if(true) {
        // note: little endian
        outName[0] = the4CC[3];
        outName[1] = the4CC[2];
        outName[2] = the4CC[1];
        outName[3] = the4CC[0];
      }
      else {
        // note: big endian
        outName[0] = the4CC[0];
        outName[1] = the4CC[1];
        outName[2] = the4CC[2];
        outName[3] = the4CC[3];
      }
      return Py_BuildValue("s", outName);
    }
  }
}

static PyObject *
audio_inputs(PyObject *self, PyObject *args)
{
  // enumerate the input devices

  ENTRY_POINT_ENTER(audio_inputs);
  ERR_DECL(audio_inputs);

  OSStatus err = noErr;
  UInt32 theSize = 0;
  AudioDeviceID * theDeviceList = NULL;
  char * name = NULL;
  char * manu = NULL;
  PyObject * ret = NULL;
  PyObject * transport = NULL;

  Boolean dummy_outWritable;

  UInt32 theNumberDevices;

  err = AudioHardwareGetPropertyInfo(kAudioHardwarePropertyDevices, &theSize, &dummy_outWritable);
  ERR_IF( err, "AudioHardwareGetPropertyInfo(kAudioHardwarePropertyDevicesfailed)" );

  theNumberDevices = theSize / sizeof(AudioDeviceID);
  theDeviceList = (AudioDeviceID*) malloc ( theNumberDevices * sizeof(AudioDeviceID) );
  ERR_IF( !theDeviceList, "malloc failed" );

  theSize = theNumberDevices * sizeof(*theDeviceList);
  err = AudioHardwareGetProperty(kAudioHardwarePropertyDevices, &theSize, theDeviceList);
  ERR_IF( err, "AudioHardwareGetProperty(kAudioHardwarePropertyDevices) failed" );

  ret = PyTuple_New(theNumberDevices);
  ERR_PYOBJECT(ret);

  // for each device
  unsigned int i;
  for( i = 0; i < theNumberDevices; ++i ) {
    AudioDeviceID const id = theDeviceList[i];

    // get device name
    err = AudioDeviceGetPropertyInfo(id, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceName, &theSize, &dummy_outWritable);
    ERR_IF( err, "AudioDeviceGetPropertyInfo(kAudioDevicePropertyDeviceName) failed" );

    name = (char *)malloc(theSize);
    ERR_IF( !name, "malloc failed" );

    err = AudioDeviceGetProperty(id, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceName, &theSize, name);
    ERR_IF( err, "AudioDeviceGetProperty(kAudioDevicePropertyDeviceName) failed");

    // get device manufacturer name
    err = AudioDeviceGetPropertyInfo(id, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceManufacturer, &theSize, &dummy_outWritable);
    ERR_IF(err, "AudioDeviceGetPropertyInfo(kAudioDevicePropertyDeviceManufacturer) failed");

    manu = (char*)malloc(theSize);
    ERR_IF( !manu, "malloc failed" );

    err = AudioDeviceGetProperty(id, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceManufacturer, &theSize, manu);
    ERR_IF(err, "AudioDeviceGetProperty(kAudioDevicePropertyDeviceManufacturer) failed");

    // get transport name
    UInt32 transportType;
    theSize = sizeof(transportType);
    OSStatus err = AudioDeviceGetProperty(id, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyTransportType, &theSize, &transportType);
    ERR_IF(err, "AudioDeviceGetProperty(kAudioDevicePropertyTransportType) failed");

    transport = transport_name(transportType);
    ERR_PYOBJECT(transport);

    // build the tuple
    PyObject * tuple = Py_BuildValue("(i,O,s,s)", id, transport, name, manu);
    transport=NULL;
    free(name);name=NULL;
    free(manu);manu=NULL;
    ERR_PYOBJECT(tuple);

    PyTuple_SET_ITEM(ret, i, tuple);
  }
  free(theDeviceList);

  ENTRY_POINT_RETURN(audio_inputs);
  return ret;

  ERR_CLEANUP();

  // our specific cleanup
  ERR_FREE(theDeviceList);
  ERR_FREE(name);
  ERR_FREE(manu);
  Py_XDECREF(ret);
  Py_XDECREF(transport);

  ERR_RETURN();
}

static PyObject *
audio_default_input(PyObject *self, PyObject *args)
{
  // get the default input device id

  ENTRY_POINT_ENTER(audio_default_input);
  ERR_DECL(audio_default_input);

  AudioDeviceID defaultId;
  UInt32 theSize = sizeof(defaultId);
  OSStatus err;
  Py_BEGIN_ALLOW_THREADS
  // ...Do some blocking I/O operation...
  err = AudioHardwareGetProperty(kAudioHardwarePropertyDefaultInputDevice, &theSize, &defaultId);
  Py_END_ALLOW_THREADS
  ERR_IF(err, "AudioHardwareGetProperty(kAudioHardwarePropertyDefaultInputDevice)");
  ENTRY_POINT_RETURN(audio_default_input);
  return Py_BuildValue("i", defaultId);

  ERR_CLEANUP();
  ERR_RETURN();
}


static PyObject *
audio_device_info(PyObject *self, PyObject *args)
{
  // get info about an audio device

  ENTRY_POINT_ENTER(audio_device_info);
  ERR_DECL(audio_device_info);

  OSStatus err = noErr;
  char * name = NULL;
  char * manu = NULL;
  UInt32 theSize;

  Boolean dummy_outWritable;

  PyObject * ret;

  int deviceId;
  ERR_PYOBJECT(PyArg_ParseTuple(args, "i", &deviceId));

  err = AudioDeviceGetPropertyInfo(deviceId, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceName, &theSize, &dummy_outWritable);
  ERR_IF(err, "AudioDeviceGetPropertyInfo(kAudioDevicePropertyDeviceNameCFString) failed");

  name = (char*)malloc(theSize);
  err = AudioDeviceGetProperty(deviceId, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceName, &theSize, name);
  ERR_IF(err, "AudioDeviceGetProperty(kAudioDevicePropertyDeviceNameCFString) failed");

  err = AudioDeviceGetPropertyInfo(deviceId, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceManufacturer, &theSize, &dummy_outWritable);
  ERR_IF(err, "AudioDeviceGetPropertyInfo(kAudioDevicePropertyDeviceManufacturer) failed");

  manu = (char*)malloc(theSize);
  err = AudioDeviceGetProperty(deviceId, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyDeviceManufacturer, &theSize, manu);
  ERR_IF(err, "AudioDeviceGetProperty(kAudioDevicePropertyDeviceManufacturer) failed");

  ret = Py_BuildValue("(s,s)", name, manu);
  free(name);
  free(manu);
  ENTRY_POINT_RETURN(audio_device_info);
  return ret;

  ERR_CLEANUP();
  ERR_FREE(name);
  ERR_FREE(manu);
  ERR_RETURN();
}


static PyObject *
audio_transport_name(PyObject *self, PyObject *args)
{
  // get the name of the transport of a device

  ENTRY_POINT_ENTER(audio_transport_name);
  ERR_DECL(audio_transport_name);

  UInt32 transportType;
  UInt32 theSize = sizeof(transportType);

  OSStatus err;

  int deviceId;
  ERR_PYOBJECT(PyArg_ParseTuple(args, "i", &deviceId));
  err = AudioDeviceGetProperty(deviceId, 0, kAudioDeviceSectionGlobal, kAudioDevicePropertyTransportType, &theSize, &transportType);
  ERR_IF(err, "AudioDeviceGetProperty(kAudioDevicePropertyTransportType) failed");

  ENTRY_POINT_RETURN(audio_transport_name);
  return transport_name(transportType);

  ERR_CLEANUP();
  ERR_RETURN();
}


// support for devices that are exposed in Python

// Vector of associations between LiveAudioInput and the callable
// PyObject where we stuff the audio events.  The index into this
// vector is the ID we expose to the Python client.
typedef std::vector<std::pair<LiveAudioInput*, PyObject*> > input_devices_t;

// first element is unused so that index=0 is never valid
static input_devices_t input_devices(1);

static int get_device_id(PyObject *args)
{
  unsigned int index;
  if( !PyArg_ParseTuple(args, "i", &index) ) {
    return 0;
  }
  if( index < 1 || index >= input_devices.size() || input_devices[index].first == NULL ) {
    PyErr_Format(PyExc_ValueError, "invalid device id %d", index);
    return 0;
  }
  return index;
}


static void process_buffer(AudioBufferList const &audio_buffer, PyObject * send_callable, bool const verbose)
{
  // function for the worker thread to use to create Python objects from a deliver()ed buffer

  // XXX we need gap support

  typedef float SampleType;

  UInt32 const num_channels = 2;
  VERIFY_TRUE( audio_buffer.mNumberBuffers == num_channels );

  // get the channel measures
  VERBOSE( float min_range = std::numeric_limits<float>::max(); )
  VERBOSE( float max_range = std::numeric_limits<float>::min(); )

  // acquire the GIL
  ASSERT( PyEval_ThreadsInitialized() );
  VERBOSE( pthread_t const me = pthread_self(); )
  VERBOSE( verbose &&  fprintf(stdout, "process_buffer: try GIL ensure 0x%08d\n", me); );
  PyGILState_STATE gstate = PyGILState_Ensure();
  // now we *can* call Python C-API

  VERBOSE( verbose &&  fprintf(stdout, "process_buffer: GIL ensure 0x%08d\n", me); );

  ASSERT( PyCallable_Check(send_callable) );

  // XXX errors

  // XXX use Numpy API to create an ndarray, shape=(num_channels, nSamples)

  // tuple that we push into the client's callable
  PyObject * ret = PyTuple_New(num_channels);
  for( UInt32 chan = 0; chan < num_channels; ++chan ) {
    AudioBuffer const &audioBuffer = audio_buffer.mBuffers[chan];
    UInt32 const nSamples = audioBuffer.mDataByteSize / sizeof(SampleType);
    ASSERT( nSamples * sizeof(SampleType) == audioBuffer.mDataByteSize );
    SampleType const * const pSamples = (SampleType const * const)audioBuffer.mData;

    // tuple for the sequence of float values for this channel
    PyObject * buf = PyTuple_New(nSamples);

    for( unsigned int i = 0; i < nSamples; ++i ) {
      PyObject * value = PyFloat_FromDouble(pSamples[i]);
      PyTuple_SetItem(buf, i, value);
    }

    // add sequence to return tuple
    PyTuple_SetItem(ret, chan, buf);
  }

  // push the return item to the client's callable
  PyObject * args = PyTuple_New(1);
  PyTuple_SetItem(args, 0, ret);
  PyObject * res = PyObject_CallObject(send_callable, args);
  Py_XDECREF(res);
  Py_XDECREF(args);

  VERBOSE( verbose && fprintf(stdout, "process_buffer: try GIL release 0x%08d\n", me); );
  PyGILState_Release(gstate);
  // now we *cannot* call Python C-API
  VERBOSE( verbose && fprintf(stdout, "process_buffer: GIL release 0x%08d\n", me); );
}

static void * audio_worker_thread_func(void * const arg)
{
  // the worker-thread function

  pthread_t const me = pthread_self();

  LiveAudioInput &recorder = *static_cast<LiveAudioInput *>(arg);
  bool const verbose = recorder.is_verbose();

  verbose && fprintf(stdout, "audio_worker_thread_func: started %8p\n", me);

  // match the recorder so as to get the send_callable
  unsigned int input_index;
  for( input_index = 1; input_index < input_devices.size(); ++input_index ) {
    if( &recorder == input_devices[input_index].first ) {
      break;
    }
  }
  // XXX error
  if( input_index == input_devices.size() ) {
    fprintf(stdout, "AudioInputProc: bogus inRefCon\n");
    pthread_exit(reinterpret_cast<void*>(0x01));
  }
  //  fprintf(stdout, "AudioInputProc: cool %d\n", input_index);

  PyObject *const send_callable = input_devices[input_index].second;

  verbose && fprintf(stdout, "audio_worker_thread_func: loop %8p\n", me);

  while( true ) {
    // blocking call where we wait for the OS audio thread to have deliver()ed data
    int const index = recorder.dude.receive();
    VERBOSE( verbose && fprintf(stdout, "audio_worker_thread_func: index %d %8p\n", index, me) );

    if( index < 0 ) {
      ASSERT( index == -1 );
      verbose && fprintf(stdout, "audio_worker_thread_func: exit %8p\n", me);
      // note: pthread_exit doesn't return
      pthread_exit(0);
    }

    // potentially slow work on the data; has to wait for Python's GIL
    process_buffer(*recorder.buffers[index], send_callable, recorder.is_verbose());
    VERBOSE( verbose && fprintf(stdout, "audio_worker_thread_func: process_buffer %d %8p\n", index, me); );
    ++processbufferCount;

    // non-block call where we release index
    recorder.dude.done(index);
    VERBOSE( verbose && fprintf(stdout, "audio_worker_thread_func: done %d %8p\n", index, me); );
    VERBOSE( fflush(stdout); );
  }
}


static PyObject *
audio_new_device(PyObject *self, PyObject *args)
{
  // get a new audio device for the given id

  ENTRY_POINT_ENTER(audio_new_device);
  ERR_DECL();

  OSStatus err;
  int ret;

  int uid;
  PyObject * send = NULL;
  PyObject * verbose = NULL;
  ERR_PYOBJECT(PyArg_ParseTuple(args, "iOO", &uid, &send, &verbose));

  // XXX should be TypeError
  ERR_IF(!PyCallable_Check(send), "Expected second argument to be callable");

  ret = input_devices.size();
  input_devices.push_back(std::make_pair(new LiveAudioInput, send));
  ERR_IF(!input_devices[ret].first, "failed to create LiveAudioInput");
  Py_INCREF(send);


  // If you don't make sure that threads are initialized, you can seg
  // fault in the PyGILState_Ensure() blocks elsewhere....
  if( !PyEval_ThreadsInitialized() ) {
    // XXX doesn't seem safe: can't be sure some other thread isn't doing the
    // same check and making the same call
    PyEval_InitThreads();
  }

  input_devices[ret].first->SetVerbose(PyObject_IsTrue(verbose));
  err = input_devices[ret].first->ConfigureAU(uid, audio_worker_thread_func /* AudioInputProc*/ );
  ERR_IF(err, "LiveAudioInput::ConfigureAU() failed");

  ENTRY_POINT_RETURN(audio_new_device);
  return Py_BuildValue("i", ret);

  ERR_CLEANUP();
  ERR_RETURN();
}

static PyObject *
audio_del_device(PyObject *self, PyObject *args)
{
  // delete the audio device

  int index;

  ENTRY_POINT_ENTER(audio_del_device);
  ERR_DECL();

  // this non-erroring check is necessary to cope with the state of
  // things if there was an error during audio_new_device() and Python
  // wrapper's __init__()
  PyObject * device = NULL;
  ERR_PYOBJECT(PyArg_ParseTuple(args, "O", &device));
  if( device == Py_None ) {
    Py_INCREF(Py_None);
    // don't increment ENTRY_POINT_RETURN
    return Py_None;
  }

  index = get_device_id(args);
  ERR_PYOBJECT(index);
  // deleting the object will Stop() it if it's running
  delete input_devices[index].first;
  Py_DECREF(input_devices[index].second);
  input_devices[index].first = NULL;
  input_devices[index].second = NULL;

  Py_INCREF(Py_None);
  ENTRY_POINT_RETURN(audio_del_device);
  return Py_None;

  ERR_CLEANUP();
  ERR_RETURN();
}

static PyObject *
audio_start_device(PyObject *self, PyObject *args)
{
  // turn on the audio device

  ENTRY_POINT_ENTER(audio_start_device);
  ERR_DECL();

  int index = get_device_id(args);
  ERR_PYOBJECT(index);

  ASSERT( PyEval_ThreadsInitialized() );

  // giving up the lock prevents dropped buffers during the startup
  Py_BEGIN_ALLOW_THREADS
    input_devices[index].first->Start();
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  ENTRY_POINT_RETURN(audio_start_device);
  return Py_None;

  ERR_CLEANUP();
  ERR_RETURN();
}

static PyObject *
audio_stop_device(PyObject *self, PyObject *args)
{
  // turn off the audio device

  // XXX we need gap support

  ENTRY_POINT_ENTER(audio_stop_device);
  ERR_DECL();

  int index = get_device_id(args);
  ERR_PYOBJECT(index);

  ASSERT( PyEval_ThreadsInitialized() );

  // we must give up the lock so that a running process_buffer() can
  // complete so the worker thread can finish
  Py_BEGIN_ALLOW_THREADS
    input_devices[index].first->Stop();
  Py_END_ALLOW_THREADS

  Py_INCREF(Py_None);
  ENTRY_POINT_RETURN(audio_stop_device);
  return Py_None;

  ERR_CLEANUP();
  ERR_RETURN();
}


static PyMethodDef AudioMethods[] = {
  {"inputs", audio_inputs, METH_VARARGS,  "Return a sequence of (integer, string, string).  Each tuple is an association of an integer id with an input device name and a manufacturer."},
  {"default_input",  audio_default_input, METH_VARARGS,  "Return the integer id currently associated with the default (system) audio input."},
  {"transport_name",  audio_transport_name, METH_VARARGS,  "Return a string indicating the transport type of the integer id audio device."},
  {"device_info",  audio_device_info, METH_VARARGS,  "Return a tuple of two strings indicating device and manufacturer."},

  {"new_device",  audio_new_device, METH_VARARGS,  "Return None."},
  {"del_device",  audio_del_device, METH_VARARGS,  "Return None."},
  {"start_device",  audio_start_device, METH_VARARGS,  "Return None."},
  {"stop_device",  audio_stop_device, METH_VARARGS,  "Return None."},

  // XXX these three are sandbox junk
  {"hughfoo",  audio_none, METH_VARARGS,  "Return NULL, heh heh."},
  {"system",  audio_system, METH_VARARGS,  "Execute a shell command."},
  {"none",  audio_none, METH_VARARGS,  "Return None."},

  {NULL, NULL, 0, NULL}        /* Sentinel */
};


ONYX_STARTUP_CODE(name1)
{
  VERBOSE( fprintf(stderr, "%s: %s: %s\n", "Startup", "_audio", "audiomodule.cpp"); )
  propertyDevicesCallbackCount = 0;
  VERBOSE( fprintf(stderr, "%s: %s %llu\n", "Startup", "audioStartTimeNanos", audioStartTimeNanos); )
}

ONYX_SHUTDOWN_CODE(name1)
{
  VERBOSE( fprintf(stderr, "%s: %s: %s\n", "Shutdown", "_audio", "audiomodule.cpp"); )
  VERBOSE( fprintf(stderr, "%s: %s %lu\n", "Shutdown", "entrypointEnterCount", entrypointEnterCount); )
  VERBOSE( fprintf(stderr, "%s: %s %lu\n", "Shutdown", "entrypointReturnCount", entrypointReturnCount); )
  VERBOSE( fprintf(stderr, "%s: %s %lu\n", "Shutdown", "entrypointErrorCount", entrypointEnterCount - entrypointReturnCount); )
  VERBOSE( fprintf(stderr, "%s: %s %lu\n", "Shutdown", "propertyDevicesCallbackCount", propertyDevicesCallbackCount); )
  VERBOSE( fprintf(stderr, "%s: %s %lu\n", "Shutdown", "processbufferCount", processbufferCount); )
  UInt64 const audioStopTimeNanos = AudioConvertHostTimeToNanos(AudioGetCurrentHostTime());
  VERBOSE( fprintf(stderr, "%s: %s %llu\n", "Shutdown", "audioStopTimeNanos", audioStopTimeNanos); )
  UInt64 const duration = audioStopTimeNanos - audioStartTimeNanos;
  UInt64 const nanosPerSec = 1000000000L;
  UInt64 const durSec = duration / nanosPerSec;
  UInt64 const durFrac = duration - durSec * nanosPerSec;
  (void)durFrac;
  VERBOSE( fprintf(stderr, "%s: %s %llu  %s %llu.(%llu / %llu)\n", "Shutdown", "durationNanos", duration, "durationSecs", durSec, durFrac, nanosPerSec); )
}

// property callbacks, not yet useful; not clear its working since these don't
// get called when the user changes the default device....

// gets called on a core audio thread's stack
static OSStatus AHPropertyListenerProc(AudioHardwarePropertyID inPropertyID, void *inClientData)
{
  OSStatus  err = noErr;
  UInt32 val = (UInt32)inClientData;

  if( inPropertyID == kAudioHardwarePropertyDevices ) {
    ++propertyDevicesCallbackCount;
    fprintf(stderr, "%s: %lu %lu %lu\n", "AHPropertyListenerProc", inPropertyID, val, propertyDevicesCallbackCount);
  }

  return err;
}

// gets called on a core audio thread's stack
static OSStatus AHPropertyListenerProc2(AudioHardwarePropertyID inPropertyID, void *inClientData)
{
  OSStatus  err = noErr;
  UInt32 val = (UInt32)inClientData;

  fprintf(stderr, "%s: %lu %lu\n", "AHPropertyListenerProc2", inPropertyID, val);

  return err;
}

OSStatus AHPropertyListenerProc3(AudioHardwarePropertyID inPropertyID, void *inClientData)
{
  ASSERT( inClientData == (void*)0x03 );
  switch (inPropertyID)
  {
    /*
     * These are the other types of notifications we might receive, however, they are beyond
     * the scope of this sample and we ignore them.
     */

  case kAudioHardwarePropertyDefaultInputDevice:
    fprintf(stderr, "AHPropertyListenerProc3: default input device changed\n");
    break;

  case kAudioHardwarePropertyDefaultOutputDevice:
    fprintf(stderr, "AHPropertyListenerProc3: default output device changed\n");
    break;

  case kAudioHardwarePropertyDefaultSystemOutputDevice:
    fprintf(stderr, "AHPropertyListenerProc3: default system output device changed\n");
    break;

  case kAudioHardwarePropertyDevices:
    fprintf(stderr, "AHPropertyListenerProc3: audio hardware changed\n");
    //[app performSelectorOnMainThread:@selector(updateDeviceList) withObject:nil waitUntilDone:NO];
    break;

  default:
    fprintf(stderr, "AHPropertyListenerProc3: unknown message\n");
    break;
  }
  return noErr;
}



static OSStatus myAudioObjectPropertyListenerProc(AudioObjectID inObjectID,
                                                  UInt32 inNumberAddresses,
                                                  const AudioObjectPropertyAddress inAddresses[],
                                                  void*inClientData)
{
  fprintf(stderr, "myAudioObjectPropertyListenerProc:\n");
  return noErr;
}


PyMODINIT_FUNC
init_audio(void)
{
  //fprintf(stderr, "init_audio(): enter\n");

  PyObject *mod = Py_InitModule("_audio", AudioMethods);
  if (mod == NULL)
    return;

  AudioError = PyErr_NewException("audio.AudioError", NULL, NULL);
  Py_INCREF(AudioError);
  PyModule_AddObject(mod, "AudioError", AudioError);

  // XXX more to  be done here:
  // must handle reload()
  // must handle being unloaded....
  //
  //AudioHardwareRemovePropertyListener(AudioHardwarePropertyID inPropertyID,
  //                                AudioHardwarePropertyListenerProc inProc);

  // install device notification callback
  AudioHardwareAddPropertyListener(kAudioHardwarePropertyDevices, AHPropertyListenerProc3, (void*)0x03);
  //  AudioHardwareAddPropertyListener(kAudioHardwarePropertyDevices, AHPropertyListenerProc3, (void*)0x03);

  OSStatus err = AudioHardwareAddPropertyListener(kAudioHardwarePropertyDevices, AHPropertyListenerProc, (void*)0x01);
  if( err == noErr ) {
    PyModule_AddObject(mod, "_kAudioHardwarePropertyDevices", Py_BuildValue("I", (UInt32)AHPropertyListenerProc));
  }
  else {
    Py_INCREF(Py_None);
    PyModule_AddObject(mod, "_kAudioHardwarePropertyDevices", Py_None);
  }

  AudioObjectPropertyAddress propaddr;
  propaddr.mSelector = 0x00;
  propaddr.mScope = 0x00;
  propaddr.mElement = 0x00;

  // XXX repeating the same function doesn't cause extra calls, but
  // each unique function gets called
  //
  // install device notification callback
  //AudioHardwareAddPropertyListener(kAudioHardwarePropertyDevices, AHPropertyListenerProc2, (void*)0x02);

  // See the thread at
  // http://lists.apple.com/archives/coreaudio-api/2006/jul/msg00249.html for an
  // explanation of why we aren't notified when the user switches the default
  // device in Preferences.  Basically it's because we're not running a main
  // application thread....
  AudioHardwareAddPropertyListener(kAudioHardwarePropertyDefaultInputDevice, AHPropertyListenerProc2, (void*)0x02);


  /*
struct AudioObjectPropertyAddress {
    AudioObjectPropertySelector mSelector;
    AudioObjectPropertyScope mScope;
    AudioObjectPropertyElement mElement;
};
  */

  /*
    kAudioObjectPropertySelectorWildcard = '****',
    kAudioObjectPropertyScopeWildcard = '****',
    kAudioObjectPropertyElementMaster
  */


  const AudioObjectPropertyAddress audioObjectHardwarePropertyAddress = {
    //kAudioHardwarePropertyDevices, kAudioDevicePropertyScopeInput, kAudioObjectPropertyElementWildcard
        kAudioHardwarePropertyDevices, kAudioObjectPropertyScopeWildcard, kAudioObjectPropertyElementWildcard
    // kAudioHardwarePropertyDefaultInputDevice, kAudioDevicePropertyScopeInput, kAudioObjectPropertyElementWildcard
    //kAudioHardwarePropertyDefaultInputDevice, kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementWildcard
    //kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeWildcard, kAudioObjectPropertyElementWildcard
    // kAudioHardwarePropertyDevices, kAudioDevicePropertyScopeInput, 0
    // kAudioHardwarePropertyDevices, kAudioDevicePropertyScopeOutput, kAudioObjectPropertyElementWildcard
  };

  AudioObjectID inObjectID = kAudioObjectSystemObject;
  const AudioObjectPropertyAddress * inAddress = &audioObjectHardwarePropertyAddress;


  Boolean res = AudioObjectHasProperty(inObjectID, inAddress);
  (void)res;
  VERBOSE( fprintf(stderr, "AudioObjectHasProperty: %x\n", res); )

  AudioObjectPropertyListenerProc inListener = myAudioObjectPropertyListenerProc;
  void* inClientData = NULL;
  err =  AudioObjectAddPropertyListener(inObjectID,
                                        inAddress,
                                        inListener,
                                        inClientData);

  (void)err;
  VERBOSE( fprintf(stderr, "AudioObjectAddPropertyListener: 0x%lx\n", err); )

  //CFRunLoopRunInMode(NULL, 1, 0);

  //fprintf(stderr, "init_audio(): exit\n");
}
