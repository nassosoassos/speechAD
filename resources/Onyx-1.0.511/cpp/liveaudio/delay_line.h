///////////////////////////////////////////////////////////////////////////
//
// File:         delay_line.h
// Date:         14-Mar-2008
// Author:       Hugh Secker-Walker
// Description:  Simple delay-line or ring buffer
//
// This file is part of Onyx   http://onyxtools.sourceforge.net
//
// Copyright 2008 The Johns Hopkins University
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

#ifndef ONYX_GUARD_DELAY_LINE_H
#define ONYX_GUARD_DELAY_LINE_H

#include "onyx_defs.h"

namespace onyx {

// A delay line of the given order and data type.  The data type must
// support a default constructor that returns the "reset" value.
// E.g.   typedef DelayLine<8u, int> int_8_delay;
template <unsigned ORDER, class TYPE>
class DelayLine 
{
  unsigned index;
  TYPE line[ORDER];
  
 public:
  DelayLine() {
    reset();
  }
  
  void reset() {
    for( int i = ORDER; --i >= 0; ) {
      line[i] = TYPE();
    }
    index = 0u;
  }
  
  // add a new value, returns the oldest
  TYPE update(TYPE const input) {
    if( ORDER == 0u ) {
      return input;
    }

    TYPE const ret = line[index];
    line[index] = input;
    if( ++index == ORDER ) {
      index = 0u;
    }
    ASSERT( index < ORDER );
    return ret;
  }

};  // class DelayLine

}  // namespace onyx

#endif  // ONYX_GUARD_DELAY_LINE_H
