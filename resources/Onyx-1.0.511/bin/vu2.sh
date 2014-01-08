#!/bin/sh -u
###########################################################################
#
# File:         vu2.sh (directory: ./bin)
# Date:         11-Jan-2009
# Author:       Ken Basye (from vu.sh by Hugh Secker-Walker)
# Description:  Shell script to run the ncurses-based chartplotter VU meter in horizontal mode
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2009 The Johns Hopkins University
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
#
###########################################################################

# Display a chart-plotter-style plot of the power from each live audio input on
# the system (on Mac).  For best results you should adjust the terminal window:
# shrink the font size a lot and then grow the window; you can do this while the
# chartplotter is running.

# $1, if given, is seconds to run, default runs forever

fifoname=/tmp/fifo.vu

rm -rf $fifoname || (echo "could not remove $fifoname"; false) || exit 1
mkfifo $fifoname || (echo "could not create fifo $fifoname"; false) || exit 1
(python ./py/onyx/audio/asciivumeter.py ${1:--1} x > $fifoname 2>&1 &) || (echo "could not start vu meter"; false) || exit 1
python ./projects/chartrecorder/chart.py $fifoname x || (echo "could not start display"; false) || exit 1
