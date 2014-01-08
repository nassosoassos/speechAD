#!/bin/sh -u
###########################################################################
#
# File:         fsdemo.sh (directory: bin)
# Date:         2009-01-14 Wed 10:51:51
# Author:       Ken Basye
# Description:  A script to fire up a framesychronous demo
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

# in other terminal windows, do something like
# bin/chart.sh /tmp/demo -T
# bin/chart.sh /tmp/log
exec python -i py/onyx/dataflow/framesynchronous.py --interactive
