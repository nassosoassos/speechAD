#!/bin/sh -u
###########################################################################
#
# File:         livedemo.sh (directory: bin)
# Date:         2009-01-08 Thu 21:50:27
# Author:       Hugh Secker-Walker
# Description:  Script to fire up a live demo
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
# bin/chart.sh /tmp/log
# bin/chart.sh /tmp/demo -T
# bin/chart.sh /tmp/demo2 -T
exec python -i sandbox/livedemo/livedemo.py -l /tmp/log -o /tmp/demo -o2 /tmp/demo2
