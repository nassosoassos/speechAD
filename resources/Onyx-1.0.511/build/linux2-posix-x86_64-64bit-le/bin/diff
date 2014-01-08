#!/bin/sh -u
###########################################################################
#
# File:         bzrwrap.sh
# Date:         12-August-2009
# Author:       Hugh Secker-Walker
# Description:  Wrapper for bzr commands to give unified 'bin/command' usage
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

# The intent is to use a symbolic link to this script for each bzr
# command for which it is useful to expose the 'bin/<command>' usage

command="`basename $0`"

args=""
if [ "$command" = diff ] ; then
  args="--diff-options --strip-trailing-cr"
fi
exec bzr "$command" $args "$@"
