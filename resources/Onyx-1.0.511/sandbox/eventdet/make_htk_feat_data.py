###########################################################################
#
# File:         make_htk_feat_data.py
# Date:         Fri 19 Dec 2008 17:18
# Author:       Ken Basye
# Description:  Simple script to use HCopy to generate feat data
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008 The Johns Hopkins University
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

from __future__ import with_statement
from os import system, makedirs
import os.path
import sys

            
def ensure_subdirs_made(path, dir_name = False):
    """
    Make sure all the subdirectories on path have been made.

    If dir_name is False, this function assumes the last thing on the path is a filename, and
    doesn't try to make that.  Otherwise, it will make subdirectories for the entire path.
    """
    if not dir_name:
        dir, filename = os.path.split(path)
    else:
        dir = path
    have_dir = os.path.isdir(dir)
    if not have_dir:
        os.makedirs(dir)


def main(scp_file, config_file, dest_prefix='.', src_prefix=None):
    print("scp_file is ", scp_file)
    print("config_file is ", config_file)
    print("dest_prefix is ", dest_prefix)
    print("src_prefix is ", src_prefix)
    with open(scp_file) as f:
        for in_name in f:
            in_name = in_name.strip()
            in_name = os.path.normpath(in_name)
            if src_prefix is None:
                rest = in_name
            elif in_name.startswith(src_prefix):
                rest = in_name[len(src_prefix)+1:]
            else:
                print("Skipping file %s which did not start with prefix" % (in_name,))
                continue
                        
            out_name = os.path.join(dest_prefix, os.path.normpath(rest))
            print("in_name = %s\nout_name = %s" % (in_name, out_name))
            ensure_subdirs_made(out_name)
            command = "HCopy -C %s %s %s" % (config_file, in_name, out_name)
            print command
            system(command)

def usage(prog_name):
    print("Usage: %s scp_file config_file [dest_prefix] [src_prefix]" % (prog_name,))
    print("No src_prefix means match every file, no dest_prefix means use ./")
    print("If src_prefix is present, it will be stripped off and dest_prefix will be added in its place")


if __name__ == '__main__':
    if not 5 >= len(sys.argv) >= 3:
        usage(sys.argv[0])
    else:
        main(*sys.argv[1:])
