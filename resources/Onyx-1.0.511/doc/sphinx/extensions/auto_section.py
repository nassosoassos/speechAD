###########################################################################
#
# File:         auto_section.py
# Date:         Tue 27 Jan 2009 15:22
# Author:       Ken Basye
# Description:  Sphinx extension for generating section headers from code
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

"""
    This file is part of the documentation generation tools.

    
"""
import sys

def auto_section_handler(app, what, name, obj, options, lines):
    if what == 'module':
        # print >> sys.stderr, "auto_section_handler - what == module"
        # print >> sys.stderr, "name = %s" % (name,)
        # print >> sys.stderr, "obj = %s" % (obj,)
        # print >> sys.stderr, "lines[0] = %s" % (lines[0],)
        line1 = ":mod:`%s` -- %s" % (name,lines[0])
        line2 = "^" * len(line1)
        lines[0] = line2
        lines.insert(0, line1)
    elif what == 'class':
        # print >>sys.stderr, "auto_section_handler - what == class"
        pass



def setup(app):
    app.connect('autodoc-process-docstring', auto_section_handler)



if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



