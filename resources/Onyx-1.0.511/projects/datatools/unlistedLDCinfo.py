###########################################################################
#
# File:         unlistedLDCinfo.py
# Date:         Wed 1 Apr 2009 12:41
# Author:       Ken Basye
# Description:  
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


# LDC2005E40     Fisher English Training Speech Phase 2, Part 1

# Keys are catalog numbers; values are a triple of (title, language, tuple-of-sources) where sources
# are strings like "newswire", "microphone speech", etc.
_UNLISTED_INFO_DICT = {
    'LDC2003E19' : ("EARS MDE RT-03F Training Corpus", "English", ("Unknown",)),
    'LDC2003E27' : ("EARS MDE RT-03 DevTest and Evaluation", "English", ("Unknown",)),
    'LDC2004E16' : ("RT-04F MDE Development Data Set Part 1", "English", ("Unknown",)),
    'LDC2004E24' : ("RT-04F MDE Consistency Study", "English", ("Unknown",)),
    'LDC2004E31' : ("RT-04 MDE Training Data V1.2", "English", ("Unknown",)),
    'LDC2005S26' : ("CSLU: 22 Languages Corpus", "Several", ("Unknown",)),
    'LDC2005E73' : ("RT-04 Eval Text", "Several", ("Unknown",)),
    'LDC2006E100' : ("Speaker Recognition Eval data for 1996-2006", "English", ("Unknown",)),
    'LDC2006E110' : ("LCTL Urdu", "Urdu", ("Unknown",)),
    'LDC2007E47' : ("LRE 2007 dev data", "English", ("Unknown",)),
    'LDC2007E64' : ("ACE 2008 XDOC Pilot Data V2.1", "English", ("Unknown",)),
    'LDC2009R30' : ("Fisher Spanish 2006 Corpus Speech and Transcripts", "Spanish", ("Unknown",)),
    'LDC2009E20' : ("TAC 2009 KBP Sample Corpus", "English", ("Unknown",)),
    'LDC2009E56' : ("TAC 2009 KBP Evaluation Generic Infoboxes", "English", ("Unknown",)),
    'LDC2009E57' : ("TAC 2009 KBP Evaluation Source Data", "English", ("Unknown",)),
    'LDC2009E58' : ("TAC 2009 KBP Evaluation Reference Knowledge Base", "English", ("Unknown",)),
    'LDC2009E64' : ("TAC 2009 KBP Evaluation Entity Linking List", "English", ("Unknown",)),
    'LDC2009E65' : ("TAC 2009 KBP Evaluation Slot Filling List", "English", ("Unknown",)),

    }

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()



