===========================================================================
=
= File:         README.txt
= Date:         Fri 1 May 2009 12:24
= Author:       Ken Basye
= Description:  
=
= This file is part of Onyx   http://onyxtools.sourceforge.net
=
= Copyright 2009 The Johns Hopkins University
=
= Licensed under the Apache License, Version 2.0 (the "License").
= You may not use this file except in compliance with the License.
= You may obtain a copy of the License at
=   http://www.apache.org/licenses/LICENSE-2.0
= 
= Unless required by applicable law or agreed to in writing, software
= distributed under the License is distributed on an "AS IS" BASIS,
= WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
= implied.  See the License for the specific language governing
= permissions and limitations under the License.
=
===========================================================================

This copy of the English Malach data transcriptions was fixed to make
the XML formatting correct.  There were several problems which caused
a subset of the files to be incorrect in their original form; the
details are below.  The files in this directory have all been
validated against the trans-13.dtd DTD file distributed with
Transcriber 1.5.1, and a few of them have been loaded into Transcriber
itself.  In addition, all the files have been parsed without error by
an internal tool.

Details of transcription fixing:

1) Two files used the token 'Speaker' in <>s as annotation; since this
   was in the set of legal XML tags, I changed both tokens to
   'SPEAKER' to avoid confusion.

   20295-001.trs:<Speaker #1Interviewer>  did you have any nicknames 
   33058-001.trs:<Speaker #1Interviewer> can you spell it <cross_talk_end>

2) Repaired XML header declarations in 249 files by inserting
   "encoding=" where it was missing.

3) Repaired the 'version_date' attribute in the Trans tag in the same
   249 files; it had been 'version_data'.

4) These same 249 files had many "commentary" tokens embedded in the
   transcriptions in <>s, e.g., "<cough>".  The remaining 535 files
   used tokens in the form &lt;cough&rt; which renders as <cough> in
   XML display tools and parsers.  Using literal <>s makes these look
   like XML tags, which causes errors in both validation and parsing.
   I replaced all occurances of such commentary tokens with the same
   token embedded in the &lt; &gt; pair so that all the files used the
   same syntax.

5) I moved an extraneous </Turn> tag from the end of line 59 up to line
   57 in the file 19989-001.trs; I don't know how this corruption was
   caused.  

6) Two files parsed correctly according to the DTD file, but were not
   well-formed in other ways.  These files are 20030_003.trs and
   20045_003.trs, both of which had turns with two speakers listed but
   no 'Who' tags to identify which speaker said what.  I listened to
   the files and added the appropriate Who tags.


