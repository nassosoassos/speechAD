#!/bin/sh -u
###########################################################################
#
# File:         display.sh
# Date:         14-Jan-2009
# Author:       Hugh Secker-Walker
# Description:  Start a visual display listener
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

# open a display processs in a new Terminal window (currently Mac OS X specific)
#
# arguments are passed to the display process
#   optional --log argument is intercepted (and passed on), makes a display for log output
#   pipe_name argument is passed on

# base for temporary filenames
selfname=`basename $0 .sh`
tempbase=`mktemp -t $selfname`

# the script that gets run
display="python `pwd`/projects/chartrecorder/chart.py $*"
# uncomment the following run a profile of the chartplotter; the profile is
# dumped into $profilename when the script successfully ends, e.g. when ESC is
# pressed in the chart window.  what was learned? it turns out that
# curses.refresh gets very expensive if there's a lot of non-blank cells,
# e.g. filled lines....
#profilename=$tempbase.prof
#display="python -m cProfile -o $profilename `pwd`/projects/chartrecorder/chart.py $*"

# default is for horizontally scrolling ascii graphics
cols=1499
rows=300
font="AmericanTypewriter-Condensed"
points=1
for arg in $* ; do
  case $arg in
    # plain vertically logging window for text
    --log)  cols=150; rows=80; font=Monaco; points=10 ;;
  esac
  displayfile=$arg
done

# create a custom .term file for Terminal
termname=$tempbase.term
cat >$termname <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
        <key>WindowSettings</key>
        <array>
                <dict>
                        <key>Shell</key>
                        <string>/bin/sh</string>
                        <key>ExecutionString</key>
                        <string>$display</string>

                        <key>Columns</key>
                        <string>$cols</string>
                        <key>Rows</key>
                        <string>$rows</string>

                        <key>NSFixedPitchFont</key>
                        <string>$font</string>
                        <key>NSFixedPitchFontSize</key>
                        <real>$points</real>

                        <key>CustomTitle</key>
                        <string>$displayfile</string>

                        <key>TitleBits</key>
                        <string>44</string>


                        <key>AutoFocus</key>
                        <string>NO</string>
                        <key>Autowrap</key>
                        <string>NO</string>
                        <key>BackgroundImagePath</key>
                        <string></string>
                        <key>Backwrap</key>
                        <string>NO</string>
                        <key>Bell</key>
                        <string>YES</string>
                        <key>BlinkCursor</key>
                        <string>NO</string>
                        <key>BlinkText</key>
                        <string>NO</string>
                        <key>CleanCommands</key>
                        <string>rlogin;telnet;ssh;slogin</string>
                        <key>CursorShape</key>
                        <string>0</string>
                        <key>DeleteKeySendsBackspace</key>
                        <string>NO</string>
                        <key>DisableAnsiColors</key>
                        <string>NO</string>
                        <key>DoubleBold</key>
                        <string>YES</string>
                        <key>DoubleColumnsForDoubleWide</key>
                        <string>NO</string>
                        <key>DoubleWideChars</key>
                        <string>NO</string>
                        <key>EnableDragCopy</key>
                        <string>NO</string>
                        <key>FontAntialiasing</key>
                        <string>NO</string>
                        <key>FontHeightSpacing</key>
                        <string>0.0</string>
                        <key>FontWidthSpacing</key>
                        <string>0.0</string>
                        <key>IsMiniaturized</key>
                        <string>NO</string>
                        <key>Meta</key>
                        <string>-1</string>
                        <key>OptionClickToMoveCursor</key>
                        <string>NO</string>
                        <key>PadBottom</key>
                        <string>1</string>
                        <key>PadLeft</key>
                        <string>1</string>
                        <key>PadRight</key>
                        <string>0</string>
                        <key>PadTop</key>
                        <string>0</string>
                        <key>RewrapOnResize</key>
                        <string>NO</string>
                        <key>SaveLines</key>
                        <string>0</string>
                        <key>ScrollRegionCompat</key>
                        <string>NO</string>
                        <key>ScrollRows</key>
                        <string>0</string>
                        <key>Scrollback</key>
                        <string>NO</string>
                        <key>Scrollbar</key>
                        <string>YES</string>
                        <key>ShellExitAction</key>
                        <string>3</string>
                        <key>StrictEmulation</key>
                        <string>NO</string>
                        <key>StringEncoding</key>
                        <string>4</string>
                        <key>TermCapString</key>
                        <string>xterm-color</string>
                        <key>TerminalOpaqueness</key>
                        <real>1</real>
                        <key>TextColors</key>
                        <string>0.000 0.000 0.000 1.000 1.000 0.714 0.000 0.000 0.000 0.000 0.000 0.000 1.000 1.000 0.714 0.000 0.000 0.000 0.667 0.667 0.667 0.333 0.333 0.333 </string>
                        <key>Translate</key>
                        <string>YES</string>
                        <key>UseCtrlVEscapes</key>
                        <string>NO</string>
                        <key>VisualBell</key>
                        <string>YES</string>
                        <key>WinLocULY</key>
                        <string>1000</string>
                        <key>WinLocX</key>
                        <string>50</string>
                        <key>WinLocY</key>
                        <string>0</string>
                        <key>WindowCloseAction</key>
                        <string>2</string>
                </dict>
        </array>
</dict>
</plist>
EOF

# Terminal runs the custom .term file
open -a Terminal $termname

# in the background, wait a while and then remove the custom .term file
(sleep 20; rm -f $termname)&
