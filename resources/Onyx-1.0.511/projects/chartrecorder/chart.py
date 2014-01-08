###########################################################################
#
# File:         chart.py
# Date:         8-December-2008
# Author:       Hugh Secker-Walker
# Description:  An ncurses-based chart plotter scrolling horizontally or vertically
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2008, 2009 The Johns Hopkins University
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
    >>> w = curses.initscr()
    >>> try: curses.endwin()
    ... except: pass
"""
from __future__ import with_statement
import os
import thread
import select
import curses
import time
from functools import partial
from itertools import izip

SPACE_CHAR = ord(' ')

class ChartDisplayer(object):
    # ChartDisplayer handles the actual drawing, scrolling and window resizing
    HORIZONTAL = 'horizontal'
    VERTICAL = 'vertical'
    def __init__(self, window, scrolling):
        self.window = window
        assert scrolling is self.HORIZONTAL or scrolling is self.VERTICAL, str(self.scrolling)
        self.horizontal_scrolling = scrolling is self.HORIZONTAL
        self.current_base = 0
        
        # set up the window
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        curses.cbreak()
        window.keypad(0)
        # setting the following seems to break the refresh...
        #window.notimeout(1)
        window.nodelay(1)
        window.leaveok(1)
        window.clear()
        window.refresh()

        max_y, max_x = self.window.getmaxyx()
        window.move(max_y-1, 0)

    def get_maxen(self):
        max_y, max_x = self.window.getmaxyx()
        if self.horizontal_scrolling:
            max_base = max_x
            pos_upper = max_y
        else:
            max_base = max_y
            pos_upper = max_x
        return max_base, pos_upper

    def clear_base_line(self):
        max_base, pos_upper = self.get_maxen()
        try:
            if self.horizontal_scrolling:
                self.window.vline(0, self.current_base, SPACE_CHAR, pos_upper)
            else:
                self.window.hline(self.current_base, 0, SPACE_CHAR, pos_upper)
        except curses.error:
            pass

    def do_line(self, line):
        # update the virtual display with the line of visuals, doesn't refresh
        max_base, pos_upper = self.get_maxen()
        if self.current_base >= max_base:
            self.current_base = 0
            self.clear_base_line()

        try:
            # draw the line
            window = self.window
            horizontal_scrolling = self.horizontal_scrolling
            current_base = self.current_base
            pos_max = pos_upper - 1
            for pos, ch in izip(xrange(pos_upper), line.rstrip()):
                if ch == SPACE_CHAR:
                    continue
                if horizontal_scrolling:
                    window.addch(pos_max - pos, current_base, ch)
                else:
                    window.addch(current_base, pos, ch)
        except curses.error:
            max_base, pos_upper = self.get_maxen()

        # update current_base
        self.current_base += 1
        if self.current_base >= max_base:
            self.current_base = 0
        self.clear_base_line()
            

def chartdisplay(window, args):

    # horizontal or vertical scrolling
    scrolling = ChartDisplayer.HORIZONTAL
    while '--log' in args:
        scrolling = ChartDisplayer.VERTICAL
        del args[args.index('--log')]

    assert len(args) == 1, str(args)

    inputfilename = args[0]
    if inputfilename == '-':
        from sys import stdin as infile
    else:
        if os.path.exists(inputfilename):
            os.remove(inputfilename)
        os.mkfifo(inputfilename)
        def writer():
            # in another thread, open the pipe for writing so that we don't hang
            # in the open for reading
            with open(inputfilename, 'wt'):
                pass
        thread.start_new_thread(writer, ())
        line_buffered = 1
        # note: this blocks until something opens the pipe for writing, hence
        # the other thread to open the pipe for writing
        infile = open(inputfilename, 'rt', line_buffered)
    do_select = partial(select.select,  [infile], [], [], 0.25)

    def check_done():
        ESC = 27
        RESIZE = 410
        c = window.getch()
        if c != -1:
            if c == ESC:
                # escape char means we're done
                return True
            elif c == RESIZE:
                # resize happens...
                return False
            else:
                # yell
                curses.flash()
                return False

    do_line = ChartDisplayer(window, scrolling).do_line
    while True:
        if check_done():
            return

        try:
            ins, outs, excpts = do_select()
            if not ins:
                # only refresh when we're caught up....
                window.refresh()
                while not ins:
                    ins, outs, excpts = do_select()
                    if check_done():
                        return
        except select.error:
            # XXX make this blanket except be more selective; it's ok to
            # continue on EINTR, which occurs when window is resized during
            # select; but other select errors are not ok; for now we always
            # continue....
            continue

        line = ins[0].readline()
        if not line:
            # we keep trying on EOF
            continue
        # updates the virtual window, but doesn't refresh the physical display
        do_line(line)

    # if we get here, there's been an uncaught exception, so wait a while before
    # finishing (since finishing clears the curses display, thus wiping out
    # potentially useful error output, e.g. stack trace)
    time.sleep(160)

if __name__ == '__main__':

    from sys import argv
    args = argv[1:]
    if args:
        curses.wrapper(chartdisplay, args)
    else:
        # this works from the command-line and an emacs buffer, but not from
        # within SCons, so the SConscript has no_test=True
        from onyx import onyx_mainstartup
        onyx_mainstartup()
