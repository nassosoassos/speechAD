###########################################################################
#
# File:         vocalsource.py (directory: py/onyx/signalprocessing)
# Date:         2008-07-21 Mon 18:01:50
# Author:       Hugh Secker-Walker
# Description:  Toying around with one-poles for a vocal source
#
# This file is part of Onyx   http://onyxtools.sourceforge.net
#
# Copyright 2007 - 2009 The Johns Hopkins University
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
Use of coincident one-pole filters to generate reasonable, reversed,
glotal pulse waveforms.
"""

def make_one_pole(alpha):
    def one_pole(alpha):
        one_minus_alpha = 1 - alpha
        y_n = 0
        while True:
            x = yield y_n
            y_n = alpha * y_n + one_minus_alpha * x
    send = one_pole(alpha).send
    send(None)
    return send

def chain(seq):
    seq = tuple(seq)
    def gen():
        x = None
        while True:
            x = yield x
            for h in seq:
                x = h(x)
    send = gen().send
    send(None)
    return send

def test():
    """
    >>> op = chain(make_one_pole(0.8) for i in xrange(2))
    >>> for x in (1,) + 30 * (0,): print ' ', ' ' * int(1000 * op(x)), '*'
                                              *
                                                                      *
                                                                                   *
                                                                                        *
                                                                                        *
                                                                                     *
                                                                                *
                                                                          *
                                                                   *
                                                            *
                                                      *
                                                *
                                          *
                                     *
                                 *
                             *
                          *
                       *
                    *
                  *
                *
               *
             *
            *
           *
          *
          *
         *
         *
        *
        *

    >>> for x in (1,) + 25 * (0,) + (.25,) + 30 * (0,): print ' ', ' ' * int(1000 * op(x)), '*'
                                                *
                                                                        *
                                                                                    *
                                                                                         *
                                                                                         *
                                                                                      *
                                                                                *
                                                                          *
                                                                   *
                                                            *
                                                      *
                                                *
                                          *
                                     *
                                 *
                             *
                          *
                       *
                    *
                  *
                *
               *
             *
            *
           *
          *
                    *
                         *
                            *
                             *
                             *
                           *
                          *
                        *
                      *
                     *
                   *
                 *
                *
              *
             *
            *
           *
           *
          *
         *
         *
         *
        *
        *
        *
        *
       *
       *
       *
       *
       *

    >>> op = chain(make_one_pole(0.6) for i in xrange(3))
    >>> for x in (1,) + 20 * (0,): print ' ', ' ' * int(1000 * op(x)), '*'
                                                                       *
                                                                                                                          *
                                                                                                                                                 *
                                                                                                                                                 *
                                                                                                                                   *
                                                                                                               *
                                                                                          *
                                                                       *
                                                       *
                                          *
                                *
                         *
                   *
               *
             *
           *
         *
        *
        *
       *
       *

    """

if __name__ == '__main__':

    from onyx import onyx_mainstartup
    onyx_mainstartup()
