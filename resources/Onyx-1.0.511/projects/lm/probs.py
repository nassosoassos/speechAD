###########################################################################
#
# File:         probs.py (directory: ./projects/lm)
# Date:         28-Mar-2008
# Author:       Chris White
# Description:  Basic language modeling tool
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
    >>> True
    True

    This module needs its own tests!
"""
from __future__ import with_statement
from __future__ import division
import re, math

# define the exportable constants
BOC = "<s>";      # special word for context at Beginning Of Corpus
EOC = "</s>";     # special word for observed token at End Of Corpus
OOV = "<OOV>";    # special word for all out-of-vocabulary words
# ======================================================================

class LanguageModel(object):
  def __init__(self):
    self.smoother = None       # type of smoother we're using
    self.lambdap = None        # lambda: parameter used by some smoothers
    self.vocab_size = None  # V: the total vocab size including OOV
    self.tokens = None      # the c(...) function
    self.progress = 0        # for the progress bar
    self.vocab = None

  def prob(self, w, x, y, z):
    "Computes a smoothed estimate of the trigram probability p(z | x,y) according to the language model."
    if self.smoother == "UNIFORM":
      return 1 / self.vocab_size
    elif self.smoother == "ADDL":
      return ((self.tokens.get(make_key(w, x, y, z), 0) + self.lambdap) /
        (self.tokens.get(make_key(w, x, y), 0) + self.lambdap * self.vocab_size))
    elif self.smoother == "BACKOFF_ADDL":     
        # ppz = P(z)
        ppz = (self.tokens.get(make_key(z), 0) + self.lambdap)/(self.tokens.get('', 0) + self.lambdap*self.vocab_size)
     
        # ppzy = p(z|y)
        ppzy = (self.tokens.get(make_key(z, y), 0) + self.lambdap*self.vocab_size * ppz)/(self.tokens.get(make_key(y), 0) + self.lambdap * self.vocab_size)
        
        # ppzxy = p(z|xy)
        ppzyx = (self.tokens.get(make_key(x, y, z), 0) + self.lambdap * self.vocab_size * ppzy)/(self.tokens.get(make_key(x, y), 0) + self.lambdap * self.vocab_size) 

        # ppzxy = p(z|wxy)
        ppzyxw = (self.tokens.get(make_key(w, x, y, z), 0) + self.lambdap * self.vocab_size * ppzyx)/(self.tokens.get(make_key(w, x, y), 0) + self.lambdap * self.vocab_size) 

        return ppzyxw
    else:
      raise ValueError("unexpected smoother value: %r" % (self.smoother,))


  def train (self, corpusPath, smoother):
    # Clear out any previous training
    self.tokens = {}
    self.vocab = {}
    w, x, y = BOC, BOC, BOC  # xy context is "beginning of corpus"
    self.tokens[make_key(w, x, y)] = 1
    self.tokens[make_key(x, y)] = 1
    self.tokens[y] = 1

    with open(corpusPath, 'rt') as corpus:
      for i, line in enumerate(corpus):      
        for z in line.split():
          self.count(w, x, y, z)
          self.show_progress()
          w=x; x=y; y=z
        self.count(w, x, y, EOC)     # count EOC token after the final context
      # show_progress uses non-newline print, so here's the newline
      print
    
    # unique words seen in this file, plus 1 for OOV
    self.vocab_size = len(self.vocab) + 1

    print "Vocabulary size is %d types including OOV and EOC" % (self.vocab_size,)
    print "Finished training on %d tokens" % (self.tokens[''],)

  def count(self, w, x, y, z):
    tokens = self.tokens
    vocab = self.vocab

    wxyz = make_key(w, x, y, z)
    tokens[wxyz] = tokens.get(wxyz, 0) + 1
    xyz = make_key(x, y, z)
    tokens[xyz] = tokens.get(xyz, 0) + 1
    yz = make_key(y, z)
    tokens[yz] = tokens.get(yz, 0) + 1
    tokens[z] = tokens.get(z, 0) + 1
    if tokens[z] == 1: vocab[z] = 1# first time we've seen unigram z
    tokens[''] = tokens.get('', 0) + 1  # the zero-gram

  def set_smoother(self, arg):
    "Sets smoother type and lambda from a string passed in by the user on the command line."
    r = re.compile('^(.*?)(-?[0-9.]*)?$')
    m = r.match(arg)
    
    if not m.lastindex:
      raise ValueError("Smoother regular expression failed for %r" % (arg,))
    else:
      smoother_name = m.group(1)
      if m.lastindex >= 2 and len(m.group(2)):
        lambda_arg = m.group(2)
        self.lambdap = float(lambda_arg)
      else:
        self.lambdap = None

    if smoother_name.lower() == 'uniform':
      self.smoother = "UNIFORM"
    elif smoother_name.lower() == 'add':
      self.smoother = "ADDL"
    elif smoother_name.lower() == 'backoff_add':
      self.smoother = "BACKOFF_ADDL"
    else:
      raise ValueError("Don't recognize smoother name %r" % (smoother_name,))

    print "Smoother set: %s" %(self.smoother)

    if self.lambdap is None and self.smoother.find('ADDL') != -1:
      raise ValueError('You must include a non-negative lambda value in smoother name %r' % (arg,))

  def show_progress(self):
    "Print a dot to stderr every 5000 calls."
    self.progress += 1
    if self.progress % 5000 == 1:
      print '.',

def make_key(*words):
  """
  Produce a key for dictionary indexes.  Due to the overhead of making a
  function call, inlining the same code where it is called often would be a good
  idea....
  """
  return ' '.join(words)
   

if __name__ == '__main__':
  from onyx import onyx_mainstartup
  onyx_mainstartup()
