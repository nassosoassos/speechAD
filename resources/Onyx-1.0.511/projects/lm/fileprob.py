###########################################################################
#
# File:         fileprob.py (directory: ./projects/lm)
# Date:         28-Mar-2008
# Author:       Chris While
# Description:  Build an LM from a file of text
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
    Exercise the simple LM builder.
    
    >>> module_dir, module_name = os.path.split(__file__)
    >>> gTrainPath=os.path.join(module_dir, 'switchboard-small')
    >>> gTestPath=os.path.join(module_dir, 'sample1')
    >>> smoother='backoff_add0.5'
    >>> go(gTrainPath,gTestPath,smoother) #doctest: +ELLIPSIS
    Smoother set: BACKOFF_ADDL
    . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    Vocabulary size is 6697 types including OOV and EOC
    Finished training on 259291 tokens
    Test File ...sample1: 1686 words
    -13173.6 ...sample1
"""
from __future__ import with_statement
import math, sys, os
import probs

# Computes the log probability of the sequence of tokens in file, according to a
# fourgram model.

def filelogprob(filePath, langmod):
  with open(filePath, 'rt') as inFile:
    logprob = 0.0
    w = probs.BOC
    x = probs.BOC
    y = probs.BOC

    count = 0
    for line in inFile:
      for z in line.split():
        count += 1
        prob = langmod.prob(w, x, y, z)
        logprob += math.log(prob)/math.log(2)  
        w = x
        x = y
        y = z
        
  logprob += math.log(langmod.prob(w, x, y, probs.EOC))/math.log(2) 
  print 'Test File %s: %d words' %(filePath,count)

  return logprob

def process_options(argv=sys.argv):
    global gInPath,gTestPath,gType

    if len(argv) < 2:
        print "Usage: %s [options] -in trainLM.txt -test test.sents"         % sys.argv[0]
        print "-type (uniform,add1,backoff_add1) (default backoff_add0.5)"
        print "\nPrints the log-probability of each file under a smoothed n-gram model."
        print "\nPossible values for smoother: uniform, add1, backoff_add1"
        print "(the \"1\" in these names can be replaced with any real lambda >= 0)"
        raise ValueError('expected more arguments: see usage message on stdout')

    # Default parameter settings
    i = 1
    gType = 'backoff_add0.5'
    while i < len(argv) - 1:
        if argv[i] == "-test":
            i += 1
            gTestPath = argv[i]
        elif argv[i] == "-in":
            i += 1
            gInPath = argv[i]
        elif argv[i] == "-type":
            i += 1
            gType = argv[i]
        else:
            print "unrecognized option"
        i += 1

def go(gInPath,gTestPath,gType):
    assert os.path.exists(gInPath), "in file not found: " + repr(gInPath)
    assert os.path.exists(gTestPath), "test file not found: " + repr(gTestPath)
    smoother = gType

    text="this will be used in a doctext"
    text_test="this will test"

    lm = probs.LanguageModel()
    lm.set_smoother(smoother)
    lm.train(gInPath, smoother)

    print "%g %s" % (filelogprob(gTestPath, lm), gTestPath)


if __name__ == '__main__':
  from onyx import onyx_mainstartup
  onyx_mainstartup()
