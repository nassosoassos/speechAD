#!/usr/bin/env python
"""
decode.py

Created by Dogan Can on 2012-02-02.
Copyright (c) 2012 SAIL. All rights reserved.
"""

import sys
from collections import defaultdict
from math import log, exp
from operator import itemgetter

def viterbi(source, n, channel, off):
    '''Return the best label sequence for a list of channel score tuples
    given an ngram source model and a maximum ngram length'''
    
    Q = defaultdict(lambda: defaultdict(lambda: float('inf')))
    H = defaultdict(lambda: defaultdict(tuple))
    N = len(channel)
    labelset = range(len(channel[0]))
    
    prev = ['<s>']
    for j in labelset:
        score, curr = lmscore(prev, [str(j)], source, n, off)
        Q[0][curr] = score + channel[0][j]
        H[0][curr] = (prev, j)
    
    for i in range(1, N):
        for k in Q[i - 1]:
            prev = k.split()
            for j in labelset:
                score, curr = lmscore(prev, [str(j)], source, n, off)
                r = Q[i - 1][k] + score + channel[i][j]
                if r < Q[i][curr]:
                    Q[i][curr] = r
                    H[i][curr] = (prev, j)
    
    for k in Q[N - 1]:
        prev = k.split()
        score, _ = lmscore(prev, ['</s>'], source, n, off)
        r = Q[N - 1][k] + score
        if r < Q[N]['</s>']:
            Q[N]['</s>'] = r
            H[N]['</s>'] = (prev, '</s>')
    
    prev = ['</s>']
    labels = []    
    for i in range(N, -1, -1):
        prev, label = H[i][" ".join(prev)]
        labels = [label] + labels
    
    return labels, Q[N]['</s>']


def readchannelscores(channelfile):
    '''Read channel score tuples from file
        - Each line corresponds to a frame
        - Each column corresponds to a source label: '0', '1', ...'''
    
    return [tuple(map(lambda x: -float(x), line.split())) for line in open(channelfile)]


def readlm(path):
    '''Read a standard ARPA format back-off language model and
    return the model, max ngram length and the vocabulary'''
    
    ln10 = log(10)
    lmfile = open(path)
    lm = defaultdict(list)
    lmfile.readline()
    clist = []
    for line in lmfile:
        if line.strip() == "":
            break
        if line.startswith("\\"):
            continue
        _, cnt = map(int, line.split()[1].split("="))
        clist += [cnt]
    nmax = len(clist)
    vocab = defaultdict(lambda: "<unk>")
    for n, cnt in enumerate(clist, 1):
        for line in lmfile:
            if line.strip() == "":
                break
            if line.startswith("\\"):
                continue
            parts = line.split()
            if n == nmax:
                lm[" ".join(parts[1:])] = [float(parts[0]) * -ln10]
            else:
                if len(parts) == n + 2:
                    lm[" ".join(parts[1:n + 1])] = [float(parts[0]) * -ln10, float(parts[-1]) * -ln10]
                else:
                    lm[" ".join(parts[1:n + 1])] = [float(parts[0]) * -ln10, 0.0]
                if n == 1:
                    vocab[parts[1]] = parts[1]
    lmfile.close()
    return lm, nmax, vocab

def lmscore(hist, curr, lm, nmax, off=False):
    '''Compute back-off language model score and the next ngram history'''
    
    score = 0
    ngram = hist + curr[0:1]
    for w in range(1, len(curr) + 1):
        ngramstr = " ".join(ngram)
        if ngramstr in lm:         # regular case
            score += lm[ngramstr][0]
            if len(ngram) == nmax:
                ngram = ngram[1:]
        else:                           # back-off
            while len(ngram) > 1:
                ngram = ngram[1:]
                ngramstr = " ".join(ngram)
                if ngramstr in lm:
                    score += sum(lm[ngramstr])
                    break
            else:                       # missing ngram
                score += float('inf')
                # sys.exit("ERROR: " + ngram[0] + " not in LM!")
        ngram += curr[w:w + 1]
    if off and score is not float('inf'):
        score = 0
    return score, " ".join(ngram)

if __name__ == "__main__":
    testlist = sys.stdin
    result = sys.stdout
    sourcefile = sys.argv[1]
    source, n, _ = readlm(sourcefile)
    for testf in testlist:
        testf = testf.rstrip('\r\n')
        channel = readchannelscores(testf)
        r, score = viterbi(source, n, channel, False)
        print >> result, r
