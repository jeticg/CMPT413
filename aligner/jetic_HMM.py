#!/usr/bin/env python
import optparse
import sys
import os
import logging
from copy import deepcopy
from jetic_IBM1 import AlignerIBM1
from collections import defaultdict


class AlignerHMM():
    def __init__(self):
        self.p0H = 0.3
        self.nullEmissionProb = 0.000005
        self.smoothFactor = 0.1
        return

    def initWithIBM(self, modelIBM1, biText):
        self.f_count = modelIBM1.f_count
        self.fe_count = modelIBM1.fe_count
        self.t = modelIBM1.t
        self.biText = biText
        return

    def forwardWithTScaled(self, a, pi, y, N, T, d):
        c_scaled = [0.0] * T
        alphaHat = [[0.0] * T] * N
        totalAlphaDoubleDot = 0

        for i in range(N + 1):
            alphaHat[i][1] = pi[i] * self.t[(y[0], d[i - 1])]
            totalAlphaDoubleDot += alphaHat[i][1]

        c_scaled[1] = 1.0 / totalAlphaDoubleDot

        for i in range(1, N + 1):
            alphaHat[i][1] = c_scaled[1] * alphaHat[i][1]

        for t in range(1, T):
            totalAlphaDoubleDot = 0
            for j in range(1, N + 1):
                total = 0
                for i in range(N + 1):
                    total += alphaHat[i][t] * a[i][j][N]
                alphaHat[j][t + 1] = self.t[(y[t], d[j - 1])] * total
                totalAlphaDoubleDot += alphaHat[j][t + 1]

            c_scaled[t + 1] = 1.0 / totalAlphaDoubleDot
            for i in range(1, N + 1):
                alphaHat[i][t + 1] = c_scaled[t + 1] * alphaHat[i][t + 1]

        return (alphaHat, c_scaled)

    def backwardWithTScaled(self, a, pi, y, N, T, d, c_scaled):
        betaHat = copy.deepcopy(c_scaled)
        for t in range(T - 1, 0, -1):
            for i in range(1, N + 1):
                total = 0
                for j in range(1, N + 1):
                    total += betaHat[j][t + 1] * a[i][j][N] * self.t[(y[t], d[j - 1])]
                betaHat[i][t] = c_scaled[t] * total
        return betaHat

    def maxTargetSentenceLength(self, biText):
        maxLength = 0
        targetLengthSet = []
        for (f, e) in biText:
            tempLength = len(e)
            if tempLength > maxLength:
                maxLength = tempLength
            targetLengthSet.append(tempLength)
        return (maxLength, targetLengthSet)

    def mapBitextToInt(self, sd_count):
        index = defaultdict(int)
        biword = defaultdict(tuple)
        i = 0
        for key in sd_count:
            index[key] = i
            biword[i] = key
            i += 1
        return (index, biword)

    def initialiseModel(self, int N):
        twoN = 2 * N
        self.a = [[[1.0 / N] * N] * twoN] * twoN
        self.pi = [1.0 / twoN] * twoN
        return

    def baumWelch(self):
        

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        if e == "null":
            return self.nullEmissionProb
        return 1.0 / v


optparser = optparse.OptionParser()
optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English (target language) filename (default=en)")
optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-v", "--num_tests", dest="num_tests", default=1000, type="int", help="Number of sentences to use for testing")
optparser.add_option("-i", "--iterations", dest="iter", default=5, type="int", help="Number of iterations to train")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

if opts.logfile:
    logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

biText = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
biText2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_tests]]

aligner = AlignerIBM1()
aligner.train(biText, opts.iter)
# aligner.decodeToStdout(biText2)
aligner.decodeToFile(biText2, "output_jetic_IBM1")
