#!/usr/bin/env python
import optparse
import sys
import os
import logging
import time
from math import log
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
        self.a = [[[1.0 / N] * (N + 1)] * (twoN + 1)] * (twoN + 1)
        self.pi = [1.0 / twoN] * (twoN + 1)
        return

    def baumWelch(self, biText=self.biText, iterations=5):
        N, self.targetLengthSet = self.maxTargetSentenceLength(biText)

        sys.stderr.write("N " + str(N))
        indexMap, biword = self.mapBitextToInt(self.fe_count)

        L = len(biText)
        sd_size = len(indexMap)
        totalGammaDeltaOverAllObservations_t_i = None
        totalGammaDeltaOverAllObservations_t_overall_states_over_dest = None

        for iteration in range(iterations):

            logLikelihood = 0

            totalGammaDeltaOverAllObservations_t_i = [0.0] * sd_size
            totalGammaDeltaOverAllObservations_t_overall_states_over_dest = defaultdict(float)
            totalGamma1OverAllObservations = [0.0] * (N + 1)
            totalC_j_Minus_iOverAllObservations = [[[0.0] * (N + 1)] * (N + 1)] * (N + 1)
            totalC_l_Minus_iOverAllObservations = [[0.0] * (N + 1)] * (N + 1)

            start0_time = time.time()

            for (f, e) in biText:
                y = f
                x = e

                T = len(y)
                N = len(x)
                c = defaultdict(float)

                if iteration == 0:
                    self.initialiseModel(N)

                alpha_hat, c_scaled = self.forwardWithTScaled(self.a, self.pi, y, N, T, x)

                beta_hat = self.backwardWithTScaled(self.a, self.pi, y, N, T, x, c_scaled)

                gamma = [[0.0] * (T + 1)] * (N + 1)

                # Setting gamma
                for t in range(1, T):
                    logLikelihood += -1 * log(c_scaled[t])
                    for i in range(1, N + 1):
                        gamma[i][t] = (alpha_hat[i][t] * beta_hat[i][t]) / c_scaled[t]
                        totalGammaDeltaOverAllObservations_t_i[indexMap[(y[t - 1], x[i - 1])]] += gamma[i][t]

                t = T
                logLikelihood += -1 * log(c_scaled[t])
                for i in range(1, N + 1):
                    gamma[i][t] = (alpha_hat[i][t] * beta_hat[i][t]) / c_scaled[t]
                    totalGammaDeltaOverAllObservations_t_i[indexMap[(y[t - 1], x[i - 1])]] += gamma[i][t]

                for t in range(1, T):
                    for i in range(1, N + 1):
                        for j in range(1, N + 1):
                            c[j - i] += = alpha_hat[i][t] * a[i][j][N] * self.t[(y[t], x[j - 1])] * beta_hat[j][t + 1]

                for i in range(1, N + 1):
                    for j in range(1, N + 1):
                        totalC_j_Minus_iOverAllObservations[i][j][N] += c[j - i]
                    for l in range(1, N + 1):
                        totalC_l_Minus_iOverAllObservations[i][N] += c[l - i]

                for i in range(1, N + 1):
                    totalGamma1OverAllObservations[i] += gamma[i][1]
            # end of loop over bitext

            start_time = time.time()

            sys.stderr.write("likelihood " + str(logLikelihood))
            N = len(totalGamma1OverAllObservations) - 1

            for k in range(sd_size):
                totalGammaDeltaOverAllObservations_t_i[k] += totalGammaDeltaOverAllObservations_t_i[k]
                f, e = biword[k]
                totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e] += totalGammaDeltaOverAllObservations_t_i[k]

            end_time = time.time()

            sys.stderr.write("time spent in the end of E-step: " + str(end_time - start_time))
            sys.stderr.write("time spent in E-step: " + str(end_time - start0_time))

            twoN = 2 * N

            # M-Step

            self.a = [[[0.0] * (N + 1)] * (twoN + 1)] * (twoN + 1)
            self.pi = [0.0] * (twoN + 1)
            self.t = defaultdict()

            sys.stderr.write("set " + str(self.targetLengthSet))
            for I in self.targetLengthSet:
                for i in range(1, I + 1):
                    for j in range(1, I + 1):
                        a[i][j][I] = totalC_j_Minus_iOverAllObservations[i][j][I] / totalC_l_Minus_iOverAllObservations[i][I]

            for i in range(1, N + 1):
                pi[i] = totalGamma1OverAllObservations[i] * (1.0 / L)

            for k in range(sd_size):
                f, e = biword[k]
                self.t[(f, e)] = totalGammaDeltaOverAllObservations_t_i[k] / totalGammaDeltaOverAllObservations_t_overall_states_over_dest[e]

            end2_time = time.time()
            sys.stderr.write("time spent in M-step: " + str(end2_time - end_time))
            sys.stderr.write("iteration " + str(iteration))

        return

    def multiplyOneMinusP0H(self):
        for I in self.targetLengthSet:
            for i in range(1, I + 1):
                for j in range(1, I + 1):
                    a[i][j][I] *= 1 - self.p0H
        for I in self.targetLengthSet:
            for i in range(1, I + 1):
                for j in range(1, I + 1):
                    a[i][i + I][I] = self.p0H
                    a[i + I][i + I][I] = self.p0H
                    a[i + I][j][I] = a[i][j][I]
        return

    def tProbability(self, f, e):
        v = 163303
        if (f, e) in self.t:
            return self.t[(f, e)]
        if e == "null":
            return self.nullEmissionProb
        return 1.0 / v

    def aProbability(self, iPrime, i, I):
        if I in targetLengthSet:
            return a[iPrime][i][I]
        return 1.0 / I

    def logViterbi(self, N, o, d):
        twoN = 2 * N
        V = [[0.0] * len(o)] * [twoN + 1]
        ptr = [[0] * len(o)] * [twoN + 1]
        newd = d + ["null"] * (len(d))
        twoLend = 2 * len(d)
        for i in range(N, twoLend):
            newd[i] = "null"

        for q in range(1, twoN + 1):
            t_o0_d_qMinus1 = self.tProbability(o[0], newd[q - 1])
            if t_o0_d_qMinus1 == 0 or pi[q] == 0:
                V[q][0] = - sys.maxint - 1
            else:
                V[q][0] = log(pi[q]) + log(t_o0_d_qMinus1)

        for t in (1, len(o)):
            for q in (1, twoN + 1):
                maximum = - sys.maxint - 1
                max_q = - sys.maxint - 1
                t_o_d_qMinus1 = self.tProbability(o[t], newd[q - 1])
                for q in range(1, twoN + 1):
                    a_q_prime_q_N = self.aProbability(q_prime, q, N)
                    if (a_q_prime_q_N != 0) and (t_o_d_qMinus1 != 0):
                        temp = V[q_prime][t - 1] + log(a_q_prime_q_N) + log(t_o_d_qMinus1)
                        if temp > maximum:
                            maximum = temp
                            max_q = q_prime
                V[q][t] = maximum
                ptr[q][t] = max_q

        max_of_V = - sys.maxint - 1
        q_of_max_of_V = 0
        for q in (1, twoN + 1):
            if V[q][len(o) - 1] > max_of_V:
                max_of_V = V[q][len(o) - 1]
                q_of_max_of_V = q

        trace = []
        trace.append(q_of_max_of_V)
        q = q_of_max_of_V
        i = len(o) - 1
        while (i > 0):
            q = ptr[q][i]
            trace = [q] + trace
            i = i - 1
        return trace

    def findBestAlignmentsForAll_AER(self, biText, fileName):
        outputFile = open(fileName, "w")
        alignmentList = []
        for (f, e) in biText:
            N = len(e)
            bestAlignment = self.logViterbi(N, f, e)
            line = ""
            for i in range(len(bestAlignment)):
                if bestAlignment[i] <= N:
                    line += str(i) + "-" + str(bestAlignment[i] - 1) + " "
            alignmentList.append(line)
            outputFile.write(line + "\n")
            # sys.stdout.write(line + "\n")
        outputFile.close()
        return alignmentList

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

alignerIBM1 = AlignerIBM1()
alignerIBM1.train(biText, opts.iter)
alignerHMM = AlignerHMM()
alignerHMM.initWithIBM(alignerIBM1, biText)
alignerHMM.baumWelch()
alignerHMM.multiplyOneMinusP0H()
alignerHMM.findBestAlignmentsForAll_AER(biText2, "output_jetic_HMM")
