import sys
import codecs
import optparse
import os

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()


class WordProbabilityDist(dict):
    "A probability distribution estimated from counts in datafile."

    def __init__(self, filename, sep='\t', N=None, missingfn=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1. / N)
        self.unknownWordDist = UnknownWordDist(self)

    def __call__(self, key):
        if key in self:
            return float(self[key]) / float(self.N)
        elif len(key) == 1:
            return self.missingfn(key, self.N)
        elif len(key) == 0:
            return 1
        else:
            score = 1
            for i in range(len(key)):
                if i == 0:
                    score *= self.unknownWordDist.startMark[key[i]]
                elif i == len(key)-1:
                    score *= self.unknownWordDist.endMark[key[i]]
                else:
                    score *= self.unknownWordDist.middleMark[key[i]]
            return score + 0.000000000000000000001


class UnknownWordDist(dict):
    def __init__(self, pDist):
        from collections import defaultdict
        startMark = defaultdict(float)
        middleMark = defaultdict(float)
        endMark = defaultdict(float)
        for key in pDist:
            if len(key) > 1:
                for i in range(len(key)):
                    if i == 0:
                        startMark[key[i]] += pDist(key)
                    if i == len(key)-1:
                        endMark[key[i]] += pDist(key)
                    if i > 0 and i < len(key)-1:
                        middleMark[key[i]] += pDist(key)
        self.startMark = startMark
        self.middleMark = middleMark
        self.endMark = endMark


class DualWordDist(dict):
    def __init__(self, filename, sep='\t', N=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))

    def __call__(self, key1, key2, pDist=None):
        if (key1 + " " + key2) in self:
            return float(self[key1 + " " + key2]) / pDist(key1)
        else:
            return pDist(key2)

'''
# This is my attempt to build a dict for unknown words, which resulted in 83 accuracy, relatively lower than the one above
class DualWordDist(dict):
    def __init__(self, filename, sep='\t', N=None):
        self.maxlen = 0
        for line in file(filename):
            (key, freq) = line.split(sep)
            try:
                utf8key = unicode(key, 'utf-8')
            except:
                raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
            self[utf8key] = self.get(utf8key, 0) + int(freq)
            self.maxlen = max(len(utf8key), self.maxlen)
        self.N = float(N or sum(self.itervalues()))
        self.unknownDualWordDist = UnknownDualWordDist(self)

    def __call__(self, key1, key2, pDist=None):
        if (key1 + " " + key2) in self:
            score = float(self[key1 + " " + key2]) / float(self.N)
            if pDist is not None:
                score += pDist(key1) * pDist(key2)
            return score
        elif pDist is not None:
            score = 0
            score += pDist(key1) * pDist(key2)
            score += self.unknownDualWordDist.prevMark[key1] * self.unknownDualWordDist.nextMark[key2] * 3
            # score += pDist(key1) * self.unknownDualWordDist.prevMark[key1]
            # score += pDist(key2) * self.unknownDualWordDist.nextMark[key2]
            return score
        else:
            return self.unknownDualWordDist.prevMark[key1] * self.unknownDualWordDist.nextMark[key2] + 0.0000000000000000000000000000000000001


class UnknownDualWordDist(dict):
    def __init__(self, dDist):
        from collections import defaultdict
        prevMark = defaultdict(float)
        nextMark = defaultdict(float)
        for key in dDist:
            key1, key2 = key.split(" ")
            prevMark[key1] += dDist(key1, key2)
            nextMark[key2] += dDist(key1, key2)
        self.prevMark = prevMark
        self.nextMark = nextMark
'''


MAXLEN = 6
Pw = WordProbabilityDist(opts.counts1w)
Dw = DualWordDist(opts.counts2w)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
with open(opts.input) as f:
    for line in f:
        from math import log
        utf8line = unicode(line.strip(), 'utf-8')
        words = [i for i in utf8line]
        score = [[0] * MAXLEN for row in range(len(words))]
        best = [[-1] * MAXLEN for row in range(len(words))]
        for i in range(min(MAXLEN-1, len(words))):
            score[i][i+1] = log(Pw("".join(words[0:i+1])))
            best[i][i+1] = 0
        for i in range(1, len(words)):
            for j in range(1, MAXLEN):
                if i-j < 0:
                    break
                word = "".join(words[i-j+1:i+1])

                # k = 1
                k = 1
                prev = "".join(words[i-j-k+1:i-j+1])

                currentScore = log(Dw(prev, word, Pw)) if i-j-k+1 < 0 else score[i-j][k] + log(Dw(prev, word, Pw))

                # print i, j, k, prev, word, currentScore
                score[i][j] = currentScore
                best[i][j] = k

                # k = 2->MAXLEN-1
                for k in range(2, MAXLEN):
                    if i+1-j-k < 0:
                        break
                    prev = "".join(words[i-j-k+1:i-j+1])

                    currentScore = log(Dw(prev, word, Pw)) if i-j-k+1 < 0 else score[i-j][k] + log(Dw(prev, word, Pw))

                    # print i, j, k, prev, word, currentScore
                    if currentScore >= score[i][j]:
                        score[i][j] = currentScore
                        best[i][j] = k

                # print "best:", i, j, best[i][j], "".join(words[i-j-best[i][j]+1:i-j+1]), word, score[i][j]
        pointer = len(words)-1
        bestScore = -100000000000000000000000000000000000000000000000000000000
        for j in range(1, MAXLEN):
            if score[pointer][j] > bestScore:
                bestScore = score[pointer][j]
                pointerLen = j

        result = ()
        while (pointer >= 0):
            result = ("".join(words[pointer-pointerLen+1:pointer+1]),) + result
            tmp = pointer
            pointer = pointer - pointerLen
            pointerLen = best[tmp][pointerLen]

        print " ".join(result)
sys.stdout = old
