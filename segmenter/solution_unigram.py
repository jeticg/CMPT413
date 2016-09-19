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
            return score + 0.0000000000000000000000000000000000001


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

    def __call__(self, key1, key2, pDist):
        if (key1 + " " + key2) in self:
            return float(self[key1 + " " + key2]) / float(self.N)
        else:
            return pDist(key1) * pDist(key2)


def unigram():
    Pw = WordProbabilityDist(opts.counts1w)

    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    with open(opts.input) as f:
        for line in f:
            from math import log
            utf8line = unicode(line.strip(), 'utf-8')
            words = [i for i in utf8line]
            score = []
            best = []
            for i in range(len(words)):
                score.append(log(Pw(words[i])))
                if i != 0:
                    score[i] = score[i] + score[i-1]
                best.append(0)
                for j in range(1, 10):
                    if i - j < 0:
                        break
                    tmp = "".join(words[i-j:i+1])
                    # print i-j, i, tmp, log(Pw(tmp, Fw))
                    if i-j == 0 and log(Pw(tmp)) >= score[i]:
                        score[i] = log(Pw(tmp))
                        best[i] = j
                    elif score[i-j-1] + log(Pw(tmp)) >= score[i]:
                        score[i] = score[i-j-1] + log(Pw(tmp))
                        best[i] = j
                # print best[i], score[i]
            pointer = len(words)-1
            result = ()
            while (pointer >= 0):
                result = ("".join(words[pointer-best[pointer]:pointer+1]),) + result
                pointer = pointer - best[pointer] - 1

            print " ".join(result)
    sys.stdout = old

if "__main__":
    unigram()
