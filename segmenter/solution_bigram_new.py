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

    def __init__(self, filename, sep='\t', N=None, missingfn=None, loadUnknow=True):
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
        if loadUnknow:
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


class BigramSegmenter():
    def __init__(self, pDist, p2Dist):
        self.segmentTable = {}
        self.pDist = pDist
        self.p2Dist = p2Dist

    def Dw(self, prev, word):
        "Conditional probability of word, given previous word."
        try:
            return float(self.p2Dist[prev + ' ' + word]) / self.pDist[prev]
        except KeyError:
            return self.pDist(word)

    def segment(self, line, prev='<S>', maxLen=5):
        if (line, prev) in self.segmentTable:
            return self.segmentTable[(line, prev)]
        if not line:
            return 0.0, []

        results = []
        for i in range(min(maxLen, len(line))):
            word = line[:i+1]
            rest = line[i+1:]
            restProb, restWords = self.segment(rest, word)
            prob = log(self.Dw(prev, word))

            results.append((prob + restProb, [word] + restWords))

        finalAns = results[0]
        for i in range(len(results)):
            if results[i][0] > finalAns[0]:
                finalAns = results[i]
        self.segmentTable[line, prev] = finalAns
        return finalAns

Pw = WordProbabilityDist(opts.counts1w)
P2w = WordProbabilityDist(opts.counts2w)

old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
with open(opts.input) as f:
    for line in f:
        from math import log
        utf8line = unicode(line.strip(), 'utf-8')

        segmenter = BigramSegmenter(Pw, P2w)

        result = segmenter.segment(utf8line)

        print " ".join(result[1])
sys.stdout = old
