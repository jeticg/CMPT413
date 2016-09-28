import sys
import codecs
import optparse
import os
from math import log

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()


class WordProbabilityDist(dict):
    "A probability distribution estimated from counts in datafile."
    "We have added an option to load up our prediction class for unknown words"
    "inside the class, called self.unknownWordDist"

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
            # Loadup unknowed word list
            self.unknownWordDist = UnknownWordDist(self)

    def __call__(self, key):
        if key in self:
            return float(self[key]) / float(self.N)
        elif len(key) == 1:
            return self.missingfn(key, self.N)
        elif len(key) == 0:
            return 1
        else:
            # when len(key) >= 2, consider how to segment
            # Calling the unknownWordDist to provide a probability score for
            # unknown word. The model description is included in our README file
            score = 1
            for i in range(len(key)):
                if i == 0:
                    score *= self.unknownWordDist.startMark[key[i]]
                elif i == len(key)-1:
                    score *= self.unknownWordDist.endMark[key[i]]
                else:
                    score *= self.unknownWordDist.middleMark[key[i]]
            # Adding a mininal const to prevent log(0)
            return score + 0.000000000000000000001


class UnknownWordDist(dict):
    # Loading the unknownWordDist using counts1w
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
    # Bigram segmenter class
    def __init__(self, pDist, p2Dist):
        self.segmentTable = {}
        self.pDist = pDist
        self.p2Dist = p2Dist

    def Dw(self, prev, word):
        "Conditional probability of word, given previous word."
        try:
            # smoothing
            # p2Dist => the count of prev+word.
            # return float(self.p2Dist[prev + ' ' + word]) / (self.pDist[prev])
            return float(0.1 +  self.p2Dist[prev + ' ' + word]) / (35 +  self.pDist[prev])

        except KeyError:
            return self.pDist(word) # probability

    def segment(self, line, prev='<S>', maxLen=5):
        if (line, prev) in self.segmentTable:
            # optimisation: use the remembered score of already calculated string
            return self.segmentTable[(line, prev)]
        if not line:
            return 0.0, []

        results = []
        for i in range(min(maxLen, len(line))):
            # Split
            word = line[:i+1]
            rest = line[i+1:]

            # Core: recursively call segment()
            restProb, restWords = self.segment(rest, word)

            prob = log(self.Dw(prev, word))
            results.append((prob + restProb, [word] + restWords))

        # Max of results
        finalAns = results[0]
        for i in range(len(results)):
            if results[i][0] > finalAns[0]:
                finalAns = results[i]

        # optimisation: remember the score of already calculated string
        self.segmentTable[line, prev] = finalAns
        return finalAns


def unigram():
    # Unigram nonrecursive algorithm
    Pw = WordProbabilityDist(opts.counts1w)

    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    with open(opts.input) as f:
        for line in f:
            utf8line = unicode(line.strip(), 'utf-8')
            words = [i for i in utf8line]
            # Score is the best score for the first i charactors
            score = []
            # Best records the position of the end of previous word of the best score
            best = []

            for i in range(len(words)):
                score.append(log(Pw(words[i])))
                if i != 0:
                    score[i] = score[i] + score[i-1]
                best.append(0)
                for j in range(1, 10):
                    if i - j < 0:
                        break

                    # Split out the current word
                    tmp = "".join(words[i-j:i+1])

                    # Compare the score(Dynamic Programming)
                    if i-j == 0 and log(Pw(tmp)) >= score[i]:
                        score[i] = log(Pw(tmp))
                        best[i] = j
                    elif score[i-j-1] + log(Pw(tmp)) >= score[i]:
                        score[i] = score[i-j-1] + log(Pw(tmp))
                        best[i] = j

            # After process, produce the result
            pointer = len(words)-1
            result = ()
            while (pointer >= 0):
                result = ("".join(words[pointer-best[pointer]:pointer+1]),) + result
                pointer = pointer - best[pointer] - 1

            print " ".join(result)
    sys.stdout = old


def bigram():
    # bigram  algorithm
    Pw = WordProbabilityDist(opts.counts1w) # occurance / total
    P2w = WordProbabilityDist(opts.counts2w)

    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    with open(opts.input) as f:
        for line in f:
            utf8line = unicode(line.strip(), 'utf-8')

            segmenter = BigramSegmenter(Pw, P2w)

            result = segmenter.segment(utf8line)

            print " ".join(result[1])
    sys.stdout = old

if "__main__":
    # unigram
    bigram()
