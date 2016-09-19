import sys
import codecs
import optparse
import os
from math import log10

optparser = optparse.OptionParser()
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
(opts, _) = optparser.parse_args()
sys.setrecursionlimit(30000)


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


def memo(f):
    "Memoize function f."
    table = {}

    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo

Pw = WordProbabilityDist(opts.counts1w)


@memo
def segment(text):
    "Return a list of words that is the best segmentation of text."
    if not text:
        return []
    candidates = ([first]+segment(rem) for first, rem in splits(text))
    return max(candidates, key=Pwords)


def splits(text, L=5):
    "Return a list of all possible (first, rem) pairs, len(first)<=L."
    return [(text[:i+1], text[i+1:])
        for i in range(min(len(text), L))]


def Pwords(words):
    "The Naive Bayes probability of a sequence of words."
    result = 1.00
    for w in words:
        result = result * Pw(w)
    return result

P2w = WordProbabilityDist(opts.counts2w)


def cPw(word, prev):
    "Conditional probability of word, given previous word."
    try:
        return P2w[prev + ' ' + word]/float(Pw[prev])
    except KeyError:
        return Pw(word)


@memo
def segment2(text, prev='<S>'):
    "Return (log P(words), words), where words is the best segmentation."
    if not text:
        return 0.0, []
    candidates = [combine(log10(cPw(first, prev)), first, segment2(rem, first))
                  for first, rem in splits(text)]
    return max(candidates)


def combine(Pfirst, first, (Prem, rem)):
    "Combine first and rem results into one (probability, words) pair."
    return Pfirst+Prem, [first]+rem


def main():
    old = sys.stdout
    sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
    with open(opts.input) as f:
        for line in f:
            utf8line = unicode(line.strip(), 'utf-8')
            print " ".join(segment2(utf8line)[1])
    sys.stdout = old

if "__main__":
    main()
