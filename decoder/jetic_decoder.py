#!/usr/bin/env python
import optparse
import sys
import models
from math import log
from collections import namedtuple


class Decoder():
    def __init__(self, tm, lm):
        self.tm = tm
        self.lm = lm
        return

    def d(self, distance):
        alpha = 0.9
        return log(distance * alpha)

    def decode(self,
               f):

        tm = self.tm
        hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
        initial_hypothesis = hypothesis(0.0, self.lm.begin(), None, None)
        stacks = [{} for _ in f] + [{}]
        stacks[0][self.lm.begin()] = initial_hypothesis
        for i, stack in enumerate(stacks[:-1]):
            for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]:  # prune
                for j in xrange(i+1, len(f)+1):
                    if f[i:j] in tm:
                        for phrase in tm[f[i:j]]:
                            logprob = h.logprob + phrase.logprob
                            lm_state = h.lm_state
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = self.lm.score(lm_state, word)
                                logprob += word_logprob
                            logprob += self.lm.end(lm_state) if j == len(f) else 0.0
                            new_hypothesis = hypothesis(logprob, lm_state, h, phrase)
                            if lm_state not in stacks[j] or stacks[j][lm_state].logprob < logprob:  # second case is recombination
                                stacks[j][lm_state] = new_hypothesis
        winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

        def extract_english(h):
            return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
        print extract_english(winner)

        if opts.verbose:
            def extract_tm_logprob(h):
                return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
            tm_logprob = extract_tm_logprob(winner)
            sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,    help="Verbose mode (default=off)")
    opts = optparser.parse_args()[0]

    # TM stands for translation model
    tm = models.TM(opts.tm, opts.k)
    # LM stands for language model
    lm = models.LM(opts.lm)
    french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

    # tm should translate unknown words as-is with probability 1
    for word in set(sum(french, ())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]

    sys.stderr.write("Decoding %s...\n" % (opts.input,))
    decoder = Decoder(tm, lm)
    for f in french:
        decoder.decode(f)
