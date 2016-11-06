#!/usr/bin/env python
import optparse
import sys
import models
from math import log
from collections import namedtuple
from copy import deepcopy

ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
phrase = namedtuple("phrase", "english, logprob")


class TargetSentence():
    def __init__(self,
                 length=0,
                 sourceMark=[],
                 targetSentenceEntity=(),
                 tmScore=0.0,
                 key=None):
        if key:
            self.sourceMark, self.targetSentenceEntity = key
            self.sourceMark = list(self.sourceMark)
            self.tmScore = tmScore
            return
        if length == 0 and sourceMark == []:
            raise ValueError("SENTENCE [ERROR]: Invalid initialisation")
        if length != 0 and sourceMark == []:
            self.sourceMark = [0 for x in range(length)]
            self.targetSentenceEntity = targetSentenceEntity
        else:
            self.sourceMark = sourceMark
            self.targetSentenceEntity = targetSentenceEntity
        self.tmScore = tmScore
        return

    def key(self):
        # Generate the unique key for the sentence
        key = (tuple(self.sourceMark), self.targetSentenceEntity)
        return key

    def overlapWithPhrase(self, phraseStartPosition, phraseEndPosition):
        # Check if the source phrase overlaps with the sentence
        if sum(self.sourceMark[phraseStartPosition:phraseEndPosition]) == 0:
            return False
        return True

    def addPhrase(self, phraseStartPosition, phraseEndPosition, targetPhrase):
        # mark positions in sourceMark as translated
        for i in range(phraseStartPosition, phraseEndPosition):
            self.sourceMark[i] = 1
        # add target phrase to sentence
        self.targetSentenceEntity = self.targetSentenceEntity + tuple(targetPhrase.english.split())
        # update translation score
        self.tmScore += targetPhrase.logprob
        return

    def lmScore(self, lm):
        # calculate language score
        lm_state = lm.begin()
        lmScore = 0.0
        for word in self.targetSentenceEntity:
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
        lmScore += lm.end(lm_state)
        return lmScore

    def totalScore(self, lm):
        return self.lmScore(lm) + self.tmScore

    def completed(self):
        if sum(self.sourceMark) == len(self.sourceMark):
            return True
        return False


class Decoder():
    def __init__(self, tm, lm):
        self.tm = tm
        self.lm = lm
        return

    def d(self, distance):
        alpha = 0.9
        return log(distance * alpha)

    def decodeDefault(self,
               sentence):

        f = sentence
        tm = self.tm
        hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase")
        initial_hypothesis = hypothesis(0.0, self.lm.begin(), None, None)
        stacks = [{} for _ in f] + [{}]
        stacks[0][self.lm.begin()] = initial_hypothesis
        for i, stack in enumerate(stacks[:-1]):
            for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]:  # prune
                for j in xrange(i+1, len(f)+1):
                    if f[i:j] in self.tm:
                        for phrase in self.tm[f[i:j]]:
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
        return

    def decode(self, sentence, max_phrase=5):
        bestScore = - sys.maxint - 1
        bestSentence = ()
        emptyTargetSentence = TargetSentence(length=len(sentence))
        stack = {}
        stack[emptyTargetSentence.key()] = emptyTargetSentence.tmScore
        newStack = {}
        for i in range(len(sentence)):
            sys.stderr.write("processing the " + str(i) + "th phrase, stack size: " + str(len(stack)) + "\n")
            # adding the ith target word/phrase
            for j in range(len(sentence)):
                # choose the jth source word as a start
                for k in range(j+1, min(len(sentence), j+max_phrase+1)):
                    # the phrase choosen to add this time is from j to k
                    sourcePhrase = sentence[j:k]
                    # Skip if the phrase doesn't exist
                    if sourcePhrase not in self.tm:
                        continue

                    for targetPhrase in self.tm[sourcePhrase]:
                        # for each translation, combine with every existing targetSentence in stack
                        for targetSentenceKey in stack:
                            # Reconstruct sentence
                            targetSentence = TargetSentence(key=targetSentenceKey,
                                                            tmScore=stack[targetSentenceKey])

                            # Check if overlapped
                            if targetSentence.overlapWithPhrase(j, k):
                                # overlapped, skip this sentence
                                continue

                            # Add phrase to sentence
                            targetSentence.addPhrase(j, k, targetPhrase)

                            # Compare with best score if translation complete, and skip adding to newStack
                            if targetSentence.completed():
                                if targetSentence.score() > bestScore:
                                    bestScore = targetSentence.score()
                                    bestSentence = targetSentence.targetSentenceEntity
                            else:
                                # Add the combined targetSentence to newStack if translation incomplete
                                key = targetSentence.key()
                                if key in newStack:
                                    if targetSentence.tmScore > newStack[key]:
                                        newStack[key] = targetSentence.tmScore
                                else:
                                    newStack[key] = targetSentence.tmScore
                            # Current targetSentences processed, proceed with next targetSentence in stack

                        # All targetSentences in stack processed, proceed with next translation

                    # All translations processed, proceed with next phrase

                # All phrases processed, proceed with next starting position for phrase

            # The ith phrase added to newStack. Exchange newStack and stack
            stack = newStack
            newStack = {}

        # All words processed, we now have the best sentence stored in bestSentence
        print bestSentence
        return


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
