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
        # There are two ways for creating a new TargetSentence instance
        # 1. use the length parameter
        # 2. use the key and tmScore parameter. (Will ignore all other parameters)
        if key:
            self.sourceMark, self.targetSentenceEntity, self.lastPos = deepcopy(key)
            self.sourceMark = list(self.sourceMark)
            self.tmScore = tmScore
            return
        if length == 0 and sourceMark == []:
            raise ValueError("SENTENCE [ERROR]: Invalid initialisation")
        if length != 0 and sourceMark == []:
            self.sourceMark = [0 for x in range(length)]
            self.targetSentenceEntity = targetSentenceEntity
            self.lastPos = -1
        else:
            self.sourceMark = sourceMark
            self.targetSentenceEntity = targetSentenceEntity
            self.lastPos = -1
        self.tmScore = tmScore
        return

    def key(self):
        # Generate the unique key for the sentence
        key = (tuple(self.sourceMark), self.targetSentenceEntity, self.lastPos)
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
        self.tmScore += targetPhrase.logprob + self.distance(self.lastPos, phraseStartPosition)
        # update lastPos
        self.lastPos = phraseEndPosition - 1
        return

    def distance(self, endOfLast, startOfCurrent):
        # d(endOfLast, startOfcurrent) = alpha ^ (abs(startOfCurrent - endOfLast - 1))
        # since all the scores are logd, assume beta = log(alpha)
        # log(d(endOfLast, startOfcurrent)) = beta * (abs(startOfCurrent - endOfLast - 1))

        # The beta value here is log(0.5)
        beta = -0.6931471805599453
        dis = abs(startOfCurrent - endOfLast - 1)
        return beta * dis

    def lmScore(self, lm):
        # calculate language score
        lm_state = lm.begin()
        logprob = 0.0
        for word in self.targetSentenceEntity:
            (lm_state, word_logprob) = lm.score(lm_state, word)
            logprob += word_logprob
        logprob += lm.end(lm_state)
        return logprob

    def totalScore(self, lm):
        return self.lmScore(lm) + self.tmScore

    def length(self):
        return sum(self.sourceMark)

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

    def decode(self, sentence, maxPhraseLen=10, maxStackSize=100, maxTranslation=sys.maxint):
        bestScore = - sys.maxint - 1
        bestSentence = ()
        emptyTargetSentence = TargetSentence(length=len(sentence))
        stack = [[] for x in range(len(sentence)+1)]
        stack[0].append((emptyTargetSentence.key(), emptyTargetSentence.tmScore))
        stackSize = 1
        newStack = [{} for x in range(len(sentence)+1)]
        for i in range(len(sentence)):
            sys.stderr.write("processing the " + str(i+1) + "th phrase of " + str(len(sentence)) + ", stack size: " + str(stackSize) + "\n")
            # adding the ith target word/phrase
            for j in range(len(sentence)):
                # choose the jth source word as a start
                for k in range(j+1, min(len(sentence), j+maxPhraseLen)+1):
                    # the phrase choosen to add this time is from j to k
                    sourcePhrase = sentence[j:k]
                    # Skip if the phrase doesn't exist
                    if sourcePhrase not in self.tm:
                        continue

                    for targetPhrase in self.tm[sourcePhrase][:maxTranslation]:
                        # print "source:", " ".join(sourcePhrase), "; target translation:", targetPhrase.english
                        # for each translation, combine with every existing targetSentence in stack
                        for stackLength in range(len(sentence)+1):
                            # check if the phrase size is getting too big
                            if stackLength + k-j > len(sentence):
                                break
                            for targetSentenceKey, targetSentenceScore in stack[stackLength]:
                                # Check if overlapped
                                if j <= targetSentenceKey[2] and targetSentenceKey[2] < k:
                                    continue
                                if sum(targetSentenceKey[0][j:k]) != 0:
                                    continue
                                # Reconstruct sentence
                                targetSentence = TargetSentence(key=targetSentenceKey,
                                                                tmScore=targetSentenceScore)

                                # Add phrase to sentence
                                targetSentence.addPhrase(j, k, targetPhrase)

                                # Compare with best score if translation complete, and skip adding to newStack
                                if targetSentence.completed():
                                    if targetSentence.totalScore(self.lm) > bestScore:
                                        bestScore = targetSentence.totalScore(self.lm)
                                        bestSentence = targetSentence.targetSentenceEntity
                                else:
                                    # print "Completion:", i+1, "out of", len(sentence), ":", " ".join(targetSentence.targetSentenceEntity)
                                    # Add the combined targetSentence to newStack if translation incomplete
                                    key = targetSentence.key()
                                    length = targetSentence.length()
                                    if key in newStack[length]:
                                        if targetSentence.tmScore <= newStack[length][key][1]:
                                            continue
                                    newStack[length][key] = targetSentence.totalScore(self.lm), targetSentence.tmScore
                                # Current stackLength processed, processed with next subStack

                            # Current targetSentences processed, proceed with next targetSentence in stack

                        # All targetSentences in stack processed, proceed with next translation

                    # All translations processed, proceed with next phrase

                # All phrases processed, proceed with next starting position for phrase

            # The ith phrase added to newStack. Exchange newStack and stack
            # do pruning
            stack = [[] for x in range(len(sentence)+1)]
            stackSize = 0
            for length in range(1, len(sentence)+1):
                # sys.stderr.write("Doing pruning for length: " + str(length) + "; size before pruning: " + str(len(newStack[length])) + "\n")
                # Sort by score
                sortedStack = sorted(newStack[length].items(), key=lambda x: x[1], reverse=True)
                for item in sortedStack[:maxStackSize]:
                    # only tmScore matters
                    stack[length].append((item[0], item[1][1]))
                    stackSize += 1
            newStack = [{} for x in range(len(sentence)+1)]

        # All words processed, we now have the best sentence stored in bestSentence
        print " ".join(bestSentence)
        return


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=sys.maxint, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=500, type="int", help="Maximum stack size (default=1)")
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
    count = 0
    for f in french:
        count += 1
        sys.stderr.write("Decoding sentence " + str(count) + " of " + str(len(french)) + "\n")
        decoder.decode(f, maxPhraseLen=10, maxStackSize=opts.s, maxTranslation=opts.k)
