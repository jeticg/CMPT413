#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

from include.target_sentence import TargetSentence

ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
phrase = namedtuple("phrase", "english, logprob")


class Decoder():
    def __init__(self, tm, lm):
        self.tm = tm
        self.lm = lm
        self.answers = []
        return

    def decode(self,
               sentence,
               maxPhraseLen=10,
               maxStackSize=100,
               maxTranslation=sys.maxint,
               saveToList=False,
               verbose=False):

        bestScore = - sys.maxint - 1
        bestSentence = ()
        emptyTargetSentence = TargetSentence(length=len(sentence))
        stack = [[] for x in range(len(sentence) + 1)]
        stack[0].append((emptyTargetSentence.key(), emptyTargetSentence.tmScore))

        for stackLength in range(len(sentence) + 1):
            # check if the phrase size is getting too big
            for targetSentenceKey, targetSentenceScore in stack[stackLength]:

                currentSentence = TargetSentence(key=targetSentenceKey,
                                                tmScore=targetSentenceScore)

                for j in range(len(sentence)):
                    # choose the jth source word as a start
                    for k in range(j + 1, min(len(sentence), j + maxPhraseLen) + 1):
                        # the phrase choosen to add this time is from j to k
                        sourcePhrase = sentence[j:k]

                        # Skip if the phrase doesn't exist
                        if sourcePhrase not in self.tm:
                            continue
                        if currentSentence.overlapWithPhrase(j, k):
                            continue

                        for targetPhrase in self.tm[sourcePhrase][:maxTranslation]:
                            # Add phrase to sentence
                            targetSentence = deepcopy(currentSentence).addPhrase(j, k, targetPhrase)

                            # Compare with best score if translation complete, and skip adding to newStack
                            if targetSentence.translationCompleted():
                                if targetSentence.totalScore(self.lm) > bestScore:
                                    bestScore = targetSentence.totalScore(self.lm)
                                    bestSentence = targetSentence.targetSentenceEntity
                                if saveToList:
                                    self.addToAnswerSet(targetSentence)
                            else:
                                # Add the combined targetSentence to newStack if translation incomplete
                                if targetSentence.key() in newStack[targetSentence.length()]:
                                    if targetSentence.tmScore <= stack[targetSentence.length()][targetSentence.key()][1]:
                                        continue
                                stack[length][key] = targetSentence.totalScore(self.lm), targetSentence.tmScore

                                # do pruning
                                sortedStack = sorted(stack[length].items(), key=lambda x: x[1], reverse=True)
                                for item in sortedStack[:maxStackSize]:
                                    # only tmScore matters
                                    stack[length].append((item[0], item[1][1]))

        # All words processed, we now have the best sentence stored in bestSentence
        print " ".join(bestSentence)
        return

    def addToAnswerSet(self, targetSentence):
        """
            add a TargetSentence instance to self.answers

        """
        self.answers.append(targetSentence)
        return

    def chooseBestAnswer(self):
        """
            return: the targetSentence with the highest score in self.answers

            # must be used with decoder option saveToList=True

            We can use a different model here to choose the best sentence. For
            example, we could call the reranker here.

        """
        bestScore = -sys.maxint - 1
        bestSentence = None
        for sentence in self.answers:
            score = sentence.totalScore(self.lm)
            if bestScore < score:
                bestScore = score
                bestSentence = sentence

        return bestSentence


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=sys.maxint, type="int", help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=500, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-p", "--max-phrase-length", dest="p", default=10, type="int", help="Maximum phrase length (default=10)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False, help="Verbose mode (default=off)")
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
        decoder.decode(f,
                       maxPhraseLen=opts.p,
                       maxStackSize=opts.s,
                       maxTranslation=opts.k,
                       saveToList=False,
                       verbose=opts.verbose)
