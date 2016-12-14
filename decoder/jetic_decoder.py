#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

from include.target_sentence import TargetSentence

ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
phrase = namedtuple("phrase", "english, logprob")


class Decoder():
    def __init__(self, tm, lm, itm=None, lex=None, ilex=None):
        self.tm = tm
        self.lm = lm
        self.itm = itm
        self.lex = lex
        self.ilex = ilex
        self.answers = []
        return

    def decode(self,
               sentence,
               maxPhraseLen=10,
               maxStackSize=100,
               maxTranslation=sys.maxint,
               saveToList=False,
               verbose=False):

        self.answers = []
        bestScore = - sys.maxint - 1
        bestSentence = ()
        emptyTargetSentence = TargetSentence(length=len(sentence))
        stack = [[] for x in range(len(sentence) + 1)]
        stack[0].append((emptyTargetSentence.key(), emptyTargetSentence.tmScore))
        stackSize = 1
        newStack = [{} for x in range(len(sentence) + 1)]
        minNewStack = [-sys.maxint for x in range(len(sentence) + 1)]

        for i in range(len(sentence)):
            if not verbose:
                sys.stderr.write("processing the " + str(i + 1) + "th phrase of " + str(len(sentence)) + ", stack size: " + str(stackSize) + "\n")
            # adding the ith target word/phrase

            # TODO: make program be able to choose discountinued word
            for j in range(len(sentence)):
                # choose the jth source word as a start
                for k in range(j + 1, min(len(sentence), j + maxPhraseLen) + 1):
                    # the phrase choosen to add this time is from j to k
                    sourcePhrase = sentence[j:k]

                    # Skip if the phrase doesn't exist
                    if sourcePhrase not in self.tm:
                        continue

                    for targetPhrase in self.tm[sourcePhrase][:maxTranslation]:
                        # print "source:", " ".join(sourcePhrase), "; target translation:", targetPhrase.english
                        # for each translation, combine with every existing targetSentence in stack
                        for stackLength in range(len(sentence) + 1):
                            # check if the phrase size is getting too big
                            if stackLength + k - j > len(sentence):
                                break
                            for targetSentenceKey, targetSentenceScore in stack[stackLength]:
                                # Check if overlapped
                                if j <= targetSentenceKey[2] and targetSentenceKey[2] < k:
                                    continue
                                if sum(targetSentenceKey[0][j:k]) != 0:
                                    continue

                                # Calculate resulting length
                                length = sum(targetSentenceKey[0]) + k - j

                                # Generate min of newStack when size of the stack hits maximum
                                # if its already generated, and if the current score is already lower than min, skip
                                '''
                                if minNewStack[length] == -sys.maxint:
                                    if (len(newStack[length]) == maxStackSize):
                                        minNewStack[length] = min(newStack[length].items(), key=lambda x: x[1][0])
                                        minNewStack[length] = minNewStack[length][1][0]
                                elif minNewStack[length] >= targetSentenceScore[0]:
                                    continue
                                '''
                                # Reconstruct sentence
                                targetSentence = TargetSentence(key=targetSentenceKey,
                                                                tmScore=targetSentenceScore)

                                # Add phrase to sentence
                                targetSentence.addPhrase(j, k,
                                                         sourcePhrase, targetPhrase,
                                                         self.itm, self.lex, self.ilex)

                                # Compare with best score if translation complete, and skip adding to newStack
                                if targetSentence.translationCompleted():
                                    if saveToList:
                                        self.addToAnswerSet(targetSentence)
                                    if targetSentence.totalScore(self.lm) > bestScore:
                                        bestScore = targetSentence.totalScore(self.lm)
                                        bestSentence = targetSentence.targetSentenceEntity
                                else:
                                    # print "Completion:", i+1, "out of", len(sentence), ":", " ".join(targetSentence.targetSentenceEntity)
                                    # Add the combined targetSentence to newStack if translation incomplete
                                    key = targetSentence.key()
                                    if key in newStack[length]:
                                        if targetSentence.tmScore[0] <= newStack[length][key][1][0]:
                                            continue
                                    newStack[length][key] = targetSentence.totalScore(self.lm), targetSentence.tmScore
                                # Current stackLength processed, processed with next subStack

                            # Current targetSentences processed, proceed with next targetSentence in stack

                        # All targetSentences in stack processed, proceed with next translation

                    # All translations processed, proceed with next phrase

                # All phrases processed, proceed with next starting position for phrase

            # The ith phrase added to newStack. Exchange newStack and stack
            # do pruning
            stack = [[] for x in range(len(sentence) + 1)]
            stackSize = 0
            for length in range(1, len(sentence) + 1):
                # sys.stderr.write("Doing pruning for length: " + str(length) + "; size before pruning: " + str(len(newStack[length])) + "\n")
                # Sort by score
                sortedStack = sorted(newStack[length].items(), key=lambda x: x[1], reverse=True)
                for item in sortedStack[:maxStackSize]:
                    # only tmScore matters
                    stack[length].append((item[0], item[1][1]))
                    stackSize += 1

            newStack = [{} for x in range(len(sentence) + 1)]
            minNewStack = [-sys.maxint - 1 for x in range(len(sentence) + 1)]

        # All words processed, we now have the best sentence stored in bestSentence
        print " ".join(bestSentence)
        return

    def addToAnswerSet(self, targetSentence):
        """
            add a TargetSentence instance to self.answers

        """
        self.answers.append(targetSentence)
        return


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
