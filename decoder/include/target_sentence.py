from math import log10
from copy import deepcopy

import utilities


class TargetSentence():
    def __init__(self,
                 length=0,
                 sourceMark=[],
                 targetSentenceEntity=(),
                 tmScore=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 # tmScore=0.0,
                 key=None):
        # There are two ways for creating a new TargetSentence instance
        # 1. use the length parameter
        # 2. use the key and tmScore parameter. (Will ignore all other parameters)
        if key:
            self.sourceMark, self.targetSentenceEntity, self.lastPos = deepcopy(key)
            self.sourceMark = list(self.sourceMark)
            self.tmScore = list(tmScore)
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

    def getWords(self):
        return list(self.targetSentenceEntity)

    def getFeatures(self, lm):
        return [self.lmScore(lm), self.tmScore[5], self.tmScore[1], self.tmScore[2], self.tmScore[3], self.tmScore[4]]

    def key(self):
        # Generate the unique key for the sentence
        key = (tuple(self.sourceMark), self.targetSentenceEntity, self.lastPos)
        return key

    def overlapWithPhrase(self, phraseStartPosition, phraseEndPosition):
        # Check if the source phrase overlaps with the sentence
        if sum(self.sourceMark[phraseStartPosition:phraseEndPosition]) == 0:
            return False
        return True

    def addPhrase(self,
                  phraseStartPosition, phraseEndPosition,
                  sourcePhrase, targetPhrase,
                  itm=None, lex=None, ilex=None):
        # mark positions in sourceMark as translated
        for i in range(phraseStartPosition, phraseEndPosition):
            self.sourceMark[i] = 1
        # add target phrase to sentence
        self.targetSentenceEntity = self.targetSentenceEntity + tuple(targetPhrase.english.split())
        # update translation score
        # self.tmScore += targetPhrase.logprob + self.distance(self.lastPos, phraseStartPosition)
        # '''
        self.tmScore[0] += targetPhrase.logprob + self.distance(self.lastPos, phraseStartPosition)
        if itm is not None:
            self.tmScore[1] += itm[(sourcePhrase, targetPhrase.english)]
        if ilex is not None:
            self.tmScore[2] += ilex[(sourcePhrase, targetPhrase.english)]
        self.tmScore[3] += targetPhrase.logprob
        if lex is not None:
            self.tmScore[4] += lex[(targetPhrase.english, sourcePhrase)]
        self.tmScore[5] += self.distance(self.lastPos, phraseStartPosition)
        # '''
        # update lastPos
        self.lastPos = phraseEndPosition - 1
        return

    def addPhraseByMask(self,
                        phraseStartPosition, phraseEndPosition, phraseMask,
                        sourcePhrase, targetPhrase,
                        itm=None, lex=None, ilex=None):

        self.sourceMark = phraseMask
        # add target phrase to sentence
        self.targetSentenceEntity = self.targetSentenceEntity + tuple(targetPhrase.english.split())
        # update translation score
        self.tmScore[0] += targetPhrase.logprob + self.distance(self.lastPos, phraseStartPosition)
        self.tmScore[1] += targetPhrase.logprob
        if itm is not None:
            self.tmScore[2] += itm[(sourcePhrase, targetPhrase.e)]
        if lex is not None:
            self.tmScore[3] += lex[(targetPhrase.e, sourcePhrase)]
        if ilex is not None:
            self.tmScore[4] += ilex[(sourcePhrase, targetPhrase.e)]
        self.tmScore[5] += self.distance(self.lastPos, phraseStartPosition)
        # update lastPos
        self.lastPos = phraseEndPosition - 1
        return

    def distance(self, endOfLast, startOfCurrent):
        # d(endOfLast, startOfcurrent) = alpha ^ (abs(startOfCurrent - endOfLast - 1))
        # since all the scores are logd, assume beta = log10(alpha)
        # log10(d(endOfLast, startOfcurrent)) = beta * (abs(startOfCurrent - endOfLast - 1))

        # The beta value here is log10(0.9)
        beta = -0.045757490560675115
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
        return self.lmScore(lm) + self.tmScore[0]

    def length(self):
        return sum(self.sourceMark)

    def translationCompleted(self):
        if sum(self.sourceMark) == len(self.sourceMark):
            return True
        return False
