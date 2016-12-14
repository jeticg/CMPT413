import math
import sys
import os
from collections import namedtuple
from copy import deepcopy

import decoder.models as models
from decoder.jetic_decoder import Decoder
from utility import get_lines


class Translator():
    def __init__(self,
                 tm, lm, itm, lex, ilex,
                 weights=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        self.decoder = Decoder(tm, lm, itm, lex, ilex)
        self.lm = lm
        self.weights = weights
        return

    def translate(self, f):
        self.decoder.decode(f,
                            maxPhraseLen=20,
                            maxStackSize=500,
                            maxTranslation=20,
                            saveToList=True,
                            verbose=False)

        sentence_list = deepcopy(self.decoder.answers)
        maxScore = -sys.maxint
        output = f
        for targetSentence in range(len(sentence_list)):
            features = targetSentence.getFeatures(self.lm)
            score = 0.0
            for i in range(len(features)):
                score += features[i] * self.weights[i]
            print score
            if score > maxScore:
                maxScore = score
                output = ' '.join(targetSentence.getWords())
        # rerank
        return output


def get_trans_pairs(source_file, target_file, k_line=5):
    def split_sentence(x):
        return tuple(x.strip().split())

    fr = get_lines(source_file, k_line=k_line, preprocess=split_sentence)
    en = get_lines(target_file, k_line=k_line, preprocess=split_sentence)
    return fr, en


def generate_TM(phrase_file, k_line=100, k_trans=1):
    phrase = namedtuple("phrase", "english, logprob")

    phrase_table = get_lines(phrase_file,
                             k_line=k_line,
                             preprocess=lambda x: x.strip().split('|||'))

    tm = {}
    itm = {}
    lex = {}
    ilex = {}

    for each in phrase_table:
        f = tuple(each[0].split())
        e = each[1].strip()
        numbers = each[2].strip().split()
        for index in range(len(numbers)):
            numbers[index] = float(numbers[index])
            if numbers[index] > 0:
                numbers[index] = math.log(numbers[index])

        tm.setdefault(f, []).append(phrase(e, numbers[0]))
        # tm[(e, f)] = numbers[0]
        itm[(f, e)] = numbers[1]
        lex[(e, f)] = numbers[2]
        ilex[(f, e)] = numbers[3]
    for f in tm:  # prune all but top k translations
        tm[f].sort(key=lambda x: -x.logprob)
        del tm[f][k_trans:]
    return tm, itm, lex, ilex

if __name__ == "__main__":
    source_file = 'nlp-data/medium/train.cn'
    target_file = 'nlp-data/medium/train.en'
    phrase_file = 'nlp-data/medium/phrase-table/phrase-table'
    lm_file = 'nlp-data/lm/en.tiny.3g.arpa'
    max_line = 5
    max_translation = 1

    fr, en = get_trans_pairs(source_file, target_file, k_line=max_line)
    sys.stderr.write("loaded source and target sentences\n")

    # load translation model and language model
    tm, itm, lex, ilex = generate_TM(phrase_file, k_line=-1, k_trans=max_translation)
    lm = models.LM(lm_file)
    for word in set(sum(fr, ())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]
            itm[((word,), word)] = 0.0
            lex[(word, (word,))] = 0.0
            ilex[((word,), word)] = 0.0

    # initialise decoder
    translator = Translator(tm, lm, itm, lex, ilex)
    sys.stderr.write("translator loaded\n")

    # start decoding
    count = 0
    for f in fr:
        count += 1
        sys.stderr.write("translating sentence " + str(count) + " of " +
                         str(len(fr)) + "\n")
        output = translator.translate(f)
        print output
