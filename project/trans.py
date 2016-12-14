import math
import sys
import os
from collections import namedtuple


import decoder.models as models
from decoder.jetic_decoder import Decoder
from utility import get_lines


def get_trans_pairs(source_file, target_file, k_line=5):
    def split_sentence(x):
        return tuple(x.strip().split())

    fr = get_lines(source_file, k_line=k_line, preprocess=split_sentence)
    en = get_lines(target_file, k_line=k_line, preprocess=split_sentence)
    return fr, en


def generate_TM(phrase_file, k_line=100, k_trans=1):
    phrase_table = get_lines(phrase_file,
                             k_line=k_line,
                             preprocess=lambda x: x.strip().split('|||'))

    tm = {}
    itm = {}
    lex = {}
    ilex = {}

    for each in phrase_table:
        f = tuple(each[0].split())
        e = each[1]
        logprob = each[2].strip().split()[0]
        tm.setdefault(tuple(f.split()), []).append(phrase(e, float(logprob)))
        itm[(e, f)] = math.log(float(each[2].strip().split()[1]))
        lex[(f, e)] = math.log(float(each[2].strip().split()[2]))
        ilex[(e, f)] = math.log(float(each[2].strip().split()[3]))
    for f in tm:  # prune all but top k translations
        tm[f].sort(key=lambda x: -x.logprob)
        del tm[f][k:]
    return tm, itm, lex, ilex

if __name__ == "__main__":
    source_file = 'nlp-data/medium/train.cn'
    target_file = 'nlp-data/medium/train.en'
    phrase_file = 'nlp-data/medium/phrase-table/phrase-table'
    lm_file = 'data/lm/en.tiny.3g.arpa'
    tm_file = 'nlp-data/temporary_tm.txt'
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

    # initialise decoder
    decoder = Decoder(tm, lm, itm, lex, ilex)
    sys.stderr.write("loaded decoder model\n")

    # start decoding
    count = 0
    for f in fr:
        count += 1
        sys.stderr.write("Decoding sentence " + str(count) + " of " +
                         str(len(fr)) + "\n")
        decoder.decode(f,
                       maxPhraseLen=20,
                       maxStackSize=500,
                       maxTranslation=20,
                       saveToList=True,
                       verbose=False)
        # sentence_list is a list of all the sentenced generated from decoder
        # each entry is a targetSentence instance
        sentence_list = decoder.answers
        for index in range(len(sentence_list)):
            targetSentence = sentence_list[i]
            sentence_list[i] = targetSentence.getWords, targetSentence.getFeatures

    # TODO: how to get n-best choice in jetic's decoder?
    # MARK: you may easily get
