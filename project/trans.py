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


def generate_TM(phrase_file, k_line=100, fname='temporary_tm.txt'):
    phrase_table = get_lines(phrase_file,
                             k_line=k_line,
                             preprocess=lambda x: x.strip().split('|||'))

    with open(fname, 'w+') as fp:
        for each in phrase_table:
            fp.write('%s|||%s||| %s\n' %
                     (each[0],
                      each[1],
                      math.log(float(each[2].strip().split()[0]))))
    return fname

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
    # tm_file = generate_TM(phrase_file, k_line=-1)  # -1 : all
    tm = models.TM(tm_file, max_translation)
    lm = models.LM(lm_file)
    for word in set(sum(fr, ())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]

    # initialise decoder
    decoder = Decoder(tm, lm)
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
        # each entry is one sentence, which is a list of words
        sentence_list = decoder.answers

    # TODO: how to get n-best choice in jetic's decoder?
    # MARK: you may easily get
