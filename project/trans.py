import math, sys, os
from collections import namedtuple


import decoder.models as models
from decoder.jetic_decoder import Decoder
from utility import get_lines


def get_trans_pairs(source_file, target_file, k_line=5):
    split_sentence = lambda x: tuple(x.strip().split())
    fr = get_lines(source_file, k_line=k_line, preprocess=split_sentence)
    en = get_lines(target_file, k_line=k_line, preprocess=split_sentence)
    return fr, en

def generate_TM( k_line=100, fname='temporary_tm.txt'):
    phrase_table = get_lines(phrase_file, k_line=k_line, preprocess=lambda x:x.strip().split('|||'))

    with open(fname, 'w+') as fp:
        for each in phrase_table:
            fp.write('%s|||%s||| %s\n' %(each[0], each[1], math.log(float(each[2].strip().split()[0]))))
    return fname

if __name__ == "__main__":
    source_file = 'nlp-data/medium/train.cn'
    target_file = 'nlp-data/medium/train.en'
    phrase_file = 'nlp-data/medium/phrase-table/phrase-table'

    fr, en = get_trans_pairs(source_file, target_file, k_line=5)
    sys.stderr.write("loaded source and target sentences\n")

    fname = generate_TM(k_line=100) # -1 : all
    tm = models.TM(fname, 1)
    lm = models.LM('lm')
    for word in set(sum(fr,())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]
    decoder = Decoder(tm, lm)
    sys.stderr.write("loaded decoder model\n")
    
    count = 0
    for f in fr:
        count += 1
        sys.stderr.write("Decoding sentence " + str(count) + " of " + str(len(fr)) + "\n")
        decoder.decode(f,
                       maxPhraseLen=20,
                       maxStackSize=500,
                       maxTranslation=20,
                       saveToList=False,
                       verbose=False)

    # TODO: how to get n-best choice in jetic's decoder?
    # MARK: you may easily get
