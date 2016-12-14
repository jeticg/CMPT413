import math
import sys
import os
import optparse
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
        for targetSentence in sentence_list:
            features = targetSentence.getFeatures(self.lm)
            score = 0.0
            for i in range(len(features)):
                score += features[i] * self.weights[i]
            print score
            if score > maxScore:
                maxScore = score
                output = ' '.join(targetSentence.getWords())

        return output

    def generateNBest(self, fr, n, fileName="duang.nbest"):
        count = 0
        fileN = open(fileName, 'w')
        for f in fr:
            self.decoder.decode(f,
                                maxPhraseLen=20,
                                maxStackSize=500,
                                maxTranslation=20,
                                saveToList=True,
                                verbose=False)

            sentence_list = deepcopy(self.decoder.answers)
            maxScore = -sys.maxint
            output = f
            printed = 0
            for targetSentence in sentence_list:
                sentence = ' '.join(targetSentence.getWords())
                features = targetSentence.getFeatures(self.lm)
                fileN.write(str(count) + " ||| " + sentence + " ||| " +
                            str(features[0]) + " " + str(features[1]) + " " +
                            str(features[2]) + " " + str(features[3]) + " " +
                            str(features[4]) + " " + str(features[5]) + "\n")
                printed += 1
                if printed == n:
                    break
            count += 1

        fileN.close()
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
    optparser = optparse.OptionParser()
    optparser.add_option("-s", "--source-file", dest="source", default='nlp-data/medium/train.cn', help="Source file")
    optparser.add_option("-t", "--target-file", dest="target", default='nlp-data/medium/train.en', help="Target file")
    optparser.add_option("-p", "--phrase-file", dest="phrase", default='nlp-data/medium/phrase-table/phrase-table', help="Phrase table file")
    optparser.add_option("-l", "--lm-file", dest="lm", default='nlp-data/lm/en.tiny.3g.arpa', help="Language model file")
    optparser.add_option("-m", "--max-line", dest="maxline", default=100, help="Translate the first m lines")
    optparser.add_option("-a", "--max-translation", dest="maxtrans", default=1, help="a translations for one source phrase")
    optparser.add_option("-g", "--generate-n-best", dest="generate", default=-1, help="generate n best")
    optparser.add_option("-w", "--weight-file", dest="weights", default=None, help="Weight filename, or - for stdin (default=use uniform weights)")
    (opts, _) = optparser.parse_args()

    fr, en = get_trans_pairs(opts.source, opts.target, k_line=opts.maxline)
    sys.stderr.write("loaded source and target sentences\n")

    # load translation model and language model
    tm, itm, lex, ilex = generate_TM(opts.phrase, k_line=-1, k_trans=opts.maxtrans)
    lm = models.LM(opts.lm)
    for word in set(sum(fr, ())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]
            itm[((word,), word)] = 0.0
            lex[(word, (word,))] = 0.0
            ilex[((word,), word)] = 0.0

    # load weight-file
    if opts.weights is not None:
        weights_file = sys.stdin if opts.weights is "-" else open(opts.weights)
        w = [float(line.strip()) for line in weights_file]
        w = map(lambda x: 1.0 if math.isnan(x) or x == float("-inf") or x == float("inf") or x == 0.0 else x, w)
        w = None if len(w) == 0 else w
    else:
        w = [1.0/6.0 for _ in xrange(6)]

    # initialise decoder
    translator = Translator(tm, lm, itm, lex, ilex, w)
    sys.stderr.write("translator loaded\n")

    # start decoding
    if opts.generate != -1:
        translator.generateNBest(fr, n=opts.generate)
    else:
        count = 0
        for f in fr:
            count += 1
            sys.stderr.write("translating sentence " + str(count) + " of " +
                             str(len(fr)) + "\n")
            output = translator.translate(f)
            print output
