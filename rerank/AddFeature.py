import optparse, sys, os
import random
import math
import logging

from collections import defaultdict, namedtuple
import numpy as np

optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
optparser.add_option("-e", "--en", dest="en", default=os.path.join("data", "train.en"), help="English input")
optparser.add_option("-f", "--fr", dest="fr", default=os.path.join("data", "train.fr"), help="French input")
optparser.add_option("-r", "--rich_nbest", dest="rich_nbest", default=os.path.join("data", "rich_train.nbest"), help="N-best file")
optparser.add_option("-i", "--IBM1", dest="IBM1", default=os.path.join("data", "train_IBM_score.nbest"), help="N-best file")

(opts, _) = optparser.parse_args()

import bleu

loadFromFile = False

#
# prefix = "test"
# class opts:
#     en = "data/%s.en" % prefix
#     fr = "data/%s.fr" % prefix
#     nbest = "data/%s.nbest" % prefix
#     rich_nbest = "data/rich_%s.nbest" % prefix
#     IBM1 = "data/%s_IBM_score.nbest" % prefix

if loadFromFile:
    from jetic_IBM1 import AlignerIBM1
    f_data = 'data/hansards.fr'
    e_data = 'data/hansards.en'

    biText = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:sys.maxint]]
    biText2 = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:sys.maxint]]

    aligner = AlignerIBM1()
    aligner.train(biText, 5)

    IBM_untrans = lambda k: sum(1 for _ in k if _[1] <= 0.5)
    IBM_score = lambda k: sum(_[1] for _ in k if _[1] >= 1)

else:
    IBM1 = open(opts.IBM1)



en_reference = {}
en_count = 0
for line in open(opts.en):
    en_reference[en_count] = line
    en_count += 1

fr_reference = {}
fr_count = 0
for line in open(opts.fr):
    fr_reference[fr_count] = line
    fr_count += 1

nbests = [[] for _ in range(0, fr_count)]
MODE_TEST = False

#
# std = namedtuple('STD', 'mean, deviation, max, min ')
#
# temp = np.load('mean.npy')
# std.mean = temp[0]
# std.deviation = temp[1]
# std.max = temp[2]
# std.min = temp[3]
#
#
# orig_feat_min = np.zeros_like(std.mean)
# orig_feat_max = np.zeros_like(std.mean)
#
# norm_feat_min = np.zeros_like(std.mean)
# norm_feat_max = np.zeros_like(std.mean)

with open(opts.rich_nbest, 'w+') as fp:
    for n, translation in enumerate(open(opts.nbest)):
        # print 'here'
        (i, sentence, features) = translation.strip().split("|||")
        i = int(i.strip())
        sentence = sentence.strip()

        features = [float(h) for h in features.strip().split()]
        old_feat = features

        model1 = float(IBM1.readline().strip())
        # new features
        total_words = len(sentence.strip().split())
        raw_untranslated = max(total_words - len(fr_reference[i].strip().split()), 0)

        features += [model1, total_words, raw_untranslated]



        sen_feat = " ".join([str(_) for _ in features])
        output2fp = "%s ||| %s ||| %s\n" % (i, sentence, sen_feat)
        fp.write(output2fp)

        if MODE_TEST:
            print translation
            print output2fp
            print old_feat
            print features
            break


        if n % 25000 == 0:
            sys.stderr.write("Reading %s lines\n"% (n))
