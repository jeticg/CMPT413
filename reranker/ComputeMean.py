#!/usr/bin/env python
import optparse, sys, os
import bleu
import random
import math
import numpy as np
from collections import namedtuple

optparser = optparse.OptionParser()


prefix = "train"
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", prefix + ".nbest"), help="N-best file")
optparser.add_option("-e", "--en", dest="en", default=os.path.join("data", prefix + ".en"), help="English input")
optparser.add_option("-f", "--fr", dest="fr", default=os.path.join("data", prefix + ".fr"), help="French input")
(opts, _) = optparser.parse_args()

NBestEntry = namedtuple('NBestEntry', 'sentence, features, bleu')


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

bleu_scores = open('data/bleu_precal.txt')

nbests = [[] for _ in range(0, fr_count)]

sys.stderr.write("Reading translation models from %s ...\n" % opts.nbest)

MODE_TEST = False

std = namedtuple('STD', 'mean, deviation, max, min')
std.mean = []
std.deviation = []

mean_feat = []

feat_min = None
feat_max = None

for n, translation in enumerate(open(opts.nbest)):
    (i, sentence, features) = translation.strip().split("|||")
    i = int(i.strip())
    sentence = sentence.strip()

    bleu_score = float(bleu_scores.readline().strip())
    features = [float(h) for h in features.strip().split()]

    np_feat = np.array(features)

    if feat_min is None:
        feat_min = np_feat
        feat_max = np_feat
    else:
        feat_min = np.minimum(np_feat, feat_min)
        feat_max = np.maximum(np_feat, feat_max)

    mean_feat.append(features)
    if MODE_TEST:
        if n > 2:
            break

    if n % 25000 == 0:
        sys.stderr.write("Reading %s lines\n"% (n))

mean_feat = np.array(mean_feat)
print mean_feat

std.mean = np.mean(mean_feat, axis=0)
std.deviation = np.std(mean_feat, axis=0)
std.min = feat_min
std.max = feat_max

print std.mean
print std.deviation
print std.min
print std.max

std_dump = np.array([std.mean, std.deviation, std.min, std.max])
np.save("%s" % ('mean'), std_dump)
