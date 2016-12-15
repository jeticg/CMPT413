#!/usr/bin/env python
import optparse, sys, os
import bleu
import random
import math
import numpy as np
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "rich_train.nbest"), help="N-best file")
optparser.add_option("-e", "--en", dest="en", default=os.path.join("data", "train.en"), help="English input")
optparser.add_option("-f", "--fr", dest="fr", default=os.path.join("data", "train.fr"), help="French input")
(opts, _) = optparser.parse_args()

tau = 5000
alpha = 0.1
xi = 150
eta = 0.1
epochs = 1

nbest = namedtuple('nbest', 'sentence, features, bleu')


en_reference = {}
en_count = 0
for line in open(opts.en):
    en_reference[en_count] = line
    en_count += 1

# train_feature7 = open('answer/train.nbest.model1')
# bleu_scores = open('data/bleu_precal.txt')

ref = [line.strip().split() for line in open(opts.en)]
system = [line.strip().split() for line in open(opts.fr)]
nbests = [[] for _ in range(0, en_count)]

sys.stderr.write("Reading translation models from nbest file...\n")
for translation in open(opts.nbest):
    (i, sentence, features) = translation.strip().split("|||")
    stats = [0 for x in xrange(10)]

    r = ref[int(i)]
    s = sentence.strip().split()

    stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(s,r))]
    bleu_score = bleu.smoothed_bleu(stats)
    # load from file instead
    # bleu_score1 = float(bleu_scores.readline().strip())

    features = [float(h) for h in features.strip().split()]
    nbests[int(i.strip())].append(nbest(sentence, features, bleu_score))

feature_number = len(features)
sys.stderr.write("Handling %s features\n" % len(features))


# Reranking
train_history = {}
theta = [1.0 / feature_number for _ in range(feature_number)]
np_theta = np.array([1.0 / feature_number for _ in xrange(feature_number)], dtype=float)

np_theta = np.random.normal(0, 1, (feature_number))
np_theta2 = np.array(np_theta, copy=True)
for i in range(0, epochs):
    mistakes = 0
    np_mistakes = 0
    np_mistakes2 = 0

    sys.stderr.write("Epoch %s...\n" % i)
    # sys.stderr.write("length of nbests is %s and first is %s\n" % (len(nbests), len(nbests[0])))
    for nbest in nbests:
        def get_sample():
            sample = []
            for j in range(0, tau):
                s1 = random.choice(nbest)
                s2 = random.choice(nbest)
                if math.fabs(s1.bleu - s2.bleu) > alpha:
                    if s1.bleu > s2.bleu:
                        sample.append((s1, s2))
                    else:
                        sample.append((s2, s1))
                else:
                    continue
            return sample

        samples = sorted(get_sample(), key=lambda h: h[0].bleu - h[1].bleu)[:xi]
        random.shuffle(samples)

        pre_grad = np.zeros_like(np_theta)
        np_delta = np.zeros_like(np_theta)
        for idx, sample in enumerate(samples):
            # np try
            s1 = sample[0]
            s2 = sample[1]

            np_feat1 = np.array(s1.features, dtype=float)
            np_feat2 = np.array(s2.features, dtype=float)

            np_score1 = np.dot(np_theta, np_feat1)
            np_score2 = np.dot(np_theta, np_feat2)

            if np_score1 <= np_score2:
                np_mistakes += 1
                # normal
                np_theta += eta * (np_feat1 - np_feat2)

            np_score1 = np.dot(np_theta2, np_feat1)
            np_score2 = np.dot(np_theta2, np_feat2)

            if np_score1 <= np_score2:
                np_mistakes2 += 1
                np_delta += eta * (np_feat1 - np_feat2)

        # perceptron
        np_theta2 += np_delta

    sys.stderr.write("Mistake in epoch %s: NP mis : %s NP2 mis : %s \n" % (i, np_mistakes, np_mistakes2))
    train_history[i] = [[mistakes,     theta],
                        [np_mistakes,  np_theta.tolist()],
                        [np_mistakes2, np_theta2.tolist()] ]

print("\n".join([str(weight) for weight in np_theta2]))
