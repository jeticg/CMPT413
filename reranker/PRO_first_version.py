
# coding: utf-8

# In[65]:

import optparse, sys, os
from collections import namedtuple
import random
import math

import numpy as np

import bleu


# In[143]:

class opts:
    reference = os.path.join("data", "train.en")
    nbest = os.path.join("data", "train.nbest")


# In[144]:

translation_candidate = namedtuple("candidate", "sentence, inverse_scores, features")
ref = [line.strip().split() for line in open(opts.reference)]


# In[150]:

nbests = []
for n, line in enumerate(open(opts.nbest)):
    (i, sentence, features) = line.strip().split("|||")
    (i, sentence) = (int(i), sentence.strip())
    features = np.array([float(it) for it in features.split()])
    if len(ref) <= i:
        break

    while len(nbests) <= i:
        nbests.append([])

    scores = tuple(bleu.bleu_stats(sentence.split(), ref[i]))
    inverse_scores = tuple([-x for x in scores])
    smoothed_score = bleu.smoothed_bleu(inverse_scores)

    nbests[i].append((translation_candidate(sentence, inverse_scores, features), smoothed_score))

    if n % 2000 == 0:
        print('loaded %d lines' % n)

    # small size for testing, delete it when release
    if n > 4000:
        break


# In[151]:

tau = 5000
alpha = 0.1
xi = 100
eta = 0.1
epochs = 5
theta = np.array([1.0 / len(features) for _ in range(len(features))])


# In[154]:

mistakes = 0
for i in range(epochs):
    for nbest in nbests:

        def get_sample():
            sample = []
            for i in range(tau):
                # s1, s2 = random.sample(nbest, 2)
                s1 = random.choice(nbest)
                s2 = random.choice(nbest)
                if math.fabs(s1[1] - s2[1]) > alpha:
                    if s1[1] > s2[1]:
                        sample.append((s1, s2))
                    else:
                        sample.append((s2, s2))
                else:
                    continue
            return sample

        samples = sorted(get_sample(), key=lambda s: s[0][1] - s[1][1], reverse=True)
        samples = samples[:xi]

        for s in samples:
            s1 = s[0][0]
            s2 = s[1][0]
            if np.dot(theta, s1.features) <= np.dot(theta, s2.features):
                mistakes += 1
                theta += eta * (s1.features - s2.features)



# In[157]:

print "\n".join([str(weight) for weight in theta])


# In[ ]:
