#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
from copy import deepcopy

import include.utilities


optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,    help="Verbose mode (default=off)")
optparser.add_option("-p", "--max-phrase-length", dest="p", default=10, type="int", help="Maximum phrase length (default=10)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]


sys.stderr.write("Decoding %s...\n" % (opts.input,))
for f in french:
    # The following code implements a monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of
    # the first i words of the input sentence. You should generalize
    # this so that they can represent translations of *any* i words.
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, predecessor, phrase, mask, start_pos, end_pos")
    initial_mask = [True for _ in f]
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, initial_mask, 0, 0)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis

    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]:  # prune
            current_mask = h.mask

            for j in xrange(len(f)):
                start_pos = j
                for k in xrange(start_pos + 1, min(len(f), start_pos + opts.p) + 1):
                    end_pos = k
                    sourcePhrase = f[start_pos:end_pos]

                    # update mask
                    new_mask = deepcopy(current_mask)
                    zlg_mask = initial_mask = [True for _ in f]

                    overlap = False
                    for idx in range(start_pos, end_pos):
                        if current_mask[idx] == False:
                            overlap = True
                            break
                        zlg_mask[idx] = new_mask[idx] = False

                    if sourcePhrase in tm and not overlap:
                        # print 'enter'
                        for phrase in tm[sourcePhrase]:
                            # score from transition model
                            logprob = h.logprob + phrase.logprob

                            # score from language model
                            lm_state = h.lm_state
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            logprob += lm.end(lm_state) if j == len(f) else 0.0

                            # score from distance
                            beta = -0.045757490560675115
                            dis = abs(start_pos - h.end_pos - 1)
                            logprob += beta * dis

                            # Insert new hypothesis
                            new_hypothesis = hypothesis(logprob, lm_state, h, phrase, new_mask, start_pos, end_pos - 1)

                            # update stacks
                            stack_idx = i + len(sourcePhrase)

                            try: # Debug usage
                                if lm_state not in stacks[stack_idx] or stacks[stack_idx][lm_state].logprob < logprob:  # second case is recombination
                                    stacks[stack_idx][lm_state] = new_hypothesis
                            except IndexError:
                                print current_mask
                                print zlg_mask
                                print new_mask

                                print overlap

                                print sourcePhrase
                                print i, len(sourcePhrase), len(stack)
                                print stack_idx
                                exit(0)

    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)


    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)


    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
            (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
