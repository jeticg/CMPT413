# Group Adam

    python answer/chunk.py -m model
    python answer/perc.py -m model > output

## Perceptron and average perceptron
Almost as pseudocode, Jetic made some performance improvement for case

    if not gold_vec == local_vec:

## Stochastic Gradient Descent
Based on Jetic's code, Lyken added SGD. We adopt batch size of 196 (which is common in deep learning), and notice a obvious speed improvement during training.

    batch_train_data = random.sample(train_data, 128)

## Trigram
Lyken modified file ```perc.py``` to implement trigram, which mades vertibie algorithm more robust.

    for tag in tagset:
        try out all possible combination of y[i-2], y[i-1]
    find the largest one

## LSTM Experiment
Inspired by paper https://arxiv.org/pdf/1603.01354v5.pdf, Lyken decides to try LSTM model for sequence tagging. Our code utilizes code from https://github.com/karpathy/char-rnn, https://github.com/harvardnlp/seq2seq-attn  and https://github.com/chilynn/sequence-labeling. We tried word-level tagging, but got a poor performance that cannot even reach the baseline. We guess it may caused by limited training data set. Char-level RNN may improve the performance, but we don't have enough compute resource. Code is attached in answer/LSTM folder.

PS : LSTM part is written in Torch/Lua. This is because Torch has a better RNN structure compared with tensorflow. And it is only a experiment, we don't use it for leaderboard. Hope it doesn't violate the rule that homework only accepts Python 2.7.
