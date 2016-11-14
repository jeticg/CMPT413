import itertools
from copy import deepcopy


def generate_by_mask(sentence, mask, max_length=10):
    '''
        An iterator to generate discountinued phrase from sentence by mask

        input:
            sentence: a tuple of words.
            mask : a tuple of boolean values. 0 means avaliable

        output:
            phrase : an phrase, represented in tuple
            new_mask : updated mask, according to chosen phrase
    '''

    sentence = list(sentence)
    mask = list(mask)

    assert (len(mask) == len(sentence)), "mask and sentence must have same length"

    avaliable_index = [idx for idx, it in enumerate(mask) if it == 0]

    for phrase_length in range(1, min(len(avaliable_index), max_length) + 1):
        for chosen_index in itertools.combinations(avaliable_index, phrase_length):
            new_mask = deepcopy(mask)
            phrase = []

            for idx in chosen_index:
                phrase.append(sentence[idx])
                new_mask[idx] = 1

            yield tuple(phrase), tuple(new_mask)


if __name__ == "__main__":
    '''
    Example Input:
        sentence : ['How', 'are', 'you']
        mask : ['0', '0', '0']
    Example Output:
        ['how']                     [1, 0, 0]
        ['are']                     [0, 1, 0]
        ['you']                     [0, 0, 1]
        ['how', 'are']              [1, 1, 0]
        ['how', 'you']              [1, 0, 1]
        ['are', 'you']              [0, 1, 1]
        ['how', 'are', 'you']       [1, 1, 1]
    '''

    sentences = ('How', 'are', 'you')
    mask = (0 for _ in sentences)
    print sentences
    for phrase, mask in generate_by_mask(sentences, mask, max_length=10):
        print phrase, mask
