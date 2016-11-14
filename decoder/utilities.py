import itertools
from copy import deepcopy


def generate_by_mask(sentence, mask):
    '''
        An iterator to generate discountinued phrase from sentence by mask

        input:
            sentence: an array of words.
            mask : an array of boolean values. True means avaliable

        output:
            phrase : an phrase, represented in array
            new_mask : updated mask, according to chosen phrase
    '''

    assert (len(mask) == len(sentence)), "mask and sentence must have same length"

    avaliable_index = [idx for idx, it in enumerate(mask) if it]
    for phrase_length in range(1, len(avaliable_index) + 1):
        for chosen_index in itertools.combinations(avaliable_index, phrase_length):
            new_mask = deepcopy(mask)
            phrase = []

            for idx in chosen_index:
                phrase.append(sentence[idx])
                new_mask[idx] = False

            yield phrase, new_mask


if __name__ == "__main__":
    '''
    Example Input:
        sentence : ['How', 'are', 'you']
        mask : ['True', 'True', 'True']
    Example Output:
        ['how']                     [False, True, True]
        ['are']                     [True, False, True]
        ['you']                     [True, True, False]
        ['how', 'are']              [False, False, True]
        ['how', 'you']              [False, True, False]
        ['are', 'you']              [True, False, False]
        ['how', 'are', 'you']       [False, False, False]
    '''

    words = "how are you"
    sentences = words.split()
    mask = [True for _ in sentences]
    print words
    for phrase, mask in generate_by_mask(sentences, mask):
        print phrase, mask
