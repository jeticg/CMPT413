# Group Adam

## Our Unigram Algorithm

We used a simple dynamic programming algorithm designed precisely for this task. `score[i]` stores the best score that the first `i` characters can get. Then, the equation would be simple:

	score[i]=max(score[j] + log(scoreOfWord(j+1, i)))

## Scoring Unknown Words

We have tried two approaches and decided to adopt the better one.

###1. The longer, the lower

Consider the unknown word to be `unknownWord`, it's score would be computed with a simple equation:

	scoreOfWord(unknownWord) = alpha * power(beta, len(unknownWord))
	
In which alpha and beta are constants. The best score we were able to get out of this method wasn't satisfactory. So we tried the following method.

###2. Prediction model using known words

In the Chinese language, it is common for some characters to composite a word and some not, sometimes it is even common for some characters to be in the exact same location in multiple words, for example the first or the last. Consider the character "李". "李" is a very common family name in the Chinese language, and in China one's family name goes before his given name. So it is common for "李" to be located at the beginning of a word.

Here, we uses the known words to produce something similar. We use `startMark[c]` to store the probability of having a word with character `c` in the beginning, `endMark[c]` to store the probability of having a word with character `c` in the end, and `middleMark[c]` for it in the middle.

Pseudocode for generating the tables:

	for all known words:
		startMark[word[0]] += scoreOfWord[word]
		endMark[word[len(word)-1]] += scoreOfWord[word]
		middleMark[word[each of the rest]] += scoreOfWord[each of the rest]
		
Pseudocode for using the tables:

	scoreOfWord(unknownWord) =
		startMark[unknownWord[0]] *
		endMark[unknownWord[len(word)-1]] *
		middleMark[unknownWord[each of the rest]] + c
		
In which c is a const.

# Out Bigram Algorithm

We split the whole sentence into two parts, first word and the rest, and recursively produce the score.

	scoreOfSentence(prev, Sentence) = max(score(prev, first) + scoreOfSentence(first, restOfSentence))
	
The scoring we used for bigram is as follow:

	score(prev, first) = count(prev+first)/count(prev)
	
And for unknown word pairs:

	score(prev, first) = scoreOfWord(first)
	
This algorithm without smoothing gave us a score of 92.463. Because one of our teammate is currently sick so we didn't have time to try out all of the smoothing methods presented in class.

### Other Approach

We have tried an unknown bigram scoring method which is similar with the one we used for unigram, marking the first word and second word according to their relative position separately, however we failed to produce a higher score. 
