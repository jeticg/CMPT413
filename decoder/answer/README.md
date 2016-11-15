# Group Adam
------------------
Our score on the leaderboard(`Adam`) was produced by the following setting:

	python jetic_decoder.py -s 1000 > output
	
Alternatively, the score of `Snoopy4President` on the leaderboard was produced by the following setting:

	python jetic_decoder.py -s 250 > output
	
## jetic_decoder.py

This is our groups's first decoder implementation. It follows the following strategy:

	for I in range(len(sentence))
		choose anyFrenchPhrase		
			combine anyFrenchPhrase with all sentences in stack, if not overlapped
			if combinedSentence is translation complete(all French words translated):
				add combinedSentence to answerSet
		prune stacks
	
	print the sentence with highest score in answerSet

We used `stack[length]` to store all sentences with `length` French words translated. The sentences in the stacks are stored with the following information:

1. positions of translated French words
2. translated part of English sentence
3. ending position of the last French phrase
4. translation score and language score (tmScore, lmScore)

This algorithm gives us `-1257.384` using stack size `1000`, and `-1262.916` with stack size `500`, `-1280.107` using stack size `250`.

## lyken_decoder.py

This is an experiement of word choice on beam model. It allows sourcePhrase to choose discountinued words from source sentence.

For example, (here I use English as source sentence, because I don't know French)

* Source sentence: ("how", "are", "you") [Take Phrase of length 2 as example]
* Original model:  ("how", "are") ("are", "you") 
* Our experiement: ("how", "are") ("are", "you") ("how", "you") 

The algorithm is like 

	for I in range(len(sentence))
		choose anyFrenchPhrase that does not overlap with translated words
			combine anyFrenchPhrase with all sentences in stack, 
			if combinedSentence is translation complete(all French words translated):
				add combinedSentence to answerSet
		prune stacks
	
	print the sentence with highest score in answerSet
	
However, this choice will produce $\sum_{i=0}^{n}{n \choose i} = 2^{n}$ combinations, far more than $n^2$. And we don't see much score improvement. Therefore it is not used for leaderboard.

