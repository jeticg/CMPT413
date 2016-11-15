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

