# How to run the code

## Adam Group

To use the code here, first one needs to generate the n-best list for the reranker to learn.

	python trans.py -s SOURCE_FILE -t TARGET_FILE -p PHRASE_FILE, -l LM_FILE -g -1
	
This will generate a file duang.txt, which is the n-best file for the reranker learner.

Then, you can use reranker/learn.py to do the training.

	cd reranker
	python learn.py -n ../duang.txt -e TARGET_FILE -f SOURCE_FILE > WEIGHT_FILE
	cd ..

After getting a trained file using reranker, you can load the translator to do the translation.
	
	python trans.py -s SOURCE_FILE -t TARGET_FILE -p PHRASE_FILE, -l LM_FILE -w WEIGHT_FILE > output
	
The output has two parts: first the output from the unranked decoder, and then the reranked output.

Additional options of the translator could be found by

	python trans.py -h
	
