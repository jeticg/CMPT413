# Current progress

1. Utilizes decoder with given Chinese-English data

# TODO
1. Generate N-best translations in jetic's decoder
2. Rerank N-best translations

# Known tricks to improve scores
1. Use [stanford segmenter](http://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.stanford_segmenter) to re-segment given data.
2. Use [filtered phrase tables](Use filtered phrase tables with your decode)
3. (Not sure) feed back loop between decoder and reranker.
