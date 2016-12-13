# Seq2seq approach to CMPT-413 final project
This approach is based by [Effective Approaches to Attention-based Neural Machine Translation](http://stanford.edu/%7Elmthang/data/papers/emnlp15_attn.pdf), Luong et al. EMNLP 2015.

We utlize code from 

* [Harvard-NLP code](https://github.com/harvardnlp/seq2seq-attn)
* [Karpathy char-rnn](https://github.com/karpathy/char-rnn)

# Dependencies
### Torch
Get [Torch](http://torch.ch) ready on your system.

### Torch Libraries
```
luarocks install hdf5
luarocks install nn
luarocks install nngraph
```

If you are going to train it by yourself
```
luarocks install cutorch
luarocks install cunn
luarocks install cudnn # for cudnn acceleration, not necessary
```

### Python 2.7
Uh, I think everyone should have it installed already.

### Python Libraries
```
pip install h5py numpy
```

# Usage (Play with)
### Download pretrained model

### run
```th evaluate.lua -model demo-model_final.t7 -src_file data/src-val.txt -output_file pred.txt 
-src_dict data/demo.src.dict -targ_dict data/demo.targ.dict``

# Usage (train your own model)
### Split data into train and validation 

### Transform data into hdf5
```
python preprocess.py --srcfile data/src-train.txt --targetfile data/targ-train.txt
--srcvalfile data/src-val.txt --targetvalfile data/targ-val.txt --outputfile data/demo
```

### Train
```
th train.lua -data_file data/demo-train.hdf5 -val_data_file data/demo-val.hdf5 -savefile demo-model \
-gpuid 0 -cudnn 1 -num_layers 4 -rnn_size 100
```
