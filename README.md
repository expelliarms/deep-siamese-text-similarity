
# Environment
- numpy 1.11.0
- tensorflow 1.2.1
- gensim 1.0.1
- nltk 3.2.2

# Preprocessing
- Download dataset from https://www.kaggle.com/c/quora-question-pairs/data
- Extract the train.csv to Data/train.csv
- Replace comma delimiters to || by running the command
```
$ sed -i 's/\",\"/\"||\"/g' Data/train.csv
```
- Get embeddings from http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
- Extract and save it in embeddings/ directory

# How to run
### Training
```
$ python train.py --is_char_based=False --word2vec_model=embeddings/glove.twitter.27B.200d.txt --embedding_dim=200 --training_files=Data/train.csv
```

### Evaluation on validation set
```
$ python eval.py --checkpoint_dir=runs/<run_id>/checkpoints/ --vocab_filepath=runs/<run_id>/checkpoints/vocab --model=runs/<run_id>/checkpoints/model-<iter>
```

### Evaluation on command line
```
$ python test_eval.py --checkpoint_dir=runs/<run_id>/checkpoints/ --vocab_filepath=runs/<run_id>/checkpoints/vocab --model=runs/<run_id>/checkpoints/model-<iter> --q1="Question 1?" --q2="Question 2?"
```
# References
1. [Learning Text Similarity with Siamese Recurrent Networks](http://www.aclweb.org/anthology/W16-16#page=162)
2. [Siamese Recurrent Architectures for Learning Sentence Similarity](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)

