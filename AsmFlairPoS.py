import nltk
import os
import sys
from nltk.corpus import indian
import string
nltk.download('indian')
nltk.download('punkt')

from pathlib import Path
import re

#!cp /content/drive/MyDrive/Colab\ Notebooks/corpus/*.txt corpus/
pos_file=Path('corpus/dataset.txt')
assert(pos_file.exists())

pct_train = 0.8
pct_dev = 0.1
pct_test = 0.1

#with open(pos_file) as f:
#    lines = f.readlines()
#    line_count = len(lines)
#    train_lines_count = int(pct_train * line_count)
#    dev_lines_count = int(pct_dev * line_count)
#    test_lines_count = line_count - (train_lines_count + dev_lines_count)
#    assert (train_lines_count + dev_lines_count + test_lines_count == line_count)

#train_lines = lines[:train_lines_count]
#dev_lines = lines[train_lines_count:train_lines_count + dev_lines_count]
#test_lines = lines[:test_lines_count]
#assert(len(train_lines) == train_lines_count)
#assert(len(dev_lines) == dev_lines_count)
#assert(len(test_lines) == test_lines_count)

#os.system((head -n $train_lines_count corpus/dataset.txt) > corpus/train.txt)
#s.system((sed -n {train_lines_count + 1},{train_lines_count + dev_lines_count}p corpus/dataset.txt) > corpus/dev.txt)
#os.system((sed -n {train_lines_count + dev_lines_count + 1},{train_lines_count + dev_lines_count+test_lines_count}p corpus/dataset.txt) > corpus/test.txt)


from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, PooledFlairEmbeddings

# define columns
columns = {0: 'text', 1: 'pos'}

# this is the folder in which train, test and dev files reside
data_folder = 'corpus/'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

     # contextual string embeddings, forward
    PooledFlairEmbeddings('multi-forward', pooling='min'),

    # contextual string embeddings, backward
    PooledFlairEmbeddings('multi-backward', pooling='min'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)



# 5. initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=1024,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)


# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7. start training
#trainer.train('resources/taggers/example-pos',
trainer.train('tagger_130521',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=150)
