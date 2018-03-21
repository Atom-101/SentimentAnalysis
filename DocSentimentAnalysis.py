import keras.backend as K
import multiprocessing
import tensorflow as tf
import json

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.models import load_model

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

data_location = 'articles.txt' #set this
nn_model_location = 'sentimentNet.h5'
wv_model_location = 'GoogleNews-vectors-negative300.bin'

with open(data_location, 'r', encoding = 'utf-8') as doc:
    sentences=[x.strip().lower() for x in doc.read().split('.')]#corpus equivalent



tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, sentence in enumerate(sentences):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(sentence) if not t.startswith('@')]
    tokenized_corpus.append(tokens)
	
vector_size = 300
max_sentence_length = 15
	
model = load_model(nn_model_location)
X_vecs = KeyedVectors.load_word2vec_format(wv_model_location, binary=True)

X = np.zeros((len(tokenized_corpus), max_sentence_length, vector_size), dtype=K.floatx())#3D np array

invalid_pos =[]
for i in range(len(tokenized_corpus)):
    invalid_word_count = 0
    for t, token in enumerate(tokenized_corpus[i]):
        if t >= max_tweet_length:
            break
        
        if token not in X_vecs:
            invalid_word_count+=1
            continue
	
    
        
        X[i, t, :] = X_vecs[token]
    if(invalid_word_count>=10):
        invalid_pos.append(i)

for i in invalid_pos:		
    del X[i,:,:]#remove sentences with fewewr than 5 valid words

predictions = model.predict(X)
print("Average sentiment of collected text is: " +str(np.average(predictions)))
print("Number of sentences processed: "+ str(len(predictions)))
