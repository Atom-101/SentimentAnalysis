import numpy as np
import keras.backend as K
import json

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from keras.models import load_model

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

from os.path import abspath,dirname

data_location = 'articles.txt' #set this
nn_model_location = '{}/sentimentNet.h5'.format(dirname(dirname(abspath(__file__))))
wv_model_location = '{}/GoogleNews-vectors-negative300.bin'.format(dirname(dirname(abspath(__file__))))

with open(data_location, 'r', encoding = 'utf-8') as doc:
    sentences=[x.strip().lower() for x in doc.read().split('.')]

tkr = RegexpTokenizer('[a-zA-Z0-9]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, sentence in enumerate(sentences):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(sentence)]
    tokenized_corpus.append(tokens)
	
vector_size = 300
max_sentence_length = 15
	
X_vecs = KeyedVectors.load_word2vec_format(wv_model_location, binary=True)

X = np.zeros((len(tokenized_corpus), max_sentence_length, vector_size), dtype=K.floatx())

invalid_ind =[]
for i in range(len(tokenized_corpus)):
    invalid_word_count = 0
    for t, token in enumerate(tokenized_corpus[i]):
        if t >= max_sentence_length:
            break
        
        if token not in X_vecs:
            invalid_word_count+=1
            continue
        
        X[i, t, :] = X_vecs[token]
    
    if(invalid_word_count>=10):
        invalid_ind.append(i)

for i in invalid_ind:		
    del X[i,:,:]#remove sentences with fewer than 5 valid words

del X_vecs


model = load_model(nn_model_location)
predictions = model.predict(X)
print("Average sentiment of collected text is: " +str(np.average(predictions)))
print("Number of sentences processed: "+ str(len(predictions)))
