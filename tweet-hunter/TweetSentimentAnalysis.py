import numpy as np
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
import argparse

from os.path import dirname, abspath

def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="Twitter Downloader")
    '''parser.add_argument("-q", "--query", dest="query", help="Query/Filter", default='-')'''
    parser.add_argument("-d", "--data-dir", dest="data_dir", help="json Data Directory")
    return parser


if __name__ == '__main__':

    parser = get_parser()
    args = parser.parse_args()

    data_location = args.data_dir #set this



    nn_model_location = '{}/sentimentNet.h5'.format(dirname(dirname(abspath(__file__))))
    wv_model_location = '{}/GoogleNews-vectors-negative300.bin'.format(dirname(dirname(abspath(__file__))))


    with open(data_location,'r', encoding ='utf-8') as t:
        tweets = []
        for i,line in enumerate(t):
            if(i%2==0):
                tweets.append(json.loads(line)["text"].strip().lower())
                    
    # Tokenize and stem
    tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
    stemmer = LancasterStemmer()

    tokenized_corpus = []

    for i, tweet in enumerate(tweets):
        tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
        tokenized_corpus.append(tokens)
            
    vector_size = 300

    max_tweet_length = 15


    word_vecs = KeyedVectors.load_word2vec_format(wv_model_location, binary=True)
    print('all models loaded')
    X = np.zeros((len(tokenized_corpus),max_tweet_length, vector_size), dtype=K.floatx())#3D np array


    for i in range(len(tokenized_corpus)):
        for t, token in enumerate(tokenized_corpus[i]):
            if t >= max_tweet_length:
                break
            
            if token not in word_vecs:
                continue
        
            
            X[i, t, :] = word_vecs[token]
    print('word vectors calculated')
    del word_vecs

    model = load_model(nn_model_location)
    predictions = model.predict(X)
    #print(predictions)
    #predictions = np.nan_to_num(predictions)
    print("Average sentiment of collected tweets is "+ str(np.average(predictions)))
    print("Number of tweets processed: "+ str(len(predictions)))
