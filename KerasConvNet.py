import numpy as np
import random
import keras.backend as K

from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam

from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer

# Set random seed (for reproducibility)
seed = 1000
np.random.seed(seed)

dataset_location = 'dataset.csv'
#model_location = 'model.h5'

corpus = []
labels = []

print('Parsing dataset')
with open(dataset_location, 'r', encoding='utf-8') as df:
    for i, line in enumerate(df):
        if i == 0:
            # Skip the header
            continue

        parts = line.strip().split(',')
        
        # Sentiment (0 = Negative, 1 = Positive)
        labels.append((int)(parts[1].strip()))
        
        # Tweet
        tweet = parts[3].strip()
        if tweet.startswith('"'):
            tweet = tweet[1:]
        if tweet.endswith('"'):
            tweet = tweet[::-1]
        
        corpus.append(tweet.strip().lower())
        
print('Corpus size: {}'.format(len(corpus)))

# Tokenize and stem
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

tokenized_corpus = []

for i, tweet in enumerate(corpus):
    tokens = [stemmer.stem(t) for t in tkr.tokenize(tweet) if not t.startswith('@')]
    tokenized_corpus.append(tokens)
    
# Gensim Word2Vec model
vector_size = 300
window_size = 10

X_vecs = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('Vectors loaded')

train_size = int(0.9*len(tokenized_corpus))
max_tweet_length = 15

print('Creating train and test sets')
# Shuffle dataset
ds = list(zip(tokenized_corpus,labels))
random.Random(seed).shuffle(ds)
tokenized_corpus,labels = zip(*ds)

X_train = tokenized_corpus[:train_size]
Y_train = labels[:train_size]
X_test = tokenized_corpus[train_size:]
Y_test = labels[train_size:]

#Get train vectors
vector_list = []
for tweet in X_train:
    build_tweet = []
    for i,token in enumerate(tweet):
        if i>=max_tweet_length:
            break
        
        if token not in X_vecs:
            build_tweet.append(np.zeros(vector_size, dtype = K.floatx()))
            continue
        
        build_tweet.append(X_vecs[token])
    vector_list.append(build_tweet)

X_train = np.array(vector_list)

#Get test vectors
vector_list = []
for tweet in X_test:
    build_tweet = []
    for i,token in enumerate(tweet):
        if i>=max_tweet_length:
            break
        
        if token not in X_vecs:
            build_tweet.append(np.zeros(vector_size, dtype = K.floatx()))
            continue
        
        build_tweet.append(X_vecs[token])
    vector_list.append(build_tweet)

X_test = np.array(vector_list)
        
# Keras CNN model
del X_vecs
batch_size = 64
nb_epochs = 50

model = Sequential()

model.add(Conv1D(64, kernel_size=3, activation='elu', padding='same', input_shape=(max_tweet_length, vector_size)))
model.add(Conv1D(128, kernel_size=3, activation='elu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=2, strides=2))

model.add(Conv1D(256, kernel_size=3, activation='elu', padding='same'))
model.add(Conv1D(512, kernel_size=3, activation='elu', padding='same'))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy', 
    optimizer=Adam(lr=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# Fit the model
print('Training model')
model.fit(
    X_train, 
    Y_train, 
    batch_size=batch_size, 
    shuffle=True, 
    epochs=nb_epochs, 
    validation_data=(X_test, Y_test),
    callbacks=[EarlyStopping(min_delta=0.00025, patience=2)]
)
		  
model.save('sentimentNet.h5')
