import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# load the dataset
model = load_model('imdb.h5')
check = input()

#max_words = 500
#X = sequence.pad_sequences(check, maxlen=max_words)


print(model.predict(check))





