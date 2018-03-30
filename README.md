# SentimentAnalysis

## About
This project aims to determine the sentiment of various forms of text input using a convolutional neural network. The trained neural network outputs a float value between 0 and 1, for every input sentence. 1 is highly positive and 0 is highly negative.

This project uses Google’s word2vec binary to convert each word in a tweet or a sentence to a 300-dimensional vector. The first 15 words from a sentence or tweet are considered and the vectors are stacked together to create a 15x300 matrix. A convolutional neural network is fed these matrices to obtain the corresponding sentiment value.  The script *KerasConvNet.py* was used to train a neural network on the sentiment140 dataset.

* **Tweet Analysis:** Relevant tweets are filtered, gathered and stored from Twitter. The neural network is run on these tweets to calculate their average sentiment. The tweet listener can keep streaming tweets in real-time. 

* **News article analysis:** While, Twitter data is really powerful for gauging public sentiment, it is also important to gauge sentiment of news articles and blogs as these play a crucial role in influencing the public’s opinion about a topic. To do this Google’s Custom Search API is used to do searches on relevant websites. The text from the top returned URLs are read using the same neural network and the average sentiment is calculated.


## Requirements
Before running the project the following python libraries have to be installed:
* numpy
* nltk
* tqdm
* Keras
* Tensorflow
* newspaper
* tweepy
* gensim
* pandas
* sklearn
* google-api-python-client

The following command can be run:
pip3 install numpy nltk tqdm Keras tensorflow newspaper tweepy gensim pandas sklearn google-api-python-client

