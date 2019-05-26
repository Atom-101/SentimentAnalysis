import tweepy
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import argparse
import string
import TweepyConfig as config
import json


def get_parser():
    parser = argparse.ArgumentParser(description="Twitter Downloader")
    parser.add_argument("-q", "--query", dest="query", help="Query/Filter", default='-')
    return parser


class MyListener(StreamListener):
    """Custom StreamListener for streaming data."""

    def __init__(self, query):
        query_fname = format_filename(query)
        self.outfile = "data/stream_%s.json" % (query_fname)
    
    def on_data(self, data):
        try:            
            with open(self.outfile, 'a') as f:
                f.write(data)
            print(data)
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
            time.sleep(5)
        return True

    def on_error(self, status):
        print(status)
        if (status == 420):
            #returning False in on_data disconnects the stream
            return False
        return True
    
    def on_status(self, status):
        if (status.retweeted_status):
            return


def format_filename(fname):
    """Convert file name into a safe string.
    Arguments:
        fname: the file name to convert
    Return:
        The converted file name
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    #Set invalid characters to '_' 
    #Removes emojis and other characters
    
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return '_'


def openStream():	
    print('opened')
    #print(config.consumer_key+'\n'+config.consumer_secret+'\n'+config.access_token+'\n'+config.access_secret+'\n')
    parser = get_parser()
    args = parser.parse_args()
    auth = OAuthHandler(config.consumer_key, config.consumer_secret)
    auth.set_access_token(config.access_token, config.access_secret)
    api = tweepy.API(auth)
    query = [x.strip() for x in args.query.split(',')]#split comma separated queries into list of queries
	
    twitter_stream = Stream(auth, MyListener(args.query))
    twitter_stream.filter(track=query)
    ''' time.sleep(60)#keep stream open for 60 seconds
    closeStream(twitter_stream)'''
	
'''
def closeStream(twitter_stream):
    twitter_stream.disconnect()
    print('closed')
    time.sleep(60*5)#give 5 minutes to allow stream to close completely
    #(call neural net for prediction)
    time.sleep(60*10)#open stream again after 10 minutes
    openStream()
'''

if __name__ == '__main__':
  openStream()
