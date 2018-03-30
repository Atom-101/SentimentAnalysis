To use this, an application has to be registered with twitter at the following website: https://apps.twitter.com/
After registering the application, a:
 i) consumer_key
 ii) consumer_secret
 iii) access_token
 iv) access_secret ,
is received.
These values have to be entered in the file TweepyConfig.py at the corresponding locations.

Following this the tweets can be streamed from the command line. 
To do this, a terminal has to be opened and navigated to the project folder. 
Now, the following command has to be used: python TweepyMain.py –q <queries>
Here, <queries> refers to strings that will be used to search for tweets. Multiple query terms can be provided by separating them by a ‘,’ (comma).

After an appropriate amount of time, the data stream can be interrupted pressing ‘Ctrl+C’. A json file containing the collected tweets is created in the data folder .

To analyze the sentiments of the collected tweets, the script TweetSentimentAnalysis.py has to be run using the command: python TweepyMain.py –d <json_directory> . Here, <json_directory> refers to path of json file on which sentiment analysis is to be run.

The script outputs the average sentiment of the collected tweets.
