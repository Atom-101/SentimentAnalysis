A google API key is required for this function. It can be obtained from the following site: 
https://developers.google.com/custom-search/json-api/v1/overview

Now the script NewsGoogleSearch.py has to be run from the command line using the following command: python NewsGoogleSearch.py –q <queries> –k <Google Api Key>
Here <queries> refers to the string to be used for google search and <Google Api Key> is the code received in the first step.

Next, the script NewstoDoc.py has to be run. This will collect the text from the corresponding URLs and write them to articles.txt.  

Finally, the script DocSentimentAnalysis.py should be run to get the average sentiment of the news articles.
