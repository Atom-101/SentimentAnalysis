import argparse
import json
from urllib.request import urlopen

def get_parser():
    """Get parser for command line arguments."""
    parser = argparse.ArgumentParser(description="URL Downloader")
    parser.add_argument("-q", "--query", dest="query", help="Query/Filter", default='-')
    parser.add_argument("-k", "--key", dest="key", help="Google API Key")
    return parser
	
if __name__ == '__main__':
	
    parser = get_parser()
    args = parser.parse_args()

    data = urlopen('https://www.googleapis.com/customsearch/v1?key={}&cx=000919482489691833361:drmpuulq2tk&q={}'.format(args.key,args.query))
    data = json.load(data)
    #print(data['items'][0]['link'])
    with open('NewsURL.txt', 'a') as f:
        for j in range(len(data['items'])):
            f.write(data['items'][j]['link']+'\n')
            print(data['items'][j]['link'])	

