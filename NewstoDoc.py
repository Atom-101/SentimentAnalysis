from newspaper import Article


outfile = 'articles.txt' #enter location of file where the extracted news text will be dumped
url_file = 'NewsURL.txt' #enter location of file where URLs of news sites are present. Each line should be a new URL

with open(url_file, 'r') as f:
	url_list = f.readlines()

articles = []
for i,url in enumerate(url_list):
	try:
		articles.append(Article(url))
	except:
	       continue
	articles[i].download()
	articles[i].parse()

with open(outfile, 'a', encoding = 'utf-8') as f:
        for article in articles:
                f.write(article.text)
	
#call doc neural net(ArticleReader.py)
