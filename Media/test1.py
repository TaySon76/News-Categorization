from bs4 import BeautifulSoup
import requests

site = 'https://www.cnn.com/business/live-news/fox-news-dominion-trial-04-18-23/h_8d51e3ae2714edaa0dace837305d03b8'
result = requests.get(site)
content = result.text

soup = BeautifulSoup(content, 'lxml')
print(soup.prettify())


