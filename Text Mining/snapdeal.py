# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:31:50 2020

@author: HP
"""


import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# creating empty reviews list 
redmi_reviews=[]
#forest = ["the","king","of","jungle"]

for i in range(1,10):
  ip=[]  
  #url="https://www.amazon.in/Apple-MacBook-Air-13-3-inch-Integrated/product-reviews/B073Q5R6VR/ref=cm_cr_arp_d_paging_btm_2?showViewpoints=1&pageNumber="+str(i)
  url = "https://www.snapdeal.com/product/redmi-rose-gold-6a-2gb/622078003148/reviews?page=2&sortBy=HELPFUL"
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content
  print(soup.prettify())
  list(soup.children)
  reviews = soup.select('div p')
  for i in range(0,len(reviews)):
    ip.append(reviews[i].get_text())
    ip[:]=(reviews.lstrip('\n') for reviews in ip)
    ip[:]=(reviews.rstrip('\n') for reviews in ip) 
    redmi_reviews=redmi_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews


# writng reviews in a text file 
with open("redmi.txt","w",encoding='utf8') as output:
    output.write(str(redmi_reviews))
    
    
    
# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(redmi_reviews)



# Removing unwanted symbols incase if exists
ip_rev_string = re.sub("[^A-Za-z" "]+"," ",ip_rev_string).lower()
ip_rev_string = re.sub("[0-9" "]+"," ",ip_rev_string)



ip_reviews_words = ip_rev_string.split(" ")
nltk.download('stopwords')
stop_words = stopwords.words('english')

with open("E:\\Bokey\\Text Mining\\sw.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")


temp = ["this","is","awsome","Data","Science"]
[i for i in temp if i not in "is"]

ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]



# Joinining all the reviews into single paragraph 
ip_rev_string = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_rev_string)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\Assignments\\Text Mining\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]



# negative words  Choose path for -ve words stored in system
with open("C:\\Users\\HP\\Desktop\\ABubakar Files\\abu_Data_Science\Assignments\\Text Mining\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)

plt.imshow(wordcloud_pos_in_pos)


