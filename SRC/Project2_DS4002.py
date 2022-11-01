#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
"""
#%%
import pandas as pd
import numpy as np
import string
import re
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


import textblob            #to import
from textblob import TextBlob


from textblob import TextBlob
#import vaderSentiment
#from vaderSentiment import SentimentIntensityAnalyzer
#from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#%%
# loading in the 3 csv files
articles1 = pd.read_csv("/Users/salemkeleta/Downloads/archive (3)/articles1.csv")
articles2 = pd.read_csv("/Users/salemkeleta/Downloads/archive (3)/articles2.csv")
articles3 = pd.read_csv("/Users/salemkeleta/Downloads/archive (3)/articles3.csv")
#%%
# combining all 3 articles
articles = pd.concat([articles1,articles2,articles3]) 
len(articles)
#%%
# just want articles from 2016
articles = articles[articles['year'] == 2016.0]
len(articles) # now we just have the 2016 articles"
#%%
# remove NAs
articles=articles.dropna(axis=0, how='any')
print(len(articles)) # down to 45,175 rows
# NAs have been removed (empty articles)
#%%
# adding a column for article bias
bias_labels = {'Atlantic': 'lean left', 'Breitbart': 'right', 'Business Insider': 'lean left', 'Buzzfeed News': 'left', 'CNN': 'left', 'Fox News': 'right',
                'Guardian': 'lean left', 'National Review': 'right', 'New York Post': 'lean right', 'New York Times': 'lean left',
                'NPR': 'left', 'Reuters': 'center', 'Talking Points Memo': 'left', 'Washington Post': 'lean left', 'Vox': 'left'}
#%%
articles['bias'] = articles['publication'].apply(lambda x: bias_labels[x]) # go through the articles publication column, for each x apply the bias label 
# now we have a new bias column
#%%
# we can make a bar graph to show how many articles from each political bias category we have
#plt.bar(articles['bias'], height=value_counts(),width=0.4)

chart1 = articles['bias'].value_counts().sort_index().plot(kind='bar', fontsize=14, figsize=(12,10))
chart1.set_xlabel('Political Bias', fontsize=12)
chart1.set_ylabel('Article Count', fontsize=12)
chart1.set_title('Article Bias Counts', fontsize=14)
#%%
stop = stopwords.words('english')
word_list = articles['title']
#%%
# this cleaning function is from https://www.dataquest.io/blog/tutorial-text-analysis-python-test-hypothesis/
def clean_text(article):
    clean1 = re.sub(r'['+string.punctuation + '’—”'+']', "", article.lower())
    return re.sub(r'\W+', ' ', clean1)
#%%
articles['clean title'] = articles['title'].map(lambda x: clean_text(x))
print(articles['clean title'])
#%%
# making a new column for the cleaned article content
articles['clean content'] = articles['content'].map(lambda x: clean_text(x))
#%%

# this was another cleaning method we used, however it resulted in the last
# 2 words of the title being mushed together (e.g. mushedtogether), so we 
# decided to use the cleaning function supplied by an article 


# Now we can clean up the text so its ready for analysis
# make all words lowercase, take out punctuation

# here, we are making a new column for the cleaned version of the titles 
#cleaned_title = []
#for i in articles['title']:
#    i = i.lower() # lowercasing
#    i = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", i)
#    i = [word for word in word_list if word not in stopwords.words('english')]
#    cleaned_title.append(i)
#articles['clean title'] = cleaned_title

# stopword removal is taking forever, dataframe is huge
# not sure if this is worth waiting around for 

#%%
# going to clean the text from the content of the articles 
#cleaned_content = []
#for i in articles['content']:
#    i = i.lower() # lowercasing
#    i = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", i)
#    cleaned_content.append(i)
#articles['clean content'] = cleaned_content
#%%
#search if title has all four candidates on it, remove if true 
clean_articles = articles[articles['clean title'].str.contains('trump',"sanders","clinton","")]

#%%
# we can start working on sentiment analysis now
# we can make a set of articles from each political bias category
# that talk about trump, and then do the same for hillary

# what is a "trump" article -- if the article title contains 'trump'
left_articles = articles[articles['bias']=='left']
#%%
#a_t = left_articles['clean title']

left_tr_articles = left_articles[left_articles['clean title'].str.contains('trump')]
#%%
# going to do the same thing for left leaning, center, right leaning, and right 
# publications
lean_left_articles = articles[articles['bias']=='lean left']
lean_left_tr_articles = lean_left_articles[lean_left_articles['clean title'].str.contains('trump')]

center_articles = articles[articles['bias']=='center']
center_tr_articles = center_articles[center_articles['clean title'].str.contains('trump')]

lean_right_articles = articles[articles['bias']=='lean right']
lean_rt_tr_articles = lean_right_articles[lean_right_articles['clean title'].str.contains('trump')]

right_articles = articles[articles['bias']=='right']
right_tr_articles = right_articles[right_articles['clean title'].str.contains('trump')]
#%%
# can make a column for the sentiment polarity of the title
# a column for the sentiment polarity of the article
#%%
import nltk
nltk.download('stopwords')


#fd = nltk.FreqDist(left_tr_articles['title']) 
#%%
# want to split up each title into a list of words
left_tr_articles['clean title'] = left_tr_articles['clean title'].map(lambda x: x.split())
#%%
lean_left_tr_articles['clean title'] = lean_left_tr_articles['clean title'].map(lambda x: x.split())
center_tr_articles['clean title'] = center_tr_articles['clean title'].map(lambda x: x.split())
lean_rt_tr_articles['clean title'] = lean_rt_tr_articles['clean title'].map(lambda x: x.split())
right_tr_articles['clean title'] = right_tr_articles['clean title'].map(lambda x: x.split())
#%%
# now going to split up all of the content for each political bias dataset
# for trump articles ... want to split up the content into a list of words
left_tr_articles['clean content'] = left_tr_articles['clean content'].map(lambda x: x.split())
lean_left_tr_articles['clean content'] = lean_left_tr_articles['clean content'].map(lambda x: x.split())
center_tr_articles['clean content'] = center_tr_articles['clean content'].map(lambda x: x.split())
lean_rt_tr_articles['clean content'] = lean_rt_tr_articles['clean content'].map(lambda x: x.split())
right_tr_articles['clean content'] = right_tr_articles['clean content'].map(lambda x: x.split())
#%%
# now we have all of the titles and all of the content split up 
# into a list of words
# for each political bias dataset filled with trump articles
#  we can now start to run sentiment analyis!
# for each political bias category (e.g., lean left) we can 
# analyze the sentiment of the article title and article content
# for each row. Then we can find the average and median sentiment
# for all of the titles, then for all of the content. 
# the question we are going to be asking is: is there a statistically
# significant difference in the setniment polarity of articles about
# trump between different political bias categories? 
# it seems someowhat common sense that publications on the left would
# speak about trump more negatively and ones on the right would 
# speak more positively, but what about the strength of the setniment? 
# for example, -0.1 vs. -0.99 is a big difference, magnitude-wise

# more specifically: the strength of the sentiment (whether it is
# negative or positive) will be equal for publications that are 
# in a similar position on the political spectrum, no matter what
# side they're on (for example, left and right would have similar
# sentiment polarity, and lean left and lean right would have 
# the same sentiment polarity). Also, the central publications
# would be the most neutral sentiment. 
#%%
#russia_articles = articles[articles['clean title'].str.contains('russia')]
#len(russia_articles)
#%%
# sentiment analysis on left trump articles 
# add a column to this dataframe that says the setniment score
# for each article
# average across all articles 

#%%
cruz_articles = articles[articles['clean title'].str.contains('cruz')]
print(len(cruz_articles))
#%%
bernie_articles = articles[articles['clean title'].str.contains('sanders')]
print(len(bernie_articles))
#%%
clinton_articles = articles[articles['clean title'].str.contains('clinton')]
print(len(clinton_articles))
#%%
trump_articles = articles[articles['clean title'].str.contains('trump')]
print(len(trump_articles))

#sample_left_tr_articles = left_tr_articles.sample(n=1000)
#sample_left_tr_articles

#%%
# Hypotheses!!
# H0: there is no difference in the average sentiment of articles
# from different poltically biased publications that talk about 
# 2016 presidential candidates (Trump, Hillary, Ted Cruz, Bernie)

# HA: 

#%%
#sentiment analysis

#left trump articles
left_tr_sent=[]
for i in left_tr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    left_tr_sent.append(a)
left_tr_articles['polarity score'] = left_tr_sent
## avg polarity score for left_tr
left_tr_sent_mean = left_tr_articles["polarity score"].mean()
len(left_tr_sent)

sample_left_tr_articles = left_tr_articles.sample(n=500)
sample_left_tr_articles
print(left_tr_sent_mean)
#histogram showing range of polarity for left_tr_art
import matplotlib.pyplot as plt
plt.hist(sample_left_tr_articles["polarity score"], bins= 30)
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,300)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Trump Articles from Left Publications")
plt.show()

#%%

#right_tr_articles
right_tr_sent=[]
for i in right_tr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    right_tr_sent.append(a)
right_tr_articles['polarity score'] = right_tr_sent
right_tr_sent_mean = right_tr_articles["polarity score"].mean()
len(right_tr_sent)
sample_right_tr_articles = right_tr_articles.sample(n=500)
print(right_tr_sent_mean)
plt.hist(sample_right_tr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,300)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Trump Articles from Right Publications")
plt.show()

#%%
lean_left_tr_sent=[]
for i in lean_left_tr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_left_tr_sent.append(a)
lean_left_tr_articles['polarity score'] = lean_left_tr_sent
lean_left_tr_sent_mean = lean_left_tr_articles["polarity score"].mean()
len(lean_left_tr_sent)
sample_lean_left_tr_articles = lean_left_tr_articles.sample(n=500)
print(lean_left_tr_sent_mean)
plt.hist(sample_lean_left_tr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,300)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Trump Articles from Lean Left Publications")
plt.show()

#%%
lean_right_tr_sent=[]
for i in lean_rt_tr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_right_tr_sent.append(a)
lean_rt_tr_articles['polarity score'] = lean_right_tr_sent
lean_right_tr_sent_mean = lean_rt_tr_articles["polarity score"].mean()
len(lean_right_tr_sent)
sample_lean_right_tr_articles = lean_rt_tr_articles.sample(n=500)

print(lean_right_tr_sent_mean)
plt.hist(sample_lean_right_tr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,300)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Trump Articles from Lean Right Publications")
plt.show()


#%%
center_tr_sent=[]
for i in center_tr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    center_tr_sent.append(a)
center_tr_articles['polarity score'] = center_tr_sent
center_tr_sent_mean = center_tr_articles["polarity score"].mean()
len(center_tr_sent)
sample_center_tr_articles = center_tr_articles.sample(n=500)

print(center_tr_sent_mean)

plt.hist(sample_center_tr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,300)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Trump Articles from Center Publications")
plt.show()


#%%
#checking assumption for ANOVA test of variance
import statistics
print(statistics.variance(left_tr_articles['polarity score']))
print(statistics.variance(right_tr_articles['polarity score']))
print(statistics.variance(lean_left_tr_articles['polarity score']))
print(statistics.variance(lean_rt_tr_articles['polarity score']))
print(statistics.variance(center_tr_articles['polarity score']))


#%%
#checking Residuals (experimental error) are approximately normally distributed (Shapiro-Wilks test)
from scipy.stats import shapiro
print(shapiro(left_tr_articles['polarity score']))
print(shapiro(right_tr_articles['polarity score']))
print(shapiro(lean_left_tr_articles['polarity score']))
print(shapiro(lean_rt_tr_articles['polarity score']))
print(shapiro(center_tr_articles['polarity score']))

#%%
print(len(left_tr_articles))
print(len(right_tr_articles))
print(len(lean_left_tr_articles))
print(len(lean_rt_tr_articles))
print(len(center_tr_articles))
#%%
from scipy.stats import f_oneway 


# Conduct the one-way ANOVA
#print(f_oneway(left_tr_sent, right_tr_sent, lean_left_tr_sent, lean_right_tr_sent,center_tr_sent))
#same sample size one way anova 
print(f_oneway(sample_left_tr_articles["polarity score"], sample_right_tr_articles["polarity score"], sample_lean_left_tr_articles["polarity score"], sample_lean_right_tr_articles["polarity score"],sample_center_tr_articles["polarity score"]))

#%%
# going to do the same thing for left leaning, center, right leaning, and right 
# publications
#clinton articles 
left_cl_articles = left_articles[left_articles['clean title'].str.contains('clinton')]
len(left_cl_articles)

right_cl_articles = right_articles[right_articles['clean title'].str.contains('clinton')]
len(right_cl_articles)

lean_left_cl_articles = lean_left_articles[lean_left_articles['clean title'].str.contains('clinton')]
len(lean_left_cl_articles)

lean_rt_cl_articles = lean_right_articles[lean_right_articles['clean title'].str.contains('clinton')]
len(lean_rt_cl_articles)

center_cl_articles = center_articles[center_articles['clean title'].str.contains('clinton')]
len(center_cl_articles)

#%%
#sentiment analysis

#left cl articles
left_cl_sent=[]
for i in left_cl_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    left_cl_sent.append(a)
left_cl_articles['polarity score'] = left_cl_sent
## avg polarity score for left_tr
left_cl_sent_mean = left_cl_articles["polarity score"].mean()
len(left_cl_sent)

sample_left_cl_articles = left_cl_articles.sample(n=200)
sample_left_cl_articles
print(left_cl_sent_mean)
#histogram showing range of polarity for left_tr_art
import matplotlib.pyplot as plt
plt.hist(sample_left_cl_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,90)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Clinton Articles from Left Publications")
plt.show()

#%%

#right_tr_articles
right_cl_sent=[]
for i in right_cl_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    right_cl_sent.append(a)
right_cl_articles['polarity score'] = right_cl_sent
right_cl_sent_mean = right_cl_articles["polarity score"].mean()
len(right_cl_sent)
sample_right_cl_articles = right_cl_articles.sample(n=200)
print(right_cl_sent_mean)
plt.hist(sample_right_cl_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,90)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Clinton Articles from Right Publications")
plt.show()

#%%
lean_left_cl_sent=[]
for i in lean_left_cl_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_left_cl_sent.append(a)
lean_left_cl_articles['polarity score'] = lean_left_cl_sent
lean_left_cl_sent_mean = lean_left_cl_articles["polarity score"].mean()
len(lean_left_cl_sent)
sample_lean_left_cl_articles = lean_left_cl_articles.sample(n=200)
print(lean_left_cl_sent_mean)
plt.hist(sample_lean_left_cl_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,90)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Clinton Articles from Lean Left Publications")
plt.show()

#%%
lean_right_cl_sent=[]
for i in lean_rt_cl_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_right_cl_sent.append(a)
lean_rt_cl_articles['polarity score'] = lean_right_cl_sent
lean_right_cl_sent_mean = lean_rt_cl_articles["polarity score"].mean()
len(lean_right_cl_sent)
sample_lean_right_cl_articles = lean_rt_cl_articles.sample(n=200)

print(lean_right_cl_sent_mean)
plt.hist(sample_lean_right_cl_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,90)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Clinton Articles from Lean Right Publications")
plt.show()


#%%
center_cl_sent=[]
for i in center_cl_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    center_cl_sent.append(a)
center_cl_articles['polarity score'] = center_cl_sent
center_cl_sent_mean = center_cl_articles["polarity score"].mean()
len(center_cl_sent)
sample_center_cl_articles = center_cl_articles.sample(n=200)

print(center_cl_sent_mean)

plt.hist(sample_center_cl_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,90)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Clinton Articles from Center Publications")
plt.show()


#%%
#checking assumption for ANOVA test of variance
import statistics
print(statistics.variance(left_cl_articles['polarity score']))
print(statistics.variance(right_cl_articles['polarity score']))
print(statistics.variance(lean_left_cl_articles['polarity score']))
print(statistics.variance(lean_rt_cl_articles['polarity score']))
print(statistics.variance(center_cl_articles['polarity score']))


#%%
#checking Residuals (experimental error) are approximately normally distributed (Shapiro-Wilks test)
from scipy.stats import shapiro
print(shapiro(left_cl_articles['polarity score']))
print(shapiro(right_cl_articles['polarity score']))
print(shapiro(lean_left_cl_articles['polarity score']))
print(shapiro(lean_rt_cl_articles['polarity score']))
print(shapiro(center_cl_articles['polarity score']))

#%%
print(len(left_cl_articles))
print(len(right_cl_articles))
print(len(lean_left_cl_articles))
print(len(lean_rt_cl_articles))
print(len(center_cl_articles))
#%%
from scipy.stats import f_oneway 


# Conduct the one-way ANOVA
#print(f_oneway(left_tr_sent, right_tr_sent, lean_left_tr_sent, lean_right_tr_sent,center_tr_sent))
#same sample size one way anova 
print(f_oneway(sample_left_cl_articles["polarity score"], sample_right_cl_articles["polarity score"], sample_lean_left_cl_articles["polarity score"], sample_lean_right_cl_articles["polarity score"],sample_center_cl_articles["polarity score"]))
#%%
cruz_articles = articles[articles['clean title'].str.contains('cruz')]
# going to do the same thing for left leaning, center, right leaning, and right 
# publications
#cruz articles 
left_cr_articles = left_articles[left_articles['clean title'].str.contains('cruz')]
len(left_cr_articles)

right_cr_articles = right_articles[right_articles['clean title'].str.contains('cruz')]
len(right_cr_articles)

lean_left_cr_articles = lean_left_articles[lean_left_articles['clean title'].str.contains('cruz')]
len(lean_left_cr_articles)

lean_rt_cr_articles = lean_right_articles[lean_right_articles['clean title'].str.contains('cruz')]
len(lean_rt_cr_articles)

center_cr_articles = center_articles[center_articles['clean title'].str.contains('cruz')]
len(center_cr_articles)
#%%
#sentiment analysis for cruz articles

#left cr articles
left_cr_sent=[]
for i in left_cr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    left_cr_sent.append(a)
left_cr_articles['polarity score'] = left_cr_sent
## avg polarity score for left_tr
left_cr_sent_mean = left_cr_articles["polarity score"].mean()
len(left_cr_sent)

sample_left_cr_articles = left_cr_articles.sample(n=30)
sample_left_cr_articles
print(left_cr_sent_mean)
#histogram showing range of polarity for left_tr_art
import matplotlib.pyplot as plt
plt.hist(sample_left_cr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Cruz Articles from Left Publications")
plt.show()

#%%

#right_tr_articles
right_cr_sent=[]
for i in right_cr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    right_cr_sent.append(a)
right_cr_articles['polarity score'] = right_cr_sent
right_cr_sent_mean = right_cr_articles["polarity score"].mean()
len(right_cr_sent)
sample_right_cr_articles = right_cr_articles.sample(n=30)
print(right_cr_sent_mean)
plt.hist(sample_right_cr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Cruz Articles from Right Publications")
plt.show()

#%%
lean_left_cr_sent=[]
for i in lean_left_cr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_left_cr_sent.append(a)
lean_left_cr_articles['polarity score'] = lean_left_cr_sent
lean_left_cr_sent_mean = lean_left_cr_articles["polarity score"].mean()
len(lean_left_cr_sent)
sample_lean_left_cr_articles = lean_left_cr_articles.sample(n=30)
print(lean_left_cr_sent_mean)
plt.hist(sample_lean_left_cr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Cruz Articles from Lean Left Publications")
plt.show()

#%%
lean_right_cr_sent=[]
for i in lean_rt_cr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_right_cr_sent.append(a)
lean_rt_cr_articles['polarity score'] = lean_right_cr_sent
lean_right_cr_sent_mean = lean_rt_cr_articles["polarity score"].mean()
len(lean_right_cr_sent)
sample_lean_right_cr_articles = lean_rt_cr_articles.sample(n=30)

print(lean_right_cr_sent_mean)
plt.hist(sample_lean_right_cr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Cruz Articles from Lean Right Publications")
plt.show()


#%%
center_cr_sent=[]
for i in center_cr_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    center_cr_sent.append(a)
center_cr_articles['polarity score'] = center_cr_sent
center_cr_sent_mean = center_cr_articles["polarity score"].mean()
len(center_cr_sent)
sample_center_cr_articles = center_cr_articles.sample(n=30)

print(center_cr_sent_mean)

plt.hist(sample_center_cr_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Cruz Articles from Center Publications")
plt.show()


#%%
#checking assumption for ANOVA test of variance
import statistics
print(statistics.variance(left_cr_articles['polarity score']))
print(statistics.variance(right_cr_articles['polarity score']))
print(statistics.variance(lean_left_cr_articles['polarity score']))
print(statistics.variance(lean_rt_cr_articles['polarity score']))
print(statistics.variance(center_cr_articles['polarity score']))


#%%
#checking Residuals (experimental error) are approximately normally distributed (Shapiro-Wilks test)
from scipy.stats import shapiro
print(shapiro(left_cr_articles['polarity score']))
print(shapiro(right_cr_articles['polarity score']))
print(shapiro(lean_left_cr_articles['polarity score']))
print(shapiro(lean_rt_cr_articles['polarity score']))
print(shapiro(center_cr_articles['polarity score']))

#%%
print(len(left_cr_articles))
print(len(right_cr_articles))
print(len(lean_left_cr_articles))
print(len(lean_rt_cr_articles))
print(len(center_cr_articles))
#%%
from scipy.stats import f_oneway 


# Conduct the one-way ANOVA
#print(f_oneway(left_tr_sent, right_tr_sent, lean_left_tr_sent, lean_right_tr_sent,center_tr_sent))
#same sample size one way anova 
print(f_oneway(sample_left_cr_articles["polarity score"], sample_right_cr_articles["polarity score"], sample_lean_left_cr_articles["polarity score"], sample_lean_right_cr_articles["polarity score"],sample_center_cr_articles["polarity score"]))

#%% 
bernie_articles = articles[articles['clean title'].str.contains('sanders')]
print(len(bernie_articles))

# going to do the same thing for left leaning, center, right leaning, and right 
# publications
#bernie articles 
left_sn_articles = left_articles[left_articles['clean title'].str.contains('sanders')]
len(left_sn_articles)

right_sn_articles = right_articles[right_articles['clean title'].str.contains('sanders')]
len(right_sn_articles)

lean_left_sn_articles = lean_left_articles[lean_left_articles['clean title'].str.contains('sanders')]
len(lean_left_sn_articles)

lean_rt_sn_articles = lean_right_articles[lean_right_articles['clean title'].str.contains('sanders')]
len(lean_rt_sn_articles)

center_sn_articles = center_articles[center_articles['clean title'].str.contains('sanders')]
len(center_sn_articles)
#%%
#sentiment analysis for cruz articles

#left bernie sanders articles
left_sn_sent=[]
for i in left_sn_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    left_sn_sent.append(a)
left_sn_articles['polarity score'] = left_sn_sent
## avg polarity score for left_tr
left_sn_sent_mean = left_sn_articles["polarity score"].mean()
len(left_sn_sent)

sample_left_sn_articles = left_sn_articles.sample(n=30)
sample_left_sn_articles
print(left_sn_sent_mean)
#histogram showing range of polarity for left_tr_art
import matplotlib.pyplot as plt
plt.hist(sample_left_sn_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Sanders Articles from Left Publications")
plt.show()

#%%

#right_sn_articles
right_sn_sent=[]
for i in right_sn_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    right_sn_sent.append(a)
right_sn_articles['polarity score'] = right_sn_sent
right_sn_sent_mean = right_sn_articles["polarity score"].mean()
len(right_sn_sent)
sample_right_sn_articles = right_sn_articles.sample(n=30)
print(right_cr_sent_mean)
plt.hist(sample_right_sn_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Sanders Articles from Right Publications")
plt.show()

#%%
lean_left_sn_sent=[]
for i in lean_left_sn_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_left_sn_sent.append(a)
lean_left_sn_articles['polarity score'] = lean_left_sn_sent
lean_left_sn_sent_mean = lean_left_sn_articles["polarity score"].mean()
len(lean_left_sn_sent)
sample_lean_left_sn_articles = lean_left_sn_articles.sample(n=30)
print(lean_left_sn_sent_mean)
plt.hist(sample_lean_left_sn_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Sanders Articles from Lean Left Publications")
plt.show()

#%%
lean_right_sn_sent=[]
for i in lean_rt_sn_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    lean_right_sn_sent.append(a)
lean_rt_sn_articles['polarity score'] = lean_right_sn_sent
lean_right_sn_sent_mean = lean_rt_sn_articles["polarity score"].mean()
len(lean_right_sn_sent)
sample_lean_right_sn_articles = lean_rt_sn_articles.sample(n=30)

print(lean_right_sn_sent_mean)
plt.hist(sample_lean_right_sn_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Sanders Articles from Lean Right Publications")
plt.show()


#%%
center_sn_sent=[]
for i in center_sn_articles['clean content']:
    txt= TextBlob(i)
    a= txt.sentiment.polarity
    center_sn_sent.append(a)
center_sn_articles['polarity score'] = center_sn_sent
center_sn_sent_mean = center_sn_articles["polarity score"].mean()
len(center_sn_sent)
sample_center_sn_articles = center_sn_articles.sample(n=30)

print(center_sn_sent_mean)

plt.hist(sample_center_sn_articles["polarity score"])
plt.grid(True)
plt.xlim(-.4,.4)
plt.ylim(0,15)
plt.xlabel("Polarity Score")
plt.ylabel("Frequency")
plt.title("Range of Sentiment for Sanders Articles from Center Publications")
plt.show()


#%%
#checking assumption for ANOVA test of variance
import statistics
print(statistics.variance(left_sn_articles['polarity score']))
print(statistics.variance(right_sn_articles['polarity score']))
print(statistics.variance(lean_left_sn_articles['polarity score']))
print(statistics.variance(lean_rt_sn_articles['polarity score']))
print(statistics.variance(center_sn_articles['polarity score']))


#%%
#checking Residuals (experimental error) are approximately normally distributed (Shapiro-Wilks test)
from scipy.stats import shapiro
print(shapiro(left_sn_articles['polarity score']))
print(shapiro(right_sn_articles['polarity score']))
print(shapiro(lean_left_sn_articles['polarity score']))
print(shapiro(lean_rt_sn_articles['polarity score']))
print(shapiro(center_sn_articles['polarity score']))

#%%
print(len(left_sn_articles))
print(len(right_sn_articles))
print(len(lean_left_sn_articles))
print(len(lean_rt_sn_articles))
print(len(center_sn_articles))
#%%
from scipy.stats import f_oneway 


# Conduct the one-way ANOVA
#print(f_oneway(left_tr_sent, right_tr_sent, lean_left_tr_sent, lean_right_tr_sent,center_tr_sent))
#same sample size one way anova 
print(f_oneway(sample_left_sn_articles["polarity score"], sample_right_sn_articles["polarity score"], sample_lean_left_sn_articles["polarity score"], sample_lean_right_sn_articles["polarity score"],sample_center_sn_articles["polarity score"]))

#%%

    
    
    