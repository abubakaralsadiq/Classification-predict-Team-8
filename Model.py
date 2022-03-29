#!/usr/bin/env python
# coding: utf-8

# # Advanced Classification Predict
# 
# ©  Explore Data Science Academy
# 
# ---
# 
# ### Honour Code
# 
# I **EDSA-Team_8**, confirm - by submitting this document - that the solutions in this notebook are a result of my own work and that I abide by the [EDSA honour code](https://drive.google.com/file/d/1QDCjGZJ8-FmJE3bZdIQNwnJyQKPhHZBn/view?usp=sharing).
# 
# Non-compliance with the honour code constitutes a material breach of contract.
# 
# ### Predict Overview: Climate Change Belief Analysis 2022
# Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received, Your company has been awarded the contract to:
# 
# - 1. analyse the supplied data;
# - 2. identify potential errors in the data and clean the existing data set;
# - 3. determine if additional features can be added to enrich the data set;
# - 4. build a model that is capable of forecasting the three hourly demand shortfalls;
# - 5. evaluate the accuracy of the best machine learning model;
# - 6. determine what features were most important in the model’s prediction decision, and
# - 7. explain the inner working of the model to a non-technical audience.
# 

# <a id="cont"></a>
# 
# ## Table of Contents
# 
# <a href=#one>1. Importing Packages</a>
# 
# <a href=#two>2. Loading Data</a>
# 
# <a href=#three>3. Exploratory Data Analysis (EDA)</a>
# 
# <a href=#four>4. Data Engineering</a>
# 
# <a href=#five>5. Modeling</a>
# 
# <a href=#six>6. Model Performance</a>
# 
# <a href=#seven>7. Model Explanations</a>

#  <a id="one"></a>
# ## 1. Importing Packages
# <a href=#cont>Back to Table of Contents</a>
# 
# ---
#     
# | ⚡ Description: Importing Packages ⚡ |
# | :--------------------------- |
# | In this section the required packages are imported, and briefly discuss, the libraries that will be used throughout the analysis and modelling. |

# In[50]:


#importing the required libraries
# Libraries for data loading, data manipulation and data visulisation
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Customise our plotting settings
sns.set_style('whitegrid')
#Libraries for data preparation and model building

# Setting global constants to ensure notebook results are reproducible


#  <a id="two"></a>
# ## 2. Loading Data
# <a href=#cont>Back to Table of Contents</a>

# We frist start by loading in our dataset, both the `training` and `testing` dataset is loaded as a pandas dataframe

# In[141]:


#load the training and test data set
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#  <a id="three"></a>
# ## 3. Exploratory Data Analysis (EDA)
# <a href=#cont>Back to Table of Contents</a>

# **Exploratory Data Analysis (EDA) :**  After loading in our dataset we first start with the vital component **EDA** to better understand the dataset we are working with and, to gain insight about the `features` and `labels` by performing `Univariate` or `Multivariate` , `Non-graphical` or `Graphical` Analysis"

# We take a look quick look at the first few rows of the `training` and `testing` dataset to have an overview of our features and labels, (using `pd.head()` method)

# In[3]:


#The first five columns of the traing dataset
train.head()


# After taking a look at the frist five  rows of the dataFrame we can see that we have `Three (3)` columns in the dataFrame.
# 
# we have two features and one label
# 
# features inludes:
# 
#     - message
#     - tweetid
# 
# label:
# 
#     - sentiment
#     
# And the test dataFrame contains only the features

# In[4]:


#The first five columns of the test dataset
test.head()


# we will take a look at the shape of the dataframe to understand the amount of data we are working with, the **rows** and the **columns**

# In[5]:


#checking the shape of the traing dataframe
train.shape


# looking at the shape of the dataframe we have `15819` rows and `3` columns

# Next up let's take a look at the data types of the dataFrame using `pd.info()`

# In[6]:


#checking the information of the dataframe
train.info()


# looking at the above output we can see that we have two `int64` and one `object` 

# In[7]:


#checking null values in the training data
train.isnull().sum()


# well it shows that we have **0** null values in the training data

# let's take a closer look on our label `sentiment` 

# In[8]:


#checking for unique values 
train['sentiment'].value_counts()


# Well it looks like we have 4 unique values in our label.
# 
# Based on the description of the data here is what each value stands for:
# 
#     1 Pro: the tweet supports the belief of man-made climate change
#     2 News: the tweet links to factual news about climate change
#     0 Neutral: the tweet neither supports nor refutes the belief of man-made climate change
#     -1 Anti: the tweet does not believe in man-made climate change

# let's count and plot the destribution of each unique value

# In[9]:


#ploting the destribution of unique label values
f, ax = plt.subplots(figsize=(8, 4))
ax = sns.countplot(x="sentiment", data=train)
plt.show()


# **Interpretation**
# 
# - The above plot comfirms that:
# 
# `1296` tweets do not believe in man-made climate change `-1`
# 
# `2353` tweets neither supports nor refutes the belief of man-made climate change `0`
# 
# `8530` Pro: the tweet supports the belief of man-made climate change `1`
# 
# `3640` News: the tweet links to factual news about climate change `2`
# 
# The plot show that the highest propotion of the tweets supports the belife of man-made climate change

# Now moving on let's explore our `features` to gain more insight  

# In[10]:


#checking the tweetid to see if there are any duplicate id's
train['tweetid'].nunique()


# Okay it looks like we don't have a value from the `tweetid`column

# Moving on let's take a closer look into the `message` column which contains the tweets

# In[11]:


#taking a colser look on the message column
train['message'].head()


# #### Using Wordcloud library, we woild crreate a visualization to see the frequent occuring words in all the tweets 

# In[21]:


from wordcloud import WordCloud


# In[29]:


# Visualize the frequent words 
all_words = " ".join([sentence for sentence in train['message']])

wordcloud= WordCloud(width=800, height=500, random_state=50, max_font_size=100).generate(all_words)
# Plot the Graph 
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# From the above visuals, the words that are most frequent are larger in size, and they are climate change, global warming and the url link 

# #### Now we would create these same word frequency but with respect to their individual sentiments 
# 
#         And we would be starting with sentiment 1, which is pro climate change 

# In[30]:


# Visualize the frequent words 
all_words = " ".join([sentence for sentence in train['message'][train['sentiment']==1]])

wordcloud= WordCloud(width=800, height=500, random_state=50, max_font_size=100).generate(all_words)
# Plot the Graph 
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# For the sentiment 1, which is in support of climate change. "Climate change" is the most frequent word

# #### For sentiment -1 Anti: the tweet does not believe in man-made climate change

# In[31]:


# Visualize the frequent words 
all_words = " ".join([sentence for sentence in train['message'][train['sentiment']==-1]])

wordcloud= WordCloud(width=800, height=500, random_state=50, max_font_size=100).generate(all_words)
# Plot the Graph 
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# For the sentiment -1, which is tweets that are anti climate change. "Climate change" and "Global warming" are  the two  most frequent words

# ####  tweets neither supports nor refutes the belief of man-made climate change  

# In[32]:


# Visualize the frequent words 
all_words = " ".join([sentence for sentence in train['message'][train['sentiment']==0]])

wordcloud= WordCloud(width=800, height=500, random_state=50, max_font_size=100).generate(all_words)
# Plot the Graph 
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# For the sentiment -1, which is tweets that are anti climate change. "Climate change" and "Global warming" are also the two  most frequent words

#  #### For sentiment 2 which is News: the tweets with  links to factual news about climate change

# In[33]:


# Visualize the frequent words 
all_words = " ".join([sentence for sentence in train['message'][train['sentiment']==2]])

wordcloud= WordCloud(width=800, height=500, random_state=50, max_font_size=100).generate(all_words)
# Plot the Graph 
plt.figure(figsize=(15,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off');


# The url link "http" is the most frequent word, that and climate change

# #### Next, we would extracts all the hashtags with respect to the various sentiments. And we would be doing this for the pro and anti climate change 

# In[118]:


def hashtag_extract(tweets):
    hashtags=[]
    tweets=tweets.to_list()
    for tweet in tweets:
        hashtag = re.findall(r"#(\w+)",tweet)
        hashtags.append(hashtag)
    return hashtags


# In[119]:


# Extract Hashtags for pro climate change 
pro_climate = hashtag_extract(train['message'][train['sentiment']==1])

# Extract Hashtags for anti climate change 
anti_climate = hashtag_extract(train['message'][train['sentiment']==-1])


# In[120]:


# un-nest the lists
pro_climate = sum(pro_climate, [])
anti_climate = sum(anti_climate, [])


# In[121]:


# checking frequency of hashtags in both pro climate change and anti climate change, using nltk.freq library
import nltk


# In[122]:


# For pro climate change
freq = nltk.FreqDist(pro_climate)
freq_df = pd.DataFrame({'hashtags':freq.keys(),
                       'counts':freq.values()})
# Display the top 10 frequent hashtags
freq_df = freq_df.nlargest(columns='counts', n=10)
plt.figure(figsize=(15,10))
sns.barplot(data=freq_df, x='hashtags',y='counts');


# In[123]:


# For anti climate change
freq = nltk.FreqDist(pro_climate)
freq_df = pd.DataFrame({'hashtags':freq.keys(),
                       'counts':freq.values()})
# Display the top 10 frequent hashtags
freq_df = freq_df.nlargest(columns='counts', n=10)
plt.figure(figsize=(15,10))
sns.barplot(data=freq_df, x='hashtags',y='counts');


# From the graph we can see that the first 10 hashtags are the same for both pro and anti climate change tweets, only the hashtags are much more in number for pro climate change tweets 

# Well based on what we are seeing the `message` column which contains the tweets has some characters, we need to clean the data

# **Next step is Data cleaning**
# 
# Before applying any ML model to a set of data we need fisrt check our data to see if the data is in the state data we want it to be or do we need clean the data, well our case we need to clean the data moving on we will start the Data cleaning process

# In[142]:


# Creating a fuction to clean the tweets 
def cleaner(tweet):
    tweet = re.sub(r"@[A-Za-z0-9]+",'',tweet) # This would remove @mentions
    tweet = re.sub(r"#",'',tweet) # This would remove the hash symbol '#'
    tweet = re.sub(r"RT[\s]+", '',tweet) # This would remove retweets RT
    tweet = re.sub(r"rt[\s]+", '',tweet) # This would remove retweets rt
    return tweet 

# Cleaning the tweets 
train['message']=train['message'].apply(cleaner)

# Show cleaned text
train


# In[143]:


# Removing all special characters, numbers and punctuation
train['message'] =  train['message'].str.replace("[^a-zA-Z#]",' ')
train.head()


# In[144]:


# CHanging all the words in the tweets to lowercase letters  
#train['message'] = [[word.lower() for word in message.split()] for message in train['message']
train['message'] = train['message'].str.lower()
train


# #### Tokenization
# ##### Basically seperating each word in a sentence to stand on their own 

# In[ ]:


from nltk.tokenize import TreebankWordTokenizer


# In[145]:


tokeniser = TreebankWordTokenizer()
train['tokens'] = train['message'].apply(tokeniser.tokenize)
train


# #### Stemming
# ##### The process of reducing a word to its simplest form 

# In[146]:


from nltk import SnowballStemmer


# In[147]:


stemmer = SnowballStemmer('english')


# In[148]:


# Function that stems every single separarted word in the tokens 
def word_stem(words,stemmer):
   return [stemmer.stem(word) for word in words]


# In[149]:


# applying the function 
train['stem'] = train['tokens'].apply(word_stem, args=(stemmer, ))


# In[150]:


# Removing stopwords
from nltk.corpus import stopwords


# In[151]:


# function to remove stopwords 
def remove_stop_words(tokens):    
    return [t for t in tokens if t not in stopwords.words('english')]


# In[152]:


train['stem'] = train['stem'].apply(remove_stop_words)


# In[161]:


# Determine our Label
for i in range(len(train['stem'])):
    train['stem'][i] = " ".join(train['stem'][i])
train['cleaned_tweets'] = train['stem']


# In[162]:


train


#  <a id="four"></a>
# ## 4. Feature Engineering
# <a href=#cont>Back to Table of Contents</a>

# In[163]:


# Extraxt features to help predict the label 
# Bag-of-words vectors built into sklearn
from sklearn.feature_extraction.text import CountVectorizer


# In[165]:


# converting text to what the computer can understand 
bow_vectorizer = CountVectorizer(max_df=0.90, max_features=1000,stop_words='english') 
bow = bow_vectorizer.fit_transform(train['cleaned_tweets'])


# In[166]:


# Splitting into training and testing set
from sklearn.model_selection import train_test_split


# In[169]:


X_train, X_test, y_train, y_test = train_test_split(
    bow, train['sentiment'], test_size=0.25, random_state=50)


#  <a id="five"></a>
# ## 5. Modeling
# <a href=#cont>Back to Table of Contents</a>

# In[171]:


# Using The logistics Regression Model 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


# In[172]:


# Model Training
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)


#  <a id="six"></a>
# ## 6. Model Performance
# <a href=#cont>Back to Table of Contents</a>

# In[173]:


# Evaluate trained model using the test set
predictions = lr_model.predict(X_test)


# In[184]:


accuracy_score(y_test,predictions)


# In[185]:


f1_score(y_test,predictions, average='macro')


#  <a id="seven"></a>
# ## 7. Model Explanations
# <a href=#cont>Back to Table of Contents</a>

# In[ ]:




