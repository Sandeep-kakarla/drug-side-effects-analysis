#!/usr/bin/env python
# coding: utf-8

# In[3]:


## Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import string
from nltk import corpus
import nltk.corpus as Corpus
import math
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings 
warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)
import plotly.graph_objs as go
import cufflinks
pd.options.display.max_columns = 30
from IPython.core.interactiveshell import InteractiveShell
import plotly.figure_factory as ff
InteractiveShell.ast_node_interactivity = 'all'
from plotly.offline import iplot
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
output_notebook()
from collections import Counter
from pprint import pprint
import gensim
from gensim import corpora
from pprint import pprint
import csv
import random
from gensim.models.wrappers import LdaMallet
from gensim import corpora
from gensim import models
from gensim.models import LdaModel
from gensim.models import TfidfModel
import pyLDAvis.gensim
from gensim.test.utils import datapath
import bs4 as bs  
import urllib.request  
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary
import plotly.express as px
import plotly.graph_objects as go
import itertools
from sklearn.naive_bayes import MultinomialNB
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from sklearn import metrics
from IPython.display import Image
from graphviz import Digraph
import graphviz
import pydotplus
import pickle
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[4]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')
get_ipython().system('pip install textblob')
get_ipython().system('pip install -U gensim')
get_ipython().system('pip install pyLDAvis')
get_ipython().system('pip install graphviz')
get_ipython().system('pip install pydotplus')
get_ipython().system('pip install --upgrade jupyter_client')
get_ipython().system('pip install six')
get_ipython().system('pip install conda graphviz')
get_ipython().system('pip install conda')
get_ipython().system('pip install wheel')
get_ipython().system('pip install flaskr-1.0.0-py3-none-any.whl')
get_ipython().system('pip install waitress')


# In[5]:


#data import
df = pd.read_csv('file:///C:/Users/sandeep/Downloads/tupakki.csv')


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


#replace null values as condotion.
df.dropna(subset=['condition'], inplace = True)


# In[7]:


df.isnull().sum()


# In[9]:


#converting review data in to list nd then to text.
''.join(df['review'].tolist())


# In[9]:


#text cleaning
def cont_to_exp(x):
    if type(x) is srt:
        x = x.replace('\\', '')
        for key in contractions:
            value = contractions[key]
            x = x.replace(key,value)
            return x
        else:
            return x


# In[10]:


def clean_text_round1():
    '''make text lowercase, remove textin square brackets, remove punctuations and remove words'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctution), '', text)
    text = re.sub['\w*\d\w*', '', text]
    return text
round1 = lambda x: clean_text_round1(x)


# In[11]:


def clean_text_round2(text):
    '''Get rid of some additional punctuation and non-sensical text was missed the first time'''
    text = re.sub('(''""...)', '', text)
    text = re.sub('\n', '', text)
    return text

round2 = lambda x: clean_text_round2(x)


# In[12]:


#feature engineering
df.head()
from textblob import TextBlob


# In[13]:


#sentiment polarity for particular text.
df['polarity'] = df['review'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[14]:


#review length.
df['review_len'] = df['review'].apply(lambda x: len(x))


# In[15]:


#cal no.of word's in reviews
df['word_count'] = df['review'].apply(lambda x: len(x.split()))


# In[16]:


# to get average word,length,in review data.
def get_avg_word_len(x):
    words = x.split()
    word_len = 0
    for word in words:
        word_len = word_len + len(word)
        
        return word_len/len(words)


# In[17]:


df['avg_word_len'] = df['review'].apply(lambda x: get_avg_word_len(x))


# In[18]:


#we can see polarity	review_len	word_count	avg_word_len if polarity is o.1 it is +ve or 0.01 it is nutral if it is 0.-1 it is -ve
df.head()


# In[10]:


df.iplot()


# In[34]:


#we can ckeck the positive nd -ve neutral reviews.
df['polarity'].iplot()


# In[37]:


df['polarity'].iplot(kind = 'hist', colors = 'red', bins = 50)


# In[45]:


df['polarity'].iplot(kind = 'hist', colors = 'red', bins = 50,
                     xTitle = 'polarity', yTitle = 'count',
                  title = 'polarity review distrabution')


# In[46]:


df['rating'].iplot(kind = 'hist', xTitle = 'rating', yTitle = 'count',
                  title = 'review rating distrabution')


# In[47]:


#review text length and word length distributoin.(most of reviews are at 750 )
df['review_len'].iplot(kind = 'hist', xTitle = 'review len', yTitle = 'review text length')


# In[50]:


#review text length and word length distributoin.(most of words are in  140 )
df['word_count'].iplot(kind = 'hist', xTitle = 'word count', yTitle = 'word count distribution')


# In[51]:


#review text length and word length distributoin.(most of age word length reviews are at 0.3 )
df['avg_word_len'].iplot(kind = 'hist', xTitle = 'average word length', yTitle = 'review text avg word length')


# In[19]:


df['condition'].value_counts()


# In[20]:


df.groupby('condition').count()


# In[54]:


df['condition'].value_counts().iplot()


# In[56]:


#3k birthcontrol people are in our data.
df['condition'].value_counts().iplot(kind = 'bar', yTitle = 'count', xTitle = 'condition',
                                    title = 'Bar chart of condition')


# In[60]:


#3k birthcontrol people are in our data.
df['condition'].value_counts().iplot(kind = 'bar')


# In[ ]:


#unigram


# In[21]:


x = ['this is the lit list this']


# In[22]:


vec = CountVectorizer().fit(x)
bow = vec.transform(x)
sum_words = bow.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
words_freq[:2]


# In[25]:


def get_top_n_words(x, n):
    vec = CountVectorizer().fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[26]:


get_top_n_words(x, 3)


# In[28]:


words = get_top_n_words(df['review'], 20)


# In[29]:


# top 20 words in review
words


# In[31]:


df1 = pd.DataFrame(words, columns = ['unigram', 'Frequency'] )
df1


# In[69]:


#bigram


# In[34]:


vec = CountVectorizer(ngram_range=(2,2)).fit(x)
bow = vec.transform(x)
sum_words = bow.sum(axis = 0)
words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
words_freq[:2]


# In[35]:


def get_top_n_words(x, n):
 vec = CountVectorizer(ngram_range=(2,2)).fit(x)
 bow = vec.transform(x)
 sum_words = bow.sum(axis = 0)
 words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
 words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
 return words_freq[:n]


# In[72]:


get_top_n_words(x, 3)


# In[36]:


words = get_top_n_words(df['review'], 20)


# In[74]:


words


# In[75]:


df1 = pd.DataFrame(words, columns = ['Bigram', 'Frequency'])
df1 = df1.set_index('Bigram') 
df1.iplot(kind = 'bar', xTitle = 'Bigram', yTitle = 'count', title = 'Top 20 Bigram words')


# In[ ]:


#Trigran


# In[37]:


def get_top_n_words(x, n):
 vec = CountVectorizer(ngram_range=(3,3)).fit(x)
 bow = vec.transform(x)
 sum_words = bow.sum(axis = 0)
 words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
 words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
 return words_freq[:n]


# In[38]:


get_top_n_words(x, 3)


# In[40]:


words = get_top_n_words(df['review'], 20)


# In[41]:


words


# In[81]:


df1 = pd.DataFrame(words, columns = ['Trigram', 'Frequency'])
df1 = df1.set_index('Trigram') 
df1.iplot(kind = 'bar', xTitle = 'Trigram', yTitle = 'count', title = 'Top 20 Trigram words')


# In[ ]:


#dist uni,bi, trigrams with out stop words 


# In[42]:


def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(1,1), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[43]:


get_top_n_words(x, 3)


# In[44]:


words = get_top_n_words(df['review'], 20)


# In[45]:


words


# In[86]:


df1 = pd.DataFrame(words, columns = ['Unigram', 'Frequency'])
df1 = df1.set_index('Unigram') 
df1.iplot(kind = 'bar', xTitle = 'Unigram', yTitle = 'count', title = 'Top 20 Unigram words')


# In[ ]:


# stop words with bigram


# In[47]:


def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(2,2), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[48]:


get_top_n_words(x, 3)


# In[49]:


words = get_top_n_words(df['review'], 20)


# In[50]:


words


# In[132]:


df1 = pd.DataFrame(words, columns = ['Bigram', 'Frequency'])
df1 = df1.set_index('Bigram') 
df1.iplot(kind = 'bar', xTitle = 'Bigram', yTitle = 'count', title = 'Top 20 Bigram words')


# In[ ]:


#stopwords with out Trigram


# In[51]:


def get_top_n_words(x, n):
    vec = CountVectorizer(ngram_range=(3,3), stop_words='english').fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in  vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


# In[52]:


x


# In[ ]:


#get_top_n_words(x, 3)


# In[53]:


words = get_top_n_words(df['review'], 20)


# In[138]:


words


# In[139]:


df1 = pd.DataFrame(words, columns = ['Trigram', 'Frequency'])
df1 = df1.set_index('Trigram') 
df1.iplot(kind = 'bar', xTitle = 'Trigram', yTitle = 'count', title = 'Top 20 Trigram words')


# In[ ]:


#Bivariate analysis


# In[20]:


#distribution of sentiment polirity
sns.pairplot(df)


# In[140]:


sns.catplot(x = 'condition', y = 'polarity', data = df)


# In[142]:


sns.catplot(x = 'condition', y = 'polarity', data = df, kind = 'box')


# In[144]:


sns.catplot(x = 'drugName', y = 'polarity', data = df)


# In[146]:


sns.catplot(x = 'drugName', y = 'polarity', data = df, kind = 'box')


# In[147]:


sns.catplot(x = 'drugName', y = 'review_len', data = df, kind = 'box')


# In[148]:


sns.catplot(x = 'condition', y = 'review_len', data = df, kind = 'box')


# In[90]:


x1 = df[df['usefulCount']==1][['usefulCount', 'polarity']] 
x1 = df[df['usefulCount']==0][['usefulCount', 'polarity']] 


# In[91]:


#Kernel Density Estimate plot
sns.jointplot(x = 'polarity', y = 'review_len', data = df, kind = 'kde')


# In[166]:


sns.jointplot(x = 'polarity', y = 'rating', data = df, kind = 'kde')


# In[21]:


#relationship analysis
corelation = df.corr()


# In[22]:


sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns
            ,annot=True)


# In[ ]:


#relation in scatter plot
sns.relplot(x='review', y='rating', hue='usefulCount', data=df)


# In[48]:


#histogram
sns.distplot(df['rating'],bins=10)


# In[24]:


sns.catplot(x='rating', kind= 'box', data= df)


# In[ ]:


#relation in scatter plot.
sns.relplot(x= 'drugName', y='review', hue='condition', data=df)


# In[26]:


df['review'].str.len().hist()


# In[27]:


df['review'].str.split().   apply(lambda x : [len(i) for i in x]).    map(lambda x: np.mean(x)).hist()


# In[54]:


nltk.download('stopwords')
stop=set(stopwords.words('english'))


# In[55]:


corpus=[]
new= df['review'].str.split()
new=new.values.tolist()
corpus=[word for i in new for word in i]

from collections import defaultdict
dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1


# In[56]:


counter=Counter(corpus)
most=counter.most_common()

x, y= [], []
for word,count in most[:40]:
    if (word not in stop):
        x.append(word)
        y.append(count)
        
sns.barplot(x=y,y=x)


# In[57]:


from nltk.util import ngrams
list(ngrams(['I' ,'went','to','the','river','bank'],2))


# In[58]:


def get_top_ngram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) 
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:10]


# In[59]:


top_n_bigrams=get_top_ngram(df['review'],2)[:10]
x,y=map(list,zip(*top_n_bigrams))
sns.barplot(x=y,y=x)


# In[96]:


top_tri_grams=get_top_ngram(df['review'],n=3)
x,y=map(list,zip(*top_tri_grams))
sns.barplot(x=y,y=x)


# In[60]:


def preprocess_news(df):
    corpus=[]
    stem=PorterStemmer()
    for df in df['review']:
        words=[w for w in (df) if (w not in stop)]
        
        words=[lem.lemmatize(w) for w in words if len(w)>2]
        
        corpus.append(words)
    return corpus

corpus=preprocess_news(df)


# In[61]:


dic=gensim.corpora.Dictionary(corpus)
bow_corpus = [dic.doc2bow(doc) for doc in corpus]


# In[62]:


condition      0
review         0
rating         0
date           0
usefulCount    0
dtype: int64
''.join(


# In[65]:


df['word_count'] = df['review'].apply(lambda x: len(str(x).split(" ")))
df[['review','word_count']].head()


# In[66]:


df['char_count'] = df['review'].str.len() ## this also includes spaces
df[['review','char_count']].head()


# In[ ]:


#RandomForestClassification


# In[63]:


df = pd.read_csv('file:///C:/Users/sandeep/Downloads/sq.csv')


# In[64]:


df.head()
sizes = df['rating'].value_counts(sort=1)
print(sizes)
df = df.dropna()
df.rating[df.rating == '10,9,8,7'] = 1
df.rating[df.rating == '7,6,5,4'] = 2
df.rating[df.rating == '3,2,1,0'] = 3
print(df.head())
y = df['rating'].values
x = df.drop(labels=['rating'], axis=1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1, random_state=20)
print(x_train)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 10, random_state=30)
model.fit(x_train, y_train)
prediction_test = model.predict(x_test)
print(prediction_test)
from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))
 
print(model.feature_importances_)
feature_list = list(x.columns)
feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
print(feature_imp)


# In[105]:


# Naive Bayes Algorithm.
df = datasets.load_iris()


# In[106]:


model = GaussianNB()


# In[107]:


model.fit(df.data, df.target)


# In[108]:


print(model)


# In[109]:


expected = df.target
predicted = model.predict(df.data)


# In[110]:


print(metrics.classification_report(expected, predicted))


# In[111]:


print(metrics.confusion_matrix(expected, predicted))


# In[1]:


#mutinomialnb algorithom NB = naviebayes 


# In[112]:


classifier = MultinomialNB()


# In[113]:


classifier.fit(x_train, y_train)


# In[114]:


pred = classifier.predict(x_test)


# In[115]:


score = metrics.accuracy_score(y_test, pred)


# In[116]:


print(score)


# In[118]:


# decision tree
x = df.iloc[:,:-1]


# In[119]:


x


# In[120]:


y = df.iloc[:,1]


# In[121]:


y


# In[122]:


laberencode_x = LabelEncoder()


# In[123]:


x = x.apply(LabelEncoder().fit_transform)


# In[124]:


x


# In[92]:


regressor = DecisionTreeClassifier()


# In[93]:


regressor.fit(x.iloc[:,1:5], y)


# In[117]:


df = pd.read_csv('file:///C:/Users/sandeep/Downloads/tempw review.csv')


# In[94]:


x_in = np.array([-1,1])


# In[95]:


y_pred = regressor.predict([x_in])


# In[96]:


y_pred


# In[97]:


dot_data = StringIO()


# In[103]:


export_graphviz(regressor, out_file = dot_data, filled = True, special_characters = True)


# In[105]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())


# In[107]:


graph.write_png('tree.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




