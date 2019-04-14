#######################################
#########                      ########
#########                      ########
#########  Project AIT 582     ########
#########  Author:Meenakshi    ########
#########  Date:12 April 2019  ########
#########                      ########
#########                      ########
#######################################
###importing all the packages#####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

#####Reading the file using pandas#####
data=pd.read_csv("Hurricane_Harvey.csv", encoding='windows-1254')
data.head()
print(data.isnull().sum())
data.shape

data=data.dropna(subset=['Time'])


##Removing the stopwords from the Tweets
from nltk.corpus import stopwords
stop=stopwords.words('english')
data['Tweet']=data["Tweet"].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
data['Tweet'].head()
#data['Tweet'].tail()



#Removing the  punctutations and unuseful characters
data['Tweet']=data['Tweet'].str.replace('[^\w\s]','')
data['Tweet'].head()

####Tokenize the data
data['Tweet'] = data.apply(lambda row: nltk.word_tokenize(row['Tweet']), axis=1)

data.head()
print(data.isnull().sum())

data['Time'] =  pd.to_datetime(data['Time'], format='%m/%d/%Y %H:%M')
data.head()

new_dates, new_times = zip(*[(d.date(), d.time()) for d in data['Time']])
data = data.assign(new_date=new_dates, new_time=new_times)

data.to_csv('Data_message.csv')
