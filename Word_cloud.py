#######################################
#########                      ########
#########                      ########
#########  Project AIT 582     ########
#########  Author:Meenakshi    ########
#########  Date:15 March 2018  ########
#########                      ########
#########                      ########
#######################################

###importing all the packages#####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wordcloudimport re 
import nltk
nltk.download('all')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize

#####Reading the file using pandas#####
data=pd.read_csv("Hurricane_Harvey.csv", encoding='windows-1254')
data.head()
data.isnull().any()

##Removing the stopwords from the Tweets#####
from nltk.corpus import stopwords
stop=stopwords.words('english')
data['Tweet']=data["Tweet"].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
data['Tweet'].head()
#data['Tweet'].tail()


#####Removing the  punctutations and unuseful characters######
data['Tweet']=data['Tweet'].str.replace('[^\w\s]','')
data['Tweet'].head()



######Full clean Tweets#########
def tweets_cleaning(raw_tweet):
    raw_tweet = " ".join(word for word in raw_tweet.split() if 'http' not in word and not word.startswith('@') and not word.startswith('pic.twitter') and word != 'RT')
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))

data['clean_tweet'] = data['Tweet'].map(lambda x: tweets_cleaning(x))


words = ' '.join(data['Tweet'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])
                            
                            
###generating the word cloud
wordcloud = WordCloud(stopwords=STOPWORDS,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=6,
                      height=3000
                     ).generate(cleaned_word)
###Plotting the word cloud####
plt.figure(1,figsize=(20, 20))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
