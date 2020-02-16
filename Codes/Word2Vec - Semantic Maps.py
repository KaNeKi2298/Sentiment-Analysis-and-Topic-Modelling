#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import pandas as pd
import gensim
# import nltk.data
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from nltk.tokenize import PunktSentenceTokenizer


# In[2]:


# import pandas as pd
import glob

path = r'enter file location here' # use your path
all_files = glob.glob(path + "/*.csv")

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
  

raw_data = pd.concat(li, axis=0, ignore_index=True)


# In[3]:



# raw_data=pd.read_csv("/Users/akhil/Desktop/My_Project/Scraping Reviews/iPhoneX.csv")

raw_data=raw_data['review_text']

sent_tokenizer= PunktSentenceTokenizer()

lemmatizer = WordNetLemmatizer()

word_tokenizer=RegexpTokenizer('[A-Za-z]\w+')

stop_words=set(stopwords.words('english'))


# In[4]:


raw_data= raw_data.sample(frac=1).reset_index(drop=True)


# In[5]:


# raw_data=raw_data.to_frame(name='reviews')
# raw_data=raw_data.reset_index(drop=True, inplace=True)  


# In[6]:


type(raw_data)


# In[7]:


# x=2
# for i in range(0,4079):
#     print(isinstance(raw_data[i],float))
    
           


# In[8]:


#all the preprocessing functions

def tokenizeit(sent):
    return word_tokenizer.tokenize(sent)


def lemmatize(tokens):
        return [lemmatizer.lemmatize(w.lower()) for w in tokens]
    
def rem_stop(tokens):
        return [w for w in tokens if w.lower() not in stop_words]    


# In[9]:


final=[]
for i in range(0,4000):
    if not isinstance(raw_data[i],float):
        raw_sentences=sent_tokenizer.tokenize(raw_data[i])
        for j in raw_sentences:
                processed=tokenizeit(j)
                processed=rem_stop(processed)
                processed=lemmatize(processed)
                final.append(processed)
    


# In[10]:


final[980]


# In[11]:


#using the word2vec model
model = gensim.models.Word2Vec(final)
model.train(final, total_examples=5, epochs=1000)


# In[12]:


# summarize vocabulary
words = list(model.wv.vocab)
# print(words)
data=model.most_similar('display')
print(data)


# In[ ]:





# In[ ]:




