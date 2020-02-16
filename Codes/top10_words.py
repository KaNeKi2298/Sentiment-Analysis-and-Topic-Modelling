#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
file1=pd.read_csv("Enter csv file location here")
file1=file1['review_text']


# In[6]:


s=""
for i in range(0,1000):
    s=s+file1[i].lower()
    


# In[14]:


import collections
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Read input file, note the encoding is specified here 
# It may be different in your text file
a= s
# Stopwords
from nltk.corpus import stopwords
stopwords=set(stopwords.words('english'))
stopwords = stopwords.union(set(['mr','mrs','one','two','said','phone','amazon','iphone','product','apple','vivo','redmi','lenovo','x',' ']))
# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}
# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
for word in a.lower().split():
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("â€œ","")
    word = word.replace("â€˜","")
    word = word.replace("*","")
    if word not in stopwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
# Print most common word
n_print = int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)
# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')


# In[ ]:




