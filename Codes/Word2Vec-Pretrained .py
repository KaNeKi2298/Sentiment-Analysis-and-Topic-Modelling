#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gensim

# Load Google's pre-trained Word2Vec model.

#IMPORTANT: to run this you have to download Google word2vec model.

model = gensim.models.KeyedVectors.load_word2vec_format( 'Download the model and copy the model address here'
, binary=True)


model.most_similar('display')


# In[ ]:




