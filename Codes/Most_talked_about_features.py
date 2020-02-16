#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk.corpus import stopwords



stop_words = stopwords.words('english')
stop_words.append('apple')
stop_words.append('iphone')
stop_words.append('amazon')
stop_words.append('product')
stop_words.append('awesome')
stop_words.append('better')
stop_words.append('amazing')
stop_words.append('everything')
stop_words.append('bought')
stop_words.append('really')
stop_words.append('always')
stop_words.append('excellent')
stop_words.append('thanks')
stop_words.append('overall')
stop_words.append('original')
stop_words.append('disappointed')
stop_words.append('little')
stop_words.append('nothing')
stop_words.append('mobile')
stop_words.append('coolpad')
stop_words.append('coolplus')



file1=pd.read_csv("Enter csv File Location Here")
file1=file1['review_text']
reviews=[]

for i in range(0,1000):
    reviews.append(file1[i].lower())


# In[115]:


def freq_words(x, terms=30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    fdist = FreqDist(all_words)
    words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

    # selecting top 20 most frequent words
    d = words_df.nlargest(columns="count", n = terms) 
    plt.figure(figsize=(20,5))
    ax = sns.barplot(data=d, x= "word", y = "count")
    ax.set(ylabel = 'Count')
    plt.show()
    
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new    

def deEmojify(inputString):
    return inputString.encode('ascii', 'ignore').decode('ascii')


# In[116]:


freq_words(reviews,30)


# In[117]:



for i in range(1000):
    reviews[i]=reviews[i].replace("[^a-zA-Z#]", " ")
    reviews[i]=' '.join([w for w in reviews[i].split() if len(w)>5])

reviews2 = [remove_stopwords(s.split()) for s in reviews]

print(reviews2[0])


# In[118]:


punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''

proc_reviews=[]

for i in range(1000):
    pro_reviews = ""
    for char in reviews2[i]:
        if char not in punctuations:
            pro_reviews = pro_reviews + char
    proc_reviews.append(deEmojify(pro_reviews))        


# In[119]:


freq_words(proc_reviews, 30)


# In[120]:


import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
tags=['NN', 'JJ']
def lemmatize(tokens):
        return [lemmatizer.lemmatize(w) for w in tokens if nltk.tag.pos_tag([w])[0][1] in tags]


# In[121]:


tokenized_reviews=[r.split() for r in proc_reviews]
print(tokenized_reviews[0])


# In[122]:


reviews_lemm=[]
for i in range(1000):
    reviews_lemm.append(lemmatize(tokenized_reviews[i]))


# In[123]:


reviews_final = []
for i in range(1000):
    reviews_final.append(' '.join(reviews_lemm[i]))


# In[124]:


print(reviews_final[0])


# In[125]:


print(type(reviews_final))


# In[126]:


import gensim
from gensim import corpora
dictionary = corpora.Dictionary(reviews_lemm)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_lemm]


# In[127]:



LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=5, random_state=100,
                chunksize=10, passes=800)


# In[128]:


ld=lda_model.print_topics()


# In[129]:


print(ld)


# In[130]:


l=[]
t=""
done=0

for i in range(0,len(ld)):
    s=ld[i][1]
    l1=[]
    for c in s:
        if (c=='"'):
            if (done==0):
                done=1
            else:
                l1.append(t)
                done=0
                t=""
        elif(done==1):
            t=t+c
    l.append(l1)        
            
                
            
        


# In[131]:


print(l)


# In[132]:



# Load Google's pre-trained Word2Vec model.
vector_model = gensim.models.KeyedVectors.load_word2vec_format( '/Users/akhil/Downloads/GoogleNews-vectors-negative300.bin'
, binary=True)


# In[133]:


print(len(l[0]))


# In[134]:


topic=[]
import numpy as np

for i in range(0,len(ld)):
    arr=vector_model[l[i][0]]
    for j in range(1,5):
        arr=np.add(arr,vector_model[l[i][j]])
    arr=np.true_divide(arr,len(l[i]))
    
    dist=np.linalg.norm(arr-vector_model[l[i][0]])
    ans=0
    
    for j in range(1,5):
        dist1=np.linalg.norm(arr-vector_model[l[i][j]])
        if (dist1<dist):
            dist=dist1
            ans=j
    topic.append(l[i][ans])        
    


# In[63]:


print(topic)


# In[64]:


raw_reviews=pd.read_csv("Enter csv file location here")


# In[65]:


raw_reviews=raw_reviews['review_text']


# In[66]:


#lower case 
for i in range(0,1000):
    raw_reviews[i]=raw_reviews[i].lower()


# In[67]:


#html tags removal
import re
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)    
    
for i in range(0,1000):
    raw_reviews[i]=remove_tags(raw_reviews[i])


# In[68]:


#handling words like can't
# %load /Users/akhil/Downloads/appos.py
appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}

for i in range(0,1000):
    raw_reviews[i]=raw_reviews[i].replace("’","'")
#     words=tokenizer.tokenize(file1[i])
#     words=word_tokenize(file1[i])
    words=raw_reviews[i].split()
    reformed = [appos[word] if word in appos else word for word in words]
    reformed = " ".join(reformed) 
    raw_reviews[i]=reformed


# In[69]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
   
sent_analyser=SentimentIntensityAnalyzer()

negdict={'heats':-2,'heating':-2,'hot':-2,'drained':-3,'draining':-3,'drain':-3,'lagging':-2,'lag':-2,'slipping':-1,'slippery':-1,'slow':-2,'heavy':-2,'issue':-2,'issues':-2,'return':-2}
posdict={'handy':2,'slim':3,'classy':2,'premium':3,'smooth':3,'responsive':2,'sharp':2,'fast':2,'budget':3,'light':2}
sent_analyser.lexicon.update(negdict)
sent_analyser.lexicon.update(posdict)


# In[70]:


topic=['Overall','camera','battery','display']+topic

print(topic)


# In[71]:


print(len(topic))


# In[72]:


sentiment_overall=0
topic_score=[0 for i in range(0,len(topic))]
topic_count=[0 for i in range(0,len(topic))]
topic_reviews=[[] for i in range(0,len(topic))]
topic_ratings=[[] for i in range(0,len(topic))]


# In[73]:


print(raw_reviews)


# In[74]:


from nltk.tokenize import sent_tokenize
for i in range(0,1000):
    sentences=sent_tokenize(raw_reviews[i])
    topic_reviews[0].append(raw_reviews[i])
    topic_count[0]+=1
    print(raw_reviews[i])
    sentiment_review=0
    sent_cnt_review=0
    for j in range(0,len(sentences)):
                    sent_cnt_review+=1
                    ss = sent_analyser.polarity_scores(sentences[j])
                    sentiment_review+=ss['compound']
                    for i in range(1,len(topic)):
                        if sentences[j].find(topic[i])!=-1:
                            topic_count[i]+=1
                            topic_score[i]+=ss['compound']
                            topic_reviews[i].append(sentences[j])
                            topic_ratings[i].append(ss['compound'])
    topic_ratings[0].append(sentiment_review/sent_cnt_review)
    topic_score[0]+=sentiment_review/sent_cnt_review 


# In[79]:


def gs(score):
    x=2.5+5*score;
    if(x<=0):
        return 1.7
    if(x<=5):
        return x;
    else:
        return 4.8;
    


# In[80]:


for i in range(0,len(topic)):
    if(topic_count[i]>0):
        score=topic_score[i]/topic_count[i]
        print("The sentiment score for ",topic[i]," is: ",gs(score))


# In[19]:


print(topic_reviews[7])


# In[ ]:





# In[ ]:




