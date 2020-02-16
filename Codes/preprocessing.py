#!/usr/bin/env python
# coding: utf-8

# In[115]:


import pandas as pd


# In[116]:


raw_reviews=pd.read_csv("Enter csv file location here")


# In[117]:


raw_reviews=raw_reviews['review_text']
raw_reviews.head()


# In[118]:


import nltk

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

print(raw_reviews[0])


# In[119]:


#lower case 
for i in range(0,800):
    raw_reviews[i]=raw_reviews[i].lower()


# In[121]:


#html tags removal
import re
TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)    
    
for i in range(0,800):
    raw_reviews[i]=remove_tags(raw_reviews[i])


# In[122]:


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

for i in range(0,800):
    raw_reviews[i]=raw_reviews[i].replace("’","'")
#     words=tokenizer.tokenize(file1[i])
#     words=word_tokenize(file1[i])
    words=raw_reviews[i].split()
    reformed = [appos[word] if word in appos else word for word in words]
    reformed = " ".join(reformed) 
    raw_reviews[i]=reformed


# In[101]:


raw_reviews[0]


# In[123]:


#removal of stop words and lemmetization
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
stop_words.remove('not')
stop_words.remove('below')
stop_words.remove('down')
stop_words.remove('very')
stop_words.remove('no')

processed_reviews=[]
lemmatizer = WordNetLemmatizer()

for i in range(0,500):
    sent_tkk=sent_tokenize(raw_reviews[i])
    st1=""
    for snt in sent_tkk:
        word_tkk=word_tokenize(snt)
        s=""
        for w in word_tkk:
            if not w in stop_words:
                s=s+" "+lemmatizer.lemmatize(w)
        st1=st1+s;
    processed_reviews.append(st1);    

    



# In[124]:


processed_reviews[0]
          


# In[125]:


sent_tokenize(processed_reviews[0])


# In[126]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
   
sent_analyser=SentimentIntensityAnalyzer()

negdict={'heats':-2,'heating':-2,'hot':-2,'drained':-3,'draining':-3,'drain':-3,'lagging':-2,'lag':-2,'slipping':-1,'slippery':-1,'slow':-2,'heavy':-2,'issue':-2,'issues':-2,'return':-2}
posdict={'handy':2,'slim':3,'classy':2,'premium':3,'smooth':3,'responsive':2,'sharp':2,'fast':2,'budget':3,'light':2}
sent_analyser.lexicon.update(negdict)
sent_analyser.lexicon.update(posdict)


# In[106]:


sentence="the phone is faulty"


# In[ ]:





# In[127]:



sentiment_score_camera=0
sentiment_score_display=0
sentiment_score_battery=0
sentiment_overall=0

display_cnt=0
cam_cnt=0
battery_cnt=0

cam_list=[]
cam_rating=[]
display_list=[]
display_rating=[]
battery_list=[]
battery_rating=[]

for i in range(0,800):
    sentences=sent_tokenize(raw_reviews[i]) 
    sentiment_review=0
    sent_cnt_review=0
    for j in range(0,len(sentences)):
                    sent_cnt_review+=1
                    ss = sent_analyser.polarity_scores(sentences[j])
                    sentiment_review+=ss['compound']
                    if sentences[j].find("camera")!=-1:
                        cam_cnt+=1
                        sentiment_score_camera+=ss['compound']
                        cam_list.append(sentences[j]);
                        cam_rating.append(ss['compound'])
                    if sentences[j].find("display")!=-1 or sentences[j].find("screen")!=-1 :
                        display_cnt+=1
                        sentiment_score_display+=ss['compound']
                        display_list.append(sentences[j])
                        display_rating.append(ss['compound'])
                    if sentences[j].find("battery")!=-1:
                        battery_cnt+=1
                        sentiment_score_battery+=ss['compound']   
                        battery_list.append(sentences[j])
                        battery_rating.append(ss['compound'])
    sentiment_overall+=sentiment_review/sent_cnt_review              

         
         
final=sentiment_score_camera/cam_cnt 

print("Sentiment score for Camera is:",2.5*final+3)

final=sentiment_score_display/display_cnt
print("Sentiment score for Display is:",2*final+3)

final=sentiment_score_battery/battery_cnt
print("Sentiment score for Battery is:",2*final+3)


final=sentiment_overall/800
print("Overall Sentiment score is:",2*final+3)





# In[108]:


for i in range(0,len(battery_list)):
  print(battery_list[i],battery_rating[i])

print(len(battery_list))


# In[ ]:




