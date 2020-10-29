import os
import pandas as pd
import numpy as np

path = os.path.dirname(os.path.realpath(__file__))

trainData = '/data/train.csv'

train = pd.read_csv(os.path.join(path,trainData), sep=',', index_col='id')

"""
Preliminary study of the train data
""" 
title1_sample = train['title1_en'][1]
title2_sample = train['title2_en'][1]
#print(title1_sample, '\n',title2_sample)

label = train['label'].unique()
#print(label)

agree = train[train['label'] == 'agreed']
#agree[0:5]

agree1_sample = agree['title1_en'][132794]
agree2_sample = agree['title2_en'][132794]
#print(agree1_sample,'\n',agree2_sample)

#print(agree.index) #length = 74238

disagree = train[train['label']=='disagreed']
#print(disagree.index) #length = 6606

neutral = train[train['label']=='unrelated']
#print(neutral.index) #length = 175598

"""
Use bag of words + cosine similarity - on an agreed pair
"""
#import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
from collections import Counter 

def word_frequency(line):
    tokens = word_tokenize(line)
    stop_words = stopwords.words('english')
    word_list = [token.lower() for token in tokens if token.isalpha()]
    clean_word_list = [word for word in word_list if not word in stop_words]
    return clean_word_list

def cosineSim(sample1,sample2):
    uVector = list(np.unique(sample1+sample2))
    s1 = [1 if w in sample1 else 0 for w in uVector]
    s2 = [1 if w in sample2 else 0 for w in uVector]
    cnum = 0
    for i in range(len(uVector)):
        cnum += s1[i]*s2[i]
    cdenum = np.sqrt(sum(s1)*sum(s2))
    cos_sim = cnum/cdenum
    return cos_sim

t1 = word_frequency(agree1_sample)

t2 = word_frequency(agree2_sample)

agree_cos_sim = cosineSim(t1,t2)
print(agree_cos_sim) #0.28867513

# Method 2 - TfidfVectorizer + cosine_simiarlity
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def cosineSim_TV(sample1,sample2):
    pairs = [sample1, sample2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(pairs)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
    return cosine_sim

pairs = [agree1_sample, agree2_sample]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(pairs)
print(tfidf_matrix[0:1])
cosine_sim = cosine_similarity(tfidf_matrix[0:1],tfidf_matrix)
print(cosine_sim) #[[1.        , 0.42720305]]


# Method 3 - CountVectorizer + TfidfTransformer + cosine_simiarlity

def cosineSim_TF(sample1,sample2):
    pairs = [sample1, sample2]
    sw = stopwords.words('english')
    vectorizer = CountVectorizer(stop_words=sw) 
    transformer = TfidfTransformer()
    tfidfArr = vz.fit_transform(pairs).toarray()
    cosine_sim = cosine_similarity(tfidfArr[0:1],tfidfArr[1:])
    return cosine_sim
    
sw = stopwords.words('english')
vz = CountVectorizer(stop_words=sw) 
tf = TfidfTransformer()

tfidfArr = vz.fit_transform(pairs).toarray() #Extract feature count (supported by CountVectorizer)
tf.fit(tfidfArr) #TF-IDF normalization
print(tf.transform(tfidfArr).toarray()) #row-wise euclidean normalization

cosine_sim2 = cosine_similarity(tfidfArr[0:1],tfidfArr[1:])
print(cosine_sim2) #0.55901699

"""
Apply models on disagreed and neutral pairs
"""

#Try with disagree title
disagree1_sample = disagree['title1_en'][36098]
disagree2_sample = disagree['title2_en'][36098]

d1 = word_frequency(disagree1_sample)
d2 = word_frequency(disagree2_sample)

disagree_cos_sim = cosineSim(d1,d2)
disagree_cos_sim #0.20100756

disagree_cos_sim_TV = cosineSim_TV(disagree1_sample,disagree2_sample)
disagree_cos_sim_TV #[[1.        , 0.30926284]]

disagree_cos_sim_TF = cosineSim_TF(disagree1_sample,disagree2_sample) 
disagree_cos_sim_TF #[[0.13363062]]

#Try with neutral title
neutral1_sample = neutral['title1_en'][191474]
neutral2_sample = neutral['title2_en'][191474]

n1 = word_frequency(neutral1_sample)
n2 = word_frequency(neutral2_sample)

neutral_cos_sim = cosineSim(n1,n2)
print(neutral_cos_sim) #0.144337

neutral_cos_sim_TV = cosineSim_TV(neutral1_sample,neutral2_sample)
print(neutral_cos_sim_TV) #[[1.         0.08546498]]

neutral_cos_sim_TF = cosineSim_TF(neutral1_sample,neutral2_sample) 
print(neutral_cos_sim_TF) #[[0.20801257]]

