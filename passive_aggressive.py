# %%
import numpy as np
import pandas as pd
import itertools
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %%
train = pd.read_csv('data/train.csv')
train["title_comb"] = train['title1_en']+train['title2_en']
test = pd.read_csv('data/test.csv')
test["title_comb"] = test['title1_en']+test['title2_en']
sub = pd.read_csv('data/sample_submission.csv')

# %%
# Data preprocessing 
def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

def remove_stopwords(text):
    '''a function for removing the stopword'''
    sw = stopwords.words('english')
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    return " ".join(text)

def remove_specialwords(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]' 
    return re.sub(pat, '', text)

def lemmatizing(text):    
    '''a function which stems each word in the given text'''
    lemmatizer  = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text.split()]
    return " ".join(text) 

train["title_comb"] = train["title_comb"].apply(remove_punctuation)
train["title_comb"] = train["title_comb"].apply(remove_stopwords)
train["title_comb"] = train["title_comb"].apply(remove_specialwords)
train["title_comb"] = train["title_comb"].apply(lemmatizing)

test["title_comb"] = test["title_comb"].apply(remove_punctuation)
test["title_comb"] = test["title_comb"].apply(remove_stopwords)
test["title_comb"] = test["title_comb"].apply(remove_specialwords)
test["title_comb"] = test["title_comb"].apply(lemmatizing)

# %%
#DataFlair - Split the dataset
x_train, y_train = train["title_comb"], train["label"] 
x_test = test["title_comb"]

# %%
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(use_idf=False ,ngram_range=(1,10))
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
# %%
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)
#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
# %%
# Submission
sub['label'] = y_pred
sub.head(10)
sub.to_csv("submission_tfidf_pac.csv", index=False)
# %%
