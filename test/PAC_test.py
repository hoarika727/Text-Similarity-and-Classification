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
import matplotlib.pyplot as plt
import scikitplot  as skplt

# %%
train = pd.read_csv('../data/train.csv')
train["title_comb"] = train['title1_en']+train['title2_en']

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

# %%
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(train['title1_en']+train['title2_en'], train['label'], test_size=0.2, random_state=7)

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
# Accuracy
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# %%
plt.rcParams.update({'font.size': 18})
skplt.metrics.plot_confusion_matrix(
    y_test, 
    y_pred,
    figsize=(15,15))
# %%
