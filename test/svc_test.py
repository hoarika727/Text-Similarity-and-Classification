import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import re
import csv
import scikitplot as skplt
import matplotlib.pyplot as plt

# %%
train = pd.read_csv('../data/train.csv') 
test = pd.read_csv('../data/test.csv')
#train.head(10)

#%%
x = train['title1_en']+train['title2_en']
regex_x = [re.sub(r'(.)\1{2,}', r'\1', title) for title in x]

# %%
#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(x, train['label'], test_size=0.2, random_state=7)

# %%
#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(max_df=0.8, ngram_range = (1,4), lowercase=True)
#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# %%
clf_SVC = LinearSVC().fit(tfidf_train, y_train)
y_real_SVC = clf_SVC.predict(tfidf_train)
train_cm = confusion_matrix(y_train,y_real_SVC)
train_score_SVC = accuracy_score(y_train,y_real_SVC)
train_f1score_SVC = f1_score(y_train,y_real_SVC, average='weighted')
print(train_cm)
print(f'SVC train score: {round(train_score_SVC*100,2)}%, {round(train_f1score_SVC*100,2)}%')

y_pred_SVC = clf_SVC.predict(tfidf_test)
cm = confusion_matrix(y_test,y_pred_SVC)
score_SVC = accuracy_score(y_test,y_pred_SVC)
f1score_SVC = f1_score(y_test,y_pred_SVC, average='weighted')  
print(cm)
print(f'SVC test score: {round(score_SVC*100,2)}%, {round(f1score_SVC*100,2)}%')

# %%
# helpful function
def plot_cm(y_test, y_pred):
    plt.rcParams.update({'font.size': 18})
    skplt.metrics.plot_confusion_matrix(
        y_test, 
        y_pred,
        figsize=(15,15))
    plt.savefig('confusion_.png')

plot_cm(y_test,y_pred_SVC)
