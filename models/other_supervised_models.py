# %%
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt

# %%
train = pd.read_csv('../data/train.csv')
#test = pd.read_csv('data/test.csv')
#train.head(10)

#%%
x = train['title1_en']+train['title2_en']

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
# Naive Bayes Classification
clf_NB = MultinomialNB().fit(tfidf_train, y_train)
y_pred_NB = clf_NB.predict(tfidf_test)

acc_NB = accuracy_score(y_test,y_pred_NB)
f1score_NB = f1_score(y_test,y_pred_NB, average='weighted')
print(f'NB Accuracy: {round(acc_NB*100,2)}%')
print(f'NB F1-score: {round(f1score_NB*100,2)}%')
# %%
#  Random Forest Classifier
clf_RF = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0).fit(tfidf_train, y_train)
y_pred_RF = clf_RF.predict(tfidf_test)
acc_RF = accuracy_score(y_test,y_pred_RF)
f1score_RF = f1_score(y_test,y_pred_RF, average='weighted')
print(f'RF Accuracy: {round(acc_RF*100,2)}%')
print(f'RF F1-score: {round(f1score_RF*100,2)}%')
# %%
# LogisticRegression(random_state=0)
clf_Logit = LogisticRegression(random_state=0).fit(tfidf_train, y_train)
y_pred_Logit = clf_Logit.predict(tfidf_test)
acc_Logit = accuracy_score(y_test,y_pred_Logit)
f1score_Logit = f1_score(y_test,y_pred_Logit, average='weighted')
print(f'Logistic Accuracy: {round(acc_Logit*100,2)}%')
print(f'Logistic F1-score: {round(f1score_Logit*100,2)}%')

# %%
# helpful function
def plot_cm(y_test, y_pred):
    plt.rcParams.update({'font.size': 18})
    skplt.metrics.plot_confusion_matrix(
        y_test, 
        y_pred,
        figsize=(15,15))
    plt.savefig('cm.png')
