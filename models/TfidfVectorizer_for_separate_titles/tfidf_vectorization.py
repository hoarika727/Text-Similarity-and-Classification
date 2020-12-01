#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import csv
import scipy

# %%
trainData = '/data/train.csv'

train = pd.read_csv(trainData, sep=',', index_col='id')

tid = train['tid1'].append(train['tid2'])
titles = train['title1_en'].append(train['title2_en'])
df_titles = pd.DataFrame({'tid': tid, 'title': titles})
df_titles = df_titles.drop_duplicates()
df_titles.index = range(len(df_titles))
df_titles

tfidf_vectorizer=TfidfVectorizer(max_df=0.8, ngram_range = (1,4), lowercase=True)

title_tfidf = tfidf_vectorizer.fit_transform(df_titles['title'])
scipy.sparse.save_npz('title_tfidf',title_tfidf)

tfidf = {}
for i in tqdm(range(len(df_titles))):
    tfidf.update({df_titles['tid'][i]:title_tfidf[i]})

tfidfs = pd.Series(tfidf)
tfidfs.to_json('title_tfidf.json', indent=2)

tfidf_x = []
for i in tqdm(range(len(train))):
    tid1 = train['tid1'][train.index[i]]
    tid2 = train['tid2'][train.index[i]]
    tfidf_combined = tfidf[tid1] + tfidf[tid2]
    tfidf_x.append(tfidf_combined)

tfidf_merged = {}
for i in tqdm(range(len(train))):
    tfidf_merged.update({train.index[i]:tfidf_x[i]})

tfidf_m = pd.Series(tfidf_merged)
tfidf_m.to_json('train_tfidf.json', indent=2)

tfidf_x_final = scipy.sparse.vstack(tfidf_x)
scipy.sparse.save_npz('train_tfidf',tfidf_x_final)

#%%
x_train,x_test,y_train,y_test=train_test_split(np.array(tfidf_m), train['label'], test_size=0.2, random_state=7)

# %%
clf_SVC = LinearSVC().fit(x_train, y_train)
y_real_SVC = clf_SVC.predict(x_train)
train_cm = confusion_matrix(y_train,y_real_SVC)
train_score_SVC = accuracy_score(y_train,y_real_SVC)
train_f1score_SVC = f1_score(y_train,y_real_SVC, average='weighted')
print(train_cm)
print(f'SVC train score: {round(train_score_SVC*100,2)}%, {round(train_f1score_SVC*100,2)}%')

y_pred_SVC = clf_SVC.predict(x_test)
cm = confusion_matrix(y_test,y_pred_SVC)
score_SVC = accuracy_score(y_test,y_pred_SVC)
f1score_SVC = f1_score(y_test,y_pred_SVC, average='weighted')  
print(cm)
print(f'SVC test score: {round(score_SVC*100,2)}%, {round(f1score_SVC*100,2)}%')

# %%
