# Follow https://github.com/epfml/sent2vec for Sent2Vec setup
# The worst case senario is to clone its repository to run this script in the cloned project

import sent2vec
import pandas as pd
import numpy as np
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import nltk
import scikitplot as skplt
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

#%%
def word_embedding_sent2vec(datapath):
    model = sent2vec.Sent2vecModel()
    # Download the pre-trained model on
    # https://github.com/epfml/sent2vec#downloading-sent2vec-pre-trained-models
    model.load_model('twitter_unigrams.bin')

    trainData = datapath #'data/train.csv'

    train = pd.read_csv(trainData, sep=',', index_col='id')

    tid = train['tid1'].append(train['tid2'])
    titles = train['title1_en'].append(train['title2_en'])
    df_titles = pd.DataFrame({'tid': tid, 'title': titles})
    df_titles = df_titles.drop_duplicates()
    df_titles.index = df_titles['tid']

    titles = df_titles['title']

    sample = titles.copy()
    tokenized_titles = []
    for title in tqdm(sample):
        token = gensim.utils.simple_preprocess(title, min_len=2, max_len=30)
        sentence = ' '.join(token) + ' \n'
        tokenized_titles.append(sentence)

    embs = model.embed_sentences(tokenized_titles)

    embs_dict = dict((titles.index[i], embs[i]) for i in range(len(embs)))
    embs_dict1 = pd.Series(embs_dict)
    #embs_dict1.to_json('title_s2v.json',orient='index',indent=2)

    result = train.copy()

    result['cosine_similarity'] = 0
    for row in tqdm(range(len(result))):
        try:
            idx = result.index[row]
            tid1 = result['tid1'][idx]
            tid2 = result['tid2'][idx]
            title1 = embs_dict[tid1].reshape(1,-1)
            title2 = embs_dict[tid2].reshape(1,-1)
            cs = cosine_similarity(title1, title2)
            result.iloc[row,result.columns.get_loc('cosine_similarity')] = cs[0]
        except:
            print(f'error at {row}')
    #result.to_csv('train_s2v_CosSim.csv')
    return embs_dict, result

def prepare_embedded_data(data, num_titles = 2, num_features=700):
    s2v_mx = np.zeros((len(data),num_titles,num_features))
    embs_dict, result = word_embedding_sent2vec(data)
    for row in tqdm(range(len(result))):
        idx = result.index[row]
        tid1 = result['tid1'][idx]
        tid2 = result['tid2'][idx]
        s2v_mx[row][0] = np.array(embs_dict[tid1])
        s2v_mx[row][1] = np.array(embs_dict[tid2])

    s2v_final_mx = s2v_mx.reshape((len(data),num_titles*num_features))
    return s2v_final_mx, result

#%%
data = pd.read_csv('data/train.csv', sep=',', index_col='id')
x, df = prepare_embedded_data(data)
x_train,x_test,y_train,y_test=train_test_split(x, df['label'], test_size=0.2, random_state=7)

#%%
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

#%%
'''
#Plot Cosine Similarity Distribution - Sent2Vec

result = pd.read_csv('WordEmbeddings/train_s2v_CosSim.csv')
agree_df = result[result['label'] == 'agreed']
disagreed_df  = result[result['label'] == 'disagreed']
neutral_df = result[result['label'] == 'unrelated']

plt.hist(agree_df['cosine_similarity'],color='orange',ec='white', lw=0.2)
plt.title("Cosine Similarity for 'agreed' titles")
plt.show()
#plt.savefig('s2v_CosSim_agreed')

plt.hist(disagreed_df['cosine_similarity'],color='brown',ec='white', lw=0.2)
plt.title("Cosine Similarity for 'disagreed' titles")
plt.show()
#plt.savefig('s2v_CosSim_disagreed')

plt.hist(neutral_df['cosine_similarity'],color='grey',ec='white', lw=0.2)
plt.title("Cosine Similarity for 'unrelated' titles")
plt.show()
#plt.savefig('s2v_CosSim_unrelated')
'''