import pandas as pd
import numpy as np
import gensim
import gensim.models as word2vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.tokenize import TweetTokenizer
import scikitplot as skplt
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

#%%
# Word2Vec word embedding for cosine similarity
def word_embedding_word2vec(datapath):
    train = pd.read_csv(datapath, sep=',', index_col='id') #datapath = 'data/train.csv'
    tid = train['tid1'].append(train['tid2'])
    titles = train['title1_en'].append(train['title2_en'])
    df_titles = pd.DataFrame({'tid': tid, 'title': titles})
    df_titles = df_titles.drop_duplicates()
    df_titles.index = df_titles['tid']
    clean_titles = df_titles['title']

    tkzr = TweetTokenizer()
    formatted_title = [sentence.lower() for sentence in clean_titles] 
    tokenized_title = [tkzr.tokenize(title) for title in formatted_title]

    # the uploaded word embedding models:
    # train_w2v: 100 dim, pre-trained + self-trained corpus
    # train_w2v_1: 300 dim, self-trained corpus
    # train_w2v_2: 300 dim, pre-trained + self-trained corpus
    model = word2vec.Word2Vec(size=300, min_count=2, workers=4, compute_loss=True)
    model.build_vocab(tokenized_title)

    # To enrich the corpus to improve word embedding
    # Download the pre-trained model on 
    # https://fasttext.cc/docs/en/english-vectors.html
    # Load the following 3 lines; otherwise, the word embedding only depends on self-trained corpus
    #pretrain_model = gensim.models.KeyedVectors.load_word2vec_format("wiki-news-300d-1M.vec", binary=False)
    #model.build_vocab([list(pretrain_model.vocab.keys())], update=True)
    #model.intersect_word2vec_format("wiki-news-300d-1M.vec", binary = False, lockf=1.0)
    model.train(tokenized_title, total_examples=model.corpus_count, epochs=10)

    #model.save('train_w2v_1')
    return model, tokenized_title

# To try the function, run the following:
# model, tokenized_title = word_embedding_word2vec('data/train.csv')

#%%
def avg_feature_vector(sentence, model, num_features, idx2word): 
    #model = model_w2v, num_features = 300, idx2word = set(model_w2v.wv.index)
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in sentence:
        if word in idx2word:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def match_word_embedding(train_dataframe, tokenized_title, model, num_features, idx2word):
    # train_dataframe = train, tokenized_titles is from word_embedding_word2vec()[1],
    # model = model_w2v, num_features = 300, idx2word = set(model_w2v.wv.index)
    title_w2v = {}
    for i in tqdm(range(len(tokenized_title))):
        result = avg_feature_vector(tokenized_title[i], model = model, num_features=num_features, idx2word=idx2word)
        title_w2v.update({int(train_dataframe['tid'].iloc[i]):result})

    w2v = {k:list(v.astype(float)) for k,v in title_w2v.items()}
    #w2v1 = pd.Series(w2v)
    #w2v1.to_json('title_w2v_cs.json',orient='index',indent=2)
    return w2v

# To try these 2 functions, run the following:
# need 'model', 'tokenized_titles' from word_embedding_word2vec()
#
# model, tokenized_title = word_embedding_word2vec('data/train.csv')
# train = pd.read_csv('data/train.csv', sep=',', index_col='id')
# embedding_model = match_word_embedding(train, tokenized_title, model, 300, set(model.wv.index))

#%%
def cosine_similarity_embedding(data, embedding_model):
    #data = train, embedding_model = w2v
    result = data.copy()
    result['cosine_similarity'] = 0
    for row in tqdm(range(len(result))):
        try:
            idx = result.index[row]
            tid1 = result['tid1'][idx]
            tid2 = result['tid2'][idx]
            cos1 = np.array(embedding_model[tid1]).reshape(1,-1)
            cos2 = np.array(embedding_model[tid2]).reshape(1,-1)
            cs = cosine_similarity(cos1, cos2)
            result.iloc[row,result.columns.get_loc('cosine_similarity')] = cs[0]
        except:
            print(f'error at {row}')

    result.to_csv('train_w2v_CosSim.csv')
    return result

# To try this function, run the following:
# train = pd.read_csv('/data/train.csv', sep=',', index_col='id')
# w2v = pd.read_json('WordEmbeddings/title_w2v_cs.json')
# CosSim_df = cosine_similarity_embedding(train, w2v)
#%%
def prepare_w2vembedded_data(data, word2vec_model):
    w2v_mx = np.zeros((len(data),2,300))
    for row in tqdm(range(len(data))):
        idx = data.index[row]
        tid1 = data['tid1'][idx]
        tid2 = data['tid2'][idx]
        w2v_mx[row][0] = np.array(word2vec_model[tid1])
        w2v_mx[row][1] = np.array(word2vec_model[tid2])

    # Concatenate the titles for modelling
    w2v_svm_mx = w2v_mx.reshape((len(data),600))
    return w2v_svm_mx

#%%
# Main function for modelling with word embedding
data = pd.read_csv('WordEmbeddings/train_w2v_CosSim.csv')
w2v = pd.read_json('WordEmbeddings/title_w2v.json')
x = prepare_w2vembedded_data(data,w2v)
x_train,x_test,y_train,y_test=train_test_split(x, data['label'], test_size=0.2, random_state=7)

# %%
from sklearn.svm import SVC

clf_SVC = SVC().fit(x_train, y_train)
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
'''
Plot Cosine Similarity Distribution - Word2Vec

result = pd.read_csv('WordEmbeddings/train_w2v_CosSim.csv')
agree_df = result[result['label'] == 'agreed']
disagreed_df  = result[result['label'] == 'disagreed']
neutral_df = result[result['label'] == 'unrelated']

plt.hist(agree_df['cosine_similarity'],color='orange',ec='white', lw=0.2)
plt.title("Cosine Similarity for 'agreed' titles")
plt.show()
#plt.savefig('w2v_CosSim_agreed')

plt.hist(disagreed_df['cosine_similarity'],color='brown',ec='white', lw=0.2)
plt.title("Cosine Similarity for 'disagreed' titles")
plt.show()
#plt.savefig('w2v_CosSim_disagreed')

plt.hist(neutral_df['cosine_similarity'],color='grey',ec='white', lw=0.2)
plt.title("Cosine Similarity for 'unrelated' titles")
plt.show()
#plt.savefig('w2v_CosSim_unrelated')
'''
