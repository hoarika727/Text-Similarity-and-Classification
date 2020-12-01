#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# %%
train = pd.read_csv('../data/train.csv', sep=',', index_col='id')

y = train['label']
y = [0 if i == 'unrelated' else (1 if i =='disagreed' else 2) for i in y]

x_train,x_test,y_train,y_test=train_test_split(train['title1_en']+train['title2_en'], y, test_size=0.2, random_state=7)

x_train,x_test,y_train,y_test=train_test_split(train.index, y, test_size=0.2, random_state=7)

#%%
# %%
# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(max_df=0.8, ngram_range = (1,4), max_features=1000, lowercase=True)

# Transform x_train, x_test as tfidf vectors 
# Set data type as float32, ready to be fed into neural network
tfidf_train=tfidf_vectorizer.fit_transform(x_train).astype(np.float32)
tfidf_test=tfidf_vectorizer.transform(x_test).astype(np.float32)

# Prepare data as tensors
x_tfidf_train = tf.convert_to_tensor(tfidf_train.toarray().reshape(tfidf_train.shape[0],1,tfidf_train.shape[1]))
x_tfidf_test = tf.convert_to_tensor(tfidf_test.toarray().reshape(tfidf_test.shape[0],1,tfidf_test.shape[1]))
y_tfidf_train = tf.convert_to_tensor(np.array(y_train).reshape(np.array(y_train).shape[0],1))
y_tfidf_test = tf.convert_to_tensor(np.array(y_test).reshape(np.array(y_test).shape[0],1))

# Prepare datasets
train_set = tf.data.Dataset.from_tensor_slices((x_tfidf_train,y_tfidf_train))
test_set = tf.data.Dataset.from_tensor_slices((x_tfidf_test,y_tfidf_test))

# %%
# Build the forward-feeding network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

#%%
model1 = tf.keras.Sequential([
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# %%
model2 = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# %%
# Compile the model
model2.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(0.1),
              metrics=['accuracy'])

#%%
# Fit the model
history = model2.fit(train_set, epochs=5,
                    validation_data=test_set, 
                    validation_steps=30)
# %%
# Test the model
test_loss, test_acc = model.evaluate(test_set)

print('Test Loss: {}'.format(test_loss)) #0.0048
print('Test Accuracy: {}'.format(test_acc)) #0.722783971
