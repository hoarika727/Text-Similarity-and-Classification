#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# %%
train = pd.read_csv('../data/train.csv', sep=',', index_col='id')

train_feature = train['title1_en']+train['title2_en']
train_feature = [row.lower() for row in train_feature]

# %%
x = np.array(train_feature).reshape(-1,1)
y = train['label']
y = [0 if i == 'unrelated' else (1 if i =='disagreed' else 2) for i in y]
y = np.array(y, dtype=np.int64).reshape(-1,1)
dataset = tf.data.Dataset.from_tensor_slices((x,y))

# reference: TensorFlow - text classification sample
# https://www.tensorflow.org/tutorials/text/text_classification_rnn
BUFFER_SIZE = 50000
BATCH_SIZE = 64
VALIDATION_SIZE = int(len(x)*0.2)

train_dataset = dataset.skip(VALIDATION_SIZE).shuffle(BATCH_SIZE)
test_dataset = dataset.take(VALIDATION_SIZE)

VOCAB_SIZE=50000
encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

#%%
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#%%
model1 = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

# %%
model2 = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# %%
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

#%%
history = model.fit(train_dataset, epochs=10, #num_epochs is limited by computing power
                    validation_data=test_dataset, 
                    validation_steps=30)
# %%
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss)) #0.0
print('Test Accuracy: {}'.format(test_acc)) #68.819530245