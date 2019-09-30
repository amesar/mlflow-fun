# Databricks notebook source
# MAGIC %md #Automatic logging from TensorFlow to MLflow
# MAGIC 
# MAGIC Blog post:
# MAGIC * [MLflow, TensorFlow, and an Open Source Show](https://databricks.com/blog/2019/08/19/mlflow-tensorflow-open-source-show.html) - Automatic logging from Keras and TensorFlow - 2019-08-19
# MAGIC 
# MAGIC This notebook demonstrates how to automatically log metrics, params, and models to MLflow from a TensorFlow program by calling a single function (`mlflow.tensorflow.autolog()`)
# MAGIC 
# MAGIC The code is lightly adapted from the [Tensorflow IMDB review classification example](https://www.tensorflow.org/tutorials/keras/basic_text_classification), which uses a `tf.Keras` model.

# COMMAND ----------

import tensorflow as tf
tf.VERSION

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enable autologging
# MAGIC 
# MAGIC To enable autologging, we simply import `mlflow.tensorflow` and call `mlflow.tensorflow.autolog()`.

# COMMAND ----------

import tensorflow as tf
from tensorflow import keras

import numpy as np
import mlflow
import mlflow.tensorflow

mlflow.tensorflow.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Preparing data
# MAGIC We'll use the IMDB review dataset included in TensorFlow. It consists of sequences of words (reviews) converted to integer sequences where each integer is the index of a word in a word dictionary.
# MAGIC We first download the dataset, then convert the integer sequences into tensors.

# COMMAND ----------

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
  
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Building model
# MAGIC These function calls build the model we'll be training.

# COMMAND ----------

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.compat.v1.train.AdamOptimizer(0.001),
              loss='binary_crossentropy',
              metrics=['acc'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Viewing logs
# MAGIC In the runs sidebar, you'll see a run corresponding to the TensorFlow training session. Post-epoch loss and accuracy have been logged, along with optimizer-related parameters and the model summary.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://i.imgur.com/97AOn4T.png"/>

# COMMAND ----------

# MAGIC %md
# MAGIC We can see, for example, post-epoch accuracy logged every two steps, as specified at the beginning:

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://i.imgur.com/TF7Wb6R.png"/>