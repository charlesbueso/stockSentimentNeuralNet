import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import io
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers

def processData():

  #download dataframe
  df = pd.read_csv('Combined_News_DJIA.csv')
  #preprocessing
  #Remove punctuation
  data = df.iloc[:,2:27]
  data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
  #Rename column name for ease of access, and lowercase words
  list1=[i for i in range(25)]
  new_Index=[str(i) for i in list1]
  data.columns= new_Index
  for index in new_Index:
    data[index]=data[index].str.lower()

  X=[]
  for i in range(0,len(data.index)):
    X.append(' '.join(str(x) for x in data.iloc[i,0:25]))


  #Splits data into 70% training, 15% testing, 15% validation
  X_train, X_test, y_train, y_test = train_test_split(X, df.Label, test_size=0.15, random_state=1)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1275, random_state=1)

  return X_train, y_train, X_test, y_test, X_val, y_val, df


def stockNeuralNet():

  #processdata
  X_train, y_train, X_test, y_test, X_val, y_val, df = processData()

  tokenizer = Tokenizer(num_words=5000)
  tokenizer.fit_on_texts(X_train)

  X_train = tokenizer.texts_to_sequences(X_train)
  X_test = tokenizer.texts_to_sequences(X_test)

  vocab_size = len(tokenizer.word_index) + 1

  #padding
  maxlen = 500
  X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
  X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

  #model
  embedding_dim = 50

  model = Sequential()
  model.add(layers.Embedding(input_dim=vocab_size, 
                           output_dim=embedding_dim, 
                           input_length=maxlen))
  model.add(layers.Flatten())
  model.add(layers.Dense(10, activation='sigmoid'))
  model.add(layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
  #training
  history = model.fit(X_train, y_train,
                    epochs=10,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)
  #testing
  loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
  print("Testing Accuracy for the StockSentiment Prediction model:  {:.4f}".format(accuracy*100))

stockNeuralNet()