import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def ProcessDataset():
    #Import data and split into 70%, 15%, 15%
    df= pd.read_csv('Combined_News_DJIA.csv',encoding='ISO-8859-1')
    train = df.iloc[:1392,:]
    development = df.iloc[1393:1690]
    test=  df.iloc[1690:,:]

    #Stop Words hashmap
    with open("stopWords.txt", "r") as f:
        stopWords = {k:v for k, *v in map(str.split, f)}  

    ##Data normalization for all 3 datasets
    #Remove punctuation TRAINING
    data= train.iloc[:,2:27]
    data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
    #Rename column name for ease of access, and lowercase words TRAINING
    list1=[i for i in range(25)]
    new_Index=[str(i) for i in list1]
    data.columns= new_Index
    for index in new_Index:
        data[index]=data[index].str.lower()

    #Remove punctuation DEVELOPMENT
    data = development.iloc[:,2:27]
    data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
    #Rename column name for ease of access, and lowercase words DEVELOPMENT
    list1=[i for i in range(25)]
    new_Index=[str(i) for i in list1]
    data.columns= new_Index
    for index in new_Index:
        data[index]=data[index].str.lower()

    #Remove punctuation TESTING
    data = test.iloc[:,2:27]
    data.replace("[^a-zA-Z]"," ",regex=True, inplace=True)
    #Rename column name for ease of access, and lowercase words TESTING
    list1=[i for i in range(25)]
    new_Index=[str(i) for i in list1]
    data.columns= new_Index
    for index in new_Index:
        data[index]=data[index].str.lower()
    
    return data

def buildKerasModel(data):
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


if __name__ == "__main__":

    processedData = ProcessDataset()
    print(processedData)
    model = buildKerasModel(processedData)

