# -*- coding: utf-8 -*-
"""


@author: Team Stonks
"""

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
import os
from keras.models import Sequential                                        #model train for open prediction
from keras.layers import Dense, LSTM,Activation,Dropout   
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                                                                               #data preprocess
def process_data(csv_file):                                                    #read csv
    df=pd.read_csv(csv_file)                      
    df['Index']=df.index
    df.replace(0,np.nan)                                                       #data cleansing
    df.dropna(how='all',axis=0)  
    df=df[['Index','Prev Close','Open','Close','Volume']]
    
    

    X=df.loc[:,['Prev Close']].values                                          #taking prev close as X Variable
    Y=df.loc[:,['Open']].values
    Z=df.loc[:,['Close']].values
    scaler = MinMaxScaler(feature_range=(0,1))                                 #data trasformation
    X1 = scaler.fit_transform(X)                                               
    Y1=scaler.fit_transform(Y)
    Z1=scaler.fit_transform(Z)
    from sklearn.model_selection import train_test_split                       #splitting data between train and test
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1,
                                                        test_size = 0.1) 
    X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))      #reshaping X train into desired array
    X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))            #reshaping X test into desired array

                                                                               #model train
                        
    model = Sequential()                                                       #it provides training and inference features on this model
    model.add(LSTM(units=80, return_sequences=True, 
                   input_shape=(X_train.shape[1],1)))                          #adding LSTM layer
    model.add(LSTM(units=50,return_sequences=True))   
    model.add(LSTM(units=40)) 
    model.add(Dense(1))

    model.compile(optimizer='adam',loss='mse')                                 #compiling model
    model.fit(X_train, y_train, epochs=4, batch_size=5, verbose=2)             #training model


    Value=Z1[-1]                                                               #taking previous close for prediction
   
    Value=np.reshape(Value,(Value.shape[0], Value.shape[0], 1))                #reshaping prediction from 1d array to 3d array
                                                                               #Predict
    predictions = model.predict(X_test)                                        #prediction value of X test
    predictions = scaler.inverse_transform(predictions)                        
    Y_test=scaler.inverse_transform(y_test)

    v=model.predict(Value)
    value = scaler.inverse_transform(v)
    
    return v,value,predictions



