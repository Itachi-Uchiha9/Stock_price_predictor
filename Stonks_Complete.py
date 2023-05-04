# -*- coding: utf-8 -*-
"""


@author: Team Stonks
"""

import numpy as np

import math
from sklearn.preprocessing import MinMaxScaler
from stonks_open import *
from pathlib import PurePath
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                                                                               #data preprocess
def Close_predict(csv_file):                                                   #reading csv file
    df=pd.read_csv(csv_file)
    df['Index']=df.index
    df.replace(0,np.nan)
    df.dropna(how='all',axis=0)
    df=df[['Index','Open','Close','Volume']]
    
    

    X=df.loc[:,['Open']].values                                                # x is open price - Independent variable
    Y=df.loc[:,['Close']].values                                               # y is close price - Target Variable

    scaler = MinMaxScaler(feature_range=(0,1))                                 #data transformation in values between 0 to 1
    X1 = scaler.fit_transform(X)
    Y1=scaler.fit_transform(Y)

    from sklearn.model_selection import train_test_split                       #data splitting in test and train datasets
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1,
                                                        test_size = 0.1)
    X_train= np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))      #reshaping X train dataset into desired array
    X_test=np.reshape(X_test,(X_test.shape[0], X_test.shape[1], 1))            #reshaping X test dataset into desired array

    print('Stonks_open running')
    value,S_open,graph=process_data(csv_file)                                  #initaillizing open prediction model
    
                                                                               #Model train
    print('Stonks_Close running')
    from keras.models import Sequential                                        #close price model train
    from keras.layers import Dense, LSTM,Activation,Dropout                    

    model = Sequential()                                                       #function for combining layers of LSTM
    model.add(LSTM(units=90, return_sequences=True, 
                   input_shape=(X_train.shape[1],1)))                          #LSTM Input Gate
    model.add(LSTM(units=50,return_sequences=True))  
    model.add(LSTM(units=40))                                              #Hidden Layer
    model.add(Dense(1))                                                        #Dense Layer for adjusting bais and weights      

    model.compile(optimizer='adam',loss='mse')                                 #compiling model
    model.fit(X_train, y_train, batch_size=5,epochs=4 ,verbose=2)              #model training dataset

                                                                               #Predict
    predictions = model.predict(X_test)                                        #predicting on X test 
    predictions = scaler.inverse_transform(predictions)                          
    Y_test=scaler.inverse_transform(y_test)                                 
    value=np.reshape(value,(value.shape[0], value.shape[0], 1))                #predicted open price of stock
    v=model.predict(value)                                                     #predicting close price from open
    
    S_close=scaler.inverse_transform(v)                                        
    return S_open[0],S_close[0]                             #returning values 
    
    

