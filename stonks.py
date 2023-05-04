"""
@author: Team Stonks
"""
import streamlit as st
from Stonks_Complete import *
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import math
import base64
from sentimental_analysis import *

def lineplot(df,value,title):                                                  #function for creating graph
    fig = px.line(                                                             #initiate line graph                                 
        df,                                                                    #Data Frame
        x = df['Date'],                                                        #Columns from the data frame
        y = df[value],
        title=title                                                            #csv file Name
           
    )
    fig.update_layout(                                                         #graph layout
            width=1400,
            height=650,
            paper_bgcolor="white",)
    fig.update_traces(line_color = "Coral")                                    #display text color on graph
    st.plotly_chart(fig)                                                       #to plot graph
   
def sentiplot(df,title):
   fig = px.histogram(       
        df,                                                                    #Data Frame
        x = df['date'],                                                        #Columns from the data frame
        y = df['Probability'],
        color=df['sentiment'],
        title=title
           
    )
   fig.update_layout(                                                          #graph layout
            width=1400,
            height=650,
            paper_bgcolor="white",)
   
   fig.update_layout(barmode='group')                                          #to transform bar graph into readable form
   st.plotly_chart(fig)
    
def main():       
                                                                              
    st.set_page_config('STONKS','LOGO_stonks.png','wide')                      # front end elements of the web page
                                                                               #for stonks logo
    main_bg = "OIP.webp"                                                       #background page
    main_bg_ext = "webp"                                                       #file extension


    st.markdown(                                                                #html script for background image
    f"""
   <style>
	.reportview-container{{
		background-image:url(./OIP.webp)
	}}
	}}
    </style>
    """,
    unsafe_allow_html=True
    )
       
    st.image('stonks_logo.png',width=250)                                       #logo file
  
    menu = [                                                           #companies names for prediction
'GAIL','BPCL','ITC','NTPC','HTC']
    
    menu.sort()                                                                #Sort in alphabetical order
    choice = st.sidebar.selectbox("List of Companies",menu)
    
    if st.sidebar.button("Prediction using historical data"):                   #Only Historical data prediction
        Open,Close=Close_predict(choice+'.csv')                                #initializing prediction module
        
        st.write("The Open Price of Stock is:",Open[0])
        st.write("The Close Price Of Stock is:",Close[0])
        
        df=pd.read_csv(choice+'.csv')
        df.replace(0,np.nan)
        df.dropna(how='all',axis=0)
        df=df[['Date','Open','Close','Volume',]] 
        data_10=df.tail(10)
        
        st.dataframe(df.tail(10)) 
        st.write('Last 10 days stock price data of ',choice) 
        st.write('Last 10 days maximum:',max(data_10['Close']))
        st.write('Last 10 days minimum:',min(data_10['Close']))  
                                                 #print recent 10 values from csv
        st.write(" ")
        st.write("Open graph of ",choice,':')
        open_title="Open graph of "+choice+':'
        lineplot(df,'Open',open_title)
        st.write(" ")
        st.write("Close graph of ",choice,':')
        close_title='Close Graph of '+choice+':'
        
        lineplot(df,'Close',close_title)
        
        
    if st.sidebar.button("Public Sentiments for Company"):                     #Only sentimental Analysis 
        st.write("Sentimental analysis of ",choice,':')
        
        analysis_title='sentimental analysis of '+choice+':'
        
        DF=Senti_analyze(choice)                                               #initializing sentimental analysis module
        
        sentiplot(DF, analysis_title)
        
        st.write('*Disclaimer :This analysis is done on tweets recently posted about the company',choice,' and are subjected to change')
        st.write('''         *  Public perception can change spontaneously positively or negatively in case of company quarterly result, news, war scenario,
                 government policies, election prediction or outcomes ,natural or man made disaster, any unpredictable event occurance.''')
    
    if st.sidebar.button("Prediction using both"):                             #Prediction and Analysis both
        
        Open,Close=Close_predict(choice+'.csv')
        
        st.write("The Open Price of Stock is:",Open[0])
        st.write("The Close Price Of Stock is:",Close[0])
        
        df=pd.read_csv(choice+'.csv')
        df.replace(0,np.nan)
        df.dropna(how='all',axis=0)
        df=df[['Date','Open','Close','Volume',]]        
        st.dataframe(df.tail(10))  
        data_10=df.tail(10)
        st.write('Last 10 days stock price data of ',choice) 
        st.write('Last 10 days maximum:',max(data_10['Close']))
        st.write('Last 10 days minimum:',min(data_10['Close']))                                             #print recent 10 values from csv
        st.write(" ")
        st.write("Open graph of ",choice,':')
        open_title="Open graph of "+choice+':'
        lineplot(df,'Open',open_title)
        st.write(" ")
        st.write("Close graph of ",choice,':')
        close_title='Close Graph of '+choice+':'
        
        lineplot(df,'Close',close_title)
        
        st.write("Sentimental analysis of ",choice,':')
        
        analysis_title='sentimental analysis of '+choice+':'
        
        DF=Senti_analyze(choice)                                               #Sentimental analysis module

        sentiplot(DF, analysis_title)
        st.write('*Disclaimer :This analysis is done on tweets recently posted about the company',choice,' and are subjected to change')
        st.write('''         *  Public perception can change spontaneously positively or negatively in case of company quarterly result, news, war scenario,
                 government policies, election prediction or outcomes ,natural or man made disaster, any unpredictable event occurance.''')
    
        
        
 
if __name__=='__main__': 
    main()