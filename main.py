import streamlit as st
import pandas as pd
import numpy as np
import os
from pycaret.classification import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)


with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Ask Your Data")
    chioce = st.radio("Navigation" , ["Upload" , "Profiling" , "Machine Learning" , "Download output"])
    st.info("This helps you to know about your data in a single click!")

if(chioce == "Upload"):
    st.title("Upload your dataset . . ")
    file = st.file_uploader("Upload your dataset")
    if file:
        df = pd.read_csv(file , index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if(chioce == "Profiling"):
    st.title("Profiling your dataset")
    report = ProfileReport(df)
    st_profile_report(report)

if(chioce == "Machine Learning"):
    st.title("Machile Learning")
    select = st.selectbox("Select your target variable .." , df.columns)
    if st.button("Run Modeling"):
        df = df.dropna()
        for col in df.columns:
            if df[col].dtype == 'category':
                df[col] = df[col].cat.codes
        st.dataframe(df)
    
        


    

