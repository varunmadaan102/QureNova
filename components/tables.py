import streamlit as st

def dataframe(df):
    st.dataframe(df, width="stretch", hide_index=True)
