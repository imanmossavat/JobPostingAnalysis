import streamlit as st
import pandas as pd
from managers import Word_Clouds_Manager
import os

# Streamlit app setup
st.title('WordCloud Generator for Job Descriptions')

# Input for output folder
output_folder = st.text_input("Enter the output folder path for the WordClouds", 'wordcloud_images')

if st.button('Generate Word Clouds'):
    try:
        # Call the Word_Clouds_Manager
        Word_Clouds_Manager(output_folder)
        
        # Display success message
        st.success(f"Wordclouds successfully extracted to reports/{output_folder}")
    except Exception as e:
        # Display error message
        st.error(f"An error occurred while generating WordClouds: {e}")