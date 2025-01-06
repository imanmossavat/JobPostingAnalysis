import streamlit as st
from managers import Topic_Modeling_Manager
# from config import topic_modeling_input_variables

from config import Config

# Default values from config
n_topics, num_top_words, epochs = Config().topic_modeling_input_variables()

# Streamlit app
st.title("NMF Topic Modeling")

# Input parameters
n_topics = st.number_input("Number of Topics", min_value=2, max_value=20, value=n_topics)
num_top_words = st.number_input("Top Words per Topic", min_value=5, max_value=20, value=num_top_words)
epochs = st.number_input("Number of Iterations", min_value=10, max_value=500, value=epochs)
output_folder = st.text_input("Output Folder", value="output_topics")

if st.button("Run"):
    # Run topic modeling and handle potential errors
    try:
        topics = Topic_Modeling_Manager(output_folder, n_topics, num_top_words, epochs)
        
        if topics is None:
            st.error("An error occurred during topic modeling. Please check the input parameters and dataset.")
        else:
            st.write(f"Topics have been successfully extracted and saved to 'reports/{output_folder}'")
    
    except Exception as e:
        # General error handling
        st.error(f"An unexpected error occurred: {e}")