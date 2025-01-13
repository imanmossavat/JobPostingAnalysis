import streamlit as st
import os

# Import the Topic_Modeling_Manager function
from managers import Topic_Modeling_Manager

from config import Config

# Default values from config
n_topics, num_top_words, epochs = Config().topic_modeling_input_variables()

# Streamlit app setup
st.title('Topic Modeling')

# Input for selecting project and subfolder
base_dir = os.path.join('data', 'registry')
projects = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Input parameters
n_topics = st.number_input("Number of Topics", min_value=2, value=n_topics)
num_top_words = st.number_input("Top Words per Topic", min_value=5, max_value=20, value=num_top_words)
epochs = st.number_input("Number of Iterations", min_value=10, max_value=500, value=epochs)

if projects:
    selected_project = st.selectbox("Select a Project", projects)
    project_path = os.path.join(base_dir, selected_project)

    subfolders = [
        os.path.relpath(os.path.join(dp, f), project_path)
        for dp, dn, filenames in os.walk(project_path)
        for f in filenames if f.endswith('.csv')
    ]
    valid_subfolders = [os.path.dirname(f) for f in subfolders]

    if valid_subfolders:
        selected_subfolder = st.selectbox("Select a Subfolder", sorted(set(valid_subfolders)))
        folder_path = os.path.join(project_path, selected_subfolder)

        output_folder = st.text_input("Enter the output folder path for the Topic Modeling", 'output_topics')

        # Call the Topic_Modeling_Manager function
        topic_modeling_manager = Topic_Modeling_Manager(folder_path, output_folder, n_topics, num_top_words, epochs)


        if st.button('Generate Topics'):
            try:
                topic_modeling_manager.main()
                # Display success message
                st.success(f"Topics successfully generated using data from: {folder_path}")
            except Exception as e:
                # Display error message
                st.error(f"An error occurred while generating topics: {e}")
    else:
        st.warning("No valid subfolders with CSV files found in the selected project.")
else:
    st.warning("No projects found in the registry.")