import streamlit as st
import os

# Import the Word_Clouds_Manager function
from managers import Word_Clouds_Manager, get_json_files_for_word_clouds

# Streamlit app setup
st.title('Word Cloud Plot Generator for Keyword Features')

# Input for selecting project and subfolder
base_dir = os.path.join('data', 'registry')
projects = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

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

        output_folder = st.text_input("Enter the output folder path for the Word Cloud Plots", 'word_cloud_images')

        # Call the Word_Clouds_Manager function
        word_cloud_manager = Word_Clouds_Manager(folder_path, output_folder)

        files = get_json_files_for_word_clouds()  # Get the list of files

        # Add dropdown for selecting files
        if files:
            selected_file = st.selectbox("Select a JSON File", files)
            
            # Send the selected file name to set_json_file_name
            if selected_file:
                word_cloud_manager.set_json_file_name(f'{selected_file}.json')  # Set the file name using the function
        else:
            selected_file = None
            st.warning("No JSON files available for Word Cloud plots.")

        if st.button('Generate Word Cloud Plots'):
            try:
                if selected_file:
                    word_cloud_manager.main()
                    # Display success message
                    st.success(f"Word Cloud plots successfully generated using data from: {folder_path}")
                else:
                    st.warning("Please select a valid JSON file.")
            except Exception as e:
                # Display error message
                st.error(f"An error occurred while generating Word Cloud Plots: {e}")
    else:
        st.warning("No valid subfolders with CSV files found in the selected project.")
else:
    st.warning("No projects found in the registry.")