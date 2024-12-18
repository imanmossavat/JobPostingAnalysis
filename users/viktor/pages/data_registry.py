import streamlit as st
from managers import DataRegistryManager

st.title("Dataset Registry Manager")

# Section: Save Dataset
st.header("Save a Dataset")

data_registry_manager_example = DataRegistryManager("", "", "")

# List existing projects and allow new project input
existing_projects = data_registry_manager_example.get_existing_projects()
project_name = st.selectbox("Select an existing project", options=[""] + existing_projects)

if project_name == "":
    project_name = st.text_input("Or enter a new project name")

# File uploader
dataset = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])

# File name input
file_name = st.text_input("Enter the file name (optional)", value=dataset.name if dataset else "")

data_registry_manager = DataRegistryManager(dataset, file_name, project_name)

if st.button("Save Dataset"):
    if not dataset:
        st.error("Please upload a dataset.")
    elif not project_name.strip():
        st.error("Please enter a valid project name.")
    else:
        message = data_registry_manager.save_dataset(dataset, project_name)
        if "Error" in message:
            st.error(message)
        else:
            st.success(message)

# Section: Remove Dataset
st.header("Remove a Dataset")

if existing_projects:
    project_to_remove = st.selectbox("Select a project", options=existing_projects)
    if project_to_remove:
        datasets_in_project = data_registry_manager.get_datasets_in_project(project_to_remove)
        dataset_to_remove = st.selectbox("Select a dataset to remove", options=[""] + datasets_in_project)

        if st.button("Remove Dataset") and dataset_to_remove:
            message = data_registry_manager.remove_dataset(project_to_remove, dataset_to_remove)
            if "Error" in message:
                st.error(message)
            else:
                st.success(message)
else:
    st.info("No existing projects found.")