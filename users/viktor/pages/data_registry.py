import streamlit as st
from managers import DataRegistryManager

st.title("Dataset and Database Registry Manager")

# Section: Save Dataset or Database Connection
st.header("Save a Dataset or Database Connection")

data_registry_manager_example = DataRegistryManager("", "", "")

existing_projects = data_registry_manager_example.get_existing_projects()
project_name = st.selectbox("Select an existing project", options=[""] + existing_projects)

if project_name == "":
    project_name = st.text_input("Or enter a new project name")

# Option to choose dataset or database
input_type = st.radio("Choose input type", options=["Dataset File", "Database Connection"])

# Handling Dataset Files
if input_type == "Dataset File":
    dataset = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "json"])
    file_name = st.text_input("Enter the file name (optional)", value=dataset.name if dataset else "")
    dataset_type = None

    if dataset:
        # Determine dataset type based on file extension
        file_extension = dataset.name.split(".")[-1].lower()
        if file_extension in ["csv", "xlsx", "json"]:
            dataset_type = file_extension
        else:
            st.warning("Unsupported file type. Please upload a CSV, Excel, or JSON file.")

    if st.button("Save Dataset"):
        if not dataset:
            st.error("Please upload a dataset.")
        elif not project_name.strip():
            st.error("Please enter a valid project name.")
        elif not dataset_type:
            st.error("Unable to determine dataset type.")
        else:
            data_registry_manager = DataRegistryManager(dataset, file_name, project_name)
            # message = data_registry_manager.save_dataset(dataset, project_name, dataset_type=dataset_type)
            message = data_registry_manager.save_dataset(dataset, project_name)
            if "Error" in message:
                st.error(message)
            else:
                st.success(message)

# Handling Database Connections
elif input_type == "Database Connection":
    db_type = st.selectbox("Database Type", options=["MySQL", "PostgreSQL", "SQLite", "MongoDB"])

    # Adjust host and port defaults based on database type
    if db_type == "MySQL":
        db_host = st.text_input("Database Host", value="127.0.0.1")
        db_port = st.text_input("Database Port", value="3306")
    elif db_type == "PostgreSQL":
        db_host = st.text_input("Database Host", value="127.0.0.1")
        db_port = st.text_input("Database Port", value="5432")
    elif db_type == "SQLite":
        db_host = st.text_input("Database Host (optional)", value="")
        db_port = st.text_input("Database Port (optional)", value="")
        st.info("SQLite doesn't require a host or port. Leave these fields blank if unnecessary.")
    elif db_type == "MongoDB":
        db_host = st.text_input("Database Host", value="127.0.0.1")
        db_port = st.text_input("Database Port", value="27017")

    db_name = st.text_input("Database Name")
    db_user = st.text_input("Username")
    db_password = st.text_input("Password", type="password")

    if st.button("Save Connection"):
        if not all([db_host, db_port, db_name, db_user, db_password]) and db_type != "SQLite":
            st.error("Please fill in all fields.")
        elif not project_name.strip():
            st.error("Please enter a valid project name.")
        else:
            # MongoDB connection string example
            if db_type == "MongoDB":
                connection_string = f"mongodb://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            else:
                connection_string = None

            db_connection = {
                "type": db_type,
                "host": db_host,
                "port": db_port,
                "name": db_name,
                "user": db_user,
                "password": db_password,
                "connection_string": connection_string  # Include for MongoDB
            }
            data_registry_manager = DataRegistryManager(None, "", project_name)
            message = data_registry_manager.save_dataset(None, project_name, db_connection=db_connection)
            if "Error" in message:
                st.error(message)
            else:
                st.success(message)

# Section: Remove Dataset
st.header("Remove a Dataset")

if existing_projects:
    project_to_remove = st.selectbox("Select a project", options=existing_projects)
    if project_to_remove:
        datasets_in_project = data_registry_manager_example.get_datasets_in_project(project_to_remove)
        dataset_to_remove = st.selectbox("Select a dataset to remove", options=[""] + datasets_in_project)

        if st.button("Remove Dataset") and dataset_to_remove:
            message = data_registry_manager_example.remove_dataset(project_to_remove, dataset_to_remove)
            if "Error" in message:
                st.error(message)
            else:
                st.success(message)
else:
    st.info("No existing projects found.")