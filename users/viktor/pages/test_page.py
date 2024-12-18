import streamlit as st
import pandas as pd
import os

# Function to change column name if "Id" exists
def rename_column(df):
    if 'Id' in df.columns:
        df.rename(columns={'Id': 'job_id'}, inplace=True)
    return df

# Function to save the dataframe to a specific location
def save_file(df, file_path):
    df.to_csv(file_path, index=False)
    st.success(f"File saved successfully to {file_path}")

# Streamlit page layout
def main():
    st.title("Dataset Upload and Save")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        # Load the dataset
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        # Show the first few rows of the dataset
        st.write("Preview of the dataset:")
        st.dataframe(df.head())
        
        # Apply column renaming if needed
        df = rename_column(df)
        
        # Display the updated dataframe
        st.write("Updated dataset:")
        st.dataframe(df.head())
        
        # Choose where to save the file
        save_path = st.text_input("Enter the path where you want to save the file (including filename):", "updated_dataset.csv")
        
        if st.button("Save Dataset"):
            if save_path:
                save_file(df, save_path)
            else:
                st.error("Please provide a valid path to save the file.")

if __name__ == "__main__":
    main()