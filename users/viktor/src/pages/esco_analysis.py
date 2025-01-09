import streamlit as st
import os
import pandas as pd
from managers import ESCOManager
from io import StringIO

# Streamlit app setup
st.title("ESCO Skills and Knowledge Extractor")

# File uploader for input CSV
uploaded_file = st.file_uploader("Choose an input CSV file", type="csv")

if uploaded_file is not None:
    # Convert the uploaded file to a DataFrame to display its content
    df = pd.read_csv(uploaded_file)
    st.write(df.head())  # Display the first few rows of the uploaded file

    # Input fields for output file and output subfolder
    # output_file = st.text_input("Enter the path for the output CSV file", "output_data.csv")
    # output_subfolder = st.text_input("Enter the path for the output subfolder", "output_reports")

    if st.button("Run ESCO Analysis"):
        try:
            # Save the uploaded file temporarily for processing
            input_file_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(input_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_subfolder = "esco_subfolder"
            # Create an instance of the manager
            manager = ESCOManager(input_file_path, output_subfolder)
            result_message = manager.run_analysis()

            # Display success message
            st.success(result_message)

            ''' # Optionally, allow users to download the output CSV
            st.download_button(
                label="Download Output CSV",
                data=open(output_file, "rb").read(),
                file_name=output_file,
                mime="text/csv"
            ) '''

        except Exception as e:
            # Display error message
            st.error(f"An error occurred: {e}")
else:
    st.warning("Please upload a CSV file to begin.")