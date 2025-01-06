# Streamlit application to upload and display a dataset
import streamlit as st
import pandas as pd

def main():
    # Title and description
    st.title("Dataset Viewer")
    st.write("Upload a CSV or Excel file to view its contents in a structured way.")

    # File upload section
    uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Determine file type and load dataset
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            # Display dataset statistics
            st.write("### Dataset Overview")
            st.dataframe(df)  # Display data table

            st.write("### Summary Statistics")
            st.write(df.describe())  # Display summary statistics

            st.write("### Dataset Shape")
            st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

            # Optionally, include a column selection filter
            st.write("### Column Selector")
            selected_columns = st.multiselect(
                "Select columns to display",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )

            if selected_columns:
                st.write("### Filtered Data")
                st.dataframe(df[selected_columns])

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

    else:
        st.info("Awaiting file upload. Please upload a file to proceed.")

if __name__ == "__main__":
    main()