import streamlit as st

# Import the Box_Plots_Manager function
from managers import Box_Plots_Manager

# Streamlit app setup
st.title('Box Plot Generator for Keyword Features')

# Input for output folder
output_folder = st.text_input("Enter the output folder path for the Box Plots", 'boxplot_images')

if st.button('Generate Box Plots'):
    
    try:
        # Call the Box_Plots_Manager function
        Box_Plots_Manager(output_folder)
        
        # Display success message
        st.success(f"Box plots successfully generated in: {output_folder}")
    except Exception as e:
        # Display error message
        st.error(f"An error occurred while generating Box Plots: {e}")