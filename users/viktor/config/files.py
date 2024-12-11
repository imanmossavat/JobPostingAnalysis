import os

# File names to look for
file_names = ["stop_words_custom.txt", "stop_words_dutch.txt", "stop_words_english.txt"]

# Get the current directory
current_directory = os.getcwd()

# Find the full paths of the files in the current folder
stopword_file_names = [os.path.join(current_directory, file) for file in file_names if os.path.isfile(os.path.join(current_directory, file))]

# Define the path to the 'data' folder inside the current directory
data_folder = os.path.join(current_directory, 'data')

raw_data_name = 'raw'

# English dataset path
english_dataset = "output_file_english_20241114_113227.csv"
english_dataset_path = os.path.join(data_folder, raw_data_name, english_dataset)

# Dutch dataset path
dutch_dataset = "output_file_dutch_20241114_113227.csv"
dutch_dataset_path = os.path.join(data_folder, raw_data_name, dutch_dataset)

# Define the path to the 'config' folder inside the current directory
config_folder = os.path.join(current_directory, 'config')

# Industries and Sectors JSON
industries_and_sectors = os.path.join(config_folder, "Industries_and_Sectors.json")

# job titles clusters JSON
job_titles_clusters = os.path.join(config_folder, "Job_Titles_Clusters.json")

# path to subfolder reports
reports_folder_path = os.path.join(data_folder, "reports")