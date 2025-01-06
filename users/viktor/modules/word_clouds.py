import pandas as pd
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
from interfaces import IWordCloudGenerator
from .text_preprocessor import TextPreprocessor

class WordCloudGenerator(IWordCloudGenerator):
    def __init__(self, df: pd.DataFrame, keyword_dict: Dict[str, List[str]], output_folder: str, name_of_topics: str, 
                 stopword_files: List[str], column: str = 'Description'):
        """
        Initialize the WordCloudGenerator.

        Parameters:
        df (pd.DataFrame): DataFrame containing job descriptions and other relevant data.
        keyword_dict (dict): Dictionary where keys are topic names and values are lists of associated keywords.
        output_folder (str): Folder where the generated WordCloud images will be saved.
        name_of_topics (str): Common title for the WordCloud topics.
        stopword_files (list): List of paths to stopword files for text preprocessing.
        column (str): Name of the column containing job descriptions in the DataFrame.
        """
        self.df = df
        self.keyword_dict = keyword_dict
        self.output_folder = output_folder
        self.name_of_topics = name_of_topics
        self.stopword_files = stopword_files
        self.column = column

    def generate_wordcloud_for_topic(self) -> List[str]:
        """
        Generate WordCloud images for each topic based on keyword frequencies in job descriptions.

        Returns:
        list: Paths to the generated WordCloud images.
        """
        # Initialize the TextPreprocessor
        text_preprocessor = TextPreprocessor(self.stopword_files)

        # Preprocess the text in the specified column
        if self.df.empty or self.df[self.column].dropna().empty:
            print("No descriptions available in the DataFrame. Skipping WordCloud generation.")
            return []

        self.df[self.column] = text_preprocessor.preprocess(self.df[self.column])

        # Ensure the output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Divide topics into manageable subgroups
        topic_groups = list(self.keyword_dict.items())
        group_size = 20
        topic_sublists = [topic_groups[i:i + group_size] for i in range(0, len(topic_groups), group_size)]
        image_paths = []

        for group_idx, topic_group in enumerate(topic_sublists):
            n_topics = len(topic_group)
            n_cols = 3
            n_rows = (n_topics + n_cols - 1) // n_cols
            fig_width = 6 * n_cols
            fig_height = 6 * n_rows + 2  # Added height for the common title

            # Create a figure with subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            axes = axes.flatten()

            # Add a common title to the figure
            fig.suptitle(
                f"WordClouds for {self.name_of_topics} - Group {group_idx + 1}",
                fontsize=24, fontweight='bold', y=0.98  # Adjusted y position for more space
            )

            subplot_idx = 0  # Track the number of used subplots

            for topic, keywords in topic_group:
                if not keywords:
                    continue  # Skip topics with an empty keyword list

                keyword_frequency = {}

                # Calculate keyword frequencies
                for text in self.df[self.column]:
                    for keyword in keywords:
                        keyword = keyword.lower()
                        if keyword in text.lower():
                            keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1

                if not keyword_frequency:
                    continue  # Skip topics without data

                ax = axes[subplot_idx]
                subplot_idx += 1

                # Generate the WordCloud
                wordcloud = WordCloud(
                    width=1000,
                    height=600,
                    background_color='white',
                    colormap='viridis'
                ).generate_from_frequencies(keyword_frequency)

                # Display the WordCloud
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title(f"{topic}", fontsize=18, pad=20)  # Increased padding for more space above title

            # Remove any unused subplots
            for j in range(subplot_idx, len(axes)):
                fig.delaxes(axes[j])

            # Adjust layout for better spacing
            if subplot_idx > 0:  # Only save if at least one subplot was used
                plt.subplots_adjust(top=0.92)  # Adjusted to move everything lower

                # Generate a unique filename with a timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"wordcloud_group_{group_idx + 1}_{timestamp}.png"
                filepath = os.path.join(self.output_folder, filename)

                # Save the figure and close it
                plt.savefig(filepath)
                image_paths.append(filepath)

            plt.close()

        return image_paths