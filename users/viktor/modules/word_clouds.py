import pandas as pd
import json
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict
from interfaces import IWordCloudGenerator
from .text_preprocessor import TextPreprocessor

class WordCloudGenerator(IWordCloudGenerator):
    def generate_wordcloud_for_topic(
        self, 
        df: pd.DataFrame, 
        keyword_dict: Dict[str, List[str]], 
        output_folder: str, 
        name_of_topics: str, 
        stopword_files: List[str],
        column: str = 'Description'
    ) -> List[str]:
        """
        Generate WordCloud images for each topic based on keyword frequencies in job descriptions.
        
        Parameters:
        df (pd.DataFrame): DataFrame containing job descriptions and extracted features.
        keyword_dict (dict): Dictionary where keys are topic names and values are lists of associated keywords.
        output_folder (str): Folder where the generated WordCloud images will be saved.
        name_of_topics (str): Common title describing the WordCloud topics.
        stopword_files (list): List of paths to stopword files for additional filtering.
        column (str): Name of the column containing job descriptions in `df` to search for keywords.
        
        Returns:
        list: Paths to the generated WordCloud images.
        """
        # Initialize the TextPreprocessor
        text_preprocessor = TextPreprocessor(stopword_files)

        # Preprocess the text in the specified column
        df[column] = text_preprocessor.preprocess(df[column])

        os.makedirs(output_folder, exist_ok=True)

        topic_groups = list(keyword_dict.items())
        group_size = 20
        topic_sublists = [topic_groups[i:i + group_size] for i in range(0, len(topic_groups), group_size)]
        image_paths = []

        for group_idx, topic_group in enumerate(topic_sublists):
            n_topics = len(topic_group)
            n_cols = 3
            n_rows = (n_topics + n_cols - 1) // n_cols
            fig_width = 6 * n_cols
            fig_height = 6 * n_rows + 2  # Added height for the common title

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            axes = axes.flatten()

            # Add common title
            fig.suptitle(
                f"WordClouds for {name_of_topics} - Group {group_idx + 1}",
                fontsize=24, fontweight='bold', y=0.98  # Adjusted y position for more space
            )

            for i, (topic, keywords) in enumerate(topic_group):
                keyword_frequency = {}

                for text in df[column]:
                    for keyword in keywords:
                        keyword = keyword.lower()
                        if keyword in text.lower():
                            keyword_frequency[keyword] = keyword_frequency.get(keyword, 0) + 1

                ax = axes[i]
                if keyword_frequency:
                    wordcloud = WordCloud(
                        width=1000,
                        height=600,
                        background_color='white',
                        colormap='viridis'
                    ).generate_from_frequencies(keyword_frequency)
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f"{topic}", fontsize=18, pad=20)  # Increased padding for more space above title
                else:
                    ax.axis('off')
                    ax.set_title(f"No data for {topic}", fontsize=18, pad=20)  # Increased padding for more space above title

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Adjust the layout to provide more space between the title and the subplots
            plt.subplots_adjust(top=0.92)  # Adjusted to move everything lower

            # Generate a unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"wordcloud_group_{group_idx + 1}_{timestamp}.png"
            filepath = os.path.join(output_folder, filename)
            plt.savefig(filepath)
            image_paths.append(filepath)
            plt.close()

        return image_paths