from interfaces import IWord2VecEmbeddingTrendAnalysis
import pandas as pd
import numpy as np
import re
import json
import os
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict
from nltk.util import ngrams
import ruptures as rpt

class Word2Vec_Embedding_Analysis(IWord2VecEmbeddingTrendAnalysis):
    """
    A class for analyzing trends in job descriptions by creating a Word2Vec embedding model
    and visualizing keyword trends over time using various techniques like Gaussian process regression and rupture detection.
    """

    def __init__(self, input_file, output_subfolder, keywords_list_file, stopwords_list):
        """
        Initializes the Word2Vec_Embedding_Analysis object with input data, output folder, keywords list, and stopwords.

        Args:
            input_file (str): Path to the CSV file containing job descriptions.
            output_subfolder (str): Directory where the output files and plots will be saved.
            keywords_list_file (str): Path to the JSON file containing the list of job keywords.
            stopwords_list (list): List of stopwords to remove during text processing.
        """
        self.input_file = input_file
        self.output_subfolder = output_subfolder

        os.makedirs(self.output_subfolder, exist_ok=True)

        # Load the keyword list
        with open(keywords_list_file, 'r') as file:
            self.job_keywords = json.load(file)
        self.job_keywords = [keyword.replace(" ", "_") for keyword in self.job_keywords]

        # Initialize trend data and stop words
        self.trends = defaultdict(list)
        self.stop_words = stopwords_list
        self.lemmatizer = WordNetLemmatizer()

        # Load dataset
        self.df = pd.read_csv(input_file)
        self.df['cleaned_text'] = self.df['Description'].apply(self.preprocess)

        # Initialize the Word2Vec model
        self.word2vec_model = None

    def preprocess(self, text: str) -> list:
        """
        Preprocesses the given job description text by converting it to lowercase, removing non-alphanumeric characters,
        lemmatizing words, and removing stopwords.

        Args:
            text (str): The text of the job description to be preprocessed.

        Returns:
            list: A list of preprocessed words from the text.
        """
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\d+', '', text)

        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]

        words = [word.replace(" ", "_") for word in words]

        # Check for multi-word keywords and replace
        for keyword in self.job_keywords:
            if " " in keyword:
                spaced_keyword = keyword.replace("_", " ")
                text = text.replace(spaced_keyword, keyword)

        words = text.split()

        # Identify bigrams
        bigrams = list(ngrams(words, 2))
        bigram_tokens = ['_'.join(bigram) for bigram in bigrams]

        for bigram in bigram_tokens:
            if bigram in self.job_keywords:
                words.append(bigram)

        return words

    def tokenize_and_train(self):
        """
        Tokenizes the job descriptions and trains a Word2Vec model on the preprocessed text data.
        The trained model is saved to the specified output folder.

        Returns:
            None
        """
        sentences = self.df['cleaned_text']
        self.word2vec_model = Word2Vec(sentences, vector_size=300, window=10, min_count=1, workers=4, epochs=30)
        
        model_path = os.path.join(self.output_subfolder, 'word2vec_model')
        self.word2vec_model.save(model_path)
        print(f"Word2Vec model saved to {model_path}")

    def produce_scatter_plot(self):
        """
        Generates a scatter plot of trends in keywords over time, using Gaussian process
        regression to smooth the trends, and marks significant events with vertical lines.

        The plot will be saved to the specified output folder.

        Returns:
            None
        """
        # List of tech keywords to track
        tech_keywords = {
            "digital_marketing": ["digital_marketing"],
            "media": ["media", "social_media", "digital_media"],
            "advertiser": ["advertiser", "advertisers"],
            "brand_management": ["brand_management"],
            "social_media": ["social_media"],
            "advertising": ["advertisement", "advertising", "marketing", "marketing_campaigns", "marketing_campaign",
                            "ads", "advertisement_campaign", "advertisement_campaign"],
            "pr": ["pr"],
            "digital_media": ["digital_media"],
            "content": ["content"],
            "customer_engagement": ["customer_engagement"],
            "brand": ["brands", "brands", "branding"],
            "branding": ["branding"],
            "market_research": ["market_research"],
            "marketing_analytics": ["marketing_analytics"]
        }

        # Convert 'CreatedAt' column to datetime and extract half-year period
        self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
        self.df['half_year'] = self.df['CreatedAt'].dt.year.astype(str) + " H" + ((self.df['CreatedAt'].dt.month - 1) // 6 + 1).astype(str)

        # Check if any of the tech keywords are in the cleaned text
        for keyword in tech_keywords:
            self.df[keyword] = self.df['cleaned_text'].apply(lambda x: any(term in x for term in tech_keywords[keyword]))

        # Aggregate trends by half-year
        trend_data = self.df.groupby('half_year')[list(tech_keywords.keys())].mean() * 100  # Convert to percentages

        # Filter out keywords that never exceed 10% at any time
        max_values = trend_data.max()
        keywords_to_plot = max_values[max_values > 10].index
        trend_data = trend_data[keywords_to_plot]

        # Plot linear graph with Gaussian process regression for smoothing
        plt.figure(figsize=(14, 7))

        for column in trend_data.columns:
            x = np.arange(len(trend_data))
            y = trend_data[column].values

            # Define Gaussian process with RBF kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

            # Fit the Gaussian process
            gp.fit(x.reshape(-1, 1), y)

            # Generate smooth predictions
            x_smooth = np.linspace(0, len(x) - 1, 500).reshape(-1, 1)
            y_smooth, sigma = gp.predict(x_smooth, return_std=True)

            # Plot the smooth curve
            line, = plt.plot(x_smooth, y_smooth, label=column)

            # Plot scatter points for the actual data
            plt.scatter(x, y, color=line.get_color(), alpha=0.7)

        # Add vertical lines in different colors (marking specific events)
        half_years = trend_data.index.tolist()
        event_positions = {
            'COVID-19 Lockdown': '2020 H1',
            'ChatGPT Launch': '2023 H1'
        }

        for event, period in event_positions.items():
            if period in half_years:
                event_idx = half_years.index(period)
                plt.axvline(event_idx, color='red' if 'COVID' in event else 'green', linestyle='--', label=event)

        # Customize plot
        plt.title('Trends of Marketing Related Keywords Over Time', pad=20, fontsize=16)
        plt.xlabel('Time Period (Half-Year)', fontsize=14)
        plt.ylabel('Percentage of Jobs Mentioning Keyword', fontsize=14)
        plt.ylim(0, None)
        plt.grid()

        # Customize x-axis with half-year labels
        plt.xticks(ticks=np.arange(len(half_years)), labels=half_years, rotation=45, fontsize=12)
        plt.yticks(fontsize=12)

        # Format y-axis to display percentages
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        # Add legend for lines and vertical markers only
        plt.legend(title='Keywords', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

        # Save the plot
        output_file = os.path.join(self.output_subfolder, 'tech_keywords_trends_filtered_gaussian_scatter_plot_half_years.png')
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Saved figure to {output_file}")

    def track_trends2(self):
        """
        Generates a trend plot similar to `produce_scatter_plot`, but focuses on a smoothed line using Gaussian process
        regression. Filters and plots keywords that have exceeded 10% mentions over time.

        Returns:
            None
        """
        # List of tech keywords to track
        tech_keywords = {
            "digital_marketing": ["digital_marketing"],
            "media": ["media", "social_media", "digital_media"],
            "advertiser": ["advertiser", "advertisers"],
            "brand_management": ["brand_management"],
            "social_media": ["social_media"],
            "advertising": ["advertisement", "advertising", "marketing", "marketing_campaigns", "marketing_campaign",
                            "ads", "advertisement_campaign", "advertisement_campaign"],
            "pr": ["pr"],
            "digital_media": ["digital_media"],
            "content": ["content"],
            "customer_engagement": ["customer_engagement"],
            "brand": ["brands", "brands", "branding"],
            "branding": ["branding"],
            "market_research": ["market_research"],
            "marketing_analytics": ["marketing_analytics"]
        }

        # Convert 'CreatedAt' column to datetime and extract half-year period
        self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
        self.df['half_year'] = self.df['CreatedAt'].dt.year.astype(str) + " H" + ((self.df['CreatedAt'].dt.month - 1) // 6 + 1).astype(str)

        # Check if any of the tech keywords are in the cleaned text
        for keyword in tech_keywords:
            self.df[keyword] = self.df['cleaned_text'].apply(lambda x: any(term in x for term in tech_keywords[keyword]))

        # Aggregate trends by half-year
        trend_data = self.df.groupby('half_year')[list(tech_keywords.keys())].mean() * 100  # Convert to percentages

        # Filter out keywords that never exceed 10% at any time
        max_values = trend_data.max()
        keywords_to_plot = max_values[max_values > 10].index
        trend_data = trend_data[keywords_to_plot]  # Only keep keywords that exceed 10% at some point

        # Ensure no empty trend data after filtering
        if trend_data.empty:
            print("No keywords exceed 10% at any point in the data.")
            return

        # Plot linear graph with smooth curves
        plt.figure(figsize=(16, 8))  # Increased figure size for better spacing

        for column in trend_data.columns:
            x = np.arange(len(trend_data))  # Use sequential numbers for x-axis positions
            y = trend_data[column].values  # Percentage trends

            # Define kernel: RBF kernel with length scale 1.0
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

            # Fit the Gaussian Process
            gp.fit(x.reshape(-1, 1), y)

            # Predict with some smoothness control
            x_pred = np.linspace(0, len(x) - 1, 300).reshape(-1, 1)  # Generate smooth x values with more points
            y_pred, y_std = gp.predict(x_pred, return_std=True)

            # Plot the predicted line
            plt.plot(x_pred, y_pred, label=column)

        # Add vertical lines in different colors (marking specific events)
        event_positions = {
            'COVID-19 Lockdown': '2020 H1',
            'ChatGPT Launch': '2023 H1'
        }
        for event, half_year in event_positions.items():
            if half_year in trend_data.index:
                plt.axvline(x=list(trend_data.index).index(half_year), color='red' if 'COVID' in event else 'green', linestyle='--', label=event)

        # Customize plot
        plt.title('Trends of Marketing Related Keywords Over Time', pad=20, fontsize=18)  # Increased title size
        plt.xlabel('Time Period (Half-Year)', fontsize=16)  # Increased label size
        plt.ylabel('Percentage of Jobs Mentioning Keyword', fontsize=16)  # Increased label size
        plt.ylim(0, None)  # Start y-axis at 0
        plt.grid()

        # Format x-axis to show half-year periods explicitly
        plt.xticks(ticks=np.arange(len(trend_data.index)), labels=trend_data.index, rotation=45, ha='right', fontsize=14)  # Adjusted tick alignment and size
        plt.yticks(fontsize=14)  # Increased tick size

        # Format y-axis to display percentages
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

        # Add legend for lines and vertical markers only
        plt.legend(title='Keywords', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14)  # Increased legend font size

        # Save the plot
        output_file = os.path.join(self.output_subfolder, 'tech_keywords_trends_gaussian_plot_halfyear.png')
        plt.tight_layout()
        plt.savefig(output_file)

    def track_trends3(self):
        """
        Similar to `track_trends2` but with the addition of detecting rupture points using the `ruptures` package. 
        This allows identifying significant changes in the trend over time.

        Returns:
            None
        """
        # List of tech keywords to track
        tech_keywords = {
            "digital_marketing": ["digital_marketing"],
            "media": ["media", "social_media", "digital_media"],
            "advertiser": ["advertiser", "advertisers"],
            "brand_management": ["brand_management"],
            "social_media": ["social_media"],
            "advertising": ["advertisement", "advertising", "marketing", "marketing_campaigns", "marketing_campaign",
                            "ads", "advertisement_campaign", "advertisement_campaign"],
            "pr": ["pr"],
            "digital_media": ["digital_media"],
            "content": ["content"],
            "customer_engagement": ["customer_engagement"],
            "brand": ["brands", "brands", "branding"],
            "branding": ["branding"],
            "market_research": ["market_research"],
            "marketing_analytics": ["marketing_analytics"]
        }

        # Convert 'CreatedAt' column to datetime and extract half-year period
        self.df['CreatedAt'] = pd.to_datetime(self.df['CreatedAt'])
        self.df['half_year'] = self.df['CreatedAt'].dt.year.astype(str) + " H" + ((self.df['CreatedAt'].dt.month - 1) // 6 + 1).astype(str)

        # Check if any of the tech keywords are in the cleaned text
        for keyword in tech_keywords:
            self.df[keyword] = self.df['cleaned_text'].apply(lambda x: any(term in x for term in tech_keywords[keyword]))

        # Aggregate trends by half-year
        trend_data = self.df.groupby('half_year')[list(tech_keywords.keys())].mean() * 100  # Convert to percentages

        # Filter out keywords that never exceed 10% at any time
        max_values = trend_data.max()
        keywords_to_plot = max_values[max_values > 10].index
        trend_data = trend_data[keywords_to_plot]

        # Handle empty trend_data case
        if trend_data.empty:
            print("No keywords exceed the threshold. Skipping plot generation.")
            return

        # Create a grid of subplots (adjust the number of rows and columns based on the number of keywords)
        n_keywords = len(trend_data.columns)
        n_cols = 2  # 2 columns for the plot grid
        n_rows = (n_keywords + 1) // n_cols  # Ensure enough rows for all subplots

        plt.figure(figsize=(max(12, 2 * len(trend_data.index)), min(6 * n_rows, 24)))  # Dynamically increase width

        # Initialize a handle for the ruptures in the legend
        rupture_line_handle = None

        for idx, column in enumerate(trend_data.columns):
            ax = plt.subplot(n_rows, n_cols, idx + 1)  # Create individual subplot
            x = np.arange(len(trend_data))
            y = trend_data[column].values

            # Define Gaussian Process with RBF kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(5, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

            # Fit the Gaussian process
            gp.fit(x.reshape(-1, 1), y)

            # Generate smooth predictions
            x_smooth = np.linspace(0, len(x) - 1, 500).reshape(-1, 1)
            y_smooth, sigma = gp.predict(x_smooth, return_std=True)

            # Plot the smooth curve
            ax.plot(x_smooth, y_smooth, label=column)

            # Mark events with vertical lines
            half_years = trend_data.index.tolist()
            event_positions = {
                'COVID-19 Lockdown': '2020 H1',
                'ChatGPT Launch': '2023 H1'
            }
            for event, period in event_positions.items():
                if period in half_years:
                    event_idx = half_years.index(period)
                    ax.axvline(event_idx, color='red' if 'COVID' in event else 'green', linestyle='--', label=event)

            # Set titles and labels with increased font sizes
            ax.set_title(f'{column} Trends', pad=20, fontsize=16)
            ax.set_xlabel('Time Period (Half-Year)', fontsize=14)
            ax.set_ylabel('Percentage of Job Descriptions Mentioning Keywords', fontsize=14)
            ax.set_ylim(0, None)
            ax.grid()

            # Customize x-axis to show half-year intervals with better spacing
            ax.set_xticks(x)
            ax.set_xticklabels(half_years, rotation=45, fontsize=12, ha='right')  # Align text for better readability

            # Format y-axis to display percentages
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

            # Add legend with increased font size
            ax.legend(title='Keywords', loc='upper left', fontsize=12)

           # Change point detection using ruptures
            algo = rpt.Pelt(model="l2").fit(y_smooth)  # PELT algorithm, l2 loss
            change_points = algo.predict(pen=100)  # Penalty parameter for controlling sensitivity

            # Plot the detected ruptures
            rupture_line_handle = None
            for cp in change_points[:-1]:  # Exclude the last point, as it's the end of the series
                rupture_time = x_smooth[cp - 1]  # Convert index to x_smooth time
                rupture_line = ax.axvline(x=rupture_time, color='orange', linestyle='-', linewidth=2)
                if rupture_line_handle is None:
                    rupture_line_handle = rupture_line

            # Add the rupture line to the legend if it exists
            if rupture_line_handle is not None:
                handles, labels = ax.get_legend_handles_labels()
                handles.append(rupture_line_handle)
                labels.append('Ruptures')
                ax.legend(handles, labels, title='Keywords', loc='upper left', fontsize=12)
            else:
                ax.legend(title='Keywords', loc='upper left', fontsize=12)

            handles, labels = ax.get_legend_handles_labels()

            if rupture_line_handle is not None:
                handles.append(rupture_line_handle)
                labels.append('Ruptures')

        # Adjust layout to prevent overlapping
        plt.tight_layout(pad=3)

        output_file = os.path.join(self.output_subfolder, 'tech_keywords_trends_half_years.png')

        try:
            plt.savefig(output_file)
            print(f"Saved figure to {output_file}")
        except SystemError as e:
            print(f"Failed to save figure: {e}. Try reducing the figure size or checking plot dimensions.")
        finally:
            plt.close()