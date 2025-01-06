import os
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .text_preprocessor import TextPreprocessor
from interfaces import ITopicModel
from datetime import datetime

class NMFModel(ITopicModel):
    """
    A class implementing Non-negative Matrix Factorization (NMF) for topic modeling.

    Attributes:
        n_topics (int): Number of topics to extract.
        data (dict): Dataset containing the descriptions to be processed.
        stopword_files (list): List of file paths to stopword files for preprocessing.
        num_top_words (int): Number of top words to display per topic.
        epochs (int): Maximum number of iterations for the NMF algorithm.
        output_subfolder (str): Directory for saving output files.
    """

    def __init__(self, n_topics, data, stopword_files, num_top_words, epochs, output_subfolder):
        """
        Initialize the NMFModel.

        Args:
            n_topics (int): Number of topics to extract.
            data (dict): Dataset containing the descriptions to process.
            stopword_files (list): File paths to stopword lists.
            num_top_words (int): Number of top words per topic to display.
            epochs (int): Maximum number of iterations for NMF.
            output_subfolder (str): Directory for saving output files.
        """
        self.n_topics = n_topics
        self.data = data
        self.num_top_words = num_top_words
        self.epochs = epochs
        self.output_subfolder = output_subfolder
        self.preprocessor = TextPreprocessor(stopword_files)
        self.vectorizer = TfidfVectorizer()
        self.model = None

        os.makedirs(self.output_subfolder, exist_ok=True)

    def fit(self):
        """
        Fit the NMF model to the data by processing text, creating a document-term matrix, and applying NMF.
        """
        processed_texts = self.preprocessor.preprocess(self.data['Description'])
        self.doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
        self.model = NMF(n_components=self.n_topics, random_state=42, max_iter=self.epochs)
        self.topic_matrix = self.model.fit_transform(self.doc_term_matrix)

    def display_topics(self):
        """
        Display the top words for each topic and save the results to a timestamped file.

        Returns:
            list: A list of topics, each represented as a list of top words.
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for idx, topic in enumerate(self.model.components_):
            topics.append([feature_names[i] for i in topic.argsort()[-self.num_top_words:]])

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_path = os.path.join(self.output_subfolder, f'topics_{timestamp}.txt')
        with open(file_path, 'w') as f:
            f.write(f"Generated on: {timestamp}\n\n")
            for i, topic in enumerate(topics, 1):
                f.write(f"Topic {i}:\n")
                f.write(", ".join(topic) + "\n\n")

        return topics

    def calculate_topic_coherence(self):
        """
        Calculate the topic coherence score based on pairwise cosine similarity of top words within each topic.

        Returns:
            float: The average topic coherence score.
        """
        coherence_scores = []
        for topic in self.model.components_:
            top_word_vectors = topic.argsort()[-self.num_top_words:]
            pairwise_similarities = [
                cosine_similarity(top_word_vectors[:, [i]].T, top_word_vectors[:, [j]].T)[0, 0]
                for i in range(len(top_word_vectors)) for j in range(i + 1, len(top_word_vectors))
            ]
            coherence_scores.append(np.mean(pairwise_similarities))
        return np.mean(coherence_scores)

    def calculate_topic_diversity(self):
        """
        Calculate the topic diversity score based on unique words across all topics.

        Returns:
            float: The diversity score, where higher values indicate more unique words across topics.
        """
        topics = self.display_topics()
        if all(topic == topics[0] for topic in topics):
            return 0.0
        total_words = sum(len(topic) for topic in topics)
        unique_words = set(word for topic in topics for word in topic)
        return len(unique_words) / total_words if total_words else 0.0

    def calculate_silhouette_score(self):
        """
        Calculate the silhouette score for the clustering of documents based on topic assignments.

        Returns:
            float: The silhouette score, which ranges from -1 (poor clustering) to 1 (well-separated clusters).
        """
        kmeans = KMeans(n_clusters=self.n_topics, random_state=42)
        labels = kmeans.fit_predict(self.topic_matrix)
        return silhouette_score(self.topic_matrix, labels)

    def evaluate_clustering_stability(self, num_runs):
        """
        Evaluate the stability of clustering by running multiple clustering trials and calculating the average silhouette score.

        Args:
            num_runs (int): Number of clustering trials to perform.

        Returns:
            float: The average silhouette score across trials.
        """
        stability_scores = []
        for _ in range(num_runs):
            model = NMF(n_components=self.n_topics, random_state=None, max_iter=self.epochs)
            topic_matrix = model.fit_transform(self.doc_term_matrix)
            kmeans = KMeans(n_clusters=self.n_topics, random_state=42)
            labels = kmeans.fit_predict(topic_matrix)
            stability_scores.append(silhouette_score(topic_matrix, labels))
        return np.mean(stability_scores)

    def calculate_cosine_similarity(self):
        """
        Calculate pairwise cosine similarity between the components of the NMF model.

        Returns:
            ndarray: A matrix of cosine similarity scores between topics.
        """
        return cosine_similarity(self.model.components_)

    def calculate_topic_percentage(self):
        """
        Calculate the percentage contribution of each topic across all documents.

        Returns:
            dict: A dictionary mapping each topic to its percentage contribution.
        """
        topic_sums = np.sum(self.topic_matrix, axis=0)
        percentages = (topic_sums / np.sum(topic_sums)) * 100
        topic_percentage_dict = {f"Topic {i+1}": percentage for i, percentage in enumerate(percentages)}

        file_path = os.path.join(self.output_subfolder, 'topic_percentages.txt')
        with open(file_path, 'w') as f:
            for topic, percentage in topic_percentage_dict.items():
                f.write(f"{topic}: {percentage:.2f}%\n")

        return topic_percentage_dict