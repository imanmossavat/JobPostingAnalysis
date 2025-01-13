import os
import json
import numpy as np
from interfaces import ITopicModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sentence_transformers.util import cos_sim
from external_systems import SSEMEmbedder

class TopicModel(ITopicModel):
    def __init__(self, embeddings, texts, n_topics, num_keywords, max_iter, model, output_subfolder):
        """
        Initialize the TopicModel class.

        Args:
            embeddings (list): List of precomputed embeddings for the text data.
            texts (list): List of textual data for topic modeling.
            n_topics (int): Number of topics to extract.
            num_keywords (int): Number of top keywords per topic.
            max_iter (int): Maximum number of iterations for the KMeans algorithm.
            model (str): Pretrained model to use for generating embeddings.
            output_subfolder (str): Directory to save the output files.
        """
        self.embeddings = embeddings
        self.texts = texts
        self.n_topics = n_topics
        self.num_keywords = num_keywords
        self.max_iter = max_iter
        self.output_subfolder = output_subfolder
        self.model = model
        self.kmeans = KMeans(n_clusters=n_topics, max_iter=max_iter, random_state=42)
        self.embedder = SSEMEmbedder(model)
        self.desc_embeddings = []
        os.makedirs(self.output_subfolder, exist_ok=True)

    def fit_model(self):
        """
        Fit the KMeans clustering model to the embeddings.

        Processes the input embeddings and fits the KMeans clustering model to 
        group the data into the specified number of topics.
        """
        for desc_embedding in self.embeddings:
            desc_embedding = np.array(eval(desc_embedding), dtype=np.float32)
            self.desc_embeddings.append(desc_embedding)

        self.kmeans.fit(self.desc_embeddings)
        self.labels = self.kmeans.labels_

    def extract_keywords(self, texts, labels, num_keywords=5):
        """
        Extract top keywords for each topic using TF-IDF.

        Args:
            texts (list): List of textual data.
            labels (list): List of cluster labels for each document.
            num_keywords (int): Number of top keywords to extract per topic.

        Returns:
            dict: A dictionary where keys are topic indices and values are lists of top keywords.
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
        
        keywords = {}
        for i in range(self.n_topics):
            topic_docs = [idx for idx, label in enumerate(labels) if label == i]
            topic_tfidf_matrix = tfidf_matrix[topic_docs]
            feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
            sum_tfidf_scores = np.asarray(topic_tfidf_matrix.sum(axis=0)).flatten()
            top_keywords_idx = sum_tfidf_scores.argsort()[-num_keywords:][::-1]
            keywords[i] = feature_names[top_keywords_idx]
        
        return keywords

    def calculate_silhouette_score(self, embeddings, labels):
        """
        Calculate the silhouette score for the clustering.

        Args:
            embeddings (list): List of embeddings for the data.
            labels (list): Cluster labels for each document.

        Returns:
            float: Silhouette score indicating the quality of clustering.
        """
        return round(silhouette_score(embeddings, labels), 2)

    def calculate_inertia(self, kmeans):
        """
        Calculate the inertia of the KMeans clustering model.

        Args:
            kmeans (KMeans): Fitted KMeans model.

        Returns:
            float: Inertia value of the clustering model.
        """
        return round(kmeans.inertia_, 2)

    def calculate_ari(self, true_labels, predicted_labels):
        """
        Calculate the Adjusted Rand Index (ARI) for the clustering.

        Args:
            true_labels (list): Ground truth labels for the data.
            predicted_labels (list): Predicted cluster labels.

        Returns:
            float: ARI score indicating the similarity between the two label sets.
        """
        return round(adjusted_rand_score(true_labels, predicted_labels), 2)

    def evaluate_clustering_stability(self, embeddings, n_clusters, n_runs=10, random_state=42):
        """
        Evaluate the stability of clustering using repeated KMeans runs.

        Args:
            embeddings (list): List of embeddings for the data.
            n_clusters (int): Number of clusters for KMeans.
            n_runs (int): Number of repeated runs to evaluate stability.
            random_state (int): Seed for random number generation.

        Returns:
            float: Average ARI score across the repeated runs.
        """
        stability_scores = []
        for i in range(n_runs):
            kmeans = KMeans(n_clusters=n_clusters, max_iter=300, random_state=random_state + i)
            kmeans.fit(embeddings)
            labels = kmeans.labels_
            if i > 0:
                ari_score = adjusted_rand_score(prev_labels, labels)
                stability_scores.append(ari_score)
            prev_labels = labels
        return round(np.mean(stability_scores), 2) if stability_scores else 1.0
    
    def calculate_topic_diversity(self, keywords, embedder, num_keywords=7):
        """
        Calculate the diversity of topics based on the top keywords.

        Args:
            keywords (dict): Dictionary of top keywords for each topic.
            embedder (SSEMEmbedder): Embedder to generate word embeddings.
            num_keywords (int): Number of top keywords to consider for diversity.

        Returns:
            float: Average diversity score across all topics.
        """
        diversity_scores = []
        for topic, words in keywords.items():
            word_embeddings = embedder.generate_embeddings(words.tolist())
            cosine_similarities = cos_sim(word_embeddings, word_embeddings)
            if not isinstance(cosine_similarities, np.ndarray):
                cosine_similarities = cosine_similarities.detach().cpu().numpy()
            upper_triangle_indices = np.triu_indices_from(cosine_similarities, k=1)
            pairwise_distances = 1 - cosine_similarities[upper_triangle_indices]
            diversity_scores.append(np.mean(pairwise_distances))
        return round(np.mean(diversity_scores), 2)

    def calculate_topic_percentages(self):
        """
        Calculate the percentage of documents assigned to each topic.

        Returns:
            dict: A dictionary where keys are topic indices and values are percentages of documents.
        """
        topic_counts = np.bincount(self.labels, minlength=self.n_topics)
        total_documents = len(self.labels)
        return {i: round((count / total_documents) * 100, 2) for i, count in enumerate(topic_counts)}
    
    def execute_topic_modeling(self):
        """
        Execute the topic modeling process, save topics and metrics to files, and print the results.

        Saves the topics to a JSON file and metrics to another JSON file. Calculates various metrics 
        including silhouette score, inertia, ARI, clustering stability, topic diversity, and topic percentages.
        """
        keywords = self.extract_keywords(self.texts, self.labels, num_keywords=self.num_keywords)
        topics_file = os.path.join(self.output_subfolder, "topics.json")
        metrics_file = os.path.join(self.output_subfolder, "metrics.json")

        # Convert keywords to Python lists for JSON serialization
        topics_data = {"topics": {int(topic): words.tolist() for topic, words in keywords.items()}}
        with open(topics_file, "w") as f:
            json.dump(topics_data, f, indent=4)

        # Round numerical metrics to two decimal places
        sil_score = round(float(self.calculate_silhouette_score(self.desc_embeddings, self.labels)), 2)
        inertia = round(float(self.calculate_inertia(self.kmeans)), 2)
        ari_score = round(float(self.calculate_ari(self.labels, self.labels)), 2)
        stability_score = round(float(self.evaluate_clustering_stability(self.desc_embeddings, n_clusters=self.n_topics)), 2)
        diversity_score = round(float(self.calculate_topic_diversity(keywords, self.embedder, num_keywords=self.num_keywords)), 2)
        topic_percentages = {int(topic): round(float(percentage), 2) for topic, percentage in self.calculate_topic_percentages().items()}

        metrics_data = {
            "silhouette_score": sil_score,
            "inertia": inertia,
            "adjusted_rand_index": ari_score,
            "clustering_stability": stability_score,
            "topic_diversity_score": diversity_score,
            "topic_percentages": topic_percentages
        }
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=4)

        print(f"Topics saved to: {topics_file}")
        print(f"Metrics saved to: {metrics_file}")

        desc_embeddings = self.desc_embeddings
        return metrics_data, keywords, desc_embeddings