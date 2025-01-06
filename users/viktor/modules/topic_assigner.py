# topic_assigner.py
import pandas as pd
import re
from interfaces import ITopicAssignment

class TopicAssigner(ITopicAssignment):
    def assign_most_likely_topic(self, df: pd.DataFrame, column: str, keyword_dict: dict) -> pd.DataFrame:
        """
        Assign the most likely topic to each row in the given DataFrame based on keyword matching.
        
        Args:
            df: DataFrame containing the text data.
            column: Column name containing the text to analyze.
            keyword_dict: Dictionary where keys are topic names and values are lists of keywords.
        
        Returns:
            DataFrame: The input DataFrame with an additional column 'Most_Likely_Topic'.
        """

        ''' def get_most_likely_topic(text: str) -> str:
            """
            Check which topic has the most keywords present in the text.
            
            Args:
                text: Text string to be analyzed.
            
            Returns:
                str: The topic with the highest keyword match, or 'Unclassified' if no matches.
            """
            topic_scores = {}
            for topic, keywords in keyword_dict.items():
                score = sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', text, flags=re.IGNORECASE))
                topic_scores[topic] = score

            # Return the topic with the highest score, or 'Unclassified' if no keywords match
            if topic_scores:
                return max(topic_scores, key=topic_scores.get) if max(topic_scores.values()) > 0 else 'Unclassified'
            return 'Unclassified' '''
        
        def get_most_likely_topic(text: str) -> str:
            """
            Check which topic has the most keywords present in the text.
            
            Args:
                text: Text string to be analyzed.
            
            Returns:
                str: The topic with the highest keyword match, or 'Unclassified' if no matches.
            """
            if not text or not isinstance(text, str):  # Check for None or non-string types
                return 'Unclassified'

            topic_scores = {}
            for topic, keywords in keyword_dict.items():
                score = sum(1 for keyword in keywords if re.search(r'\b' + re.escape(keyword) + r'\b', text, flags=re.IGNORECASE))
                topic_scores[topic] = score

            # Return the topic with the highest score, or 'Unclassified' if no keywords match
            if topic_scores:
                return max(topic_scores, key=topic_scores.get) if max(topic_scores.values()) > 0 else 'Unclassified'
            return 'Unclassified'

        # Apply the function to each row in the selected column
        df['Most_Likely_Topic'] = df[column].apply(get_most_likely_topic)
        return df