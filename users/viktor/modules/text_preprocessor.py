from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from interfaces import ITextPreprocessor

class TextPreprocessor(ITextPreprocessor):
    """
    A class for preprocessing text by removing stopwords and tokenizing words.

    Attributes:
        stopwords (set): A set of stopwords loaded from NLTK and additional files.
    """

    def __init__(self, stopword_files):
        """
        Initialize the TextPreprocessor with custom stopword files.

        Args:
            stopword_files (list): List of file paths to additional stopword files.
        """
        self.stopwords = self.load_stopwords(stopword_files)

    def load_stopwords(self, stopword_files):
        """
        Load stopwords from NLTK and additional stopword files.

        Args:
            stopword_files (list): List of file paths to stopword files.

        Returns:
            set: A set containing all stopwords.
        """
        stopwords_set = set(stopwords.words('english'))
        for file in stopword_files:
            with open(file, 'r', encoding='utf-8') as f:
                stopwords_set.update(f.read().splitlines())
        return stopwords_set

    def preprocess(self, texts):
        """
        Preprocess a list of text strings by tokenizing, converting to lowercase, 
        removing non-alphabetic tokens, and filtering out stopwords.

        Args:
            texts (list): List of text strings to preprocess.

        Returns:
            list: A list of preprocessed text strings.
        """
        processed_texts = []
        for text in texts:
            tokens = word_tokenize(text.lower())
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stopwords]
            processed_texts.append(' '.join(filtered_tokens))
        return processed_texts