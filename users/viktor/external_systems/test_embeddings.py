import numpy as np
import pandas as pd
from ssem_embedder import SSEMEmbedder

# Initialize embedder
embedder = SSEMEmbedder(model_name="all-mpnet-base-v2")

df = pd.read_csv('file_english.csv')

# Generate embeddings for the 'description' column
descriptions = df["Description"].tolist()
embeddings = embedder.generate_embeddings(descriptions)

# Save embeddings to a file
np.save("description_embeddings.npy", embeddings)