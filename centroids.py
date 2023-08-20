"""
script for computing centroids
do not run often unless really needed
"""

import config
import constants
from llama_index.embeddings import OpenAIEmbedding
import numpy as np


def compute_query_centroids(queries):
    """
    Given a list of queries as strings, returns the mean of the embeddings as a numpy array
    """
    embed_model = OpenAIEmbedding()  # defaults to ada-002
    query_embeddings = embed_model._get_text_embeddings(queries)
    centroid = np.mean(query_embeddings, axis=0)

    return centroid


if __name__ == "__main__":
    pass
