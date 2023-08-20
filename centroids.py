"""
script for computing centroids
do not run often unless really needed
"""

import config
import constants
from llama_index.embeddings import OpenAIEmbedding
import numpy as np
import openai

openai.api_key = config.jason_key


def compute_query_centroids(queries, path="./query_centroids.npy"):
    """
    Given a list of queries as strings, returns the mean of the embeddings as a numpy array
    """
    embed_model = OpenAIEmbedding()  # defaults to ada-002
    query_embeddings = embed_model._get_text_embeddings(queries)
    centroid = np.mean(query_embeddings, axis=0)

    np.save(path, centroid)


if __name__ == "__main__":
    compute_query_centroids(constants.questions)
