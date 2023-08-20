"""
script for computing centroids
do not run often unless really needed
"""

import config
import constants
from llama_index.embeddings import OpenAIEmbedding
import numpy as np
import openai
import json

openai.api_key = config.jason_key


def compute_query_centroids(queries, path="./query_centroid.npy"):
    """
    Given a list of queries as strings, returns the mean of the embeddings as a numpy array
    """
    embed_model = OpenAIEmbedding()  # defaults to ada-002
    query_embeddings = embed_model._get_text_embeddings(queries)
    centroid = np.mean(query_embeddings, axis=0)

    np.save(path, centroid)


def compute_index_centroid(
    index_path="./indices/arize_1000", path="./arize_1000_centroid.npy"
):
    """
    Given an index, returns the mean of the embeddings as a numpy array
    """
    if path is not None:
        index_name = index_path.split("/")[-1]
        path = "./" + index_name + "_centroid.npy"
    vector_store_json = json.load(open(index_path + "/vector_store.json"))
    document_embeddings = list(vector_store_json["embedding_dict"].values())
    centroid = np.mean(document_embeddings, axis=0)
    np.save(path, centroid)


if __name__ == "__main__":
    index_path = "./indices/arize_1000"
    compute_query_centroids(constants.questions)
    compute_index_centroid()
