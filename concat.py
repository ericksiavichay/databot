import numpy as np
import pickle
from llama_index_qa import plot_graphs
import os

with open("./arize_1000_centroid.npy", "rb") as f:
    arize_1000_centroid = np.load(f)

with open("./query_centroid.npy", "rb") as f:
    query_centroid = np.load(f)

with open("./experiment_data/arize/arize_all_data.pkl", "rb") as file:
    all_data = pickle.load(file)

with open("./experiment_data/arize_1000/arize_all_data.pkl", "rb") as file:
    all_data_1000 = pickle.load(file)

all_data_with_debiased = {}

for chunk_size in all_data_1000:
    all_data_with_debiased[chunk_size] = {
        "original": all_data[chunk_size]["original"],
        "original_rerank": all_data[chunk_size]["original_rerank"],
        "original_debias": all_data[chunk_size]["original"],
    }
save_dir = f"./experiment_data/arize_debiased/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
with open(f"{save_dir}arize_all_data_debiased.pkl", "wb") as f:
    pickle.dump(all_data_with_debiased, f)

plot_graphs(all_data_with_debiased, 4, "./experiment_data/arize_debiased/", show=False)
