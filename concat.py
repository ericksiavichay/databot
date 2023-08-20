import numpy as np
import pickle
from llama_index_qa import plot_graphs

with open("./experiment_data/arize/arize_all_data.pkl", "rb") as file:
    all_data = pickle.load(file)

plot_graphs(all_data, 4, save_dir=f"./experiment_data/arize/", show=False)
