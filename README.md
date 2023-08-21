# Important to know before running anything
- ./query_centroid.npy contains the centroid of the query embeddings
- ./arize_1000_centroid.npy contains the centroid of the document chunk embeddings with a chunk size of 1000. This was an arbitrary choice and you can get the centroid of any chunk size document. I chose 1000 for the sake of time and amount of info that is able to be stored in a chunk
- ./raw_arize_docs.pkl contains the raw arize docs as a list of strings
- ./indices contains the document embeddings sorted by chunk size. If you want the raw embeddings, they are in here
- ./experiment_data contains dataframes and graphs
  - arize contains a pkl file for the data and graphs for the following experiments: chunk size, k = 4, original/original reranked
  - arize_1000 contains only the debiased data (data and graphs). You'll need to use an editted version of llama_index for this to run
  - arize_debiased contains all of the experiments comparing the original, original_rerank, and original_debias retrieval methods

# How to run experiments
## No debias

1. Set up your favorite virtual env with Python 3.10 (eg. conda)
2. pip install -r requirements.txt
3. Make changes to the experiment's main function (eg. chunk sizes)
   a. Allowable query transformations: 'original', 'original_rerank', 'hyde', 'hyde_rerank'. This list can be the empty list if you do not want to apply any transformation
5. ```bash
   python llama_index_qa.py
   ```
   Expect this to take many hours (at 2-14 hours depeneding on how many configurations you are choosing). Blame OpenAI for the rate limits :D
## With debias
Honestly just message me so I can tell you which llama_index library file you need to edit to subtract the centroids. Llamaindex may have a way to insert a custom similarity function but it's really messy to do so so it's easier to just edit the codebase mounted on your workspace in something like VS code.
