"""
Llama Index implementation of a chatbot
"""
from llama_index.callbacks import CallbackManager, LlamaDebugHandler, CBEventType

from sklearn.metrics import ndcg_score
from ast import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import cohere
import constants
from concurrent.futures import ThreadPoolExecutor

from llama_index.indices.postprocessor.cohere_rerank import CohereRerank

import config
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from llama_index.query_engine.multistep_query_engine import MultiStepQueryEngine
from llama_index.indices.query.query_transform.base import StepDecomposeQueryTransform


from llama_index import (
    ListIndex,
    QuestionAnswerPrompt,
    RefinePrompt,
    StorageContext,
    download_loader,
    TreeIndex,
    VectorStoreIndex,
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    Response,
    load_index_from_storage,
)

from llama_index.llms import OpenAI
import requests
from bs4 import BeautifulSoup
import asyncio
from aiohttp import ClientSession
from llama_index.node_parser import SimpleNodeParser
from llama_index.indices.query.query_transform import HyDEQueryTransform
from llama_index.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
import os
import openai
from llama_index.llms import OpenAI
from llama_index.evaluation import QueryResponseEvaluator

openai.api_key = (
    config.jason_key
)  # replace with the string containing the API key if needed
cohere.api_key = config.erick_cohere_key
EVALUATION_SYSTEM_MESSAGE = "You will be given a query and a reference text. You must determine whether the reference text contains an answer to the input query. Your response must be binary (NO or YES) and should not contain any text or characters aside from NO or YES. NO means that the reference text does not contain an answer to the query. YES means the reference text contains an answer to the query."
QUERY_CONTEXT_PROMPT_TEMPLATE = """# Query: {query}

# Reference: {reference}

# Binary: """

JASON_INIT_EVALUATION_SYSTEM_MESSAGE = """
    You are comparing a reference text to a question and trying to determine if the reference text contains information relevant to answering the question. Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {query}
    ************
    [Reference text]: {context}
    [END DATA]
    
    Compare the Question above to the Reference text. You must determine whether the Reference text contains information that can answer the Question. 
    Please focus on if the very specific question can be answered by the information in the Reference text.
    Your response must be a single word, either "Yes" for relevant or "No" for irrelevant,
    and should not contain any text or characters aside from that word. 
    "No" means that the reference text does not contain an answer to the query.
    "Yes" means the reference text contains an answer to the query.
"""

questions = constants.questions


def get_urls(base_url):
    if not base_url.endswith("/"):
        base_url = base_url + "/"
    page = requests.get(f"{base_url}sitemap.xml")
    scraper = BeautifulSoup(page.content, "xml")

    urls_from_xml = []

    loc_tags = scraper.find_all("loc")

    for loc in loc_tags:
        urls_from_xml.append(loc.get_text())

    return urls_from_xml


async def get_website_document(url):
    name = "BeautifulSoupWebReader"
    BeautifulSoupWebReader = download_loader(name)
    loader = BeautifulSoupWebReader()

    document = loader.load_data(urls=[url])


async def get_documents_from_urls(urls):
    tasks = []

    async with ClientSession() as session:
        for url in urls:
            task = asyncio.ensure_future(get_website_document(url))
            tasks.append(task)
        return await asyncio.gather(*tasks)


def get_documents_from_url(base_url):
    urls = get_urls(base_url)
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(get_documents_from_urls(urls))
    return loop.run_until_complete(future)


def compute_precision_at_i(eval_scores, i):
    return sum(eval_scores[:i]) / i


def compute_average_precision_at_i(evals, cpis, i):
    if np.sum(evals[:i]) == 0:
        return 0
    subset = cpis[:i]
    return (np.array(evals[:i]) @ np.array(subset)) / np.sum(evals[:i])


def compute_mean_precision(df, i):
    return df[f"precision_at_{i}"].mean()


def compute_mean_precisions(df):
    """
    given a df with columns named precision_at_{i}, compute
    the columns' mean precisions
    """
    mean_precisions = df[[f"precision_at_{i}" for i in range(1, k + 1)]].mean()
    return mean_precisions


def plot_mrr_graphs(all_data, k, save_dir="./", show=True):
    for i in range(1, k + 1):
        plt.figure()

        mrrs_dict = {}
        for chunk_size, method_data in all_data.items():
            for method, df in method_data.items():
                if method == "multistep":
                    continue
                mrr_i = (1 / df[f"rank_at_{i}"]).mean()
                if method not in mrrs_dict:
                    mrrs_dict[method] = {}
                mrrs_dict[method][chunk_size] = mrr_i

        # Convert the mean_evaluations_dict to a DataFrame for easier plotting
        df_mrrs = pd.DataFrame.from_dict(mrrs_dict)

        # Plot the grouped bar graph
        df_mrrs.plot(kind="bar", width=0.8, figsize=(10, 6))
        plt.xlabel("Chunk Size")
        plt.ylabel(f"MRR @ {i}")
        plt.title(f"MRR @ {i} for Different Chunk Sizes and Methods")
        plt.legend(title="Method", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mrr_at_{i}.png")
        if show:
            plt.show()


def plot_ndcg_graphs(all_data, k, save_dir="./", show=True):
    for i in range(1, k + 1):
        plt.figure()

        average_ndcgs_dict = {}
        for chunk_size, method_data in all_data.items():
            for method, df in method_data.items():
                if method == "multistep":
                    continue
                average_ndcg_i = df[f"ndcg_at_{i}"].mean()
                if method not in average_ndcgs_dict:
                    average_ndcgs_dict[method] = {}
                average_ndcgs_dict[method][chunk_size] = average_ndcg_i

        # Convert the mean_evaluations_dict to a DataFrame for easier plotting
        df_average_ndcgs = pd.DataFrame.from_dict(average_ndcgs_dict)

        # Plot the grouped bar graph
        df_average_ndcgs.plot(kind="bar", width=0.8, figsize=(10, 6))
        plt.xlabel("Chunk Size")
        plt.ylabel(f"Average NDCG @ {i}")
        plt.title(f"Average NDCG @ {i} for Different Chunk Sizes and Methods")
        plt.legend(title="Method", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/average_ndcg_at_{i}.png")
        if show:
            plt.show()


def plot_precision_graphs(all_data, k, save_dir="./", show=True):
    for i in range(1, k + 1):
        mean_average_precisions_dict = {}
        for chunk_size, method_data in all_data.items():
            for method, df in method_data.items():
                if method == "multistep":
                    continue
                macp_i = df[f"average_context_precision_at_{i}"].mean()
                if method not in mean_average_precisions_dict:
                    mean_average_precisions_dict[method] = {}
                mean_average_precisions_dict[method][chunk_size] = macp_i

        # Convert the mean_evaluations_dict to a DataFrame for easier plotting
        df_mean_average_precisions = pd.DataFrame.from_dict(
            mean_average_precisions_dict
        )

        # Plot the grouped bar graph
        df_mean_average_precisions.plot(kind="bar", width=0.8, figsize=(10, 6))
        plt.xlabel("Chunk Size")
        plt.ylabel(f"MACP @ {i}")
        plt.title(f"MACP @ {i} Different Chunk Sizes and Methods")
        plt.legend(title="Method", bbox_to_anchor=(1, 1))
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mean_avg_p_at_{i}.png")
        if show:
            plt.show()


def plot_latency_graphs(all_data, save_dir="./", show=True):
    # Create an empty dictionary to store the mean latency for each method and chunk size
    mean_latency_dict = {}

    # Iterate through the input dictionary to compute the mean latency for each method and chunk size
    for chunk_size, method_data in all_data.items():
        for method, df in method_data.items():
            mean_latency = df["response_latency"].mean()
            if method not in mean_latency_dict:
                mean_latency_dict[method] = {}
            mean_latency_dict[method][chunk_size] = mean_latency

    # Convert the mean_latency_dict to a DataFrame for easier plotting
    df_mean_latency = pd.DataFrame.from_dict(mean_latency_dict)

    # Plot the grouped bar graph
    df_mean_latency.plot(kind="bar", width=0.8, figsize=(10, 6))
    plt.xlabel("Chunk Size (tokens)")
    plt.ylabel("Mean Latency (seconds)")
    plt.title("Mean Latency for Different Chunk Sizes and Methods")
    plt.legend(title="Method", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"{save_dir}/latency.png")
    if show:
        plt.show()
    else:
        plt.close()


# def plot_response_evaluation_graphs(all_data, save_dir="./", show=True):
#     # Create an empty dictionary to store the mean evaluations for each method and chunk size
#     mean_evaluations_dict = {}

#     # Iterate through the input dictionary to compute the mean evaluations for each method and chunk size
#     for chunk_size, method_data in all_data.items():
#         for method, df in method_data.items():
#             mean_evaluations = df["response_evaluation"].mean()
#             if chunk_size not in mean_evaluations_dict:
#                 mean_evaluations_dict[chunk_size] = {}
#             mean_evaluations_dict[chunk_size][method] = mean_evaluations

#     # Convert the mean_evaluations_dict to a DataFrame for easier plotting
#     df_mean_evaluations = pd.DataFrame.from_dict(mean_evaluations_dict)

#     # Plot the grouped bar graph
#     df_mean_evaluations.plot(kind="bar", width=0.8, figsize=(10, 6))
#     plt.xlabel("Chunk Size")
#     plt.ylabel("Mean Response Evaluation")
#     plt.title("Mean Response Evaluation for Different Chunk Sizes and Methods")
#     plt.legend(title="Method", bbox_to_anchor=(1, 1))
#     plt.tight_layout()
#     plt.savefig(f"{save_dir}/evaluation.png")

#     if show:
#         plt.show()


def plot_graphs(all_data, k, save_dir="./", show=True):
    plot_latency_graphs(all_data, save_dir, show)
    plot_precision_graphs(all_data, k, save_dir, show)
    plot_ndcg_graphs(all_data, k, save_dir, show)
    plot_mrr_graphs(all_data, k, save_dir, show)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def evaluate_query_and_retrieved_context(
    query: str, contexts, model_name: str, evaluation_template: str
) -> str:
    evals = []

    for context in contexts:
        prompt = evaluation_template.format(
            query=query,
            context=context,
        )
        try:
            response = openai.ChatCompletion.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You can only output the words YES or NO.",
                    },
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=0.6,
            )
        except:
            print("EVALUATION FAILED. RETRYING AFTER 60 s")
            time.sleep(61)
        eval = response["choices"][0]["message"]["content"]
        evals.append(eval)
    return evals


def format_evals(evals):
    evals_as_int = []
    for eval in evals:
        if eval.lower() == "yes":
            evals_as_int.append(1)
        else:
            evals_as_int.append(0)
    return evals_as_int


def get_transformation_query_engine(index, name, k):
    if name == "original_rerank":
        cohere_rerank = CohereRerank(api_key=cohere.api_key, top_n=k)
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model="gpt-4")
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k * 2,
            response_mode="refine",  # response mode can also be parameterized
            service_context=service_context,
            node_postprocessors=[cohere_rerank],
        )
        return query_engine
    elif name == "hyde":
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model="gpt-4")  # change to model
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k, response_mode="refine", service_context=service_context
        )
        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(query_engine, hyde)

        return hyde_query_engine

    elif name == "hyde_rerank":
        cohere_rerank = CohereRerank(api_key=cohere.api_key, top_n=k)

        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model="gpt-4"),
            callback_manager=callback_manager,
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k * 2,
            response_mode="compact",
            service_context=service_context,
            node_postprocessors=[cohere_rerank],
        )
        hyde = HyDEQueryTransform(include_original=True)
        hyde_rerank_query_engine = TransformQueryEngine(query_engine, hyde)

        return hyde_rerank_query_engine

    elif name == "multistep":
        gpt4 = OpenAI(temperature=0.6, model="gpt-4")
        service_context_gpt4 = ServiceContext.from_defaults(llm=gpt4)

        step_decompose_transform = StepDecomposeQueryTransform(
            LLMPredictor(llm=gpt4), verbose=True
        )

        multi_query_engine = MultiStepQueryEngine(
            query_engine=index.as_query_engine(
                service_context=service_context_gpt4, similarity_top_k=k
            ),
            query_transform=step_decompose_transform,
            index_summary="documentation",  # llama index isn't really clear on how this works
        )

        return multi_query_engine

    else:
        return


def get_rank(evals):
    for i, eval in enumerate(evals):
        if eval == 1:
            return i + 1

    return np.inf


def run_experiments(
    documents, queries, chunk_sizes, query_transformations, k, web_title, rerank=False
):
    all_data = {}

    for chunk_size in chunk_sizes:
        print(f"PARSING WITH CHUNK SIZE {chunk_size}")
        persist_dir = f"./indices/{web_title}_{chunk_size}"
        if os.path.isdir(persist_dir):
            print("EXISTING INDEX FOUND, LOADING...")
            # Rebuild storage context
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)

            # Load index from the storage context
            index = load_index_from_storage(storage_context)
        else:
            print("BUILDING INDEX...")
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size, chunk_overlap=0
            )  # you can also experiment with the chunk overlap too
            nodes = node_parser.get_nodes_from_documents(documents)
            index = VectorStoreIndex(nodes, show_progress=True)
            index.storage_context.persist(persist_dir)

        engines = {}
        # query cosine similarity to nodes engine
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        callback_manager = CallbackManager([llama_debug])
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model="gpt-4"),
            callback_manager=callback_manager,
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k,
            response_mode="compact",
            service_context=service_context,
        )  # response mode can also be parameterized
        engines["original"] = query_engine

        # create different query transformation engines
        for name in query_transformations:
            this_engine = get_transformation_query_engine(index, name, k)
            engines[name] = this_engine

        query_transformation_data = {name: [] for name in engines}

        for name in engines:
            engine = engines[name]
            # these take some time to compute...

            for i, query in enumerate(queries):
                print("-" * 50)

                time_start = time.time()
                response = engine.query(query)
                time_end = time.time()
                response_latency = time_end - time_start

                print(f"{name.upper()} RESPONSE: ", response, "\n")
                print(f"LATENCY: {response_latency:.2f}", "\n")

                # special case if the query transformation is the multistep
                # only log latency, response evaluation
                # if name == "multistep":
                #     res_eval = format_evals([evaluator.evaluate(query, response)])
                #     print(f"{name} EVAL: ", res_eval, "\n")
                #     row = [query, response, res_eval[0], response_latency]
                #     query_transformation_data[name].append(row)

                #     continue

                # evals = evaluator.evaluate_source_nodes(
                #     query, response
                # )  # evaluates if the retrieved nodes contain an answer to the query
                contexts = [
                    source_node.node.get_content()
                    for source_node in response.source_nodes
                ]
                scores = [source_node.score for source_node in response.source_nodes]
                evals = evaluate_query_and_retrieved_context(
                    query,
                    contexts,
                    "gpt-4",
                    evaluation_template=JASON_INIT_EVALUATION_SYSTEM_MESSAGE,
                )
                formatted_evals = format_evals(evals)

                print("CONTEXT EVALS: ", formatted_evals)

                # context precision at i
                cpis = [
                    compute_precision_at_i(formatted_evals, i) for i in range(1, k + 1)
                ]

                # average context precision at k for this query
                acpk = [
                    compute_average_precision_at_i(formatted_evals, cpis, i)
                    for i in range(1, k + 1)
                ]

                ndcgis = [
                    ndcg_score([formatted_evals], [scores], k=i)
                    for i in range(1, k + 1)
                ]

                ranki = [get_rank(formatted_evals[:i]) for i in range(1, k + 1)]

                # get the evaluation of the response
                # res_eval = format_evals([evaluator.evaluate(query, response)]) # don't need this for now

                row = (
                    [query, response.response]
                    + cpis
                    + acpk
                    + ndcgis
                    + ranki
                    + formatted_evals
                    # + res_eval
                    + [response_latency]
                    + contexts
                )
                query_transformation_data[name].append(row)

                print("-" * 50)

        columns = (
            ["query", "response"]
            + [f"context_precision_at_{i}" for i in range(1, k + 1)]
            + [f"average_context_precision_at_{i}" for i in range(1, k + 1)]
            + [f"ndcg_at_{i}" for i in range(1, k + 1)]
            + [f"rank_at_{i}" for i in range(1, k + 1)]
            + [f"context_{i}_evaluation" for i in range(1, k + 1)]
            + ["response_latency"]
            + [f"retrieved_context_{i}" for i in range(1, k + 1)]
        )
        all_data[chunk_size] = {}
        for name, data in query_transformation_data.items():
            if name == "multistep":
                df = pd.DataFrame(
                    data,
                    columns=[
                        "query",
                        "response",
                        "response_evaluation",
                        "response_latency",
                    ],
                )
                all_data[chunk_size][name] = df
            else:
                df = pd.DataFrame(data, columns=columns)
            all_data[chunk_size][name] = df

    return all_data


def main():
    name = "BeautifulSoupWebReader"
    BeautifulSoupWebReader = download_loader(name)

    # if loading from scratch, change these two
    web_title = "arize"  # nickname for this website, used for saving purposes
    base_url = "https://docs.arize.com/arize"
    # urls = get_urls(base_url)
    # print(f"LOADED {len(urls)} URLS")

    print("GRABBING DOCUMENTS")
    # two options here, either get the documents from scratch or load one from disk
    # loader = BeautifulSoupWebReader()
    # documents = loader.load_data(urls=urls)  # may take some time
    with open("raw_documents.pkl", "rb") as file:
        documents = pickle.load(file)

    chunk_sizes = [
        300,
        # 500,
        # 1000,
        # 2000,
        # 2500,
    ]  # change this, perhaps experiment from 500 to 3000 in increments of 500
    k = 4  # num documents to retrieve

    transformations = ["original_rerank"]

    all_data = run_experiments(
        documents,
        questions,
        chunk_sizes,
        transformations,
        k,
        web_title,
        rerank=True,
    )

    # save data to disk
    save_dir = f"./experiment_data/{web_title}_300/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}{web_title}_all_data.pkl", "wb") as f:
        pickle.dump(all_data, f)
    # with open(f"{save_dir}{web_title}_all_data.pkl", "rb") as f:
    #     all_data = pickle.load(f)
    plot_graphs(all_data, k, save_dir, show=False)


program_start = time.time()
main()
program_end = time.time()
total_time = (program_end - program_start) / (60 * 60)
print(f"EXPERIMENTS FINISHED: {total_time:.2f} hrs")
