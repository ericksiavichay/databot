"""
Llama Index implementation of a chatbot
"""
from ast import List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import cohere
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

questions = [
    "How do I use the SDK to upload a ranking model?",
    "What drift metrics are supported in Arize?",
    "Does Arize support batch models?",
    "Does Arize support training data?",
    "How do I configure a threshold if my data has seasonality trends?",
    "How are clusters in the UMAP calculated? When are the clusters refreshed?",
    "How does Arize calculate AUC?",
    "Can I send truth labels to Arize separately?",
    "How do I send embeddings to Arize?",
    "Can I copy a dashboard?",
    "Can I copy a dashboard to a new space?",
    "How do I configure permissions for GBQ?",
    "How often does Arize query the table for table import jobs?",
    "Can you configure the query interval for table import jobs?",
    "Do you need to have a prediction label for classification models?",
    "Do you need a prediction ID for the training set?",
    "How do you set up PagerDuty alerts?",
    "Does the ingestion job for GBQ tables detect changes in the table schema?",
    "How do I send in extra metadata with each record?",
    "What is the current retention period of the data in Arize (if any), and can we customize this? (e.g. could we choose to set a specific retention period of, say 90-days, so that all data older than 90 days is deleted from Arize systems?)",
    "Does Arize store the individual records (rows) somewhere, or does it only store the aggregations calculated from the data?",
    "What happens if I upload actuals twice?",
    "What format should the prediction timestamp be?",
    "How often do the monitors run and evaluate?",
    "Do we have the ability to resolve a monitor?",
    "Does Arize support Microsoft Teams alerting?",
    "What should I do if I sent in duplicate prediction IDs?",
    "Why does Arize use UMAP over t-SNE?",
    "Can I export a dashboard as PDF?",
    "How does Arize's surrogate explainability model work?",
    "Can I update my predictions or features on Arize?",
    "What happens if my model schema changes after I deploy a new version of the model?",
    "How does Arize integrate with SageMaker?",
    "Does the ingestion job for GBQ tables detect changes in schema?",
    "Can I configure the Arize data sampling policy?",
    "About how long should it take for delayed actuals to link to predictions in the UI?",
    "Can I change strings to numeric in Arize?",
    "What is the definition of a model or a prediction in Arize?",
    "How do I pass in delayed ground truth?",
    "Can I pass in my own metrics within Arize?",
    "How large should my file sizes be when uploading data?",
    "How long does it take to ingest my data?",
    "How do I edit the frequency that my table import job runs?",
    "What permissions are needed to import my files from cloud storage?",
    "How do I grant permissions to import my GBQ table?",
    "Does Arize ingest null values?",
    "Which file or dataframe columns can be null?",
    "What file types are supported for cloud storage uploads?",
    "Is prediction_id required?",
    "How do I need to format timestamps?",
    "Why do I need a timestamp?",
    "What time unit is a timestamp?",
    "Does Arize support timestamps that are pandas format?",
    "Can I create any string format that is a timestamp?",
    "Can I send latent ground truth for ranking models?",
    "Does Arize count duplicate prediction IDs as a single prediction?",
    "Does Arize sample the data on ingestion of files?",
    "When connecting to a table, is the data copied into Arize or does Arize just run off of the table?",
    "Is the entire data set copied when connecting to data in files?",
    "How long does it take for data to show up in the platform?",
    "Does Arize support PSI as a drift metric?",
    "Arize help",
    "Dataset not showing up",
    "Is it possible for me to change the threshold for PSI for the drift tab, as in what I can configure for each monitor?",
    "I don't have actuals",
    "I see data in the data ingestion tab but none of the charts are showing data. What's going wrong?",
    "My monitor's latest status is green even though the chart shows the threshold is crossed. What does this mean?",
    "What is a managed monitor?",
    "What is Euclidean distance? I thought it is the distance between points. What does it mean on the page?",
    "Can I deploy Arize on my own Kubernetes cluster?",
    "How are the records sent to Arize secured? How does Arize handle sensitive data?",
    "Data ingestion page does not show the correct number of records",
    "How do I link my actuals to predictions?",
    "Can I create one FileImporter job with both predictions and latent actuals?",
    "Why is my FileImport job failing on uploading actuals?",
    "What is a score categorical model?",
    "I am sending images with my embeddings but the images don't load. What's going wrong?",
    "I am sending videos in link_to_data but they don't show up. Why?",
    "Does Arize support timeseries data?",
    "Where can I find HIPAA reporting?",
    "How do I get feature importance on data I upload in the File Importer?",
    "How can I move a model from one space to another?",
    "How do I get an Arize API key?",
    "What is a space key?",
    "How do I rotate my credentials?",
    "I can't recover my password, how do I do that?",
    "What is Arize's retention policy?",
    "How do I delete a space?",
    "How do I delete a space and organization?",
    "How do I change my email address?",
    "I sent the wrong records to Arize. How do I delete them?",
    "I sent 8000 records but I only see 1000 in the UI. Why?",
    "I can't find my feature in any of the dropdowns.",
    "How do I use custom metrics on monitors?",
    "How do I unsubscribe from a monitor?",
    "Do predictions from deleted models count against my plan usage?",
    "How do I change current SAML auth to remove the email and only authorize using the first name and last name?",
    "What counts against my plan usage?",
    "What happens if I go over my plan's allocated volume?",
    "What is the change_timestamp on table import?",
    "Do I need to upload timestamps?",
    "How do I load private images into Arize?",
    "What do I do if I sent in a feature whose type changed?",
    "What's the difference between Arize's Pro and Enterprise plans?",
    "How to share customized dashboard?",
    "How to create custom metric for ROI?",
    "Where do I find the API and Space keys?",
    "How can I run Arize on my own hardware?",
    "Is my prediction data shared with any services except for Arize?",
    "How do I change fields on a prediction?",
    "Can I create monitors with an API?",
    "Can I create dashboards with an API?",
    "How do I delete data?",
    "I can't see the points in the UMAP. How do I make the points bigger?",
    "I want to cancel my account. How do I do that?",
    "I want to delete my data. Help.",
    "How do I update my predictions?",
    "The File Importer job failed. How do I restart it?",
    "Does Arize support Snowflake?",
    "How do I ingest CSV data?",
    "How do I get access to my embeddings?",
    "I don't see errors in the SDK, but my records don't show up. How do I troubleshoot?",
    "Can I download my data?",
    "How can I change the threshold of my metric?",
    "How do I duplicate a dashboard?",
    "My monitor is noisy. How do I fix?",
    "How do I download my data?",
    "I got a 200 from my SDK request, but my data never showed up",
    "Is there a way to automatically infer which columns serve which purpose during the ingestion process?",
    "How much does the Arize platform cost and how do you charge?",
    "What is the price of the Arize platform?",
    "How much does Arize charge and how do you price, is it per model?",
    "What would Arize cost annually and what is the likely ROI? I assume it is quite high ROI",
    "Do you have a pricing calculator that can help me understand the price of Arize relative to the various deployment options?",
    "What is the cost of the Arize platform?",
    "Is there cost for Arize beyond an annual subscription price?",
    "What is the price per model or per volume and how much does the price discount as the volume goes up?",
    "What is the annual cost of a VPC deployment option? How does that price scale?",
    "How expensive is the Arize platform and how do you charge?",
    "Does Arize support object segmentation use cases?",
    "If I am using an object segmentation model, should I apply my own segmentation mask to the image before uploading the image, or will the platform do that for me?",
    "Can you give me an example schema I could use for uploading inference data from an image segmentation model?",
    "What's the difference between image segmentation and object detection?",
    "How do you recommend I create embeddings for an object segmentation model?",
    "Is it possible to upload multiple masks for the same image in a segmentation use case?",
    "What evaluation metrics are supported for image segmentation use cases?",
    "Do you have an example image segmentation notebook?",
    "How many classes are supported for image segmentation?",
    "Do you support IoU for image segmentation?",
    "This is a test question?",
    "?",
    "This is a question?",
    "My service is a hosting service designed for hosting your website. You can put your website on our service and host it with accelerated CDN delivery, tracking of usage data for running your website. Our service is one of the best on the internet in terms of delivery and experience.",
    "What is a timestamp?",
    "What is a prediction ID?",
    "What are actuals?",
    "Can I log single events?",
    "Can I log batches of data?",
    "What happens if I don't have ground truths?",
    "Can Arize be deployed inside my own cloud environment?",
    "What data do I need to send to Arize?",
    "What if I don't have a timestamp?",
    "Do I need to send in input features along with my ground truth, if I'm sending my ground truth data later?",
    "What if I don't have my prediction label or prediction score for my training data?",
    "How do I send training data vs production data?",
    "How do I connect Arize to data that exists somewhere else?",
    "What is the validation environment used for?",
    "Can I set permissions for my users?",
    "Can I set up monitors programmatically, or am I only able to set them up through the UI?",
    "How can I get feature importance values?",
    "Do you only support SHAP for feature importance?",
]


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


def plot_precision_graphs(all_data, k, web_title="arize", show=True):
    for i in range(1, k + 1):
        plt.figure()

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
        plt.savefig(f"./experiment_data/{web_title}/{web_title}_mean_avg_p_at_{i}.png")
        if show:
            plt.show()


def plot_latency_graphs(all_data, web_title="arize", show=True):
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
    plt.savefig(f"./experiment_data/{web_title}/{web_title}_latency.png")
    if show:
        plt.show()


def plot_response_evaluation_graphs(all_data, web_title="arize", show=True):
    # Create an empty dictionary to store the mean evaluations for each method and chunk size
    mean_evaluations_dict = {}

    # Iterate through the input dictionary to compute the mean evaluations for each method and chunk size
    for chunk_size, method_data in all_data.items():
        for method, df in method_data.items():
            mean_evaluations = df["response_evaluation"].mean()
            if chunk_size not in mean_evaluations_dict:
                mean_evaluations_dict[chunk_size] = {}
            mean_evaluations_dict[chunk_size][method] = mean_evaluations

    # Convert the mean_evaluations_dict to a DataFrame for easier plotting
    df_mean_evaluations = pd.DataFrame.from_dict(mean_evaluations_dict)

    # Plot the grouped bar graph
    df_mean_evaluations.plot(kind="bar", width=0.8, figsize=(10, 6))
    plt.xlabel("Chunk Size")
    plt.ylabel("Mean Response Evaluation")
    plt.title("Mean Response Evaluation for Different Chunk Sizes and Methods")
    plt.legend(title="Method", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(f"./experiment_data/{web_title}/{web_title}_evaluation.png")

    if show:
        plt.show()


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
            llm=OpenAI(temperature=0.6, model_name="gpt-4")
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k * 2,
            response_mode="compact",  # response mode can also be parameterized
            service_context=service_context,
            node_postprocessors=[cohere_rerank],
        )
        return query_engine
    elif name == "hyde":
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model_name="gpt-4")
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k, response_mode="compact", service_context=service_context
        )
        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(query_engine, hyde)

        return hyde_query_engine

    elif name == "hyde_rerank":
        cohere_rerank = CohereRerank(api_key=cohere.api_key, top_n=k)

        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model_name="gpt-4")
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


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def aquery(engine, query, index):
    try:
        print(f"TRYING REQUEST {index}")
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, engine.query, query)
        return response

    except:
        print(f"REQUEST {index} FAILED. RETRYING")


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
        service_context = ServiceContext.from_defaults(
            llm=OpenAI(temperature=0.6, model_name="gpt-4")
        )
        query_engine = index.as_query_engine(
            similarity_top_k=k,
            response_mode="compact",
            service_context=service_context,
        )  # response mode can also be parameterized
        engines["original"] = query_engine

        if rerank:
            cohere_rerank = CohereRerank(api_key=cohere.api_key, top_n=k)

            engines["original_rerank"] = query_engine

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

                @retry(
                    wait=wait_random_exponential(min=1, max=60),
                    stop=stop_after_attempt(6),
                )
                def query_with_retry():
                    print(f"TRYING REQUEST {i + 1}")
                    print(f"QUERY:\n{query}\n")
                    try:
                        response = engine.query(query)
                        return response
                    except:
                        print(f"REQUEST {i + 1} FAILED. RETRYING")

                time_start = time.time()
                response = query_with_retry()
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
                scores = [source_node.score for source_node in response.source_node]
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

                # get the evaluation of the response
                # res_eval = format_evals([evaluator.evaluate(query, response)]) # don't need this for now

                row = (
                    [query, response]
                    + cpis
                    + acpk
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
        2000,
        # 2500,
    ]  # change this, perhaps experiment from 500 to 3000 in increments of 500
    k = 5  # num documents to retrieve

    # use direct call to openai instead, leaving here incase needed in the futre
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4"))
    # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    # evaluator = QueryResponseEvaluator(service_context=service_context)

    # query transformers
    # transformations = ["hyde", "multistep"] # ignore this one for now
    transformations = ["original_rerank", "hyde", "hyde_rerank"]

    all_data = run_experiments(
        documents,
        questions[:5],
        chunk_sizes,
        transformations,
        k,
        web_title,
        rerank=True,
    )

    # save data to disk
    save_dir = f"./experiment_data/{web_title}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{save_dir}/{web_title}_all_data.pkl", "wb") as f:
        pickle.dump(all_data, f)
    plot_precision_graphs(all_data, k, web_title, show=False)
    plot_latency_graphs(all_data, web_title, show=False)
    # plot_response_evaluation_graphs(all_data, web_title) # work in progress


program_start = time.time()
main()
program_end = time.time()
total_time = (program_end - program_start) / (60 * 60)
print(f"EXPERIMENTS FINISHED: {total_time:.2f} hrs")
