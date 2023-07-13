"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""


import os
from typing import Dict, List, Optional, Tuple
import numpy as np

import time

import openai

from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, Chroma
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import YoutubeLoader, GitbookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks import OpenAICallbackHandler

import pandas as pd

class ChromaWrapper(Chroma):
    query_text_to_document_score_tuples = {}

    def __init__(self, callback=None, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter=None, namespace=None
    ):
        # print("INSIDE WRAPPER SIM SEARCH BY QUERY CALL")
        # print("WRAPER SIM QUERY:", query)
        embedding = self._embedding_function.embed_query(query)
        document_score_tuples = self.similarity_search_by_vector_with_relevance_scores(
            embedding=embedding,
            k=k,
            filter=filter,
            namespace=namespace,
        )
        # print("WRAPPER SIM SEARCH RETRIEVED DOCUMENTS AND SCORES", document_score_tuples)

        if self.callback:
            self.callback.query = query
            self.callback.query_embedding = embedding
            
            document_texts = []
            document_embeddings = []
            document_scores = []
            for doc, score in document_score_tuples:
                document_texts.append(doc.page_content)
                document_embeddings.append(self._embedding_function.embed_query(doc.page_content)) # may not have to do this if chroma is set up right
                document_scores.append(score)

            self.callback.document_texts = document_texts
            self.callback.document_embeddings = document_embeddings
            self.callback.document_scores = document_scores

        # print(document_score_tuples)
        return document_score_tuples

class RetrievalCallbackHandler(OpenAICallbackHandler):
    def __init__(self, embedding_model=None):
        # super().__init__()
        self.embedding_model = embedding_model
        self.retrieval_data = {
            'queries': [],
            'query_embeddings': [],
            'responses': [],
            'response_embeddings': [],
            'total_completion_costs': [],
            'retrieved_contexts': [],
            'retrieved_context_embeddings': [],
            'retrieved_context_cosine_similarities': [],
            'retrieved_context_relevancy_scores': [],
            'precision_at_ks': [],
            'latencies': []
        }
        self.MODEL_COST_PER_1K_TOKENS = {
                                # GPT-4 input
                                "gpt-4": 0.03,
                                "gpt-4-0314": 0.03,
                                "gpt-4-0613": 0.03,
                                "gpt-4-32k": 0.06,
                                "gpt-4-32k-0314": 0.06,
                                "gpt-4-32k-0613": 0.06,
                                # GPT-4 output
                                "gpt-4-completion": 0.06,
                                "gpt-4-0314-completion": 0.06,
                                "gpt-4-0613-completion": 0.06,
                                "gpt-4-32k-completion": 0.12,
                                "gpt-4-32k-0314-completion": 0.12,
                                "gpt-4-32k-0613-completion": 0.12,
                                # GPT-3.5 input
                                "gpt-3.5-turbo": 0.0015,
                                "gpt-3.5-turbo-0301": 0.0015,
                                "gpt-3.5-turbo-0613": 0.0015,
                                "gpt-3.5-turbo-16k": 0.003,
                                "gpt-3.5-turbo-16k-0613": 0.003,
                                # GPT-3.5 output
                                "gpt-3.5-turbo-completion": 0.002,
                                "gpt-3.5-turbo-0301-completion": 0.002,
                                "gpt-3.5-turbo-0613-completion": 0.002,
                                "gpt-3.5-turbo-16k-completion": 0.004,
                                "gpt-3.5-turbo-16k-0613-completion": 0.004,
                                # Others
                                "gpt-35-turbo": 0.002,  # Azure OpenAI version of ChatGPT
                                "text-ada-001": 0.0004,
                                "ada": 0.0004,
                                "text-babbage-001": 0.0005,
                                "babbage": 0.0005,
                                "text-curie-001": 0.002,
                                "curie": 0.002,
                                "text-davinci-003": 0.02,
                                "text-davinci-002": 0.02,
                                "code-davinci-002": 0.02,
                                "ada-finetuned": 0.0016,
                                "babbage-finetuned": 0.0024,
                                "curie-finetuned": 0.012,
                                "davinci-finetuned": 0.12,
                                }


    # def on_chain_start(self, serialized, inputs, **kwargs):
    #     # print("IN CHAIN START")
    #     # print(inputs)
    #     # print(kwargs)
    #     self.query = inputs["query"]
    #     # embedding = self.embedding_model.embed_query(self.query)
    #     # self.query_embeddings.append(embedding)
    #     # print("query embedding", embedding)

    

    def standardize_model_name(self,
        model_name: str,
        is_completion: bool = False,
    ) -> str:
        """
        Standardize the model name to a format that can be used in the OpenAI API.
        Args:
            model_name: Model name to standardize.
            is_completion: Whether the model is used for completion or not.
                Defaults to False.

        Returns:
            Standardized model name.

        """
        model_name = model_name.lower()
        if "ft-" in model_name:
            return model_name.split(":")[0] + "-finetuned"
        elif is_completion and (
            model_name.startswith("gpt-4") or model_name.startswith("gpt-3.5")
        ):
            return model_name + "-completion"
        else:
            return model_name


    def get_openai_token_cost_for_model(self,
        model_name: str, num_tokens: int, is_completion: bool = False
    ) -> float:
        """
        Get the cost in USD for a given model and number of tokens.

        Args:
            model_name: Name of the model
            num_tokens: Number of tokens.
            is_completion: Whether the model is used for completion or not.
                Defaults to False.

        Returns:
            Cost in USD.
        """
        model_name = self.standardize_model_name(model_name, is_completion=is_completion)
        if model_name not in self.MODEL_COST_PER_1K_TOKENS:
            raise ValueError(
                f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
                "Known models are: " + ", ".join(self.MODEL_COST_PER_1K_TOKENS.keys())
            )
        return self.MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)

    def summarize_system(self):
        # column_names = [name for name in self.retrieval_data.keys()]
        # df = pd.DataFrame(self.retrieval_data, columns=column_names)

        # key stats
        # average precision at each k
        # total cost

        avg_precisions = np.mean(self.retrieval_data["precision_at_ks"], axis=0)
        total_cost = np.sum(self.retrieval_data["total_completion_costs"])
        avg_latency = np.mean(self.retrieval_data["latencies"])

        print("Average Precisions at each k: ", avg_precisions)
        print(f"[WIP] Total System Cost: ${total_cost:.2f}")
        print(f"Average Response Latency: {avg_latency:.2f}s")

    def evaluate_query_and_context(self, query, context):
        EVALUATION_SYSTEM_MESSAGE = "You will be given a query and a reference text. You must determine whether the reference text contains an answer to the input query. Your response must be binary (0 or 1) and should not contain any text or characters aside from 0 or 1. 0 means that the reference text does not contain an answer to the query. 1 means the reference text contains an answer to the query."
        QUERY_CONTEXT_PROMPT_TEMPLATE = """Query: {query}
        Reference: {reference}
        """


        prompt = QUERY_CONTEXT_PROMPT_TEMPLATE.format(
            query=query,
            reference=context,
        )
        res = openai.ChatCompletion.create(
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4",
            temperature=0
        )
        response = res["choices"][0]["message"]["content"]
        return int(response)

    def compute_embedding_price(self, text):
        pass

    def on_chain_start(
            self,
            serialized,
            inputs,
            *,
            run_id,
            parent_run_id,
            tags: Optional[List[str]] = None,
            metadata,
            **kwargs,
    ):
        self.time_start = time.time()

    def on_chain_end(self, outputs, **kwargs):
        self.response = outputs["result"]
        self.time_end = time.time()
        self.latency = self.time_end - self.time_start
        print(f"RESPONSE LATENCY: {self.latency:.2f}")
        embedding = self.embedding_model.embed_query(self.response)
        self.response_embedding = embedding

        self.evals = []
        self.p_at_ks = []

        # compute relevancy score via LLM and precision at k for each k
        for index, retrieved_context in enumerate(self.document_texts):
            eval = self.evaluate_query_and_context(self.query, retrieved_context)
            self.evals.append(eval)

            current_k = index + 1
            p_at_current_k = sum(self.evals) / current_k
            self.p_at_ks.append(p_at_current_k)

        row = (self.query, self.query_embedding, self.response, self.response_embedding, self.total_cost, self.document_texts, self.document_embeddings, self.document_scores, self.evals, self.p_at_ks, self.latency)
        for key, data in zip(self.retrieval_data, row):
            self.retrieval_data[key].append(data)

    # def on_llm_start(self, serialized, prompts, **kwargs):
    #     print("IN LLM START")
    #     # print("PROMPTS:", prompts[0])
    #     # print(kwargs)


    def on_llm_end(self, response, **kwargs):
        # print("IN LLM END")
        # print("LLM RESPONSE:", response.generations[0][0].text)
        # print("LLM PROMPT TOKENS:", response.llm_output["token_usage"]["prompt_tokens"])
        # print("LLM RESPONSE TOKENS:", response.llm_output["token_usage"]["completion_tokens"])
        # print("LLM CURRENT RUN TOTAL TOKEN USAGE:", response.llm_output["token_usage"]["total_tokens"])
        model_name = self.standardize_model_name(response.llm_output.get("model_name", ""))
        if model_name in self.MODEL_COST_PER_1K_TOKENS:
            completion_cost = self.get_openai_token_cost_for_model(
                model_name, response.llm_output["token_usage"]["completion_tokens"], is_completion=True
            )
            prompt_cost = self.get_openai_token_cost_for_model(model_name, response.llm_output["token_usage"]["prompt_tokens"])
            self.total_cost = prompt_cost + completion_cost

        print("LLM NET GENERATION COST:", self.total_cost)
        # print(response)

   

class ChatBot:
    def __init__(self, name="Arize AI", embedding_model=None, llm=None, memory=None, k=4):
        self.name = name
        self.embedding_model = embedding_model
        self.llm = llm
        self.memory = memory
        self.k = k
        self.vectorstore = None

    def vectorstore_from_url(self, url, persist_directory="./chroma_db"):
        """
        Assumes URL is a Gitbook url. Can take a long time. This function can be modified to load
        other types of data listed here:
        https://github.com/hwchase17/langchain/tree/04001ff0778d88a644fd20accf2bdaef0ef3258d/langchain/document_loaders
        """
        assert self.embedding_model is not None, "Error: there's no embedding_model"
        print("Loading documents...")
        documents = GitbookLoader(url, load_all_paths=True).load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        print("Chunking...")
        document_chunks = text_splitter.split_documents(documents)
        print("Generating embeddings...")
        self.vectorstore = Chroma.from_documents(
            document_chunks, self.embedding_model, persist_directory=persist_directory
        )
        self.vectorstore.persist()

    def vectorstore_from_disk(self, persist_directory="./chroma_db", callback=None):
        assert self.embedding_model is not None, "Error: there's no embedding_model"
        self.vectorstore = ChromaWrapper(
            embedding_function=self.embedding_model,
            persist_directory=persist_directory,
            callback=callback,
        )

    def build_chain(self, callbacks=None):
        assert self.llm is not None, "Error: no LLM"
        assert self.vectorstore is not None, "Error: no vectorstore loaded"
        if self.memory:
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                self.llm,
                self.vectorstore.as_retriever(k=self.k),
                memory=self.memory,
            )
        else:
            self.qa_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k":self.k}),
                callbacks=callbacks,
            )


    def chat(self):
        assert self.qa_chain is not None, "Error: no chain"
        while True:
            user_input = input("\n\nYou:\n")
            output = self.qa_chain.run(user_input)
            print(f"\n\n{self.name}: \n{output:<5}")

    def debug_function(self):
        pass


def main():
    # openai.api_key = "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" # pretty sure this is jason's, i might have hit the limit rate, lo siento
    # openai.api_key = (
    #     "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"  # burch's key
    # )
    openai.api_key = "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" # my key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    # pinecone_environment = "us-west1-gcp-free"
    # pinecone.api_key = os.environ["PINECONE_API_KEY"]
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # pinecone.init(api_key=pinecone.api_key, environment=pinecone_environment)

    # string_list_read = []
    # with zipfile.ZipFile('./output.zip', 'r') as zip_file:
    #     for filename in zip_file.namelist():
    #         with zip_file.open(filename, 'r') as file:
    #             string = file.read().decode('utf-8')
    #             string_list_read.append(string)

    # while True:
    #     user_input = input("\n\nYou:\n")
    #     output = qa_chain({"question": user_input})["answer"]
    #     print("\n\nArize Chat Bot: ", output)
    # answers = []
    # for question in data["Question"]:
    #     print("Trying: ", question)
    #     answers.append(qa_chain.run(question))

    url = "https://docs.arize.com/arize/"
    embedding_model_name = "text-embedding-ada-002"
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    retrieval_callback_handler = RetrievalCallbackHandler(
        embedding_model=embedding_model
    )

    llm_model_name = "gpt-3.5-turbo"
    llm = OpenAI(
        model_name=llm_model_name, temperature=0, callbacks=[retrieval_callback_handler]
    )

    inputs = {
        "embedding_model": embedding_model,
        "llm": llm,
        "k": 2
    }

    # initialize the chatbot
    chat_bot = ChatBot(**inputs)

    # documentation chunk vectorstore loading option 1: from disk
    # path = "/content/drive/MyDrive/Professional/Arize AI/data/chroma_db"
    path = "./chroma_db"
    chat_bot.vectorstore_from_disk(path, callback=retrieval_callback_handler)

    # vector store loading option 2: from url (may take a while)
    # chat_bot.vectorstore_from_url(url)

    # build the question answering retrieval chain
    chat_bot.build_chain(callbacks=[retrieval_callback_handler])

    sheet_data = pd.read_csv("arize_docs_questions.csv")

    for question in sheet_data["Question"]:
        print("\n\n")
        print("ATTEMPTING QUESTION:", question)
        print(chat_bot.qa_chain.run(question))
        print("\n\n")

    retrieval_callback_handler.summarize_system()

    print("done")


if __name__ == "__main__":
    main()
