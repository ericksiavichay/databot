"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""

import itertools
import zipfile
import os
import time
import json
import gzip
import requests
import argparse
from pdb import set_trace

import openai
import pinecone

from typing import Dict, List, Optional, Tuple

from langchain.llms import OpenAI
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

from youtube_transcript_api import YouTubeTranscriptApi as yt
from youtube_transcript_api.formatters import TextFormatter

# from youtube_transcript_api import YouTubeTranscriptApi as yt
# from youtube_transcript_api.formatters import TextFormatter


class ChromaWrapper(Chroma):
    query_text_to_document_score_tuples = {}

    def __init__(self, callback, **kwargs):
        super().__init__(**kwargs)
        self.callback = callback

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter=None, namespace=None
    ):
        # print("INSIDE WRAPPER SIM SEARCH BY QUERY CALL")
        # print("WRAPER SIM QUERY:", query)
        document_score_tuples = super().similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            namespace=namespace,
        )
        self.query_text_to_document_score_tuples[query] = document_score_tuples
        # print(document_score_tuples)
        return document_score_tuples

    

    @property
    def retrieval_dataframe(self) -> pd.DataFrame:
        query_texts = []
        document_texts = []
        retrieval_ranks = []
        scores = []
        for (
            query_text,
            document_score_tuples,
        ) in self.query_text_to_document_score_tuples.items():
            for retrieval_rank, (document, score) in enumerate(document_score_tuples):
                query_texts.append(query_text)
                document_texts.append(document.page_content)
                retrieval_ranks.append(retrieval_rank)
                scores.append(score)
        return pd.DataFrame.from_dict(
            {
                "query_text": query_texts,
                "document_text": document_texts,
                "retrieval_rank": retrieval_ranks,
                "score": scores,
            }
        )


class RetrievalCallbackHandler(OpenAICallbackHandler):
    def __init__(self, evaluator=None, embedding_model=None):
        # super().__init__()
        self.evaluator = evaluator
        self.embedding_model = embedding_model
        self.retrieval_data = []  # eg: [query, response, documents, cost, score]
        self.query_embeddings = []
        self.response_embeddings = []
        self.total_cost = 0

    def on_chain_start(self, serialized, inputs, **kwargs):
        # print("IN CHAIN START")
        # print(inputs)
        # print(kwargs)
        self.query = inputs["query"]
        # embedding = self.embedding_model.embed_query(self.query)
        # self.query_embeddings.append(embedding)
        # print("query embedding", embedding)

    def on_chain_end(self, outputs, **kwargs):
        print("IN CHAIN END")
        # print(outputs)
        # self.response = outputs["result"]
        # embedding = self.embedding_model.embed_query(self.response)
        # self.response_embeddings.append(embedding)
        # print("response embedding", embedding)

        # TODO: evaluate the response using the evaluator given the question, answer, retrieved documents, correct answer, etc

    def on_llm_end(self, response, **kwargs):
        pass
        print("IN LLM END")
        self.total_cost = super().total_cost
        print(self.total_cost)

   

class ChatBot:
    def __init__(self, name="Arize AI", embedding_model=None, llm=None, memory=None):
        self.name = name
        self.embedding_model = embedding_model
        self.llm = llm
        self.memory = memory
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
                self.vectorstore.as_retriever(search_kwargs={"k": 2}),
                memory=self.memory,
            )
        else:
            self.qa_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
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
    openai.api_key = (
        "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"  # burch's key
    )
    # openai.api_key = "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" # my key
    os.environ["OPENAI_API_KEY"] = openai.api_key
    pinecone_environment = "us-west1-gcp-free"
    # pinecone.api_key = os.environ["PINECONE_API_KEY"]
    # openai.api_key = os.environ["OPENAI_API_KEY"]
    # pinecone.init(api_key=pinecone.api_key, environment=pinecone_environment)

    # string_list_read = []
    # with zipfile.ZipFile('output.zip', 'r') as zip_file:
    #     for filename in zip_file.namelist():
    #         with zip_file.open(filename, 'r') as file:
    #             string = file.read().decode('utf-8')
    #             string_list_read.append(string)

    # while True:
    #     user_input = input("\n\nYou:\n")
    #     output = qa_chain({"question": user_input})["answer"]
    #     print("\n\nArize Chat Bot: ", output)
    # data = pd.read_csv("arize_docs_questions.csv")
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
    chat_bot.qa_chain.run("How do I grant permissions to import my GBQ table?")


if __name__ == "__main__":
    main()
