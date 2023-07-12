"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""


import os
from typing import Dict, List, Optional, Tuple

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
        print("INSIDE WRAPPER SIM SEARCH BY QUERY CALL")
        print("WRAPER SIM QUERY:", query)
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
                document_embeddings.append(self._embedding_function.embed_query(doc.page_content))
                document_scores.append(score)

            self.callback.document_texts = document_texts
            self.callback.document_embeddings = document_embeddings
            self.callback.document_scores = document_scores

        return document_score_tuples

class RetrievalCallbackHandler(OpenAICallbackHandler):
    def __init__(self, embedding_model=None):
        # super().__init__()
        self.embedding_model = embedding_model
        self.retrieval_data = []  # eg: [query, response, documents, cost, score]

    # def on_chain_start(self, serialized, inputs, **kwargs):
    #     # print("IN CHAIN START")
    #     # print(inputs)
    #     # print(kwargs)
    #     self.query = inputs["query"]
    #     # embedding = self.embedding_model.embed_query(self.query)
    #     # self.query_embeddings.append(embedding)
    #     # print("query embedding", embedding)

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
        # return response["choices"][0]["message"]["content"]

    def compute_embedding_price(self, text):
        pass

    def on_chain_end(self, outputs, **kwargs):
        print("IN CHAIN END")
        self.response = outputs["result"]
        embedding = self.embedding_model.embed_query(self.response)
        self.response_embedding = embedding
        self.completion_cost = self.total_cost

        self.eval = self.evaluate_query_and_context(self.query, self.response)
        print("EVAL:", self.eval)

        data = (self.query, self.query_embedding, self.response, self.response_embedding, self.document_texts, self.document_embeddings, self.document_scores, self.eval)
        self.retrieval_data.append(data)
        print("done")


    # def on_llm_end(self, response, **kwargs):
    #     print("IN LLM END")
    #     self.total_cost = super().total_cost
    #     print(self.total_cost)

   

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
    chat_bot.qa_chain.run("what is the arize SDK?")

    print("done")


if __name__ == "__main__":
    main()
