"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""

import itertools
import zipfile

import openai
import cohere
import pinecone

from langchain.llms import OpenAI, Cohere
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.vectorstores import Pinecone, Chroma
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import YoutubeLoader, GitbookLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import json
import gzip
import requests
import argparse
from pdb import set_trace

# from youtube_transcript_api import YouTubeTranscriptApi as yt
# from youtube_transcript_api.formatters import TextFormatter

def batch_embeddings(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size.
    
    From Pinecone's documentation"""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

class ChatBot:
    def __init__(self, chain, name="Arize AI", data_path=None):
        self.name = name
        self.data_path = data_path
        self.chain = chain

    def chat(self):
        while True:
            user_input = input("You:\n")
            print("\n\n")
            output = self.chain.predict(human_input=user_input)
            print(f"{self.name}: ", output)


def main():
    pinecone_environment = "us-west1-gcp-free"
    pinecone.api_key = os.environ["PINECONE_API_KEY"]
    os.environ["OPENAI_API_KEY"] = "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"
    openai.api_key = os.environ["OPENAI_API_KEY"]
    cohere.api_key = os.environ["COHERE_API_KEY"]
    pinecone.init(api_key=pinecone.api_key, environment=pinecone_environment)

    # yt_url = "https://www.youtube.com/watch?v=ibjUpk9Iagk"
    # video_id = yt_url.split("watch?v=")[1]
    # youtube_documents = YoutubeLoader(video_id).load()

    

    # docs_url = "https://docs.arize.com/arize/"
    # index_name = "arize-docs-ada-002" # the name of the index containing your docs' embeddings, for pinecone

    embedding_model_name = "text-embedding-ada-002"
    embedding_model = OpenAIEmbeddings(model=embedding_model_name)
    # load and parse data
    print("Loading data...")
    # documents = GitbookLoader(docs_url, load_all_paths=True).load()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # document_chunks = text_splitter.split_documents(documents)
    # texts = [chunk.page_content for chunk in document_chunks]
    
    # create embeddings for each chunk of text
    # print("Generating embeddings...")
    
    # embedded_documents = embedding_model.embed_documents(texts)
    # embed_dim = len(embedded_documents[0])

    # create a pinecone index, connect to it, and upload emebddings
    # print("Uploading embeddings to Pincone...")
    # pinecone.create_index(index_name, 1536, metric="cosine")
    # pinecone.Index(index_name)
    # vectorstore = Pinecone.from_texts(texts, embedding_model, index_name=index_name)
    # vectorstore = Chroma.from_documents(youtube_documents, embedding_model)
    string_list_read = []
    with zipfile.ZipFile('output.zip', 'r') as zip_file:
        for filename in zip_file.namelist():
            with zip_file.open(filename, 'r') as file:
                string = file.read().decode('utf-8')
                string_list_read.append(string)


    # vectorstore = Chroma.from_texts(string_list_read, embedding_model, persist_directory="./chroma_db")
    # vectorstore.persist() # save to disk

    # load from disk
    vectorstore = Chroma(embedding_function=embedding_model, persist_directory="./chroma_db")
    print("Done")


    llm_model_name = "gpt-3.5-turbo"

    # llm = ChatOpenAI(temperature=0.2)
    llm = OpenAI(model_name=llm_model_name, temperature=0.2)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, vectorstore.as_retriever(), memory=memory
    )

    while True:
        user_input = input("\n\nYou:\n")
        output = qa_chain({"question": user_input})["answer"]
        print("\n\nArize Chat Bot: ", output)

    # chat_bot = ChatBot(qa_chain)
    # chat_bot.chat()


if __name__ == "__main__":
    main()
