"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""

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
    openai.api_key = os.environ["OPENAI_API_KEY"]
    cohere.api_key = os.environ["COHERE_API_KEY"]
    pinecone.init(api_key=pinecone.api_key, environment=pinecone_environment)

    # set_trace()
    # yt_url = "https://www.youtube.com/watch?v=ECLJ95XNxA4"
    # video_id = yt_url.split("watch?v=")[1]
    # documents = YoutubeLoader(video_id).load()

    docs_url = "https://docs.arize.com/arize/"
    documents = GitbookLoader(docs_url, load_all_paths=True).load()
    set_trace()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = Chroma.from_documents(document_chunks, embeddings)

    embedding_dim = 1024
    llm_model_name = "gpt-3.5-turbo"

    # chat_model = ChatOpenAI(model_name=llm_model_name, temperature=0.2)
    llm = OpenAI(temperature=0.2)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, vectorstore.as_retriever(), memory=memory
    )

    chat_history = []
    while True:
        user_input = input("You:\n")
        output = qa_chain({"question": user_input})["answer"]
        print("\n\nYoutube bot: ", output)

        chat_history.append((user_input, output))
    # chat_bot = ChatBot(qa_chain)
    # chat_bot.chat()


if __name__ == "__main__":
    main()
