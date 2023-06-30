"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""

import openai
import cohere
from langchain.llms import Cohere
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
import os
import time
import json
import gzip
import requests
import argparse
from pdb import set_trace
from youtube_transcript_api import YouTubeTranscriptApi as yt
from youtube_transcript_api.formatters import TextFormatter


def download_youtube_transcript(url):
    video_id = url.split("watch?v=")[1]
    print(video_id)
    transcript = yt.get_transcript(video_id)
    formatter = TextFormatter()
    formatted_transcript = formatter.format_transcript(transcript)
    with open(f"./data/{video_id}.txt", "w", encoding="utf-8") as text_file:
        text_file.write(formatted_transcript)


class ChatBot:
    def __init__(self, name="Arize AI"):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        cohere.api_key = os.environ["COHERE_API_KEY"]
        self.name = name

        template = """Assistant is a large language model trained by Cohere.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

        {history}`
        Human: {human_input}
        Assistant:"""

        prompt = PromptTemplate(
            input_variables=["history", "human_input"], template=template
        )
        self.chain = LLMChain(
            llm=Cohere(
                model="command-light", cohere_api_key=cohere.api_key, temperature=0.1
            ),
            prompt=prompt,
            verbose=False,
            memory=ConversationBufferWindowMemory(k=10),
        )

    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()

    def chat(self):
        while True:
            user_input = input("You:\n")
            print("")
            output = self.chain.predict(human_input=user_input)
            print(f"{self.name}: ", output)


def main():
    url = "https://www.youtube.com/watch?v=vvTOfV38MnM"
    download_youtube_transcript(url)

    # chat_bot = ChatBot()
    # chat_bot.chat()


if __name__ == "__main__":
    main()
