"""
Author: Erick Siavichay
Inspiration from Arize's chatbot files
A simple chatbot over a dataset. 
"""

import openai
import anthropic
import os
import time
import json
import gzip
import requests
import argparse
from pdb import set_trace


class ChatBot:
    def __init__(self, data_filename, name="Arize AI", forget=True):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.name = name
        self.forget = forget
      


    @staticmethod
    def read_file(file_path):
        with open(file_path, "r") as file:
            return file.read()

    def get_anthropic_response(self, human_prompt=None):
        if human_prompt:
            if self.forget:
                final_prompt = (
                    self.context_anthropic
                    + f"{anthropic.HUMAN_PROMPT} {human_prompt} {anthropic.AI_PROMPT}"
                )
            else:
                self.context_anthropic += (
                    f"{anthropic.HUMAN_PROMPT} {human_prompt} {anthropic.AI_PROMPT}"
                )
                final_prompt = self.context_anthropic
        c = anthropic.Client(self.anthropic_api_key)
        response = c.completion(
            prompt=final_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            model=self.model,
            max_tokens_to_sample=40000,
        )

        if not self.forget:
            self.context_anthropic += response["completion"]
        return response

    def get_openai_response(self, prompt=None):
        if prompt:
            self.context_openai.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.context_openai,
            temperature=0,
        )
        ai_message = response["choices"][0]["message"]["content"].strip()
        self.context_openai.append({"role": "assistant", "content": ai_message})
        return response

    def chat(self):
        while True:
            if self.company == "openai":
                prompt = input(f"\n[{num_tokens}/{self.max_tokens}]You: ")
                start_time = time.perf_counter()
                response = self.get_openai_response(prompt=prompt)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                num_tokens = response["usage"]["total_tokens"]
                model = response["model"]
                print(
                    f"\n[{num_tokens}/{self.max_tokens}][{total_time:.0f}s]{model}: "
                    + self.context_openai[-1]["content"]
                )
            elif self.company == "anthropic":
                print(
                    f"[{anthropic.count_tokens(self.context_anthropic)}/{self.get_max_tokens(self.model)}] Human:"
                )
                prompt = anthropic.HUMAN_PROMPT + " " + input()

                start_time = time.perf_counter()
                response = self.get_anthropic_response(human_prompt=prompt)
                end_time = time.perf_counter()
                total_time = end_time - start_time
                num_tokens = len(self.context_anthropic.split(" "))
                
                print(anthropic.AI_PROMPT, " " + response["completion"])


def main():
    chat_bot = ChatBot()
    chat_bot.chat()

if __name__ == "__main__":
    main()
