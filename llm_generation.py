#!/usr/bin/env python3
"""
llm_generation.py
OpenAI Responses API backend for RAG system.
"""
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

class LLMGenerator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Wrap the OpenAI Responses API.
        """
        self.client = OpenAI()
        self.model = model

    def generate(
    self,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
    system_message: Optional[str] = None,
    ) -> str:
        if system_message is None:
            system_message = (
                "You are a helpful assistant that answers questions ONLY using "
                "the provided context documents. If the documents are not "
                "enough to answer, say you are not sure."
            )

        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        return (response.output_text or "").strip()