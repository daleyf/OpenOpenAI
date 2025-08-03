"""Transparent AI core module.

Provides a simple wrapper around the OpenAI Chat Completions API that exposes
exactly which messages are sent to the model and returns logprobs for full
transparency.

Usage:
    from transparent_ai import TransparentAI
    ai = TransparentAI()
    result = ai.generate("Explain transformers in 2 sentences.")
    print(result["output"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from openai import OpenAI


@dataclass
class TransparentAI:
    """Small wrapper around OpenAI's API exposing prompts and logprobs."""

    model: str = "gpt-4o"
    system_prompt: str = "You are a helpful assistant."
    rag_fn: Optional[Callable[[str], List[str]]] = None
    client: OpenAI = field(default_factory=OpenAI)

    def build_messages(self, user_input: str, dev_mode: bool) -> List[Dict[str, str]]:
        """Constructs the message list sent to the model.

        If ``dev_mode`` is True, only the raw user message is included. Otherwise
        the system prompt and any retrieved RAG context is prepended.
        """
        messages: List[Dict[str, str]] = []
        if dev_mode:
            messages.append({"role": "user", "content": user_input})
            return messages

        messages.append({"role": "system", "content": self.system_prompt})
        if self.rag_fn:
            for chunk in self.rag_fn(user_input):
                messages.append({"role": "system", "content": chunk})
        messages.append({"role": "user", "content": user_input})
        return messages

    def generate(self, user_input: str, dev_mode: bool = False) -> Dict[str, object]:
        """Generates a response from the model.

        Returns a dictionary containing the final output, the model used, the
        messages that were sent to the API and the log probabilities for each
        generated token.
        """
        messages = self.build_messages(user_input, dev_mode)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
        )
        choice = response.choices[0]
        return {
            "model": response.model,
            "output": choice.message.content,
            "logprobs": choice.logprobs,
            "messages": messages,
        }


def simple_rag(query: str) -> List[str]:
    """Very small placeholder RAG function.

    In a real system this would issue a search and return relevant context
    chunks. Here we simply return an empty list to illustrate the interface.
    """
    return []


if __name__ == "__main__":
    import os
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else "What is RAG?"
    dev_mode = bool(os.environ.get("DEV_MODE"))
    ai = TransparentAI(rag_fn=simple_rag)
    result = ai.generate(question, dev_mode=dev_mode)
    print("MODEL:", result["model"])
    print("OUTPUT:", result["output"])
    print("TOKENS:", result["logprobs"])
    print("MESSAGES:", result["messages"])
