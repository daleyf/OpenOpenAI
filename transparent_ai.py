"""Transparent AI core module.

Provides a wrapper around the OpenAI Chat Completions API (and optional local
``ollama`` models) that exposes exactly which messages are sent to the model,
returns log probabilities and token counts, and supports a developer bypass
mode.

Usage::

    from transparent_ai import TransparentAI
    ai = TransparentAI()
    result = ai.generate("Explain transformers in 2 sentences.")
    print(result["output"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import json
import urllib.request

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

    def generate(self, user_input: str, dev_mode: bool = False) -> Dict[str, Any]:
        """Generates a response from either OpenAI or a local ``ollama`` model.

        Returns a dictionary containing the final output, model used, messages
        sent and log probabilities/token counts when available. Local models are
        selected by prefixing the model name with ``"ollama/"``.
        """
        messages = self.build_messages(user_input, dev_mode)
        if self.model.startswith("ollama/"):
            local_model = self.model.split("/", 1)[1]
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": local_model, "prompt": prompt}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            output = data.get("response", "")
            return {
                "model": local_model,
                "output": output,
                "logprobs": None,
                "messages": messages,
                "tokens": None,
            }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
        )
        choice = response.choices[0]
        token_count = len(choice.logprobs["content"]) if choice.logprobs else 0
        return {
            "model": response.model,
            "output": choice.message.content,
            "logprobs": choice.logprobs,
            "messages": messages,
            "tokens": token_count,
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
    print("TOKENS:", result.get("tokens"))
    print("LOGPROBS:", result.get("logprobs"))
    print("MESSAGES:", result["messages"])
