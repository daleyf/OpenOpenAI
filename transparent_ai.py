"""Transparent AI module.
Provides TransparentAI class that interfaces with language models
and exposes full context and model details for transparency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - openai may not be installed
    OpenAI = None  # type: ignore


@dataclass
class TransparentAI:
    """A small helper around the OpenAI API exposing full context.

    Parameters
    ----------
    model: str
        Model name to query, e.g. ``"gpt-4o"``.
    system_prompt: str
        Optional system instruction used when ``dev_mode`` is ``False``.
    dev_mode: bool
        If ``True`` only the user message is sent to the model, mimicking
        direct access without additional instructions or retrieved context.
    stub_response: str
        Text returned when no OpenAI API key is configured. This allows the
        module to be run in environments without network access during tests.
    """

    model: str = "gpt-4o"
    system_prompt: str = "You are a helpful assistant."
    dev_mode: bool = False
    stub_response: str = "[stubbed response]"

    def _call_model(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to the model and return the text completion.

        Falls back to ``stub_response`` when the OpenAI client is not available
        or no API key is configured.
        """
        if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
            return self.stub_response

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            logprobs=True,
        )
        return response.choices[0].message.content

    def chat(self, user_input: str, context_docs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate a response and return full transparency data.

        Parameters
        ----------
        user_input: str
            End user's message.
        context_docs: list of str, optional
            Additional context strings retrieved via RAG or search.

        Returns
        -------
        dict
            Dictionary with ``model``, ``messages`` and ``output`` keys.
        """
        messages: List[Dict[str, str]] = []
        if not self.dev_mode:
            messages.append({"role": "system", "content": self.system_prompt})
            if context_docs:
                context = "\n".join(context_docs)
                messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": user_input})

        output_text = self._call_model(messages)
        return {"model": self.model, "messages": messages, "output": output_text}
