"""Minimal command-line interface for TransparentAI."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, List

from transparent_ai import TransparentAI


def build_rag_from_file(path: Path) -> Callable[[str], List[str]]:
    """Return a RAG function that feeds the entire file as context."""

    text = path.read_text(encoding="utf-8")

    def _rag(_query: str) -> List[str]:
        return [text]

    return _rag


def main() -> None:
    parser = argparse.ArgumentParser(description="TransparentAI CLI")
    parser.add_argument("prompt", nargs="?", default="What is RAG?", help="Question to ask")
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--dev", action="store_true", help="Bypass system prompt and RAG")
    parser.add_argument("--docs", type=Path, help="Path to text file for RAG context")
    args = parser.parse_args()

    rag_fn = build_rag_from_file(args.docs) if args.docs else None

    ai = TransparentAI(model=args.model, rag_fn=rag_fn)
    result = ai.generate(args.prompt, dev_mode=args.dev)
    print("MODEL:", result["model"])
    print("OUTPUT:\n", result["output"])
    print("MESSAGES:\n", result["messages"])
    print("LOGPROBS:\n", result["logprobs"])


if __name__ == "__main__":
    main()
