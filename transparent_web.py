"""Gradio web interface for TransparentAI.

Allows non-technical users to interact with the wrapper from a browser.
"""

from __future__ import annotations

from typing import List, Tuple

import gradio as gr

from transparent_ai import TransparentAI


def ask(question: str, model: str, dev_mode: bool, docs: str) -> Tuple[str, List[dict], dict, str, int]:
    """Run a query through ``TransparentAI`` and return detailed data."""
    rag_fn = (lambda _query: [docs]) if docs else None
    ai = TransparentAI(model=model, rag_fn=rag_fn)
    result = ai.generate(question, dev_mode=dev_mode)
    return (
        result["output"],
        result["messages"],
        result.get("logprobs"),
        result["model"],
        result.get("tokens", 0),
    )


def main() -> None:
    with gr.Blocks() as demo:
        gr.Markdown("# Transparent AI")
        with gr.Row():
            with gr.Column():
                question = gr.Textbox(label="Question", lines=2)
                docs = gr.Textbox(
                    label="RAG Context (optional)",
                    lines=2,
                    placeholder="Paste reference text here",
                )
                model = gr.Textbox(label="Model", value="gpt-4o")
                dev = gr.Checkbox(label="Dev mode (bypass system prompt)")
                run = gr.Button("Run")
                used_model = gr.Textbox(label="Model used", interactive=False)
            with gr.Column():
                output = gr.Textbox(label="Output", lines=10)
            with gr.Column():
                messages = gr.JSON(label="Messages")
                logprobs = gr.JSON(label="Log probabilities")
                tokens = gr.Number(label="Token count", precision=0)
        run.click(
            ask,
            [question, model, dev, docs],
            [output, messages, logprobs, used_model, tokens],
        )
    demo.launch()


if __name__ == "__main__":
    main()
