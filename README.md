# Transparent AI

Tools to inspect exactly what prompts and context reach a model and what it
returns. The goal is to make model interactions auditable for developers.
It can query both OpenAI hosted models and local ``ollama`` models.

## Design Goals

- **Which model**: easily switch between GPT-4o or any other supported model.
- **What context**: show the final system prompt, retrieved documents and user
  message.
- **How output is generated**: expose the raw completion and token logprobs.
- **Bypass guardrails**: a dev mode sends only the user message when needed.

## Command Line

```bash
export OPENAI_API_KEY="sk-..."
python transparent_cli.py "What is RAG?"
python transparent_cli.py --docs docs.txt "Who wrote these docs?"
```

Use `--dev` to bypass the system prompt and any RAG context. Models may also be
prefixed with `ollama/` to call a local model, e.g. `--model ollama/llama2`.

## Web Interface

For non-technical users install `gradio` and run the browser UI:

```bash
pip install gradio
python transparent_web.py
```

Paste optional reference text into the RAG context box and toggle **Dev mode**
for direct model access. The right-hand panel displays the full message list,
token logprobs and total token count for each response.

## Extending

`TransparentAI` accepts a `rag_fn` callback for retrieving context. The included
`simple_rag` is a placeholder; replace it with a search or database lookup to
build a complete RAG system.
