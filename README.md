# Transparent AI

Tools to inspect exactly what prompts and context reach a model and what it
returns.  The goal is to make model interactions auditable for developers.

## Design Goals

- **Which model**: easily switch between GPT-4o or any other supported model.
- **What context**: show the final system prompt, retrieved documents and user
  message.
- **How output is generated**: expose the raw completion and token logprobs.
- **Bypass guardrails**: a dev mode sends only the user message when needed.

## Example

```bash
export OPENAI_API_KEY="sk-..."
python transparent_ai.py "What is RAG?"
```

Set `DEV_MODE=1` to bypass the system prompt and RAG pipeline.

## Extending

`TransparentAI` accepts a `rag_fn` callback for retrieving context.  The
included `simple_rag` is a placeholder; replace it with a search or database
lookup to build a complete RAG system.
