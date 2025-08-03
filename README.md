# Chatbot and Dev tools to actually know what's going on inside the model
Access to direct model if needed, or you **choose** to turn on, or send more context :D

## All outputs must be clearly explained as far as system prompt and RAG or search or anything the model does
Should be some combination of
- which model
- what context (huge)
- output

## TransparentAI script

`transparent_ai.py` exposes a `TransparentAI` class that forwards messages to an
OpenAI chat model while returning the full message context. A small CLI in
`main.py` demonstrates how the system works.

Example:

```bash
python main.py "Explain transformers in two sentences" --context "Transformers are attention based" --dev-mode
```

If the `OPENAI_API_KEY` environment variable is not configured the script will
return a stubbed response so the repository can be tested offline.
