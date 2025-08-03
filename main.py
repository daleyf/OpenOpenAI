"""Simple CLI demonstrating TransparentAI usage."""

import argparse
from transparent_ai import TransparentAI


def main() -> None:
    parser = argparse.ArgumentParser(description="Transparent AI demo")
    parser.add_argument("message", help="User message to send to the model")
    parser.add_argument("--model", default="gpt-4o", help="Model to query")
    parser.add_argument("--dev-mode", action="store_true", help="Bypass system prompt and context")
    parser.add_argument(
        "--context",
        nargs="*",
        default=None,
        help="Optional context documents to supply",
    )
    args = parser.parse_args()

    ai = TransparentAI(model=args.model, dev_mode=args.dev_mode)
    result = ai.chat(args.message, context_docs=args.context)

    print("MODEL:", result["model"])
    print("MESSAGES:")
    for msg in result["messages"]:
        print(f"- {msg['role']}: {msg['content']}")
    print("OUTPUT:", result["output"])


if __name__ == "__main__":
    main()
