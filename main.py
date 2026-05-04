from __future__ import annotations

import argparse

from config import DEFAULT_THREAD_ID
from document_loader import load_and_split_documents
from rag_agent import RagAgent
from vector_store import ingest_documents


TEST_QUESTIONS = [
    "What is the company's leave policy?",
    "How many vacation days do employees get?",
    "What are the steps in the offboarding process?",
    "What are the IT security requirements for new employees?",
    "What is the performance review process?",
    "How do I submit travel expenses for reimbursement?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="HR RAG Chatbot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load documents into ChromaDB")
    ingest_parser.add_argument("--reset", action="store_true", help="Rebuild ChromaDB")

    chat_parser = subparsers.add_parser("chat", help="Start CLI chatbot")
    chat_parser.add_argument("--thread-id", default=DEFAULT_THREAD_ID)

    test_parser = subparsers.add_parser("test", help="Run homework test questions")
    test_parser.add_argument("--thread-id", default="homework-test-session")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(reset=args.reset)
    elif args.command == "chat":
        run_chat(thread_id=args.thread_id)
    elif args.command == "test":
        run_tests(thread_id=args.thread_id)


def run_ingest(reset: bool = False) -> None:
    documents = load_and_split_documents()
    ingest_documents(documents, reset=reset)
    print(f"Ingested {len(documents)} chunks into ChromaDB using OpenRouter free embeddings.")


def run_chat(thread_id: str) -> None:
    chatbot = RagAgent(thread_id=thread_id)
    print("HR RAG Chatbot started. Type 'exit' to quit.")

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            answer = chatbot.ask(question)
            print(f"Bot: {answer}")
    finally:
        chatbot.close()


def run_tests(thread_id: str) -> None:
    chatbot = RagAgent(thread_id=thread_id)
    print("Running homework test questions.\n")

    try:
        for question in TEST_QUESTIONS:
            print(f"You: {question}")
            print(f"Bot: {chatbot.ask(question)}\n")

        print("Short-term memory check:")
        first_question = "What is the leave policy?"
        follow_up = "What about sick leave?"
        print(f"You: {first_question}")
        print(f"Bot: {chatbot.ask(first_question)}")
        print(f"You: {follow_up}")
        print(f"Bot: {chatbot.ask(follow_up)}")
    finally:
        chatbot.close()


if __name__ == "__main__":
    main()
