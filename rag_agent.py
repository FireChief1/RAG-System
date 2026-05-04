from __future__ import annotations

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.postgres import PostgresSaver

from config import (
    CHAT_MODEL,
    DEFAULT_THREAD_ID,
    OPENROUTER_BASE_URL,
    load_settings,
)
from vector_store import get_retriever


SYSTEM_PROMPT = """
You are a helpful HR document assistant.
Use the document search tool before answering HR policy questions.
Keep every answer short, ideally 2-3 sentences.
Always cite the source document using the file_name metadata.
If the documents do not contain the answer, say that you could not find it.
""".strip()


def build_retriever_tool():
    retriever = get_retriever()

    @tool
    def search_hr_documents(query: str) -> str:
        """Search HR documents and return relevant snippets with source files."""
        documents = retriever.invoke(query)
        if not documents:
            return "No relevant HR document snippets found."

        snippets = []
        for index, document in enumerate(documents, start=1):
            file_name = document.metadata.get("file_name", "unknown")
            page_number = document.metadata.get("page_number", "")
            source = f"{file_name}"
            if page_number:
                source = f"{source}, page {page_number}"

            snippets.append(
                "\n".join(
                    [
                        f"[Snippet {index}]",
                        f"Source: {source}",
                        document.page_content,
                    ]
                )
            )

        return "\n\n".join(snippets)

    return search_hr_documents


class RagAgent:
    def __init__(self, thread_id: str = DEFAULT_THREAD_ID):
        settings = load_settings()
        api_key = settings["openrouter_api_key"]
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for real agent mode.")
        if not CHAT_MODEL.endswith(":free") and CHAT_MODEL != "openai:openrouter/free":
            raise ValueError("Only OpenRouter free chat models are allowed.")

        self.thread_id = thread_id
        self._checkpointer_context = PostgresSaver.from_conn_string(settings["db_uri"])
        self.checkpointer = self._checkpointer_context.__enter__()
        self.checkpointer.setup()

        model = init_chat_model(
            CHAT_MODEL,
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
        )
        self.agent = create_agent(
            model=model,
            tools=[build_retriever_tool()],
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )

    def ask(self, question: str) -> str:
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": self.thread_id}},
        )
        messages = result.get("messages", [])
        if not messages:
            return "I could not produce an answer."
        return messages[-1].content

    def close(self) -> None:
        self._checkpointer_context.__exit__(None, None, None)
