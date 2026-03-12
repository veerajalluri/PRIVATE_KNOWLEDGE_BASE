from dataclasses import dataclass

from injector import inject, singleton
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.chat_engine.types import (
    BaseChatEngine,
)
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.sql.mcp_bi_service import MCPBiService
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk
from private_gpt.settings.settings import Settings


class Completion(BaseModel):
    response: str
    sources: list[Chunk] | None = None


class CompletionGen(BaseModel):
    response: TokenGen
    sources: list[Chunk] | None = None


@dataclass
class ChatEngineInput:
    system_message: ChatMessage | None = None
    last_message: ChatMessage | None = None
    chat_history: list[ChatMessage] | None = None

    @classmethod
    def from_messages(cls, messages: list[ChatMessage]) -> "ChatEngineInput":
        # Detect if there is a system message, extract the last message and chat history
        system_message = (
            messages[0]
            if len(messages) > 0 and messages[0].role == MessageRole.SYSTEM
            else None
        )
        last_message = (
            messages[-1]
            if len(messages) > 0 and messages[-1].role == MessageRole.USER
            else None
        )
        # Remove from messages list the system message and last message,
        # if they exist. The rest is the chat history.
        if system_message:
            messages.pop(0)
        if last_message:
            messages.pop(-1)
        chat_history = messages if len(messages) > 0 else None

        return cls(
            system_message=system_message,
            last_message=last_message,
            chat_history=chat_history,
        )


@singleton
class ChatService:
    settings: Settings

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        mcp_bi_service: MCPBiService,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self._mcp_bi_service = mcp_bi_service

    def _chat_engine(self, system_prompt: str | None = None) -> BaseChatEngine:
        return SimpleChatEngine.from_defaults(
            system_prompt=system_prompt,
            llm=self.llm_component.llm,
        )

    def stream_chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> CompletionGen:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        if use_context:
            # NOTE: "use_context" (RAG mode in the UI) is repurposed as the
            # text-to-SQL path. Vector retrieval is bypassed entirely;
            # BISqlService translates the question to DuckDB SQL and executes
            # it against the pre-loaded bi.db written by prepare_bi_data.py.
            result = self._mcp_bi_service.query(last_message or "")
            return CompletionGen(response=iter([result]), sources=[])

        chat_engine = self._chat_engine(system_prompt=system_prompt)
        streaming_response = chat_engine.stream_chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in streaming_response.source_nodes]
        completion_gen = CompletionGen(
            response=streaming_response.response_gen, sources=sources
        )
        return completion_gen

    def chat(
        self,
        messages: list[ChatMessage],
        use_context: bool = False,
        context_filter: ContextFilter | None = None,
    ) -> Completion:
        chat_engine_input = ChatEngineInput.from_messages(messages)
        last_message = (
            chat_engine_input.last_message.content
            if chat_engine_input.last_message
            else None
        )
        system_prompt = (
            chat_engine_input.system_message.content
            if chat_engine_input.system_message
            else None
        )
        chat_history = (
            chat_engine_input.chat_history if chat_engine_input.chat_history else None
        )

        if use_context:
            # NOTE: "use_context" (RAG mode in the UI) is repurposed as the
            # text-to-SQL path. Vector retrieval is bypassed entirely;
            # BISqlService translates the question to DuckDB SQL and executes
            # it against the pre-loaded bi.db written by prepare_bi_data.py.
            result = self._mcp_bi_service.query(last_message or "")
            return Completion(response=result, sources=[])

        chat_engine = self._chat_engine(system_prompt=system_prompt)
        wrapped_response = chat_engine.chat(
            message=last_message if last_message is not None else "",
            chat_history=chat_history,
        )
        sources = [Chunk.from_node(node) for node in wrapped_response.source_nodes]
        completion = Completion(response=wrapped_response.response, sources=sources)
        return completion
