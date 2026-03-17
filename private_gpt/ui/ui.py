"""Simplified BI chat UI — chatbot only, always uses the MCP SQL pipeline."""

import base64
import logging
import time
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore

# Patch gradio-client 1.3.0 bug: boolean JSON Schemas (e.g. additionalProperties: true)
# are not handled in get_type() or _json_schema_to_python_type(). Fixed in gradio 5+.
try:
    import gradio_client.utils as _gc_utils  # type: ignore

    _orig_get_type = _gc_utils.get_type

    def _patched_get_type(schema: object) -> str:
        if isinstance(schema, bool):
            return "any"
        return _orig_get_type(schema)  # type: ignore[arg-type]

    _gc_utils.get_type = _patched_get_type

    _orig_json_schema = _gc_utils._json_schema_to_python_type  # type: ignore[attr-defined]

    def _patched_json_schema(schema: object, defs: object = None) -> str:
        if isinstance(schema, bool):
            return "any"
        return _orig_json_schema(schema, defs)  # type: ignore[arg-type]

    _gc_utils._json_schema_to_python_type = _patched_json_schema  # type: ignore[attr-defined]
except Exception:
    pass

from collections.abc import Iterable

from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "BI Insights"
SOURCES_SEPARATOR = "<hr>Sources: \n"


@singleton
class PrivateGptUi:
    @inject
    def __init__(self, chat_service: ChatService) -> None:
        self._chat_service = chat_service
        self._ui_block = None

    def _chat(self, message: str, history: list[list[str]]) -> Iterable[str]:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            for delta in completion_gen.response:
                if isinstance(delta, str):
                    full_response += delta
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)
            yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []
            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]

        query_stream = self._chat_service.stream_chat(
            messages=all_messages,
            use_context=True,
        )
        yield from yield_deltas(query_stream)

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=(
                ".logo { display:flex; background-color: #C7BAFF; height: 80px;"
                " border-radius: 8px; align-content: center; justify-content: center;"
                " align-items: center; }"
                ".logo img { height: 25% }"
                ".contain { display: flex !important; flex-direction: column !important; }"
                "#component-0, #component-3, #component-10, #component-8 { height: 100% !important; }"
                "#chatbot { flex-grow: 1 !important; overflow: auto !important; }"
                "#col { height: calc(100vh - 112px - 16px) !important; }"
                "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFF; }"
                ".avatar-image { background-color: antiquewhite; border-radius: 2px; }"
                ".footer { text-align: center; margin-top: 20px; font-size: 14px;"
                " display: flex; align-items: center; justify-content: center; }"
            ),
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=BI-Insights></div")

            with gr.Row(equal_height=False):
                with gr.Column(scale=10, elem_id="col"):
                    gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label="Business Intelligence Assistant",
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(None, AVATAR_BOT),
                        ),
                        textbox=gr.Textbox(
                            placeholder="Ask a question about your business data…",
                            container=False,
                            scale=7,
                        ),
                    )

            with gr.Row():
                avatar_byte = AVATAR_BOT.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
                gr.HTML(
                    f"<div class='footer'>"
                    f"Powered by Claude + DuckDB"
                    f"<img style='height:20px;margin-left:8px;border-radius:2px;background:antiquewhite' "
                    f"src='{f_base64}' alt=bot>"
                    f"</div>"
                )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
