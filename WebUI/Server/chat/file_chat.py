from fastapi import Body, File, Form, UploadFile
from WebUI.Server.chat.utils import History
from WebUI.Server.utils import (wrap_done, get_ChatOpenAI,
                        BaseResponse, get_prompt_template, run_in_thread_pool)

from typing import AsyncIterable, List, Optional


CHUNK_SIZE = 250
OVERLAP_SIZE = 50
ZH_TITLE_ENHANCE = False

def upload_temp_docs(
    files: List[UploadFile] = File(..., description="upload fiele, support multiple files."),
    prev_id: str = Form(None, description="Prev knowledge ID"),
    chunk_size: int = Form(CHUNK_SIZE, description="chunk size"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="overlap size"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="zh title enhance"),
) -> BaseResponse:
    pass


async def file_chat(query: str = Body(..., description="User input: ", examples=["chat"]),
        knowledge_id: str = Body(..., description="Knowledge ID"),
        top_k: int = Body(3, description="matching vector count"),
        score_threshold: float = Body(0.6, description="Knowledge base matching relevance threshold, with a range between 0 and 1. A smaller SCORE indicates higher relevance, and setting it to 1 is equivalent to no filtering. It is recommended to set it around 0.5", ge=0, le=2),
        history: List[History] = Body([],
                                    description="History chat",
                                    examples=[[
                                        {"role": "user",
                                        "content": "Who are you?"},
                                        {"role": "assistant",
                                        "content": "I am AI."}]]
                                    ),
        stream: bool = Body(False, description="stream output"),
        model_name: str = Body("", description="LLM Model"),
        temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(None, description="max tokens."),
        prompt_name: str = Body("default", description=""),
    ):
    pass