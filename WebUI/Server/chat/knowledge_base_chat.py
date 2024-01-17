from fastapi import Body, Request
from WebUI.Server.chat.utils import History
from typing import AsyncIterable, List, Optional

async def knowledge_base_chat(query: str = Body(..., description="User input: ", examples=["chat"]),
        knowledge_base_name: str = Body(..., description="kb name", examples=["samples"]),
        top_k: int = Body(3, description="matching vector count"),
        score_threshold: float = Body(
            0.6,
            description="Knowledge base matching relevance threshold, with a range between 0 and 1. A smaller SCORE indicates higher relevance, and setting it to 1 is equivalent to no filtering. It is recommended to set it around 0.5",
            ge=0,
            le=2
        ),
        history: List[History] = Body(
            [],
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
        max_tokens: Optional[int] = Body(
            None,
            description="max tokens."
        ),
        prompt_name: str = Body(
            "default",
            description=""
        ),
        request: Request = None,
    ):
    pass