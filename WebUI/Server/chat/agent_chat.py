from fastapi import Body, File, Form, UploadFile
from WebUI.Server.chat.utils import History
from WebUI.Server.utils import (wrap_done, get_ChatOpenAI,
                        BaseResponse, get_prompt_template, run_in_thread_pool)

from typing import AsyncIterable, List, Optional

async def agent_chat(query: str = Body(..., description="User input: ", examples=["chat"]),
    history: List[History] = Body([],
                                description="History chat",
                                examples=[[
                                    {"role": "user", "content": "Please call a tool to check the weather today."},
                                    {"role": "assistant",
                                    "content": "Using a weather query tool, today's weather in Beijing is cloudy with a temperature range of 10-14 degrees Celsius, a northeast wind at 2 meters per second. Prone to catching a cold."}]]
                                ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body("", description="LLM Model"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default",
                            description="")
    ):
    pass