from fastapi import Body
from WebUI.configs import GetProviderByName
from fastapi.responses import StreamingResponse
from WebUI.Server.chat.utils import History
from langchain.docstore.document import Document
from langchain.chains import LLMChain
import asyncio
import json
import os
from WebUI.Server.utils import wrap_done, get_ChatOpenAI
from fastapi.concurrency import run_in_threadpool
from WebUI.Server.utils import get_prompt_template
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from langchain.prompts.chat import ChatPromptTemplate
from typing import AsyncIterable, Dict, List, Optional

async def code_interpreter_chat():
    pass