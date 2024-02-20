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

def bing_search(text, search_url, api_key, result_len, **kwargs):
    search = BingSearchAPIWrapper(bing_subscription_key=api_key,
                                  bing_search_url=search_url)
    return search.results(text, result_len)


def duckduckgo_search(text, search_url, api_key, result_len, **kwargs):
    search = DuckDuckGoSearchAPIWrapper()
    return search.results(text, result_len)

def metaphor_search(
        text: str,
        search_url: str,
        api_key: str,
        result_len: int,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
) -> List[Dict]:
    from exa_py import Exa
    from markdownify import markdownify
    from strsimpy.normalized_levenshtein import NormalizedLevenshtein

    highlights_options  = {
        "num_sentences": 7, # how long our highlights should be
        "highlights_per_url": 1, # just get the best highlight for each URL
    }

    info_for_llm = []
    exa = Exa(api_key=api_key)
    search_response = exa.search_and_contents(text, highlights=highlights_options, num_results=result_len, use_autoprompt=True)
    info = [sr for sr in search_response.results]
    for x in info:
        x.highlights[0] = markdownify(x.highlights[0])
    info_for_llm = info

    docs = [{"snippet": x.highlights[0],
                "link": x.url,
                "title": x.title}
            for x in info_for_llm]
    return docs

SEARCH_ENGINES = {
    "bing": bing_search,
    "duckduckgo": duckduckgo_search,
    "metaphor": metaphor_search,
    }

def search_result2docs(search_results):
    docs = []
    for result in search_results:
        doc = Document(page_content=result["snippet"] if "snippet" in result.keys() else "",
                       metadata={"source": result["link"] if "link" in result.keys() else "",
                                 "filename": result["title"] if "title" in result.keys() else ""})
        docs.append(doc)
    return docs


async def lookup_search_engine(
        query: str,
        search_engine_name: str,
        top_k: int = 3,
):
    search_engine = SEARCH_ENGINES[search_engine_name]
    
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    config = webui_config.get("SearchEngine").get(search_engine_name)
    api_key = config.get("api_key", "")
    search_url = config.get("search_url", "")
    if search_engine_name == "bing":
        if api_key == "" or api_key == "YOUR_API_KEY":
            api_key = os.environ.get("BING_SUBSCRIPTION_KEY", "")
        if search_url == "":
            search_url = os.environ.get("BING_SEARCH_URL", "")
    elif search_engine_name == "metaphor":
        if api_key == "" or api_key == "YOUR_API_KEY":
            api_key = os.environ.get("METAPHOR_API_KEY", "")
    results = await run_in_threadpool(search_engine, query, search_url=search_url, api_key=api_key, result_len=top_k)
    docs = search_result2docs(results)
    return docs

async def search_engine_chat(
    query: str = Body(..., description="User input: ", examples=["chat"]),
    search_engine_name: str = Body(..., description="search engine name", examples=["duckduckgo"]),
    history: List[dict] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body("", description="model name"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default", description=""),
) -> StreamingResponse:
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    searchengine = webui_config.get("SearchEngine")
    top_k = searchengine.get("top_k", 3)

    history = [History.from_data(h) for h in history]

    async def search_engine_chat_iterator(query: str,
                                          search_engine_name: str,
                                          top_k: int,
                                          history: Optional[List[History]],
                                          stream: bool,
                                          model_name: str = "",
                                          temperature: float = 0.7,
                                          max_tokens: Optional[int] = None,
                                          prompt_name: str = prompt_name,
                                          ) -> AsyncIterable[str]:
        nonlocal webui_config
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        provider = GetProviderByName(webui_config, model_name)

        model = get_ChatOpenAI(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        docs = await lookup_search_engine(query, search_engine_name, top_k)
        context = "\n".join([doc.page_content for doc in docs])

        prompt_template = get_prompt_template("search_engine_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = [
            f"""from [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
            for inum, doc in enumerate(docs)
        ]

        if len(source_documents) == 0:
            source_documents.append(f"""<span style='color:red'>No relevant information were found. This response is generated based on the LLM Model '{model_name}' itself!</span>""")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return StreamingResponse(search_engine_chat_iterator(query=query,
                                                           search_engine_name=search_engine_name,
                                                           top_k=top_k,
                                                           history=history,
                                                           stream=stream,
                                                           model_name=model_name,
                                                           temperature=temperature,
                                                           max_tokens=max_tokens,
                                                           prompt_name=prompt_name),
                               )