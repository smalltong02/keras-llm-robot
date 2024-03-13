from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from WebUI.configs import (DEF_TOKENS, USE_RERANKER, GetProviderByName, GetRerankerModelPath)
from WebUI.Server.knowledge_base.utils import SCORE_THRESHOLD
from WebUI.Server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
import asyncio
import json
from urllib.parse import urlencode
#from sse_starlette.sse import EventSourceResponse
from fastapi.concurrency import run_in_threadpool
from WebUI.Server.utils import BaseResponse, get_prompt_template, detect_device
from WebUI.Server.chat.utils import History
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from langchain.callbacks import AsyncIteratorCallbackHandler
from WebUI.Server.knowledge_base.kb_doc_api import search_docs
from WebUI.Server.knowledge_base.kb_service.base import KBServiceFactory
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.Server.reranker.reranker import LangchainReranker
from langchain.prompts.chat import ChatPromptTemplate
from typing import AsyncIterable, List, Optional

async def knowledge_base_chat(
        query: str = Body(..., description="User input: ", examples=["chat"]),
        knowledge_base_name: str = Body(..., description="kb name", examples=["samples"]),
        top_k: int = Body(3, description="matching vector count"),
        score_threshold: float = Body(
            SCORE_THRESHOLD,
            description="Knowledge base matching relevance threshold, with a range between 0 and 1. A smaller SCORE indicates higher relevance, and setting it to 1 is equivalent to no filtering. It is recommended to set it around 0.5"),
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
        imagesdata: List[str] = Body([], description="image data", examples=["image"]),
        speechmodel: dict = Body({}, description="speech model config"),
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
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Not found Knowledge Base '{knowledge_base_name}'")
    if len(imagesdata):
        return BaseResponse(code=404, msg=f"Not support Multimodal Model '{model_name}' now!")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            knowledge_base_name: str,
            top_k: int,
            score_threshold: float,
            history: Optional[List[History]],
            stream: bool,
            model_name: str = model_name,
            imagesdata: List[str] = [],
            speechmodel: dict = {},
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            prompt_name: str = prompt_name,
            request: Request = None,
    ) -> AsyncIterable[str]:
        
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        async_callback = AsyncIteratorCallbackHandler()
        callbackslist = [async_callback]
        if len(speechmodel):
            modeltype = speechmodel.get("type", "")
            provider = speechmodel.get("provider", "")
            #spmodel = speechmodel.get("model", "")
            spspeaker = speechmodel.get("speaker", "")
            speechkey = speechmodel.get("speech_key", "")
            speechregion = speechmodel.get("speech_region", "")
            if modeltype == "local" or modeltype == "cloud":
                speak_handler = StreamSpeakHandler(run_place=modeltype, provider=provider, synthesis=spspeaker, subscription=speechkey, region=speechregion)
                callbackslist.append(speak_handler)
        if len(imagesdata):
            if max_tokens is None:
                max_tokens = DEF_TOKENS
        provider = GetProviderByName(webui_config, model_name)
        model = get_ChatOpenAI(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbackslist,
        )
        docs = await run_in_threadpool(search_docs,
                                       query=query,
                                       knowledge_base_name=knowledge_base_name,
                                       top_k=top_k,
                                       score_threshold=score_threshold)

        # 加入reranker
        if USE_RERANKER:
            reranker_model_path = GetRerankerModelPath()
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=top_k,
                                            device=detect_device(),
                                            max_length=1024,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=query)
            print("---------after rerank------------------")
            print(docs)
        context = "\n".join([doc.page_content for doc in docs])

        if len(docs) == 0:
            prompt_template = get_prompt_template("knowledge_base_chat", "Empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            async_callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""from [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:
            source_documents.append(f"<span style='color:red'>No relevant documents were found. This response is generated based on the LLM Model '{model_name}' itself!</span>")

        if stream:
            async for token in async_callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in async_callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return StreamingResponse(knowledge_base_chat_iterator(query=query, 
                                                            knowledge_base_name=knowledge_base_name, 
                                                            top_k=top_k,
                                                            score_threshold=score_threshold,
                                                            history=history,
                                                            stream=stream,
                                                            model_name=model_name,
                                                            imagesdata=imagesdata,
                                                            speechmodel=speechmodel,
                                                            temperature=temperature,
                                                            max_tokens=max_tokens,
                                                            prompt_name=prompt_name,
                                                            request=request))