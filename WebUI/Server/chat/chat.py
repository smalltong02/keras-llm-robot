import asyncio
import json
from fastapi import Body
from fastapi.responses import StreamingResponse
from WebUI.configs import DEF_TOKENS, SAVE_CHAT_HISTORY
from WebUI.Server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
from WebUI.configs import (GetProviderByName, generate_new_query, ModelType, ModelSize, ModelSubType, ToolsType,
                            use_new_search_engine, use_knowledge_base, use_new_function_calling, use_new_toolboxes_calling, use_code_interpreter,
                            GetUserAnswerForCurConfig, GetCurrentRunningCfg, ExtractJsonStrings, GetModelInfoByName, GetModelConfig, GetSystemPromptForCurrentRunningConfig,
                            GetSystemPromptForSupportTools, CallingExternalToolsForCurConfig, GetNewAnswerForCurConfig,)
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional, Union, Any, Dict
from WebUI.configs import USE_RERANKER, GetRerankerModelPath
from WebUI.Server.chat.utils import History
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from WebUI.Server.utils import get_prompt_template, detect_device
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse

async def CreateChatHistoryFromCallCalling(query: str = "", new_answer: str = "", history: list[dict] = []) ->list[dict]:
    user_msg = {'role': "user", 'content': f"{query}"}
    assistant_msg = {'role': "assistant", 'content': f"{new_answer}"}
    chat_history = history + [user_msg] + [assistant_msg]
    await asyncio.sleep(0.1)
    return chat_history

async def CreateChatPromptFromCallCalling(query: str = "", new_answer: str = "", history: list[History] = []) ->list[History]:
    user_msg = History(role="user", content=f"{query}")
    assistant_msg = History(role="assistant", content=f"{new_answer}")
    chat_history = history + [user_msg] + [assistant_msg]
    await asyncio.sleep(0.1)
    return chat_history

async def GetChatPromptFromFromSearchEngine(json_lists: list = [], se_name: str = "", query: str = "") ->Union[str, Any, Any]:
    from WebUI.Server.chat.search_engine_chat import lookup_search_engine
    if not json_lists or not se_name or not query:
        return None, "", []
    se_query = query
    try:
        for item in json_lists:
            item_json = json.loads(item)
            arguments = item_json.get("arguments", {})
            if arguments:
                first_key = next(iter(arguments))
                first_value = arguments[first_key]
                if isinstance(first_value, str):
                    se_query = first_value
                    break
    except Exception as _:
        pass
    docs = await lookup_search_engine(se_query, se_name)
    source_documents = [
        f"""from [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
        for inum, doc in enumerate(docs)
    ]
    context = "\n".join([doc.page_content for doc in docs])
    if not context:
        return None, "", []
    new_query = f"""The user's question has been searched on the internet. Here is all the content retrieved from the search engine:
    {context}\n
"""
    return new_query, se_name, source_documents

async def GetChatPromptFromKnowledgeBase(json_lists: list = [], kb_name: str = "", query: str = "") ->Union[str, Any, Any]:
    from urllib.parse import urlencode
    from fastapi.concurrency import run_in_threadpool
    from WebUI.Server.reranker.reranker import LangchainReranker
    from WebUI.Server.knowledge_base.kb_doc_api import search_docs
    from WebUI.Server.knowledge_base.kb_service.base import KBServiceFactory
    from WebUI.Server.knowledge_base.utils import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
    if not json_lists or not kb_name or not query:
        return None, "", []
    kb_query = query
    try:
        for item in json_lists:
            item_json = json.loads(item)
            arguments = item_json.get("arguments", {})
            if arguments:
                first_key = next(iter(arguments))
                first_value = arguments[first_key]
                if isinstance(first_value, str):
                    kb_query = first_value
                    break
    except Exception as _:
        pass
    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is None:
        return None, "", []
    docs = await run_in_threadpool(search_docs,
            query=kb_query,
            knowledge_base_name=kb_name,
            top_k=VECTOR_SEARCH_TOP_K,
            score_threshold=SCORE_THRESHOLD)
    if USE_RERANKER:
            reranker_model_path = GetRerankerModelPath()
            print("-----------------model path------------------")
            print(reranker_model_path)
            reranker_model = LangchainReranker(top_n=VECTOR_SEARCH_TOP_K,
                                            device=detect_device(),
                                            max_length=1024,
                                            model_name_or_path=reranker_model_path
                                            )
            print(docs)
            docs = reranker_model.compress_documents(documents=docs,
                                                     query=kb_query)
            print("---------after rerank------------------")
            print(docs)
    source_documents = []
    for inum, doc in enumerate(docs):
        filename = doc.metadata.get("source")
        parameters = urlencode({"knowledge_base_name": kb_name, "file_name": filename})
        base_url = "/"
        url = f"{base_url}knowledge_base/download_doc?" + parameters
        text = f"""from [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
        source_documents.append(text)
    context = "\n".join([doc.page_content for doc in docs])
    if not context:
        return None, "", []
    new_query = f"""The user's issue has been searched through the knowledge base. Here is all the content retrieved:
    {context}\n
"""
    return new_query, kb_name, source_documents

async def GetChatPromptFromFunctionCalling(json_lists: list = []) ->Union[str, Any, Any]:
    from WebUI.Server.funcall.funcall import RunNormalFunctionCalling
    if not json_lists:
        return None, "", []
    result_list = []
    func_name = []
    for item in json_lists:
        name, result = RunNormalFunctionCalling(item)
        if result:
            func_name.append(name)
            result_list.append(result)
    source_documents = []
    for result in enumerate(result_list):
        source_documents.append(f"function - {func_name[0]}()\n\n{result}")
    context = "\n".join(result_list)
    if not context:
        return None, "", []
    new_query = f"""The function ''{func_name[0]}'' has been executed, and the result is as follows:
    {context}\n
"""
    await asyncio.sleep(0.1)
    return new_query, func_name[0], source_documents

async def GetChatPromptFromCodeInterpreter(json_lists: list = []) ->Union[str, Any, Any]:
    from WebUI.Server.funcall.funcall import RunCodeInterpreter
    if not json_lists:
        return None, "", []
    result_list = []
    func_name = []
    for item in json_lists:
        name, result = RunCodeInterpreter(item)
        if result:
            func_name.append(name)
            result_list.append(result)
    source_documents = []
    for result in enumerate(result_list):
        source_documents.append(f"The code running result - {func_name[0]}()\n\n{result}")
    context = "\n".join(result_list)
    if not context:
        return None, "", []
    new_query = f"""The python code has been executed, and the result is as follows:
    {context}\n
"""
    await asyncio.sleep(0.1)
    return new_query, func_name[0], source_documents

async def GetChatPromptFromToolBoxes(json_lists: list = []) ->Union[str, Any, Any]:
    from WebUI.Server.funcall.google_toolboxes.credential import RunFunctionCallingInToolBoxes
    if not json_lists:
        return None, "", []
    result_list = []
    func_name = []
    for item in json_lists:
        name, result = RunFunctionCallingInToolBoxes(item)
        if result:
            func_name.append(name)
            result_list.append(result)
    source_documents = []
    for result in enumerate(result_list):
        source_documents.append(f"function - {func_name[0]}()\n\n{result}")
    context = "\n".join(result_list)
    if not context:
        return None, "", []
    new_query = f"""The function '{func_name[0]}' has been executed, and the result is as follows:
    {context}\n
"""
    await asyncio.sleep(0.1)
    return new_query, func_name[0], source_documents

async def GetQueryFromExternalToolsForCurConfig(answer: str, query: str) ->Union[str, Any, Any, Any]:
    if not answer:
        return None, "", [], ToolsType.Unknown
    config = GetCurrentRunningCfg(True)
    if not config:
        return None
    json_lists = ExtractJsonStrings(answer)
    if not json_lists:
        return None, "", [], ToolsType.Unknown
    if config["search_engine"]["name"] and use_new_search_engine(json_lists):
        new_query, tool_name, docs = await GetChatPromptFromFromSearchEngine(json_lists, config["search_engine"]["name"], query)
        return new_query, tool_name, docs, ToolsType.ToolSearchEngine
    if config["knowledge_base"]["name"] and use_knowledge_base(json_lists):
        new_query, tool_name, docs = await GetChatPromptFromKnowledgeBase(json_lists, config["knowledge_base"]["name"], query)
        return new_query, tool_name, docs, ToolsType.ToolKnowledgeBase
    if config["normal_calling"]["enable"] and use_new_function_calling(json_lists):
        new_query, tool_name, docs = await GetChatPromptFromFunctionCalling(json_lists)
        return new_query, tool_name, docs, ToolsType.ToolFunctionCalling
    if config["code_interpreter"]["name"] and use_code_interpreter(json_lists):
        new_query, tool_name, docs = await GetChatPromptFromCodeInterpreter(json_lists)
        return new_query, tool_name, docs, ToolsType.ToolCodeInterpreter
    if use_new_toolboxes_calling(json_lists):
        new_query, tool_name, docs = await GetChatPromptFromToolBoxes(json_lists)
        return new_query, tool_name, docs, ToolsType.ToolToolBoxes
    return None, "", [], ToolsType.Unknown

async def RunAllEnableToolsInString(func_name: str="", args: dict={}, query: str=""):
    if not func_name or not query:
        return None, "", [], ToolsType.Unknown
    json_data = json.dumps({"name": func_name, "arguments": args})
    new_query, tool_name, docs, tooltype = await GetQueryFromExternalToolsForCurConfig(json_data, query)
    return new_query, tool_name, docs, tooltype

async def chat(query: str = Body(..., description="User input: ", examples=["chat"]),
    imagesdata: List[str] = Body([], description="image data", examples=["image"]),
    audiosdata: List[str] = Body([], description="audio data", examples=["audio"]),
    videosdata: List[str] = Body([], description="video data", examples=["video"]),
    imagesprompt: List[str] = Body([], description="image prompt", examples=["prompt"]),
    history: List[History] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body("", description="model name"),
    speechmodel: dict = Body({}, description="speech model config"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default", description=""),
    ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            imagesdata: List[str] = [],
                            audiosdata: List[str] = [],
                            videosdata: List[str] = [],
                            imagesprompt: List[str] = [],
                            history: List[History] = [],
                            stream: bool = True,
                            model_name: str = "",
                            speechmodel: dict = {},
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = None,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        model_info : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
        model_info["mtype"], model_info["msize"], model_info["msubtype"] = GetModelInfoByName(webui_config, model_name)
        model_info["mname"] = model_name
        model_config = GetModelConfig(webui_config, model_info)
        support_tools = model_config.get("support_tools", False)
        system_msg = []
        if not support_tools:
            tools_system_prompt = GetSystemPromptForCurrentRunningConfig()
        else:
            tools_system_prompt = GetSystemPromptForSupportTools()
        if tools_system_prompt:
            system_msg = History(role="system", content=tools_system_prompt)
            if history and history[0].role == "system":
                history[0].content = history[0].content + "\n\n" + tools_system_prompt
            else:
                history = [system_msg] + history
        docs = []
        btalk = True
        while btalk:
            btalk = False
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
            # def tool_chain(model_output):
            #     tool_map = {tool.name: tool for tool in funcall_tools}
            #     chosen_tool = tool_map[model_output["name"]]
            #     return itemgetter("arguments") | chosen_tool

            if len(imagesdata):
                from langchain.schema import HumanMessage
                content=[{
                        "type": "text",
                        "text": f"{query}"}]
                for imagedata in imagesdata:
                    content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{imagedata}"}})
                message = HumanMessage(content=content)

                task = asyncio.create_task(wrap_done(
                    model.ainvoke([message]),
                    async_callback.done),
                )
            else:
                if imagesprompt:
                    query = generate_new_query(query, imagesprompt)

                prompt_template = get_prompt_template("llm_chat", prompt_name)
                input_msg = History(role="user", content=prompt_template).to_msg_template(False)
                chat_prompt = ChatPromptTemplate.from_messages([i.to_msg_template() for i in history] + [input_msg])
                #print("chat_prompt: ", chat_prompt)
                chain = LLMChain(prompt=chat_prompt, llm=model)
                # Begin a task that runs in the background.
                task = asyncio.create_task(wrap_done(
                    chain.acall({"input": query}),
                    async_callback.done),
                )

            answer = ""
            chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)
            if stream:
                async for token in async_callback.aiter():
                    answer += token
                    if not btalk:
                        btalk, new_answer = CallingExternalToolsForCurConfig(answer)
                        if btalk:
                            new_query, tool_name, docs, tooltype = await GetQueryFromExternalToolsForCurConfig(answer=answer, query=query)
                            if not new_query:
                                btalk = False
                            else:
                                btalk = True
                                new_answer = GetNewAnswerForCurConfig(new_answer, tool_name, tooltype)
                                history.append(History(role="user", content=query))
                                history.append(History(role="assistant", content=new_answer))
                                yield json.dumps(
                                    {"clear": new_answer, "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                                user_answer = GetUserAnswerForCurConfig(tool_name, tooltype)
                                yield json.dumps(
                                    {"user": user_answer, "tooltype": tooltype.value},
                                    ensure_ascii=False)
                                query = new_query
                    # Use server-sent-events to stream the response
                    if not btalk:
                        yield json.dumps(
                            {"text": token, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
            else:
                async for token in async_callback.aiter():
                    answer += token
                if not btalk:
                    btalk, new_answer = CallingExternalToolsForCurConfig(answer)
                    if btalk:
                        new_query, tool_name, docs, tooltype = await GetQueryFromExternalToolsForCurConfig(answer=answer, query=query)
                        if not new_query:
                            btalk = False
                        else:
                            btalk = True
                            new_answer = GetNewAnswerForCurConfig(new_answer, tool_name, tooltype)
                            history.append({'role': "user",'content': query})
                            history.append({'role': "assistant", 'content': new_answer})
                            yield json.dumps(
                                {"clear": new_answer, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            user_answer = GetUserAnswerForCurConfig(tool_name, tooltype)
                            yield json.dumps(
                                {"user": user_answer, "tooltype": tooltype.value},
                                ensure_ascii=False)
                            query = new_query
                    if not btalk:
                        yield json.dumps(
                            {"text": answer, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
            if not btalk and docs:
                yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                docs = []
                tooltype = ToolsType.Unknown

            if SAVE_CHAT_HISTORY and len(chat_history_id) > 0:
                update_chat_history(chat_history_id, response=answer)
        await task

    return StreamingResponse(chat_iterator(query=query,
                                           imagesdata=imagesdata,
                                           audiosdata=audiosdata,
                                           videosdata=videosdata,
                                           imagesprompt=imagesprompt,
                                           history=history,
                                           stream=stream,
                                           model_name=model_name,
                                           speechmodel=speechmodel,
                                           temperature=temperature,
                                           max_tokens=max_tokens,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")