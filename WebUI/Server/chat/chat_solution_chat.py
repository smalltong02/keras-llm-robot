import json
import asyncio
from fastapi import Body
from langchain.chains import LLMChain
from fastapi.responses import StreamingResponse
from WebUI.Server.chat.utils import History
from langchain.prompts.chat import ChatPromptTemplate
from WebUI.configs import SAVE_CHAT_HISTORY
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import AsyncIteratorCallbackHandler
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from WebUI.configs.basicconfig import (GetProviderByName, GetSpeechModelInfo, GetSpeechForChatSolution, GetSystemPromptForChatSolution)
from WebUI.Server.utils import wrap_done, get_ChatOpenAI, get_prompt_template, GetModelApiBaseAddress
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.configs.basicconfig import (ModelType, ModelSize, ModelSubType, GetModelInfoByName, ExtractJsonStrings, use_new_search_engine, use_knowledge_base, use_new_function_calling)
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from typing import AsyncIterable, Dict, List, Optional, Union, Any

def CallingExternalTools(text: str) -> bool:
    if not text:
        return False
    json_lists = ExtractJsonStrings(text)
    if not json_lists:
        return False
    if use_new_search_engine(json_lists):
        return True
    if use_knowledge_base(json_lists):
        return True
    if use_new_function_calling(json_lists):
        return True
    return False

async def GetChatPromptFromFromSearchEngine(query: str = "", history: list[History] = [], chat_solution: Any = None) ->Union[ChatPromptTemplate, Any]:
    from WebUI.Server.chat.search_engine_chat import lookup_search_engine
    if not query or not history or not chat_solution:
        return None
    search_engine_name = chat_solution["config"]["search_engine"]["name"]
    docs = await lookup_search_engine(query, search_engine_name)
    context = "\n".join([doc.page_content for doc in docs])
    if not context:
        return None
    prompt_template = f"""The user's question has been searched on the internet. Here is all the content retrieved from the search engine:
    {context}\n
    The user's original question is: {query}
"""
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    return chat_prompt

async def GetChatPromptFromKnowledgeBase(query: str = "", history: list[History] = [], chat_solution: Any = None) ->Union[ChatPromptTemplate, Any]:
    if not query or not history or not chat_solution:
        return None

async def GetChatPromptFromFunctionCalling(json_lists: list = [], query: str = "", history: list[History] = []) ->Union[ChatPromptTemplate, Any]:
    from WebUI.Server.funcall.funcall import RunFunctionCalling
    if not json_lists or not history:
        return None
    result_list = []
    for item in json_lists:
        result = RunFunctionCalling(item)
        if result:
            result_list.append(result)
    context = "\n".join(result_list)
    if not context:
        return None
    prompt_template = f"""The function has been executed, and the result is as follows:
    {context}\n
    The user's original question is: {query}
"""
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    await asyncio.sleep(0.1)
    return chat_prompt

async def GetChatPromptFromExternalTools(text: str, query: str, history: List[History], chat_solution: Any) ->Union[ChatPromptTemplate, Any]:
    if not text:
        return None
    json_lists = ExtractJsonStrings(text)
    if not json_lists:
        return None
    if use_new_search_engine(json_lists):
        chat_prompt = await GetChatPromptFromFromSearchEngine(query, history, chat_solution)
        return chat_prompt
    if use_knowledge_base(json_lists):
        chat_prompt = await  GetChatPromptFromKnowledgeBase(query, history, chat_solution)
        return chat_prompt
    if use_new_function_calling(json_lists):
        chat_prompt = await  GetChatPromptFromFunctionCalling(json_lists, query, history)
        return chat_prompt
    return None

async def chat_solution_chat(
    query: str = Body(..., description="User input: ", examples=["chat"]),
    imagesdata: List[str] = Body([], description="image data", examples=["image"]),
    audiosdata: List[str] = Body([], description="audio data", examples=["audio"]),
    videosdata: List[str] = Body([], description="video data", examples=["video"]),
    history: List[History] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    chat_solution: dict = Body({}, description="Chat Solution"),
    temperature: float = Body(0.7, description="Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    ) -> StreamingResponse:
    history = [History.from_data(h) for h in history]

    async def chat_solution_chat_iterator(query: str,
        imagesdata: List[str] = [],
        audiosdata: List[str] = [],
        videosdata: List[str] = [],
        history: List[History] = [],
        stream: bool = True,
        chat_solution: dict = {},
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        ) -> AsyncIterable[str]:
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        async_callback = AsyncIteratorCallbackHandler()
        callbackslist = [async_callback]
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None
        model_name = chat_solution["config"]["llm_model"]
        modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
        modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
        modelinfo["mname"] = model_name
        speak_handler = None
        speech = GetSpeechForChatSolution(chat_solution)
        if speech:
            config = GetSpeechModelInfo(webui_config, speech.get("model", ""))
            if len(config):
                modeltype = config["type"]
                speechkey = config.get("speech_key", "")
                if speechkey == "[Your Key]":
                    speechkey = ""
                speechregion = config.get("speech_region", "")
                if speechregion == "[Your Region]":
                    speechregion = ""
                provider = config.get("provider", "")
                spspeaker = speech.get("speaker", "")
                if modeltype == "local" or modeltype == "cloud":
                    speak_handler = StreamSpeakHandler(run_place=modeltype, provider=provider, synthesis=spspeaker, subscription=speechkey, region=speechregion)
        system_prompt = GetSystemPromptForChatSolution(chat_solution)
        if system_prompt:
            system_msg = {
                'role': "system",
                'content': system_prompt
            }
            history = [History.from_data(system_msg)] + history
        if modelinfo["mtype"] == ModelType.Local:
            provider = GetProviderByName(webui_config, model_name)
            model = get_ChatOpenAI(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbackslist,
                )
        else:
            api_base = GetModelApiBaseAddress(modelinfo)
            model = ChatOpenAI(
                streaming=True,
                verbose=False,
                openai_api_key="EMPTY",
                openai_api_base=api_base,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbackslist,
            )
        prompt_template = get_prompt_template("llm_chat", "default")
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        
        btalk = True
        while btalk:
            btalk = False
            chain = LLMChain(prompt=chat_prompt, llm=model)

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
                        btalk = CallingExternalTools(answer)
                        if btalk:
                            new_chat_prompt = await GetChatPromptFromExternalTools(text=answer, query=query, history=history, chat_solution=chat_solution)
                            if not new_chat_prompt:
                                btalk = False
                            else:
                                chat_prompt = new_chat_prompt
                                yield json.dumps(
                                    {"cmd": "clear", "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                                async_callback.done.clear()
                                break
                    if speak_handler: 
                        speak_handler.on_llm_new_token(token)
                    yield json.dumps(
                        {"text": token, "chat_history_id": chat_history_id},
                        ensure_ascii=False)
                if speak_handler: 
                    speak_handler.on_llm_end(None)
            else:
                async for token in async_callback.aiter():
                    answer += token
                    if not btalk:
                        btalk = CallingExternalTools(answer)
                        if btalk:
                            new_chat_prompt = await GetChatPromptFromExternalTools(text=answer, query=query, history=history, chat_solution=chat_solution)
                            if not new_chat_prompt:
                                btalk = False
                            else:
                                chat_prompt = new_chat_prompt
                                async_callback.done.clear()
                                break
                if not btalk:
                    yield json.dumps(
                        {"text": answer, "chat_history_id": chat_history_id},
                        ensure_ascii=False)
                    if speak_handler: 
                        speak_handler.on_llm_new_token(answer)
                        speak_handler.on_llm_end(None)
            await task
        print("chat_solution_chat_iterator exit!")
        if SAVE_CHAT_HISTORY and len(chat_history_id) > 0:
            update_chat_history(chat_history_id, response=answer)

    return StreamingResponse(chat_solution_chat_iterator(query=query,
            imagesdata=imagesdata,
            audiosdata=audiosdata,
            videosdata=videosdata,
            history=history,
            stream=stream,
            chat_solution=chat_solution,
            temperature=temperature,
            max_tokens=max_tokens),
            media_type="text/event-stream")