import re
import asyncio
import json
from fastapi import Body
from fastapi.responses import StreamingResponse
from WebUI.configs import DEF_TOKENS, SAVE_CHAT_HISTORY, is_function_calling_enable
from WebUI.Server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
from WebUI.configs import GetProviderByName, generate_new_query, GetToolsSystemPrompt, is_toolboxes_enable, call_calling
#from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional
#from operator import itemgetter
from WebUI.Server.chat.utils import History
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from WebUI.Server.utils import get_prompt_template
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
#from WebUI.Server.funcall.funcall import funcall_tools

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
        async_callback = AsyncIteratorCallbackHandler()
        calling_enable = is_function_calling_enable()
        toolboxes_enable = is_toolboxes_enable()
        if toolboxes_enable:
            from WebUI.Server.funcall.google_toolboxes.credential import init_credential
            init_credential()
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
        system_msg = []
        if calling_enable:
            tools_system_prompt = GetToolsSystemPrompt()
            system_msg = History(role="system", content=tools_system_prompt)
            if history and history[0].role == "system":
                history[0].content = history[0].content + "\n\n" + tools_system_prompt
            else:
                history = [system_msg] + history

        btalk = True
        while btalk:
            btalk = False
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
                    if not btalk and calling_enable:
                        new_query, new_answer = call_calling(answer)
                        if new_query:
                            btalk = True
                            pattern = r'\"(.*?)\"'
                            match = re.search(pattern, new_query)
                            if match:
                                function_call = match.group(1)
                            new_answer = new_answer + "\n" + f'It is necessary to call the function `{function_call}` to get more information.'
                            yield json.dumps(
                                {"clear": new_answer, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            yield json.dumps(
                                {"user": f'The function `{function_call}` was called.', "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            new_history = await CreateChatPromptFromCallCalling(query=query, new_answer=new_answer, history=history)
                            history = new_history
                            query = new_query
                            break
                    # Use server-sent-events to stream the response
                    yield json.dumps(
                        {"text": token, "chat_history_id": chat_history_id},
                        ensure_ascii=False)
            else:
                async for token in async_callback.aiter():
                    answer += token
                if not btalk and calling_enable:
                    new_query, new_answer = call_calling(answer)
                    if new_query:
                        btalk = True
                        pattern = r'\"(.*?)\"'
                        match = re.search(pattern, new_query)
                        if match:
                            function_call = match.group(1)
                        new_answer = new_answer + "\n" + f'It is necessary to call the function `{function_call}` to get more information.'
                        yield json.dumps(
                            {"clear": new_answer, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                        yield json.dumps(
                            {"user": f'The function `{function_call}` was called.', "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                        new_history = await CreateChatPromptFromCallCalling(query=query, new_answer=new_answer, history=history)
                        history = new_history
                        query = new_query
                        break
                    else:
                        yield json.dumps(
                            {"text": answer, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                else:
                    yield json.dumps(
                        {"text": answer, "chat_history_id": chat_history_id},
                        ensure_ascii=False)

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