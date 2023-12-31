from fastapi import Body
from fastapi.responses import StreamingResponse
from WebUI.configs import LLM_MODELS, TEMPERATURE, DEF_TOKENS, SAVE_CHAT_HISTORY
from WebUI.Server.utils import wrap_done, get_ChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional
from WebUI.Server.chat.utils import History
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from WebUI.Server.utils import get_prompt_template
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history

async def chat(query: str = Body(..., description="User input: ", examples=["chat"]),
    imagesdata: List[str] = Body([], description="image data", examples=["image"]),
    history: List[History] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body(LLM_MODELS[0], description="model name"),
    speechmodel: dict = Body({}, description="speech model config"),
    temperature: float = Body(TEMPERATURE, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default", description=""),
    ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            imagesdata: List[str] = [],
                            history: List[History] = [],
                            stream: bool = True,
                            model_name: str = LLM_MODELS[0],
                            speechmodel: dict = {},
                            temperature: float = TEMPERATURE,
                            max_tokens: Optional[int] = None,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
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
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbackslist,
        )

        if len(imagesdata):
            from langchain.chat_models import ChatOpenAI
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
            prompt_template = get_prompt_template("llm_chat", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
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
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "chat_history_id": chat_history_id},
                    ensure_ascii=False)
        else:
            async for token in async_callback.aiter():
                answer += token
            yield json.dumps(
                {"text": answer, "chat_history_id": chat_history_id},
                ensure_ascii=False)

        if SAVE_CHAT_HISTORY and len(chat_history_id) > 0:
            update_chat_history(chat_history_id, response=answer)
        await task

    return StreamingResponse(chat_iterator(query=query,
                                           imagesdata=imagesdata,
                                           history=history,
                                           stream=stream,
                                           model_name=model_name,
                                           speechmodel=speechmodel,
                                           temperature=temperature,
                                           max_tokens=max_tokens,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")