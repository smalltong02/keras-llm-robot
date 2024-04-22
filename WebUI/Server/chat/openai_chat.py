from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional
import openai
from WebUI.Server.utils import get_model_worker_config, fschat_openai_api_address
from pydantic import BaseModel
from WebUI.Server.utils import FastAPI
from fastchat.protocol.openai_api_protocol import ChatCompletionRequest
from fastchat.constants import ErrorCode
from fastchat.protocol.openai_api_protocol import ErrorResponse

class OpenAiMessage(BaseModel):
    role: str = "user"
    content: str = "hello"


class OpenAiChatMsgIn(BaseModel):
    model: str = ""
    messages: List[OpenAiMessage]
    temperature: float = 0.7
    n: int = 1
    max_tokens: Optional[int] = None
    stop: List[str] = []
    stream: bool = False
    presence_penalty: int = 0
    frequency_penalty: int = 0


async def openai_chat(msg: OpenAiChatMsgIn):
    config = get_model_worker_config(msg.model)
    openai.api_key = config.get("api_key", "EMPTY")
    print(f"{openai.api_key=}")
    openai.api_base = config.get("api_base_url", fschat_openai_api_address())
    print(f"{openai.api_base=}")
    print(msg)

    async def get_response(msg):
        data = msg.dict()

        try:
            response = await openai.ChatCompletion.acreate(**data)
            if msg.stream:
                async for data in response:
                    if choices := data.choices:
                        if chunk := choices[0].get("delta", {}).get("content"):
                            print(chunk, end="", flush=True)
                            yield chunk
            else:
                if response.choices:
                    answer = response.choices[0].message.content
                    print(answer)
                    yield(answer)
        except Exception as e:
            msg = f"Get ChatCompletion error: {e}"
            print(f'{e.__class__.__name__}: {msg}')

    return StreamingResponse(
        get_response(msg),
        media_type='text/event-stream',
    )

def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(
        ErrorResponse(message=message, code=code).dict(), status_code=400
    )

def check_requests(request) -> Optional[JSONResponse]:
    # Check all params
    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    if request.top_k is not None and (request.top_k > -1 and request.top_k < 1):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_k} is out of Range. Either set top_k to -1 or >=1.",
        )
    if request.stop is not None and (
        not isinstance(request.stop, str) and not isinstance(request.stop, list)
    ):
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )
    return None

def completion_stream_generator(app: FastAPI, request: ChatCompletionRequest):
    async def chat_completion_stream_generator(app: FastAPI, request: ChatCompletionRequest):
        import json
        import shortuuid
        from typing import Dict
        from WebUI.configs.basicconfig import ModelType, ModelSize, ModelSubType, GetModelInfoByName, ConvertCompletionRequestToHistory
        from WebUI.configs.specialmodels import special_chat_iterator
        from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
        from fastchat.protocol.openai_api_protocol import ChatCompletionResponseStreamChoice, DeltaMessage, ChatCompletionStreamResponse
        print("completions: ", request)
        model_name = request.model
        history, query = ConvertCompletionRequestToHistory(request)
        temperature = request.temperature
        max_tokens = request.max_tokens
        stream = request.stream
        
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
        modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
        modelinfo["mname"] = model_name

        response = special_chat_iterator(
            model=app._model,
            tokenizer=app._tokenizer,
            async_callback=app._streamer,
            modelinfo=modelinfo,
            query=query,
            imagesdata=[],
            audiosdata=[],
            videosdata=[],
            imagesprompt=[],
            history=history,
            stream=stream,
            speechmodel={},
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_name="default",
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=DeltaMessage(role="assistant"),
            finish_reason=None,
        )
        id = f"chatcmpl-{shortuuid.random()}"
        chunk = ChatCompletionStreamResponse(
            id=id, choices=[choice_data], model=model_name
        )
        chunk = chunk.json(exclude_unset=True, ensure_ascii=False)
        #print("chunk: ", chunk)
        yield f"data: {chunk}\n\n"

        new_sentence = ""
        async for tokens in response:
            if not tokens:
                continue
            data = json.loads(tokens)["text"]
            token_list = data.split('\n')
            length = len(token_list)
            if length != 0:
                for i in range(length - 1):
                    new_sentence += token_list[i]
                    if len(new_sentence) == 0:
                        continue
                    choice_data = ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(content=new_sentence),
                            finish_reason=None,
                        )
                    chunk = ChatCompletionStreamResponse(
                        id=id, choices=[choice_data], model=model_name
                    )
                    chunk = chunk.json(exclude_unset=True, ensure_ascii=False)
                    #print("chunk: ", chunk)
                    yield f"data: {chunk}\n\n"
                    new_sentence = ""
                    choice_data = ChatCompletionResponseStreamChoice(
                            index=0,
                            delta=DeltaMessage(content='\n'),
                            finish_reason=None,
                        )
                    chunk = ChatCompletionStreamResponse(
                        id=id, choices=[choice_data], model=model_name
                    )
                    chunk = chunk.json(exclude_unset=True, ensure_ascii=False)
                    #print("chunk: ", chunk)
                    yield f"data: {chunk}\n\n"
                new_sentence = token_list[length-1]
            if len(new_sentence):
                choice_data = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=DeltaMessage(content=new_sentence),
                        finish_reason=None,
                    )
                chunk = ChatCompletionStreamResponse(
                    id=id, choices=[choice_data], model=model_name
                )
                chunk = chunk.json(exclude_unset=True, ensure_ascii=False)
                #print("chunk: ", chunk)
                yield f"data: {chunk}\n\n"
                new_sentence = ""
        yield "data: [DONE]\n\n"

    from WebUI.Server.chat.openai_chat import (create_error_response, check_requests)
    if app._model_name != request.model:
        return create_error_response(
            ErrorCode.INVALID_MODEL,
            f"the model `{request.model}` is not loaded!",
        )
    error_check_ret = check_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    generator = chat_completion_stream_generator(app, request)
    return StreamingResponse(generator, media_type="text/event-stream")