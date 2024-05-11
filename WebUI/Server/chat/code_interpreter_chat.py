import json
import asyncio
from fastapi import Body
from fastapi.responses import StreamingResponse
from WebUI.configs.basicconfig import (ModelType, ModelSize, ModelSubType, GetModelInfoByName)
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from typing import AsyncIterable, Dict

async def code_interpreter_chat(query: str = Body(..., description="User input: ", examples=["chat"]),
    stream: bool = Body(False, description="stream output"),
    interpreter_id: str = Body(..., description="interpreter id"),
    model_name: str = Body("", description="model name"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    ):
    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
    modelinfo["mname"] = model_name
    codeinterpreter = webui_config.get("CodeInterpreter")
    offline = codeinterpreter.get("offline", False)
    auto_run = codeinterpreter.get("auto_run", True)
    safe_mode = codeinterpreter.get("safe_mode", False)

    async def code_interpreter_chat_iterator(query: str,
        stream: bool = True,
        interpreter_id: str = "",
        model_name: str = "",
        temperature: float = 0.7,
        offline: bool = False,
        auto_run: bool = True,
        safe_mode: bool = False,
        ) -> AsyncIterable[str]:
        from WebUI.Server.utils import GetModelApiBaseAddress
        nonlocal webui_config
        nonlocal codeinterpreter
        if interpreter_id == "Keras Interpreter":
            from WebUI.Server.interpreter_wrapper.keras_interpreter_wrapper import KerasInterpreter, SafeModeType
            custom_instructions = codeinterpreter.get(interpreter_id).get("custom_instructions")
            system_message = codeinterpreter.get(interpreter_id).get("system_message")
            if system_message == "[default]":
                system_message = None
            if custom_instructions == "[default]":
                custom_instructions = None
            if system_message is None and custom_instructions is None:
                interpreter = KerasInterpreter(
                    model_name=model_name,
                    offline=offline,
                    auto_run=auto_run,
                    safe_mode=SafeModeType.AskMode if safe_mode else SafeModeType.OffMode,
                    llm_base=GetModelApiBaseAddress(modelinfo),
                )
            elif system_message is None:
                interpreter = KerasInterpreter(
                    model_name=model_name,
                    offline=offline,
                    auto_run=auto_run,
                    safe_mode=SafeModeType.AskMode if safe_mode else SafeModeType.OffMode,
                    llm_base=GetModelApiBaseAddress(modelinfo),
                    custom_instructions=custom_instructions,
                )
            elif custom_instructions is None:
                interpreter = KerasInterpreter(
                    model_name=model_name,
                    offline=offline,
                    auto_run=auto_run,
                    safe_mode=SafeModeType.AskMode if safe_mode else SafeModeType.OffMode,
                    llm_base=GetModelApiBaseAddress(modelinfo),
                    system_message=system_message,
                )
            else:
                interpreter = KerasInterpreter(
                    model_name=model_name,
                    offline=offline,
                    auto_run=auto_run,
                    safe_mode=SafeModeType.AskMode if safe_mode else SafeModeType.OffMode,
                    llm_base=GetModelApiBaseAddress(modelinfo),
                    custom_instructions=custom_instructions,
                    system_message=system_message,
                )

            for chunk in interpreter.chat(message=query, stream=stream):
                print("chunk", chunk)
                chunk = chunk.get("content", "")
                yield json.dumps(
                    {"text": chunk},
                    ensure_ascii=False)
                await asyncio.sleep(0.1)               

    return StreamingResponse(code_interpreter_chat_iterator(
                    query=query,
                    stream=stream,
                    interpreter_id=interpreter_id,
                    model_name=model_name,
                    temperature=temperature,
                    offline=offline,
                    auto_run=auto_run,
                    safe_mode=safe_mode),
            media_type="text/event-stream")