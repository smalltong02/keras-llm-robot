import os
import re
import json
import asyncio
import base64
import threading
from pathlib import Path
import google.generativeai as genai
from fastapi.responses import StreamingResponse
from WebUI.configs.basicconfig import (TMP_DIR, ToolsType, ModelType, ModelSize, ModelSubType, GetModelInfoByName, GetProviderByName, GetModelConfig, GetGGUFModelPath, generate_new_query, GeneratePresetPrompt, 
                                       GetSystemPromptForSupportTools, GetSystemPromptForCurrentRunningConfig, GetGoogleNativeTools, GetOpenaiNativeTools, CallingExternalToolsForCurConfig, GetNewAnswerForCurConfig,
                                       GetUserAnswerForCurConfig)
from WebUI.configs.codemodels import code_model_chat
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from langchain.chains import LLMChain
from WebUI.Server.utils import FastAPI
from WebUI.Server.chat.chat import GetQueryFromExternalToolsForCurConfig, RunAllEnableToolsInString
from typing import List, Dict, Any, Optional, AsyncIterable

def clean_special_text(text : str, prompttemplate: dict):
    anti_prompt = prompttemplate.get("anti_prompt", "")
    if len(anti_prompt):
        def is_substring(strs_list, search_string):
            search_string_lower = search_string.lower()
            for str in strs_list:
                if str.lower() in search_string_lower:
                    return True
            return False
        return is_substring(anti_prompt, text)
    return False

def chat_completion_request(client,model="", messages="", tools=None, tool_choice=None, stream=False):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=stream
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e

def init_cloud_models(model_name):
    if model_name is None:
        return None
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
    if modelinfo["mtype"] != ModelType.Online:
        return None
    modelinfo["mname"] = model_name
    provider = GetProviderByName(webui_config, model_name)
    if provider is None or model_name is None:
        return None
    if provider != "google-api":
        return None
    return None

def load_pipeline_model(app: FastAPI, model_name, model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
    from langchain.llms.huggingface_pipeline import HuggingFacePipeline
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=device, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    chatconfig = webui_config.get("ChatConfiguration")
    temperature = chatconfig.get("temperature")
    top_p = chatconfig.get("top_p")
    repetition_penalty = chatconfig.get("repetition_penalty")["cur"]
    tokens_length = chatconfig.get("tokens_length")["cur"]
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=tokens_length,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        streamer=streamer
    )
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    if model_name.startswith("phi"):
        pipe.model.config.eos_token_id = tokenizer.eos_token_id
        pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    elif model_name.startswith("Meta-Llama-3"):
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        pipe.model.config.eos_token_id = terminators
        pipe.model.config.pad_token_id = tokenizer.eos_token_id
    llm_model = HuggingFacePipeline(pipeline=pipe)
    app._model = llm_model
    app._tokenizer = tokenizer
    app._streamer = streamer
    app._model_name = model_name

def load_llamacpp_model(app: FastAPI, model_name, model_path):
    from langchain.llms.llamacpp import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from WebUI.Server.chat.StreamHandler import LlamacppStreamCallbackHandler
    from transformers import AutoTokenizer
    async_callback = LlamacppStreamCallbackHandler()
    callback_manager = CallbackManager([async_callback])
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    chatconfig = webui_config.get("ChatConfiguration")
    temperature = chatconfig.get("temperature")
    top_p = chatconfig.get("top_p")
    tokens_length = chatconfig.get("tokens_length")["cur"]
    modellist = GetGGUFModelPath(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    path = model_path + "/" + modellist[0]
    if len(modellist):
        llm_model = LlamaCpp(
            model_path=path,
            do_sample=True,
            temperature=temperature,
            n_ctx=tokens_length,
            max_tokens=tokens_length,
            top_p=top_p,
            verbose=True,
            callback_manager=callback_manager,
            n_threads=4,
            streaming=True,
        )
        app._model = llm_model
        app._tokenizer = tokenizer
        app._streamer = async_callback
        app._model_name = model_name

def init_special_models(app: FastAPI, args):
    model_name = args.model_names[0]
    model_path = args.model_path
    if len(model_name) == 0 or len(model_path) == 0:
        return
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    model_info : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    model_info["mtype"], model_info["msize"], model_info["msubtype"] = GetModelInfoByName(webui_config, model_name)
    model_info["mname"] = model_name
    model_config = GetModelConfig(webui_config, model_info)
    load_type = model_config.get("load_type", "")
    if load_type == "pipeline":
        load_pipeline_model(app=app, model_name=model_name, model_path=model_path, device=args.device)
    elif load_type == "llamacpp":
        load_llamacpp_model(app=app, model_name=model_name, model_path=model_path)

async def special_chat_iterator(model: Any,
    tokenizer: Any,
    query: str,
    imagesdata: List[str],
    audiosdata: List[str],
    videosdata: List[str],
    imagesprompt: List[str],
    history: List[dict] = [],
    async_callback: Any = None,
    stream: bool = True,
    speechmodel: dict = {},
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    modelinfo: Any = None,
    prompt_name: str = "default",
    ) -> AsyncIterable[str]:

    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    model_config = GetModelConfig(webui_config, modelinfo)
    support_tools = model_config.get("support_tools", False)
    model_name = modelinfo["mname"]
    speak_handler = None
    if len(speechmodel):
        modeltype = speechmodel.get("type", "")
        provider = speechmodel.get("provider", "")
        #spmodel = speechmodel.get("model", "")
        spspeaker = speechmodel.get("speaker", "")
        speechkey = speechmodel.get("speech_key", "")
        speechregion = speechmodel.get("speech_region", "")
        if modeltype == "local" or modeltype == "cloud":
            speak_handler = StreamSpeakHandler(run_place=modeltype, provider=provider, synthesis=spspeaker, subscription=speechkey, region=speechregion)

    btalk = True
    answer = ""
    chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)
    if imagesprompt:
        query = generate_new_query(query, imagesprompt)
    system_msg = {}
    tools_system_prompt = ""
    if not support_tools:
        tools_system_prompt = GetSystemPromptForCurrentRunningConfig()
    else:
        tools_system_prompt = GetSystemPromptForSupportTools()
    if tools_system_prompt:
        if history and history[0].get("role", "") == "system":
            history[0]["content"] = tools_system_prompt + "\n\n" + history[0]["content"] 
        else:
            system_msg = {
                'role': "system",
                'content': tools_system_prompt
            }
            history = [system_msg] + history
    if modelinfo["mtype"] == ModelType.Special:
        from langchain.prompts import PromptTemplate
        modelconfig = GetModelConfig(webui_config, modelinfo)
        loadtype = modelconfig["load_type"]

        docs = []
        btalk = True
        while btalk:
            answer = ""
            btalk = False
            prompt = history.copy()
            prompt.append({'role': "user",
                            'content': "{{input}}"})

            messages = tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False, 
                    add_generation_prompt=True
            )
            prompt = PromptTemplate(template=messages, template_format="jinja2", input_variables=["input"])
            if loadtype == "pipeline":
                chain = LLMChain(prompt=prompt, llm=model)
                def running_chain(chain, query):
                    chain.run(query)
                    print("running_chain exit!")
                
                thread = threading.Thread(target=running_chain, args=(chain, query))
                thread.start()
                if loadtype == "pipeline":
                    streamer = async_callback
                    for chunk in streamer:
                        if chunk is not None:
                            answer += chunk
                            print(chunk, end="")
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
                                    {"text": chunk, "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                                if speak_handler: 
                                    speak_handler.on_llm_new_token(chunk)
                        await asyncio.sleep(0.1)
                    print("async_callback exit!")
                if speak_handler: 
                    speak_handler.on_llm_end(None)
                if not btalk and docs:
                    yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                    docs = []
                    tooltype = ToolsType.Unknown
            elif loadtype == "llamacpp":
                presetname = modelconfig["preset"]
                prompttemplate = GeneratePresetPrompt(presetname)
                chain = LLMChain(prompt=prompt, llm=model)
                def running_chain(chain, query):
                    chain.run(query)
                    print("running_chain exit!")
                
                thread = threading.Thread(target=running_chain, args=(chain, query))
                thread.start()
                noret = False
                while True:
                    chunk = async_callback.get_tokens()
                    if chunk is not None:
                        answer += chunk
                        print(chunk, end="")
                        if noret is False:
                            noret = clean_special_text(chunk, prompttemplate)
                            if noret is False:
                                if not btalk:
                                    btalk, new_answer = CallingExternalToolsForCurConfig(answer)
                                if btalk:
                                    new_query, tool_name, docs, tooltype = await GetQueryFromExternalToolsForCurConfig(answer=answer, query=query)
                                    if not new_query:
                                        btalk = False
                                    else:
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
                                        {"text": chunk, "chat_history_id": chat_history_id},
                                        ensure_ascii=False)
                                    if speak_handler: 
                                        speak_handler.on_llm_new_token(chunk)
                    await asyncio.sleep(0.1)
                    if not thread.is_alive():
                        print("async_callback exit!")
                        break
                if speak_handler: 
                    speak_handler.on_llm_end(None)
                if not btalk and docs:
                    yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                    docs = []
                    tooltype = ToolsType.Unknown
    elif modelinfo["mtype"] == ModelType.Online:
        provider = GetProviderByName(webui_config, model_name)
        if provider == "google-api":
            apikey = model_config.get("apikey", "[Your Key]")
            if apikey == "[Your Key]" or apikey == "":
                apikey = os.environ.get('GOOGLE_API_KEY')
            if apikey is None:
                apikey = "EMPTY"
            genai.configure(api_key=apikey)
            calling_tools =[]
            if support_tools:
                calling_tools = GetGoogleNativeTools()
            model = genai.GenerativeModel(model_name=model_name, tools=calling_tools)
            generation_config = {'temperature': temperature}

            updated_history = [
                {'parts': entry['content'], **({'role': 'model'} if entry['role'] == 'assistant' else {'role': "user"})}
                for entry in history
            ]
            if len(imagesdata):
                from io import BytesIO
                import PIL.Image
                content=[]
                content.append(query)
                for imagedata in imagesdata:
                    decoded_data = base64.b64decode(imagedata)
                    imagedata = BytesIO(decoded_data)
                    content.append(PIL.Image.open(imagedata))
                response = model.generate_content(content, generation_config=generation_config, stream=stream)
            else:
                chat = model.start_chat(history=updated_history)
                response = chat.send_message(query, generation_config=generation_config, stream=stream)
            docs=[]
            repeat = True
            while repeat:
                answer = ""
                repeat = False    
                if stream:
                    for chunk in response:
                        if hasattr(chunk, 'parts'):
                            for part in chunk.parts:
                                function_name = ""
                                function_response = ""
                                if fn := part.function_call:
                                    args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                                    args_dict = {key: val for key, val in fn.args.items()}
                                    function_text = f"{fn.name}({args})"
                                    print(function_text)
                                    if function_text:
                                        function_name = fn.name
                                        function_response, tool_name, docs, tooltype = await RunAllEnableToolsInString(fn.name, args_dict, query)
                                        break
                                elif text_text := part.text:
                                    answer += text_text
                                    yield json.dumps(
                                        {"text": text_text, "chat_history_id": chat_history_id},
                                        ensure_ascii=False)
                                    if speak_handler: 
                                        speak_handler.on_llm_new_token(text_text)
                            if function_name and function_response:
                                repeat = True
                                new_answer = GetNewAnswerForCurConfig("", tool_name, tooltype)
                                yield json.dumps(
                                    {"clear": new_answer, "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                                user_answer = GetUserAnswerForCurConfig(tool_name, tooltype)
                                yield json.dumps(
                                    {"user": user_answer, "tooltype": tooltype.value},
                                    ensure_ascii=False)
                                response = chat.send_message(
                                    genai.protos.Content(
                                    parts=[genai.protos.Part(
                                        function_response = genai.protos.FunctionResponse(
                                        name=function_name,
                                        response={'result': function_response}))]))
                        elif hasattr(chunk, 'text'):
                            answer = chunk.text
                            yield json.dumps(
                                {"text": answer, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            if speak_handler: 
                                speak_handler.on_llm_new_token(chunk.text)
                        await asyncio.sleep(0.1)
                else:
                    if hasattr(response, 'parts'):
                        for part in response.parts:
                            function_name = ""
                            function_response = ""
                            if fn := part.function_call:
                                #btalk = True
                                args = ", ".join(f"{key}={val}" for key, val in fn.args.items())
                                args_dict = {key: val for key, val in fn.args.items()}
                                function_text = f"{fn.name}({args})"
                                print(function_text)
                                if function_text:
                                    function_name = fn.name
                                    function_response, tool_name, docs, tooltype = await RunAllEnableToolsInString(fn.name, args_dict)
                                    repeat = True
                                    break
                            elif text_text := part.text:
                                answer += text_text
                                yield json.dumps(
                                    {"text": text_text, "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                                if speak_handler: 
                                    speak_handler.on_llm_new_token(text_text)
                        if function_name and function_response:
                            new_answer = GetNewAnswerForCurConfig("", tool_name, tooltype)
                            yield json.dumps(
                                {"clear": new_answer, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            user_answer = GetUserAnswerForCurConfig(tool_name, tooltype)
                            yield json.dumps(
                                {"user": user_answer, "tooltype": tooltype.value},
                                ensure_ascii=False)
                            response = chat.send_message(
                                genai.protos.Content(
                                parts=[genai.protos.Part(
                                    function_response = genai.protos.FunctionResponse(
                                    name=function_name,
                                    response={'result': function_response}))]))
                    elif hasattr(response, 'text'):
                        answer = response.text
                        yield json.dumps(
                            {"text": answer, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                        if speak_handler: 
                            speak_handler.on_llm_new_token(response.text)
                await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_end(None)
            if not repeat and docs:
                yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                docs = []
                tooltype = ToolsType.Unknown
        elif provider == "ali-cloud-api":
            from http import HTTPStatus
            from dashscope import Generation
            from dashscope.api_entities.dashscope_response import Role
            apikey = model_config.get("apikey", "[Your Key]")
            if apikey == "[Your Key]" or apikey == "":
                apikey = os.environ.get('ALI_API_KEY')
            if apikey is None:
                apikey = "EMPTY"
            docs = []
            btalk = True
            while btalk:
                answer = ""
                btalk = False
                messages = history.copy()
                messages.append({'role': Role.USER,
                            'content': query})
                if stream is True:
                    responses = Generation.call(model=model_name,
                                    messages=messages,
                                    result_format='message',
                                    api_key=apikey,
                                    stream=True,
                                    incremental_output=True,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    )
                    for response in responses:
                        if response.status_code == HTTPStatus.OK:
                            print(response)
                            answer += response.output.choices[0]['message']['content']
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
                                    {"text": response.output.choices[0]['message']['content'], "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                            await asyncio.sleep(0.1)
                            if speak_handler: 
                                speak_handler.on_llm_new_token(response.output.choices[0]['message']['content'])
                        else:
                            error_message = f'Request id: {response.request_id}, \
                            Status code: {response.status_code}, \
                            error code: {response.code}, \
                            error message: {response.message}'
                            yield json.dumps(
                                {"text": error_message, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            await asyncio.sleep(0.1)
                            break
                else:
                    response = Generation.call(model=model_name,
                                messages=messages,
                                result_format='message',
                                api_key=apikey,
                                temperature=temperature,
                                    max_tokens=max_tokens,
                                )
                    if response.status_code == HTTPStatus.OK:
                        print(response)
                        answer += response.output.choices[0]['message']['content']
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
                                await asyncio.sleep(0.1)
                        else:
                            yield json.dumps(
                                {"text": response.output.choices[0]['message']['content'], "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            await asyncio.sleep(0.1)
                        if speak_handler: 
                            speak_handler.on_llm_new_token(response.output.choices[0]['message']['content'])
                    else:
                        error_message = f'Request id: {response.request_id}, \
                        Status code: {response.status_code}, \
                        error code: {response.code}, \
                        error message: {response.message}'
                        yield json.dumps(
                            {"text": error_message, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                    await asyncio.sleep(0.1)
                if speak_handler: 
                    speak_handler.on_llm_end(None)
                if not btalk and docs:
                    yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                    docs = []
                    tooltype = ToolsType.Unknown

        elif provider == "openai-api" or provider == "kimi-cloud-api":
            from openai import OpenAI
            calling_tools =[]
            tool_choice = None
            if support_tools:
                calling_tools = GetOpenaiNativeTools()
                if calling_tools:
                    tool_choice = "auto"
                    stream = False
                else:
                    calling_tools = None
            if provider == "openai-api":
                apikey = model_config.get("api_key", "[Your Key]")
                if apikey == "[Your Key]":
                    apikey = os.environ.get('OPENAI_API_KEY')
            elif provider == "kimi-cloud-api":
                apikey = model_config.get("api_key", "[Your Key]")
                if apikey == "[Your Key]":
                    apikey = os.environ.get('KIMI_API_KEY')
            if not apikey:
                apikey = "EMPTY"
            if provider == "openai-api":
                client = OpenAI(api_key=apikey)
            else:
                proxy = model_config.get("baseurl", "")
                client = OpenAI(api_key=apikey, base_url=proxy)
            docs = []
            btalk = True
            message = history.copy()
            message.append({"role": "user", "content": query})
            while btalk:
                answer = ""
                btalk = False
                chat_response = chat_completion_request(
                    client=client,
                    model=model_name,
                    messages=message,
                    tools=calling_tools,
                    tool_choice=tool_choice,
                    stream=stream
                )
                if hasattr(chat_response, 'status_code') and chat_response.status_code != 200:
                    yield json.dumps(
                        {"text": chat_response.message, "chat_history_id": chat_history_id},
                        ensure_ascii=False)
                    await asyncio.sleep(0.1)
                else:
                    if stream:
                        for response in chat_response:
                            response_delta = response.choices[0].delta
                            token = response_delta.content
                            if token:
                                answer += token
                                yield json.dumps(
                                    {"text": token, "chat_history_id": chat_history_id},
                                    ensure_ascii=False)
                                if speak_handler: 
                                    speak_handler.on_llm_new_token(token)
                                await asyncio.sleep(0.1)
                    else:
                        response_message = chat_response.choices[0].message
                        tool_calls = response_message.tool_calls
                        answer = response_message.content
                        if answer:
                            yield json.dumps(
                                {"text": answer, "chat_history_id": chat_history_id},
                                ensure_ascii=False)
                            if speak_handler: 
                                speak_handler.on_llm_new_token(token)
                        if tool_calls:
                            for tool_call in tool_calls:
                                function_name = tool_call.function.name
                                function_response = ""
                                args = tool_call.function.arguments
                                try:
                                    args_dict = json.loads(args)
                                except Exception as _:
                                    args_dict = {}
                                function_text = f"{function_name}({args})"
                                print(function_text)
                                if function_text:
                                    function_response, tool_name, docs, tooltype = await RunAllEnableToolsInString(function_name, args_dict, query)
                                    if function_name and function_response:
                                        btalk = True
                                        new_answer = GetNewAnswerForCurConfig("", tool_name, tooltype)
                                        yield json.dumps(
                                            {"clear": new_answer, "chat_history_id": chat_history_id},
                                            ensure_ascii=False)
                                        user_answer = GetUserAnswerForCurConfig(tool_name, tooltype)
                                        yield json.dumps(
                                            {"user": user_answer, "tooltype": tooltype.value},
                                            ensure_ascii=False)
                                        if provider == "openai-api":
                                            if tooltype == ToolsType.ToolCodeInterpreter:
                                                tool_choice = "none"
                                            message.append({"tool_call_id": tool_call.id, "role": "function", "name": function_name,"content": function_response})
                                        else:
                                            message.append({"role": "assistant", "content": new_answer})
                                            message.append({"role": "user", "content": function_response})
                await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_end(None)
            if docs:
                yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                docs = []
                tooltype = ToolsType.Unknown

        elif provider == "baidu-cloud-api":
            import qianfan
            apikey = model_config.get("apikey", "[Your Key]")
            if apikey == "[Your Key]" or apikey == "":
                apikey = os.environ.get('QIANFAN_ACCESS_KEY')
            if apikey is None:
                apikey = "EMPTY"
            secretkey = model_config.get("secretkey", "[Your Key]")
            if secretkey == "[Your Key]" or secretkey == "":
                secretkey = os.environ.get('QIANFAN_SECRET_KEY')
            if secretkey is None:
                secretkey = "EMPTY"

            chat_comp = qianfan.ChatCompletion(ak=apikey, sk=secretkey)

            docs=[]
            btalk = True
            while btalk:
                btalk = False
                answer = ""
                system_list = []
                if history and history[0]["role"] == "system":
                    system_list = [history.pop(0)]
                    system_list[0]['role'] = "user"
                    system_list.append({"role": "assistant", "content": "OK!"})

                messages = system_list + history
                messages.append({'role': "user",
                            'content': query})
                responses = chat_comp.do(model=model_name, messages=messages, stream=True)
                for response in responses:
                    print(response)
                    answer += response['result']
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
                            {"text": response['result'], "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                    await asyncio.sleep(0.1)
                    if speak_handler: 
                        speak_handler.on_llm_new_token(response.output.choices[0]['message']['content'])
                if speak_handler: 
                    speak_handler.on_llm_end(None)
                if not btalk and docs:
                    yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                    docs = []
                    tooltype = ToolsType.Unknown
        
        elif provider == "groq-api":
            from groq import Groq
            apikey = model_config.get("apikey", "[Your Key]")
            if apikey == "[Your Key]" or apikey == "":
                apikey = os.environ.get('GROQ_API_KEY')
            if apikey is None:
                apikey = "EMPTY"
            calling_tools =[]
            tool_choice = None
            if support_tools:
                calling_tools = GetOpenaiNativeTools()
                if calling_tools:
                    tool_choice = "auto"
                else:
                    calling_tools = None
            client = Groq(
                api_key=apikey,
            )
            docs = []
            btalk = True
            message = history.copy()
            message.append({"role": "user", "content": query})
            while btalk:
                answer = ""
                btalk = False
                chat_response = chat_completion_request(
                    client=client,
                    model=model_name,
                    messages=message,
                    tools=calling_tools,
                    tool_choice=tool_choice,
                )
                if hasattr(chat_response, 'status_code') and chat_response.status_code != 200:
                    yield json.dumps(
                        {"text": chat_response.message, "chat_history_id": chat_history_id},
                        ensure_ascii=False)
                    await asyncio.sleep(0.1)
                else:
                    response_message = chat_response.choices[0].message
                    tool_calls = response_message.tool_calls
                    answer = response_message.content
                    if answer:
                        yield json.dumps(
                            {"text": answer, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                        if speak_handler: 
                            speak_handler.on_llm_new_token(token)
                    if tool_calls:
                        for tool_call in tool_calls:
                            function_name = tool_call.function.name
                            function_response = ""
                            args = tool_call.function.arguments
                            try:
                                args_dict = json.loads(args)
                            except Exception as _:
                                args_dict = {}
                            function_text = f"{function_name}({args})"
                            print(function_text)
                            if function_text:
                                function_response, tool_name, docs, tooltype = await RunAllEnableToolsInString(function_name, args_dict, query)
                                if function_name and function_response:
                                    btalk = True
                                    new_answer = GetNewAnswerForCurConfig("", tool_name, tooltype)
                                    yield json.dumps(
                                        {"clear": new_answer, "chat_history_id": chat_history_id},
                                        ensure_ascii=False)
                                    user_answer = GetUserAnswerForCurConfig(tool_name, tooltype)
                                    yield json.dumps(
                                        {"user": user_answer, "tooltype": tooltype.value},
                                        ensure_ascii=False)
                                    message.append({"role": "assistant", "content": new_answer})
                                    message.append({"role": "user", "content": function_response})
                                    tool_choice = "none"
                await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_end(None)
            if docs:
                yield json.dumps({"tooltype": tooltype.value, "docs": docs}, ensure_ascii=False)
                docs = []
                tooltype = ToolsType.Unknown

    update_chat_history(chat_history_id, response=answer)

def special_model_chat(
    model: Any,
    tokenizer: Any,
    async_callback: Any,
    modelinfo: Any,
    query: str,
    imagesdata: List[str],
    audiosdata: List[str],
    videosdata: List[str],
    imagesprompt: List[str],
    history: List[dict],
    stream: bool,
    speechmodel: dict,
    temperature: float,
    max_tokens: Optional[int],
    prompt_name: str,
):    
    return StreamingResponse(special_chat_iterator(
                                            model=model,
                                            tokenizer=tokenizer,
                                            async_callback=async_callback,
                                            query=query,
                                            imagesdata=imagesdata,
                                            audiosdata=audiosdata,
                                            videosdata=videosdata,
                                            imagesprompt=imagesprompt,
                                            history=history,
                                            stream=stream,
                                            speechmodel=speechmodel,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            modelinfo=modelinfo,
                                            prompt_name=prompt_name),
                             media_type="text/event-stream")

def model_knowledge_base_chat(
    app: FastAPI,
    query: str,
    knowledge_base_name: str,
    top_k: int,
    score_threshold: float,
    history: List[dict],
    stream: bool,
    imagesdata: List[str],
    speechmodel: dict,
    temperature: float,
    max_tokens: Optional[int],
    prompt_name: str,
):
    pass

def load_causallm_model(app: FastAPI, model_name, model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if model_name == "cogvlm-chat-hf":
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map=device, low_cpu_mem_usage=True, trust_remote_code=True).eval()
    app._model = model
    app._tokenizer = tokenizer
    app._model_name = model_name

def load_automodel_model(app: FastAPI, model_name, model_path, device):
    import torch
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, device_map=device, trust_remote_code=True, torch_dtype=torch.float16)
    model = model.to(device=device)
    app._model = model
    app._tokenizer = tokenizer
    app._model_name = model_name

def load_video_diffusion_model(app: FastAPI, model_name, model_path, device):
    import torch
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config
    def load_model(config: str, model_path: str, device: str, num_frames: int, num_steps: int):
        config = OmegaConf.load(config)
        config.model.params.conditioner_config.params.emb_models[0].params.open_clip_embedding_config.params.init_device = device
        config.model.params.sampler_config.params.num_steps = num_steps
        config.model.params.sampler_config.params.guider_config.params.num_frames = (num_frames)
        config.model.params.ckpt_path = model_path
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval().requires_grad_(False)
        return model
    num_frames = 25
    num_steps = 30
    if model_name == "stable-video-diffusion-img2vid":
        model_config = "generative-models/scripts/sampling/configs/svd.yaml"
        model_path += "/svd.safetensors"
    elif model_name == "stable-video-diffusion-img2vid-xt":
        model_config = "generative-models/scripts/sampling/configs/svd_xt.yaml"
        model_path += "/svd_xt.safetensors"
    model = load_model(model_config, model_path, device, num_frames, num_steps)
    model.conditioner.cpu()
    model.first_stage_model.cpu()
    model.model.to(dtype=torch.float16)
    torch.cuda.empty_cache()
    model = model.requires_grad_(False)
    
    app._model = model
    app._model_name = model_name

def init_multimodal_models(app: FastAPI, args):
    model_name = args.model_names[0]
    model_path = args.model_path
    if len(model_name) == 0 or len(model_path) == 0:
        return
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    model_info : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    model_info["mtype"], model_info["msize"], model_info["msubtype"] = GetModelInfoByName(webui_config, model_name)
    model_info["mname"] = model_name
    model_config = GetModelConfig(webui_config, model_info)
    load_type = model_config.get("load_type", "")
    if load_type == "causallm":
        load_causallm_model(app=app, model_name=model_name, model_path=model_path, device=args.device)
    elif load_type == "automodel":
        load_automodel_model(app=app, model_name=model_name, model_path=model_path, device=args.device)
    elif load_type == "vdiffusion":
        load_video_diffusion_model(app=app, model_name=model_name, model_path=model_path, device=args.device)

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N, T, device, dtype=None):
    import torch
    import math
    from einops import repeat
    batch = {}
    batch_uc = {}
    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device, dtype=dtype)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device, dtype=dtype),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]
    if T is not None:
        batch["num_video_frames"] = T
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def generating_video(
    model,
    imagesdata: List[str],
    resize_image: bool = False,
    num_frames: Optional[int] = None,
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = str(TMP_DIR),
):
    import torch
    from io import BytesIO
    from PIL import Image
    from glob import glob
    import os
    import cv2
    import numpy as np
    from einops import rearrange, repeat
    from torchvision.transforms import ToTensor
    from torchvision.transforms import functional as TF
    torch.manual_seed(seed)
    all_out_paths = []
    for encode_data in imagesdata:
        decoded_data = base64.b64decode(encode_data)
        imagedata = BytesIO(decoded_data)
        with Image.open(imagedata).convert('RGB') as image:
            if resize_image and image.size != (720, 720):
                print(f"Resizing {image.size} to (720, 720)")
                image = TF.resize(TF.resize(image, 720), (720, 720))
            w, h = image.size
            if h % 64 != 0 or w % 64 != 0:
                width, height = map(lambda x: x - x % 64, (w, h))
                image = image.resize((width, height))
                print(
                    f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                )
            image = ToTensor()(image)
            image = image * 2.0 - 1.0

        image = image.unsqueeze(0).to(device)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (num_frames, C, H // F, W // F)
        if (H, W) != (720, 720):
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )
        if fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")
        if fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug
        value_dict["cond_frames_without_noise"] = image
        value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
        value_dict["cond_aug"] = cond_aug
        # low vram mode
        model.conditioner.cpu()
        model.first_stage_model.cpu()
        torch.cuda.empty_cache()
        model.sampler.verbose = True

        with torch.no_grad():
            with torch.autocast(device):
                model.conditioner.to(device)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [1, num_frames],
                    T=num_frames,
                    device=device,
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )
                model.conditioner.cpu()
                torch.cuda.empty_cache()

                # from here, dtype is fp16
                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
                for k in uc.keys():
                    uc[k] = uc[k].to(dtype=torch.float16)
                    c[k] = c[k].to(dtype=torch.float16)

                randn = torch.randn(shape, device=device, dtype=torch.float16)
                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(2, num_frames).to(device)
                additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                for k in additional_model_inputs:
                    if isinstance(additional_model_inputs[k], torch.Tensor):
                        additional_model_inputs[k] = additional_model_inputs[k].to(dtype=torch.float16)

                def denoiser(input, sigma, c):
                    return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

                samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
                samples_z.to(dtype=model.first_stage_model.dtype)
                model.en_and_decode_n_samples_a_time = decoding_t
                model.first_stage_model.to(device)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                model.first_stage_model.cpu()
                torch.cuda.empty_cache()

                os.makedirs(output_folder, exist_ok=True)
                base_count = len(glob(os.path.join(output_folder, "*.mp4")))
                video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
                writer = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"MP4V"),
                    fps_id + 1,
                    (samples.shape[-1], samples.shape[-2]),
                )
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                for frame in vid:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    writer.write(frame)
                writer.release()
                all_out_paths.append(video_path)
    return all_out_paths

def multimodal_model_chat(
        model: Any,
        tokenizer: Any,
        modelinfo: Any,
        query: str,
        imagesdata: List[str],
        audiosdata: List[str],
        videosdata: List[str],
        imagesprompt: List[str],
        history: List[dict],
        stream: bool,
        speechmodel: dict,
        temperature: float,
        max_tokens: Optional[int],
        prompt_name: str,
):
    if modelinfo is None:
        return json.dumps(
            {"text": "Unusual error!", "chat_history_id": 123},
            ensure_ascii=False)
    
    async def multimodal_chat_iterator(model: Any,
                            tokenizer: Any,
                            query: str,
                            imagesdata: List[str],
                            audiosdata: List[str],
                            videosdata: List[str],
                            imagesprompt: List[str],
                            history: List[dict] = [],
                            modelinfo: Any = None,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
        import torch
        import uuid
        from io import BytesIO
        from PIL import Image
        from WebUI.Server.utils import detect_device
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        model_config = GetModelConfig(webui_config, modelinfo)
        support_tools = model_config.get("support_tools", False)
        model_name = modelinfo["mname"]
        speak_handler = None
        if len(speechmodel):
                modeltype = speechmodel.get("type", "")
                provider = speechmodel.get("provider", "")
                #spmodel = speechmodel.get("model", "")
                spspeaker = speechmodel.get("speaker", "")
                speechkey = speechmodel.get("speech_key", "")
                speechregion = speechmodel.get("speech_region", "")
                if modeltype == "local" or modeltype == "cloud":
                    speak_handler = StreamSpeakHandler(run_place=modeltype, provider=provider, synthesis=spspeaker, subscription=speechkey, region=speechregion)

        answer = ""
        chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)
        modelconfig = GetModelConfig(webui_config, modelinfo)
        device = modelconfig.get("device", "auto")
        device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
        if model_name == "cogvlm-chat-hf":
            # Only support one image.
            image1 = []
            if len(imagesdata):
                decoded_data = base64.b64decode(imagesdata[0])
                imagedata = BytesIO(decoded_data)
                image1 = [Image.open(imagedata).convert('RGB')]
            formated_history = []
            if len(history):
                formatted_history = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history), 2) if i < len(history)-1 and history[i]['role'] == 'user']
            inputs = model.build_conversation_input_ids(tokenizer, query=query, history=formated_history, images=image1)
            if len(imagesdata):
                inputs = {
                    'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                    'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                    'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
                }
            else:
                inputs = {
                    'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
                    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
                    'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
                    'images': [],
                }
            gen_kwargs = {"max_length": 2048, "do_sample": False}
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                answer =tokenizer.decode(outputs[0])

            yield json.dumps(
                {"text": answer, "chat_history_id": chat_history_id},
                ensure_ascii=False)
            await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_new_token(answer)
            if speak_handler: 
                speak_handler.on_llm_end(None)
        elif model_name == "Qwen-VL-Chat" or model_name == "Qwen-VL-Chat-Int4":
            img_list = []
            for image in imagesdata:
                decoded_data = base64.b64decode(image)
                imagedata = BytesIO(decoded_data)
                image_rgb = Image.open(imagedata).convert('RGB')
                image_path = str(TMP_DIR / Path(str(uuid.uuid4()) + ".jpg"))
                image_rgb.save(image_path)
                img_list.append(image_path)
            query_list = []
            for img in img_list:
                img_dict = {'image': img}
                query_list.append(img_dict)
            query_dict = {'text': query}
            query_list.append(query_dict)
            query_tkz = tokenizer.from_list_format(query_list)
            formatted_history = []
            if len(history):
                formatted_history = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history), 2) if i < len(history)-1 and history[i]['role'] == 'user']
            answer, history = model.chat(tokenizer, query=query_tkz, history=formatted_history)
            yield json.dumps(
                {"text": answer, "chat_history_id": chat_history_id},
                ensure_ascii=False)
            await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_new_token(answer)
            if speak_handler: 
                speak_handler.on_llm_end(None)
        elif model_name == "stable-video-diffusion-img2vid" or model_name == "stable-video-diffusion-img2vid-xt":
            import random
            answer = "video"
            seed = random.randint(0, 2**32)
            seed = int(seed)
            resize_image = True
            num_frames = 25
            fps_id=6
            motion_bucket_id=127
            cond_aug=0.02
            decoding_t=2
            all_out_paths = generating_video(
                model=model,
                imagesdata=imagesdata,
                resize_image=resize_image,
                num_frames=num_frames,
                fps_id=fps_id,
                motion_bucket_id=motion_bucket_id,
                cond_aug=cond_aug,
                seed=seed,
                decoding_t=decoding_t,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
                device=device,
            )

            for _, video_path in enumerate(all_out_paths):
                video_path_h264 = video_path[:-4] + "_h264.mp4"
                os.system(f"ffmpeg -i {video_path} -c:v libx264 {video_path_h264}")
                yield json.dumps(
                    {"text": video_path_h264, "chat_history_id": chat_history_id},
                    ensure_ascii=False)
                await asyncio.sleep(0.01)

        elif model_name == "Qwen-Audio-Chat":
            audio_list = []
            for audio in audiosdata:
                decoded_data = base64.b64decode(audio)
                #audiodata = BytesIO(decoded_data)
                audiopath = str(TMP_DIR / Path(str(uuid.uuid4()) + ".wav"))
                with open(audiopath, "wb") as file:
                    file.write(decoded_data)
                audio_list.append(audiopath)
            query_list = []
            for ado in audio_list:
                ado_dict = {'audio': ado}
                query_list.append(ado_dict)
            query_dict = {'text': query}
            query_list.append(query_dict)
            query_tkz = tokenizer.from_list_format(query_list)
            formatted_history = []
            if len(history):
                formatted_history = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history), 2) if i < len(history)-1 and history[i]['role'] == 'user']
            answer, history = model.chat(tokenizer, query=query_tkz, history=formatted_history)
            yield json.dumps(
                {"text": answer, "chat_history_id": chat_history_id},
                ensure_ascii=False)
            await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_new_token(answer)
            if speak_handler: 
                speak_handler.on_llm_end(None)

        elif model_name == "Phi-3-vision-128k-instruct":
            from transformers import AutoProcessor
            configinst = InnerJsonConfigWebUIParse()
            model_id = model_config.get("path")
            device = model_config.get("device", "auto")
            device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
            if not model_id:
                model_id = model_config.get("Huggingface")
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

            if imagesdata:
                img_list = []
                count = 1
                image_prompt = f"<|image_{count}|>"
                for image in imagesdata:
                    decoded_data = base64.b64decode(image)
                    imagedata = BytesIO(decoded_data)
                    image_rgb = Image.open(imagedata)#.convert('RGB')
                    img_list.append(image_rgb)
                    if image_prompt:
                        image_prompt = f"<|image_{count}|>"
                    else:
                        image_prompt += "\n" + f"<|image_{count}|>"
                    count += 1

                chat_prompt = {"role": "user", "content": f"{image_prompt}\n{query}"}
                message = history + [chat_prompt]
                prompt = processor.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                inputs = processor(prompt, img_list, return_tensors="pt").to(device)
            else:
                chat_prompt = {"role": "user", "content": f"{query}"}
                message = history + [chat_prompt]
                prompt = processor.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                inputs = processor(prompt, return_tensors="pt").to(device)
            generation_args = { 
                "max_new_tokens": 500, 
                "temperature": temperature, 
                "do_sample": True, 
            } 
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
            # remove input tokens 
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            yield json.dumps(
                {"text": response, "chat_history_id": chat_history_id},
                ensure_ascii=False)
            await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_new_token(answer)
            if speak_handler: 
                speak_handler.on_llm_end(None)

        elif model_name == "MiniCPM-Llama3-V-2_5":
            img_list = []
            for image in imagesdata:
                decoded_data = base64.b64decode(image)
                imagedata = BytesIO(decoded_data)
                image_rgb = Image.open(imagedata).convert('RGB')
                img_list.append(image_rgb)
            chat_prompt = {'role': 'user', 'content': query}
            message = history + [chat_prompt]
            if img_list:
                response = model.chat(
                    image=img_list[0],
                    msgs=message,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=temperature,
                    stream=True
                )
            else:
                response = model.chat(
                    image=None,
                    msgs=message,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=temperature,
                    stream=True
                )

            generated_text = ""
            for new_text in response:
                generated_text += new_text
                yield json.dumps(
                    {"text": new_text, "chat_history_id": chat_history_id},
                    ensure_ascii=False)
                await asyncio.sleep(0.1)
                if speak_handler: 
                    speak_handler.on_llm_new_token(new_text)
            if speak_handler: 
                speak_handler.on_llm_end(None)
        
        elif model_name == "glm-4v-9b":
            img_list = []
            for image in imagesdata:
                decoded_data = base64.b64decode(image)
                imagedata = BytesIO(decoded_data)
                image_rgb = Image.open(imagedata).convert('RGB')
                img_list.append(image_rgb)
            if img_list:
                inputs = tokenizer.apply_chat_template([{"role": "user", "image": img_list[0], "content": query}],
                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                    return_dict=True)
            else:
                inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                    add_generation_prompt=True, tokenize=True, return_tensors="pt",
                    return_dict=True)
            device = model_config.get("device", "auto")
            device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
            inputs = inputs.to(device)
            gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
            answer = ""
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                answer = tokenizer.decode(outputs[0])
            yield json.dumps(
                {"text": answer, "chat_history_id": chat_history_id},
                ensure_ascii=False)
            await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_new_token(answer)
            if speak_handler: 
                speak_handler.on_llm_end(None)

        update_chat_history(chat_history_id, response=answer)
        
    return StreamingResponse(multimodal_chat_iterator(
                                            model=model,
                                            tokenizer=tokenizer,
                                            query=query,
                                            imagesdata=imagesdata,
                                            audiosdata=audiosdata,
                                            videosdata=videosdata,
                                            imagesprompt=imagesprompt,
                                            history=history,
                                            modelinfo=modelinfo,
                                            prompt_name=prompt_name),
                             media_type="text/event-stream")

def model_chat(
        app: FastAPI,
        query: str,
        imagesdata: List[str],
        audiosdata: List[str],
        videosdata: List[str],
        imagesprompt: List[str],
        history: List[dict],
        stream: bool,
        speechmodel: dict,
        temperature: float,
        max_tokens: Optional[int],
        prompt_name: str,
):
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    model_name = app._model_name
    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
    modelinfo["mname"] = model_name
    modelinfo["config"] = GetModelConfig(webui_config, modelinfo)
    if modelinfo["mtype"] == ModelType.Special or modelinfo["mtype"] == ModelType.Online:
        return special_model_chat(app._model, app._tokenizer, app._streamer, modelinfo, query, imagesdata, audiosdata, videosdata, imagesprompt, history, stream, speechmodel, temperature, max_tokens, prompt_name)
    elif modelinfo["mtype"] == ModelType.Code:
        return code_model_chat(app._model, app._tokenizer, app._streamer, modelinfo, query, imagesdata, audiosdata, videosdata, imagesprompt, history, False, speechmodel, temperature, max_tokens, prompt_name)
    elif modelinfo["mtype"] == ModelType.Multimodal:
        return multimodal_model_chat(app._model, app._tokenizer, modelinfo, query, imagesdata, audiosdata, videosdata, imagesprompt, history, False, speechmodel, temperature, max_tokens, prompt_name)
    
def special_model_search_engine_chat(
    model: Any,
    tokenizer: Any,
    async_callback: Any,
    modelinfo: Any,
    query: str,
    search_engine_name: str,
    history: List[dict],
    stream: bool,
    temperature: float,
    max_tokens: Optional[int],
    prompt_name: str,
):
    async def special_search_chat_iterator(model: Any,
        tokenizer: Any,
        async_callback: Any,
        query: str,
        search_engine_name: str,
        history: List[dict] = [],
        modelinfo: Any = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        prompt_name: str = "default",
        ) -> AsyncIterable[str]:
        from WebUI.Server.utils import get_prompt_template
        from WebUI.Server.chat.search_engine_chat import lookup_search_engine

        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        model_name = modelinfo["mname"]

        answer = ""
        searchengine = webui_config.get("SearchEngine")
        top_k = searchengine.get("top_k", 3)
        docs = await lookup_search_engine(query, search_engine_name, top_k)
        context = "\n".join([doc.page_content for doc in docs])
        
        source_documents = [
            f"""from [{inum + 1}] [{doc.metadata["source"]}]({doc.metadata["source"]}) \n\n{doc.page_content}\n\n"""
            for inum, doc in enumerate(docs)
        ]
        if len(source_documents) == 0:
            source_documents.append(f"""<span style='color:red'>No relevant information were found. This response is generated based on the LLM Model '{model_name}' itself!</span>""")

        prompt_template = get_prompt_template("search_engine_chat", prompt_name)
        new_query = prompt_template.replace('{{ context }}', context).replace('{{ question }}', query)
        response = special_chat_iterator(
                model=model,
                tokenizer=tokenizer,
                async_callback=async_callback,
                query=new_query,
                modelinfo=modelinfo,
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
        async for tokens in response:
            if not tokens:
                continue
            answer = json.loads(tokens)["text"]
            yield json.dumps({"answer": answer,
                            "docs": []},
                            ensure_ascii=False)
        yield json.dumps({"answer": "",
                            "docs": source_documents},
                            ensure_ascii=False)

        
    return StreamingResponse(special_search_chat_iterator(
                                            model=model,
                                            tokenizer=tokenizer,
                                            async_callback=async_callback,
                                            query=query,
                                            search_engine_name=search_engine_name,
                                            history=history,
                                            modelinfo=modelinfo,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            prompt_name=prompt_name),
                             media_type="text/event-stream")

def model_search_engine_chat(
    app: FastAPI,
    query: str,
    search_engine_name: str,
    history: List[dict],
    stream: bool,
    temperature: float,
    max_tokens: Optional[int],
    prompt_name: str, 
):
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    model_name = app._model_name
    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
    modelinfo["mname"] = model_name
    if modelinfo["mtype"] == ModelType.Online or modelinfo["mtype"] == ModelType.Special:
        return special_model_search_engine_chat(app._model, app._tokenizer, app._streamer, modelinfo, query, search_engine_name, history, stream, temperature, max_tokens, prompt_name)
    elif modelinfo["mtype"] == ModelType.Code:
        return json.dumps(
            {"text": "Unsupport code model!"},
            ensure_ascii=False)
    elif modelinfo["mtype"] == ModelType.Multimodal:
        return json.dumps(
            {"text": "Unsupport multimodal model!"},
            ensure_ascii=False)