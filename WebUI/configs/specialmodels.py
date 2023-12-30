import json
import asyncio
import threading
from WebUI.Server.utils import wrap_done
import google.generativeai as genai
from fastapi.responses import StreamingResponse
from WebUI.configs.basicconfig import *
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from langchain.chains import LLMChain
from WebUI.Server.utils import FastAPI

def clean_special_text(text : str, prompttemplate: dict):
    anti_prompt = prompttemplate["anti_prompt"]
    if len(anti_prompt):
        def is_substring(strs_list, search_string):
            search_string_lower = search_string.lower()
            for str in strs_list:
                if str.lower() in search_string_lower:
                    return True
            return False
        return is_substring(anti_prompt, text)
    return False

def init_cloud_models(model_name):
    if model_name is None:
        return None
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
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
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.2,
        streamer=streamer
    )
    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    llm_model = HuggingFacePipeline(pipeline=pipe)
    app._model = llm_model
    app._streamer = streamer
    app._model_name = model_name

def load_llamacpp_model(app: FastAPI, model_name, model_path):
    from langchain.llms.llamacpp import LlamaCpp
    from langchain.callbacks.manager import CallbackManager
    from WebUI.Server.chat.StreamHandler import LlamacppStreamCallbackHandler
    async_callback = LlamacppStreamCallbackHandler()
    callback_manager = CallbackManager([async_callback])
    modellist = GetGGUFModelPath(model_path)
    path = model_path + "/" + modellist[0]
    if len(modellist):
        llm_model = LlamaCpp(
            model_path=path,
            temperature=0.7,
            max_tokens=512,
            top_p=1,
            verbose=True,
            callback_manager=callback_manager,
            n_threads=4,
            streaming=True,
        )
        app._model = llm_model
        app._streamer = async_callback
        app._model_name = model_name

def init_special_models(app: FastAPI, args):
    model_name = args.model_names[0]
    model_path = args.model_path
    if len(model_name) == 0 or len(model_path) == 0:
        return
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    model_info = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    model_info["mtype"], model_info["msize"], model_info["msubtype"] = GetModelInfoByName(webui_config, model_name)
    model_info["mname"] = model_name
    model_config = GetModelConfig(webui_config, model_info)
    load_type = model_config.get("load_type", "")
    if load_type == "pipeline":
        load_pipeline_model(app=app, model_name=model_name, model_path=model_path, device=args.device)
    elif load_type == "llamacpp":
        load_llamacpp_model(app=app, model_name=model_name, model_path=model_path)

def special_model_chat(
        model: Any,
        model_name: str,
        async_callback: Any,
        query: str,
        history: List[dict],
        stream: bool,
        speechmodel: dict,
        temperature: float,
        max_tokens: Optional[int],
        prompt_name: str,
):
    async def special_chat_iterator(query: str,
                            history: List[dict] = [],
                            model_name: str = "",
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:
    
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        modelinfo = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
        modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)

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
        if modelinfo["mtype"] == ModelType.Special:
            from langchain.prompts import PromptTemplate
            modelinfo["mname"] = model_name
            modelconfig = GetModelConfig(webui_config, modelinfo)
            loadtype = modelconfig["load_type"]
            presetname = modelconfig["preset"]
            presetconfig = GetPresetConfig(presetname)
            if loadtype != "llamacpp" and loadtype != "pipeline":
                pass
            else:
                prompttemplate = GeneratePresetPrompt(presetname)
                if len(prompttemplate):
                    input_variables = prompttemplate["input_variables"]
                    prompt = PromptTemplate(template=prompttemplate["prompt_templates"], input_variables=input_variables)
                    chain = LLMChain(prompt=prompt, llm=model)
                    def running_chain(chain, input_variables, query):
                        chain.run(GenerateModelPrompt(input_variables, query))
                        print("running_chain exit!")
                    
                    thread = threading.Thread(target=running_chain, args=(chain, input_variables, query))
                    thread.start()
                    noret = False
                    if loadtype == "pipeline":
                        streamer = async_callback
                        for chunk in streamer:
                            if chunk is not None:
                                print(chunk, end="")
                                if noret is False:
                                    noret = clean_special_text(chunk, prompttemplate)
                                    if noret is False:
                                        if speak_handler: speak_handler.on_llm_new_token(chunk)
                                        yield json.dumps(
                                            {"text": chunk, "chat_history_id": chat_history_id},
                                            ensure_ascii=False)
                                        await asyncio.sleep(0.1)
                        print("async_callback exit!")
                    elif loadtype == "llamacpp":
                        while True:
                            chunk = async_callback.get_tokens()
                            if chunk is not None:
                                print(chunk, end="")
                                if noret is False:
                                    noret = clean_special_text(chunk, prompttemplate)
                                    if noret is False:
                                        if speak_handler: speak_handler.on_llm_new_token(chunk)
                                        yield json.dumps(
                                            {"text": chunk, "chat_history_id": chat_history_id},
                                            ensure_ascii=False)
                                        await asyncio.sleep(0.1)
                            if not thread.is_alive():
                                print("async_callback exit!")
                                break
                    if speak_handler: speak_handler.on_llm_end(None)
        elif modelinfo["mtype"] == ModelType.Online:
            provider = GetProviderByName(webui_config, model_name)
            if provider == "google-api":
                modelinfo["mname"] = model_name
                model_config = GetModelConfig(webui_config, modelinfo)
                apikey = model_config.get("apikey", "[Your Key]")
                if apikey == "[Your Key]" or apikey == "":
                    apikey = os.environ.get('GOOGLE_API_KEY')
                if apikey == None:
                    apikey = "EMPTY"
                genai.configure(api_key=apikey)
                model = genai.GenerativeModel(model_name=model_name)
                updated_history = [
                    {'parts': entry['content'], **({'role': 'model'} if entry['role'] == 'assistant' else {'role': entry['role']})}
                    for entry in history
                ]
                chat = model.start_chat(history=updated_history)
                generation_config = {'temperature': temperature}
                response = chat.send_message(query, generation_config=generation_config, stream=stream)
                if stream is True:
                    for chunk in response:
                        if speak_handler: speak_handler.on_llm_new_token(chunk.text)
                        yield json.dumps(
                            {"text": chunk.text, "chat_history_id": chat_history_id},
                            ensure_ascii=False)
                        await asyncio.sleep(0.1)
                    if speak_handler: speak_handler.on_llm_end(None)
                else:
                    for chunk in response:
                        if speak_handler: speak_handler.on_llm_new_token(chunk.text)
                        answer += chunk.text
                    yield json.dumps(
                        {"text": answer, "chat_history_id": chat_history_id},
                        ensure_ascii=False)
                    await asyncio.sleep(0.1)
                    if speak_handler: speak_handler.on_llm_end(None)
        
        update_chat_history(chat_history_id, response=answer)
        
    return StreamingResponse(special_chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")