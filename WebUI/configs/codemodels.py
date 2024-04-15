
import json
import asyncio
from WebUI.configs.basicconfig import (ModelType, ModelSize, ModelSubType, GetModelInfoByName, GetModelConfig)
from fastapi.responses import StreamingResponse
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from WebUI.Server.utils import FastAPI
from typing import Dict, List, Any, Optional, AsyncIterable

def load_causallm_model(app: FastAPI, model_name, model_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", device_map=device)
    app._model = model
    app._tokenizer = tokenizer
    app._model_name = model_name

def load_llama_model(app: FastAPI, model_name, model_path, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map=device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=0.2,
        torch_dtype=torch.float16,
        streamer=streamer
    )
    # pipe.model.config.pad_token_id = pipe.model.config.eos_token_id
    # llm_model = HuggingFacePipeline(pipeline=pipe)
    app._model = pipe
    app._streamer = streamer
    app._tokenizer = tokenizer
    app._model_name = model_name

def init_code_models(app: FastAPI, args):
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
    elif load_type == "llama":
        load_llama_model(app=app, model_name=model_name, model_path=model_path, device=args.device)

def code_model_chat(
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
    if modelinfo is None:
        return json.dumps(
            {"text": "Unusual error!", "chat_history_id": 123},
            ensure_ascii=False)
    
    async def code_chat_iterator(model: Any,
                            tokenizer: Any,
                            async_callback: Any,
                            query: str,
                            imagesdata: List[str],
                            audiosdata: List[str],
                            videosdata: List[str],
                            imagesprompt: List[str],
                            history: List[dict] = [],
                            modelinfo: Any = None,
                            temperature: float = 0.7,
                            max_tokens: Optional[int] = 2048,
                            prompt_name: str = prompt_name,
                            ) -> AsyncIterable[str]:

        from WebUI.Server.utils import detect_device
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
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
        if model_name == "stable-code-3b":
            if max_tokens is None:
                max_tokens = 512
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            tokens = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True,
            )
            answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
            #answer = answer.replace(query, '').strip()
            #while(answer.startswith(("'", '"', ' ', ',', '.', '!', '?'))):
            #    answer = answer[1:].strip()
            answer = "```python\n" + answer + "\n```"
            if speak_handler: 
                speak_handler.on_llm_new_token(answer)
            yield json.dumps(
                {"text": answer, "chat_history_id": chat_history_id},
                ensure_ascii=False)
            await asyncio.sleep(0.1)
            if speak_handler: 
                speak_handler.on_llm_end(None)

        elif model_name == "CodeLlama-7b-Python-hf" or \
             model_name == "CodeLlama-13b-Python-hf" or \
             model_name == "CodeLlama-7b-Instruct-hf" or \
             model_name == "CodeLlama-13b-Instruct-hf":
             sequences = model(
                query,
                #do_sample=True,
                temperature=0.2,
                top_p=0.95,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=512,
             )
             answer = ""
             yield json.dumps(
                    {"text": "```python\n", "chat_history_id": chat_history_id},
                    ensure_ascii=False)
             for seq in sequences:
                yield json.dumps(
                    {"text": seq['generated_text'], "chat_history_id": chat_history_id},
                    ensure_ascii=False)
             await asyncio.sleep(0.1)
             yield json.dumps(
                    {"text": "\n```", "chat_history_id": chat_history_id},
                    ensure_ascii=False)
             if speak_handler: 
                speak_handler.on_llm_new_token(answer)
                speak_handler.on_llm_end(None)
        
        update_chat_history(chat_history_id, response=answer)
        
    return StreamingResponse(code_chat_iterator(
                                            model=model,
                                            tokenizer=tokenizer,
                                            async_callback=async_callback,
                                            query=query,
                                            imagesdata=imagesdata,
                                            audiosdata=audiosdata,
                                            videosdata=videosdata,
                                            imagesprompt=imagesprompt,
                                            history=history,
                                            modelinfo=modelinfo,
                                            temperature=temperature,
                                            max_tokens=max_tokens,
                                            prompt_name=prompt_name),
                             media_type="text/event-stream")
