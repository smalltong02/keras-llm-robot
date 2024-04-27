from fastapi import Body
from WebUI.Server.knowledge_base.utils import SCORE_THRESHOLD
from WebUI.configs import HTTPX_DEFAULT_TIMEOUT
from WebUI.Server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config, get_vtot_worker_config, get_speech_worker_config,
                          get_image_recognition_worker_config, get_image_generation_worker_config,
                          get_music_generation_worker_config)
import json
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.configs.basicconfig import (ModelType, ModelSize, ModelSubType, GetSizeName, GetSubTypeName)
from fastapi.responses import StreamingResponse
from typing import List, Optional, AsyncIterable

def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"), 
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            models = r.json()["models"]
            data = {m: get_model_config(m).data for m in models}
            return BaseResponse(data=data)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get current model, error: {e}")

def get_running_models(
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"),
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/list_models")
            models = r.json()["models"]
            if len(models):
                data = {m: get_model_config(m).data for m in models}
                return BaseResponse(data=data)
            else:
                workerconfig = get_model_worker_config()
                worker_address = "http://" + workerconfig["host"] + ":" + str(workerconfig["port"])
                with get_httpx_client() as client:
                    try:
                        r = client.post(worker_address + "/get_name",
                            json={})
                        name = r.json().get("name", "")
                        if name != "":
                            models = [name]
                    except Exception as _:
                        pass
                return BaseResponse(data=models)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get all running models from controller: {controller_address} error: {e}")


def list_config_models() -> BaseResponse:
    '''
    read mode all list
    '''
    configs = {}
    online_list = []
    list_models = list_config_llm_models()
    for name, config in list_models["online"].items():
        for k, v in config.items():
            if k == "modellist":
                online_list += v
                break
    configs['online'] = online_list

    local_list = []
    for name, path in list_models["local"].items():
        local_list += [name]
    configs['local'] = local_list
    return BaseResponse(data=configs)


def get_model_config(
    model_name: str = Body(description="LLM Model name"),
    placeholder: str = Body(description="Unused")
) -> BaseResponse:
    config = {}
    
    for k, v in get_model_worker_config(model_name=model_name).items():
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            config[k] = v

    return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="Stop Model", examples=[""]),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}. error: {e}")
    
async def chat_llm_model(
    query: str = Body(..., description="User input: ", examples=["chat"]),
    imagesdata: List[str] = Body([], description="image data", examples=["image"]),
    audiosdata: List[str] = Body([], description="audio data", examples=["audio"]),
    videosdata: List[str] = Body([], description="video data", examples=["video"]),
    imagesprompt: List[str] = Body([], description="prompt data", examples=["prompt"]),
    history: List[dict] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body("", description="model name"),
    speechmodel: dict = Body({}, description="speech model"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default", description=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> StreamingResponse:
    
    controller_address = controller_address or fschat_controller_address()
    async def fake_json_streamer() -> AsyncIterable[str]:
        import asyncio
        with get_httpx_client() as client:
            response = client.stream(
                "POST",
                url=controller_address + "/text_chat",
                json={
                    "query": query,
                    "imagesdata": imagesdata,
                    "audiosdata": audiosdata,
                    "videosdata": videosdata,
                    "imagesprompt": imagesprompt,
                    "history": history,
                    "stream": stream,
                    "model_name": model_name,
                    "speechmodel": speechmodel,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "prompt_name": prompt_name,
                    },
            )
            with response as r:
                for chunk in r.iter_text(None):
                    if not chunk:
                        continue
                    yield chunk
                    await asyncio.sleep(0.1)
    return StreamingResponse(fake_json_streamer(), media_type="text/event-stream")

async def llm_knowledge_base_chat(
    query: str = Body(..., description="User input: ", examples=["chat"]),
    knowledge_base_name: str = Body(..., description="knowledge base name"),
    top_k: int = Body(3, description="matching vector count"),
    score_threshold: float = Body(
            SCORE_THRESHOLD,
            description="Knowledge base matching relevance threshold, with a range between 0 and 1. A smaller SCORE indicates higher relevance, and setting it to 1 is equivalent to no filtering. It is recommended to set it around 0.5"),
    history: List[dict] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body("", description="model name"),
    imagesdata: List[str] = Body([], description="image data", examples=["image"]),
    speechmodel: dict = Body({}, description="speech model"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default", description=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> StreamingResponse:
    controller_address = controller_address or fschat_controller_address()
    async def fake_json_streamer() -> AsyncIterable[str]:
        import asyncio
        with get_httpx_client() as client:
            response = client.stream(
                "POST",
                url=controller_address + "/knowledge_base_chat",
                json={
                    "query": query,
                    "knowledge_base_name": knowledge_base_name,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "history": history,
                    "stream": stream,
                    "model_name": model_name,
                    "imagesdata": imagesdata,
                    "speechmodel": speechmodel,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "prompt_name": prompt_name,
                    },
            )
            with response as r:
                for chunk in r.iter_text(None):
                    if not chunk:
                        continue
                    yield chunk
                    await asyncio.sleep(0.1)
    return StreamingResponse(fake_json_streamer(), media_type="text/event-stream")
    
def change_llm_model(
    model_name: str = Body(..., description="Change Model", examples=""),
    new_model_name: str = Body(..., description="Switch to new Model", examples=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to switch LLM model from controller: {controller_address}. error: {e}")

def get_webui_configs(
        controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        configinst = InnerJsonConfigWebUIParse()
        return BaseResponse(data = configinst.dump())
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to get webui configration, error: {e}")
    
def save_model_config(
        mtype: int = Body(..., description="Model Type"),
        msize: int = Body(..., description="Model Size"),
        msubtype: int = Body(..., description="Model Sub Type"),
        model_name: str = Body(..., description="Model Name"),
        config: dict = Body(..., description="Model configration information"),
        controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        if mtype == ModelType.Local.value:
            msize = GetSizeName(ModelSize(msize))
            provider = "LLM Model"
        elif mtype == ModelType.Special.value:
            msize = GetSizeName(ModelSize(msize))
            provider = "Special Model"
        elif mtype == ModelType.Multimodal.value:
            msize = GetSubTypeName(ModelSubType(msubtype))
            provider = "Multimodal Model"
        elif mtype != ModelType.Online.value:
            return BaseResponse(
                code=500,
                msg="failed to save model configration, error mtype!")

        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            if mtype == ModelType.Online.value:
                jsondata["ModelConfig"]["OnlineModel"][model_name].update(config)
            else:
                jsondata["ModelConfig"]["LocalModel"][provider][msize][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")
    
async def download_llm_model(
    model_name: str = Body(..., description="Model Name"),
    hugg_path: str = Body(..., description="Huggingface Path"),
    local_path: str = Body(..., description="Local Path"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> StreamingResponse:
    controller_address = controller_address or fschat_controller_address()
    async def fake_json_streamer() -> AsyncIterable[str]:
        import asyncio
        with get_httpx_client() as client:
            response = client.stream(
                "POST",
                url=controller_address + "/download_llm_model",
                json={"model_name": model_name,
                    "hugg_path": hugg_path,
                    "local_path": local_path,
                    },
            )
            with response as r:
                for chunk in r.iter_text(None):
                    if not chunk:
                        continue
                    yield chunk
                    await asyncio.sleep(0.1)
    return StreamingResponse(fake_json_streamer(), media_type="text/event-stream")

# Voice Model

def get_vtot_model_config(
        model_name: str = Body(description="Vtot Model name"),
        placeholder: str = Body(description="Unused")
) -> BaseResponse:
    try:
        config = get_vtot_worker_config(model_name=model_name)
        return BaseResponse(data=config)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")
    
def save_voice_model_config(
    model_name: str = Body(..., description="Save Model Config"),
    config: dict = Body(..., description="Model configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])    
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["VtoTModel"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save local model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")

def get_vtot_model(
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"), 
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/get_vtot_model")
            model = r.json()["model"]
            return BaseResponse(data=model)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get current model, error: {e}")
    
def get_vtot_data(
    voice_data: str = Body(..., description="voice data"),
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/get_vtot_data",
                json={"voice_data": voice_data},
                )
            data = r.json()["text"]
            code = r.json()["code"]
            if code == 200:
                return BaseResponse(data=data)
            else:
                return BaseResponse(
                    code=500,
                    data="",
                    msg="failed to translate voice data, error!")
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data="",
            msg=f"failed to translate voice data, error: {e}")
    
def stop_vtot_model(
    model_name: str = Body(..., description="Stop Model"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_vtot_model",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to stop Voice model {model_name} from controller: {controller_address}. error: {e}")
    
def change_vtot_model(
    model_name: str = Body(..., description="Change Model", examples=""),
    new_model_name: str = Body(..., description="Switch to new Model", examples=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_vtot_model",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to switch Voice model from controller: {controller_address}. error: {e}")

# Speech Model
    
def get_speech_model_config(
        model_name: str = Body(description="Speech Model name"),
        placeholder: str = Body(description="Unused")
) -> BaseResponse:
    try:
        config = get_speech_worker_config(model_name=model_name)
        return BaseResponse(data=config)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")
    
def save_speech_model_config(
    model_name: str = Body(..., description="Save Model Config"),
    config: dict = Body(..., description="Model configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])    
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["TtoVModel"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save local model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")

def get_speech_model(
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"), 
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/get_speech_model")
            model = r.json()["model"]
            speaker = r.json()["speaker"]
            return BaseResponse(data={"model": model, "speaker": speaker})
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get current model, error: {e}")
    
def get_speech_data(
    text_data: str = Body(..., description="speech data"),
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    speech_type: str = Body("", description="synthesis")
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/get_speech_data",
                json={"text_data": text_data, "speech_type": speech_type},
                )
            code = r.json()["code"]
            if code == 200:
                return r.json()
            else:
                return {
                    "code": 500,
                    "speech_data": ""}
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return {
            "code": 500,
            "speech_data": ""}
    
def stop_speech_model(
    model_name: str = Body(..., description="Stop Model"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_speech_model",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to stop Voice model {model_name} from controller: {controller_address}. error: {e}")
    
def change_speech_model(
    model_name: str = Body(..., description="Change Model", examples=""),
    new_model_name: str = Body(..., description="Switch to new Model", examples=""),
    speaker: str = Body(..., description="Speaker", examples=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_speech_model",
                json={"model_name": model_name, "new_model_name": new_model_name, "speaker": speaker},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to switch speech model from controller: {controller_address}. error: {e}")
    
# Image Recognition interface
    
def get_image_recognition_model_config(
        model_name: str = Body(description="Image Recognition Model name"),
        placeholder: str = Body(description="Unused")
) -> BaseResponse:
    try:
        config = get_image_recognition_worker_config(model_name=model_name)
        return BaseResponse(data=config)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")
    
def save_image_recognition_model_config(
    model_name: str = Body(..., description="Save Model Config"),
    config: dict = Body(..., description="Model configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])    
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["ImageRecognition"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save local model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")

def get_image_recognition_model(
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"), 
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/get_image_recognition_model")
            model = r.json()["model"]
            return BaseResponse(data=model)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get current model, error: {e}")
    
def get_image_recognition_data(
    imagedata: str = Body(..., description="image recognition data", examples=["image"]),
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/get_image_recognition_data",
                json={
                    "imagedata": imagedata,
                    "imagetype": "jpeg",
                    },
                )
            data = r.json()["text"]
            code = r.json()["code"]
            if code == 200:
                return BaseResponse(data=data)
            else:
                return BaseResponse(
                    code=500,
                    data="",
                    msg="failed to translate voice data, error!")
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data="",
            msg=f"failed to translate voice data, error: {e}")
    
def eject_image_recognition_model(
    model_name: str = Body(..., description="Stop Model"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_image_recognition_model",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to stop Image Recognition model {model_name} from controller: {controller_address}. error: {e}")
    
def change_image_recognition_model(
    model_name: str = Body(..., description="Change Model", examples=""),
    new_model_name: str = Body(..., description="Switch to new Model", examples=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_image_recognition_model",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to switch image recognition model from controller: {controller_address}. error: {e}")
    
# Image Generation interface

def get_image_generation_model_config(
        model_name: str = Body(description="Image Generation Model name"),
        placeholder: str = Body(description="Unused")
) -> BaseResponse:
    try:
        config = get_image_generation_worker_config(model_name=model_name)
        return BaseResponse(data=config)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")
    
def save_image_generation_model_config(
    model_name: str = Body(..., description="Save Model Config"),
    config: dict = Body(..., description="Model configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])    
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["ImageGeneration"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save local model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")

def get_image_generation_model(
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"), 
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/get_image_generation_model")
            model = r.json()["model"]
            return BaseResponse(data=model)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get current model, error: {e}")
    
def get_image_generation_data(
    prompt_data: str = Body(..., description="prompt data"),
    negative_prompt: str = Body(..., description="negative prompt"),
    btranslate_prompt: bool = Body(False, description=""),
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/get_image_generation_data",
                json={"prompt_data": prompt_data, "negative_prompt": negative_prompt, "btranslate_prompt": btranslate_prompt},
                )
            code = r.json()["code"]
            image = r.json()["image"]
            if code == 200:
                return BaseResponse(data=image)
            else:
                return {
                    "code": 500,
                    "image": ""}
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return {
            "code": 500,
            "image": ""}
    
def eject_image_generation_model(
    model_name: str = Body(..., description="Stop Model"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_image_generation_model",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to stop Image Generation model {model_name} from controller: {controller_address}. error: {e}")
    
def change_image_generation_model(
    model_name: str = Body(..., description="Change Model", examples=""),
    new_model_name: str = Body(..., description="Switch to new Model", examples=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_image_generation_model",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to switch speech model from controller: {controller_address}. error: {e}")
    
# Music Generation interface

def get_music_generation_model_config(
        model_name: str = Body(description="Music Generation Model name"),
        placeholder: str = Body(description="Unused")
) -> BaseResponse:
    try:
        config = get_music_generation_worker_config(model_name=model_name)
        return BaseResponse(data=config)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")
    
def save_music_generation_model_config(
    model_name: str = Body(..., description="Save Model Config"),
    config: dict = Body(..., description="Model configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])    
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["MusicGeneration"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save local model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")

def get_music_generation_model(
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
    placeholder: str = Body(None, description="Not use"), 
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(controller_address + "/get_music_generation_model")
            model = r.json()["model"]
            return BaseResponse(data=model)
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get current model, error: {e}")
    
def get_music_generation_data(
    prompt_data: str = Body(..., description="prompt data"),
    btranslate_prompt: bool = Body(False, description=""),
    controller_address: str = Body(None, description="Fastchat controller adress", examples=[fschat_controller_address()]),
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/get_music_generation_data",
                json={"prompt_data": prompt_data, "btranslate_prompt": btranslate_prompt},
                )
            code = r.json()["code"]
            audio = r.json()["audio"]
            if code == 200:
                return BaseResponse(data=audio)
            else:
                return {
                    "code": 500,
                    "audio": ""}
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return {
            "code": 500,
            "audio": ""}
    
def eject_music_generation_model(
    model_name: str = Body(..., description="Stop Model"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_music_generation_model",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to stop Music Generation model {model_name} from controller: {controller_address}. error: {e}")
    
def change_music_generation_model(
    model_name: str = Body(..., description="Change Model", examples=""),
    new_model_name: str = Body(..., description="Switch to new Model", examples=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    try:
        controller_address = controller_address or fschat_controller_address()
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_music_generation_model",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to switch speech model from controller: {controller_address}. error: {e}")

# chat interface

def save_chat_config(
        config: dict = Body(..., description="Chat configration information"),
        controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ChatConfiguration"].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save chat configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save chat configration, error: {e}")
    
def save_search_engine_config(
    config: dict = Body(..., description="Search Engine configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["SearchEngine"].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save chat configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save chat configration, error: {e}")
    
def llm_search_engine_chat(
    query: str = Body(..., description="User input: ", examples=["chat"]),
    search_engine_name: str = Body(..., description="Search engine name"),
    history: List[dict] = Body([],
                                  description="History chat",
                                  examples=[[
                                      {"role": "user", "content": "Who are you?"},
                                      {"role": "assistant", "content": "I am AI."}]]
                                  ),
    stream: bool = Body(False, description="stream output"),
    model_name: str = Body("", description="model name"),
    temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
    max_tokens: Optional[int] = Body(None, description="max tokens."),
    prompt_name: str = Body("default", description=""),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
):
    controller_address = controller_address or fschat_controller_address()
    async def fake_json_streamer() -> AsyncIterable[str]:
        import asyncio
        with get_httpx_client() as client:
            response = client.stream(
                "POST",
                url=controller_address + "/llm_search_engine_chat",
                json={
                    "query": query,
                    "search_engine_name": search_engine_name,
                    "history": history,
                    "stream": stream,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "prompt_name": prompt_name,
                    },
            )
            with response as r:
                for chunk in r.iter_text(None):
                    if not chunk:
                        continue
                    yield chunk
                    await asyncio.sleep(0.1)
    return StreamingResponse(fake_json_streamer(), media_type="text/event-stream")

def list_search_engines() -> BaseResponse:
    pass
    #from server.chat.search_engine_chat import SEARCH_ENGINES

    #return BaseResponse(data=list(SEARCH_ENGINES))

def save_code_interpreter_config(
    config: dict = Body(..., description="Code Interpreter configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["CodeInterpreter"].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save chat configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save chat configration, error: {e}")
    
def save_function_calling_config(
    function_calling: dict = Body(..., description="Function Calling configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        with open("WebUI/configs/webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["FunctionCalling"]=function_calling
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg="success save chat configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save chat configration, error: {e}")