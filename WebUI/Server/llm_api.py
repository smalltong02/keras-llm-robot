from fastapi import Body
from configs import LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from WebUI.Server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config, get_vtot_worker_config)
from copy import deepcopy
import json
from WebUI.configs.webuiconfig import *

def get_running_models(
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

def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()]),
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
    model_name: str = Body(..., description="Stop Model", examples=[LLM_MODELS[0]]),
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
        model_name: str = Body(..., description="Change Model"),
        config: dict = Body(..., description="Model configration information"),
        controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        with open(".\WebUI\configs\webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["LocalModel"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg=f"success save local model configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save local model configration, error: {e}")

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
    model_name: str = Body(..., description="Change Model"),
    config: dict = Body(..., description="Model configration information"),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])    
) -> BaseResponse:
    try:
        with open(".\WebUI\configs\webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ModelConfig"]["VtoTModel"][model_name].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg=f"success save local model configration!")
            
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
                    msg=f"failed to translate voice data, error: {e}")
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

def save_chat_config(
        config: dict = Body(..., description="Chat configration information"),
        controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    try:
        with open(".\WebUI\configs\webuiconfig.json", 'r+') as file:
            jsondata = json.load(file)
            jsondata["ChatConfiguration"].update(config)
            file.seek(0)
            json.dump(jsondata, file, indent=4)
            file.truncate()
        return BaseResponse(
            code=200,
            msg=f"success save chat configration!")
            
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
        return BaseResponse(
            code=500,
            msg=f"failed to save chat configration, error: {e}")


def list_search_engines() -> BaseResponse:
    pass
    #from server.chat.search_engine_chat import SEARCH_ENGINES

    #return BaseResponse(data=list(SEARCH_ENGINES))