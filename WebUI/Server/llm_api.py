from fastapi import Body
from configs import LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from WebUI.Server.utils import (BaseResponse, fschat_controller_address, list_config_llm_models,
                          get_httpx_client, get_model_worker_config)
from copy import deepcopy

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
            if k == "model_list":
                online_list += v
                break
    configs['online'] = online_list

    local_list = []
    for name, path in list_models["local"].items():
        local_list += [name]
    configs['local'] = local_list
    return BaseResponse(data=configs)


def get_model_config(
    model_name: str = Body(description="配置中LLM模型的名称"),
    placeholder: str = Body(description="占位用，无实际效果")
) -> BaseResponse:
    '''
    获取LLM模型配置项（合并后的）
    '''
    config = {}
    # 删除ONLINE_MODEL配置中的敏感信息
    for k, v in get_model_worker_config(model_name=model_name).items():
        if not (k == "worker_class"
            or "key" in k.lower()
            or "secret" in k.lower()
            or k.lower().endswith("id")):
            config[k] = v

    return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller address", examples=[fschat_controller_address()])
) -> BaseResponse:
    '''
    向fastchat controller请求停止某个LLM模型。
    注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
    '''
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
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


def change_llm_model(
    model_name: str = Body(..., description="Current Model", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="Switch to new Model", examples=[LLM_MODELS[0]]),
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
            msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")


def list_search_engines() -> BaseResponse:
    pass
    #from server.chat.search_engine_chat import SEARCH_ENGINES

    #return BaseResponse(data=list(SEARCH_ENGINES))