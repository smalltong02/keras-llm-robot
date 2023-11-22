import streamlit as st
import httpx
import json
import contextlib
from pprint import pprint
from typing import *
from WebUI.Server.utils import get_httpx_client
from WebUI.configs.serverconfig import API_SERVER
from WebUI.configs import HTTPX_DEFAULT_TIMEOUT

def api_address() -> str:
    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"

class ApiRequest:
    def __init__(self, base_url, timeout: float):
        self.base_url = base_url
        self.timeout = timeout
        self._use_async = False
        self._client = None

    @property
    def client(self):
        if self._client is None or self._client.is_closed:
            self._client = get_httpx_client(base_url=self.base_url,
                                            use_async=self._use_async,
                                            timeout=self.timeout)
        return self._client
    
    def get(
        self,
        url: str,
        params: Union[Dict, List[Tuple], bytes] = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any,
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("GET", url, params=params, **kwargs)
                else:
                    return self.client.get(url, params=params, **kwargs)
            except Exception as e:
                msg = f"error when get {url}: {e}"
                print(f'{e.__class__.__name__}: {msg}')
                retry -= 1

    def post(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    return self.client.post(url, data=data, json=json, **kwargs)
            except Exception as e:
                print(self._client)
                msg = f"error when post {url}: {e}"
                print(f'{e.__class__.__name__}: {msg}')
                retry -= 1

    def delete(
        self,
        url: str,
        data: Dict = None,
        json: Dict = None,
        retry: int = 3,
        stream: bool = False,
        **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    return self.client.stream("DELETE", url, data=data, json=json, **kwargs)
                else:
                    return self.client.delete(url, data=data, json=json, **kwargs)
            except Exception as e:
                msg = f"error when delete {url}: {e}"
                print(f'{e.__class__.__name__}: {msg}')
                retry -= 1

    def chat_chat(
        self,
        query: str,
        history: List[Dict] = [],
        stream: bool = True,
        model: str = "",
        temperature: float = 0.7,
        max_tokens: int = None,
        prompt_name: str = "default",
        **kwargs,
    ):
        '''
        对应api.py/chat/chat接口 #TODO: 考虑是否返回json
        '''
        data = {
            "query": query,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        print(f"received input message:")
        pprint(data)

        response = self.post("/chat/chat", json=data, stream=True, **kwargs)
        return self._httpx_stream2generator(response, as_json=True)

    def _httpx_stream2generator(
        self,
        response: contextlib._GeneratorContextManager,
        as_json: bool = False,
    ):
        '''
        将httpx.stream返回的GeneratorContextManager转化为普通生成器
        '''
        async def ret_async(response, as_json):
            try:
                async with response as r:
                    async for chunk in r.aiter_text(None):
                        if not chunk: # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                data = json.loads(chunk)
                                pprint(data, depth=1)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                print(f'{e.__class__.__name__}: {msg}')
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                print(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                print(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                print(f'{e.__class__.__name__}: {msg}')
                yield {"code": 500, "msg": msg}

        def ret_sync(response, as_json):
            try:
                with response as r:
                    for chunk in r.iter_text(None):
                        if not chunk: # fastchat api yield empty bytes on start and end
                            continue
                        if as_json:
                            try:
                                data = json.loads(chunk)
                                pprint(data, depth=1)
                                yield data
                            except Exception as e:
                                msg = f"接口返回json错误： ‘{chunk}’。错误信息是：{e}。"
                                print(f'{e.__class__.__name__}: {msg}')
                        else:
                            # print(chunk, end="", flush=True)
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"无法连接API服务器，请确认 ‘api.py’ 已正常启动。({e})"
                print(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API通信超时，请确认已启动FastChat与API服务（详见Wiki '5. 启动 API 服务或 Web UI'）。（{e}）"
                print(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API通信遇到错误：{e}"
                print(f'{e.__class__.__name__}: {msg}')
                yield {"code": 500, "msg": msg}

        if self._use_async:
            return ret_async(response, as_json)
        else:
            return ret_sync(response, as_json)
        
    def get_server_configs(self, **kwargs) -> Dict:
        response = self.post("/server/configs", **kwargs)
        return self._get_response_value(response, as_json=True)

    def list_search_engines(self, **kwargs) -> List:
        response = self.post("/server/list_search_engines", **kwargs)
        return self._get_response_value(response, as_json=True, value_func=lambda r: r["data"])
    
    def get_prompt_template(
        self,
        type: str = "llm_chat",
        name: str = "default",
        **kwargs,
    ) -> str:
        data = {
            "type": type,
            "name": name,
        }
        response = self.post("/server/get_prompt_template", json=data, **kwargs)
        return self._get_response_value(response, value_func=lambda r: r.text)
    

    def get_running_models(self, controller_address: str = None):
        data = {
            "controller_address": controller_address,
        }
        response = self.post(
            "/llm_model/get_running_models",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", []))
    
    def list_running_models(self, controller_address: str = None,):
        data = {
            "controller_address": controller_address,
        }

        response = self.post(
            "/llm_model/list_running_models",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", []))
    
    def list_config_models(self) -> Dict[str, List[str]]:
        '''
        {"type": [model_name1, model_name2, ...], ...}。
        '''
        response = self.post(
            "/llm_model/list_config_models",
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", {}))
    
    def get_model_config(
            self,
            model_name: str = None,
        ) -> Dict:

        data={
            "model_name": model_name,
        }
        response = self.post(
            "/llm_model/get_model_config",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", {}))
    
    def get_webui_config(
            self,
        ) -> Dict:

        response = self.post(
            "/server/get_webui_config",
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", {}))
    
    def change_llm_model(
        self,
        model_name: str,
        new_model_name: str,
        controller_address: str = None,
    ):
        if not new_model_name:
            return {
                "code": 500,
                "msg": f"name for the new model is None."
            }

        def ret_sync():
            running_models = self.get_running_models()
            if new_model_name == model_name or new_model_name in running_models:
                return {
                    "code": 200,
                    "msg": "Not necessary to switch models."
                }

            config_models = self.list_config_models()
            if new_model_name not in config_models["local"]:
                return {
                    "code": 500,
                    "msg": f"The new Model '{new_model_name}' is not configured in the configs."
                }

            data = {
                "model_name": model_name,
                "new_model_name": new_model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/change",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        async def ret_async():
            running_models = await self.get_running_models()
            if new_model_name == model_name or new_model_name in running_models:
                return {
                    "code": 200,
                    "msg": "Not necessary to switch models."
                }

            config_models = await self.list_config_models()
            if new_model_name not in config_models["local"]:
                return {
                    "code": 500,
                    "msg": f"The new Model '{new_model_name}' is not configured in the configs."
                }

            data = {
                "model_name": model_name,
                "new_model_name": new_model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/change",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        if self._use_async:
            return ret_async()
        else:
            return ret_sync()
        
    def eject_llm_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": f"name for the new model is None."
            }
        
        def ret_sync():
            running_models = self.get_running_models()
            if model_name not in running_models:
                return {
                    "code": 200,
                    "msg": f"the model '{model_name}' is not running."
                }

            data = {
                "model_name": model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/stop",
                json=data,
            )
            return self._get_response_value(response, as_json=True)

        async def ret_async():
            running_models = self.get_running_models()
            if model_name not in running_models:
                return {
                    "code": 200,
                    "msg": f"the model '{model_name}' is not running."
                }

            data = {
                "model_name": model_name,
                "controller_address": controller_address,
            }

            response = self.post(
                "/llm_model/stop",
                json=data,
            )
            return self._get_response_value(response, as_json=True)
        
        if self._use_async:
            return ret_async()
        else:
            return ret_sync()
    
    def _get_response_value(self, response: httpx.Response, as_json: bool = False, value_func: Callable = None,):
        
        def to_json(r):
            try:
                return r.json()
            except Exception as e:
                msg = "error jason format: " + str(e)
                print(f'{e.__class__.__name__}: {msg}')
                return {"code": 500, "msg": msg, "data": None}
        
        if value_func is None:
            value_func = (lambda r: r)

        async def ret_async(response):
            if as_json:
                return value_func(to_json(await response))
            else:
                return value_func(await response)

        if self._use_async:
            return ret_async(response)
        else:
            if as_json:
                return value_func(to_json(response))
            else:
                return value_func(response)
            
    
class AsyncApiRequest(ApiRequest):
    def __init__(self, base_url: str = api_address(), timeout: float = HTTPX_DEFAULT_TIMEOUT):
        super().__init__(base_url, timeout)
        self._use_async = True


def check_error_msg(data: Union[str, dict, list], key: str = "errorMsg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if isinstance(data, dict):
        if key in data:
            return data[key]
        if "code" in data and data["code"] != 200:
            return data["msg"]
    return ""


def check_success_msg(data: Union[str, dict, list], key: str = "msg") -> str:
    '''
    return error message if error occured when requests API
    '''
    if (isinstance(data, dict)
        and key in data
        and "code" in data
        and data["code"] == 200):
        return data[key]
    return ""

if __name__ == "__main__":
    api = ApiRequest()
    aapi = AsyncApiRequest()

