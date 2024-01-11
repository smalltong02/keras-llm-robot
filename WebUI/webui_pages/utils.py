import httpx
import json
import base64
import contextlib
from pprint import pprint
from typing import *
from WebUI.configs import *
from WebUI.Server.utils import get_httpx_client
from WebUI.configs.serverconfig import API_SERVER
from WebUI.configs import HTTPX_DEFAULT_TIMEOUT
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse

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

    def chat_feedback(
        self,
        chat_history_id: str,
        score: int,
        reason: str = "",
    ) -> int:
        '''
        feedback user reviews
        '''
        data = {
            "chat_history_id": chat_history_id,
            "score": score,
            "reason": reason,
        }
        resp = self.post("/chat/feedback", json=data)
        return self._get_response_value(resp)

    def chat_chat(
        self,
        query: str,
        imagesdata: List[bytes] = [],
        history: List[dict] = [],
        stream: bool = True,
        model: str = "",
        speechmodel: dict = {"model": "", "speaker": ""},
        temperature: float = 0.7,
        max_tokens: int = None,
        prompt_name: str = "default",
        **kwargs,
    ):
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
        modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model)
        config = GetSpeechModelInfo(webui_config, speechmodel.get("model", ""))
        if len(config):
            speechmodel["type"] = config["type"]
            speechmodel["speech_key"] = config.get("speech_key", "")
            if speechmodel["speech_key"] == "[Your Key]":
                speechmodel["speech_key"] = ""
            speechmodel["speech_region"] = config.get("speech_region", "")
            if speechmodel["speech_region"] == "[Your Region]":
                speechmodel["speech_region"] = ""
            speechmodel["provider"] = config.get("provider", "")
        else:
            speechmodel["type"] = ""
            speechmodel["speech_key"] = ""
            speechmodel["speech_region"] = ""
            speechmodel["provider"] = config.get("provider", "")

        dataslist = []
        if len(imagesdata):
            for imagedata in imagesdata:
                dataslist.append(base64.b64encode(imagedata).decode('utf-8'))
        data = {
            "query": query,
            "imagesdata": dataslist,
            "history": history,
            "stream": stream,
            "model_name": model,
            "speechmodel": speechmodel,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        print(f"received input message:")
        pprint(data)

        if modelinfo["mtype"] == ModelType.Local:
            response = self.post("/chat/chat", json=data, stream=True, **kwargs)
            return self._httpx_stream2generator(response, as_json=True)
        
        elif modelinfo["mtype"] == ModelType.Special:
            response = self.post(
               "/llm_model/chat",
               json=data,
               stream=True,
               **kwargs
            )
            return self._httpx_stream2generator(response, as_json=True)
        elif modelinfo["mtype"] == ModelType.Online:
            provider = GetProviderByName(webui_config, model)
            if provider is not None:
                if provider == "google-api":
                    response = self.post(
                        "/llm_model/chat",
                        json=data,
                        stream=True
                    )
                    return self._httpx_stream2generator(response, as_json=True)
                else:
                    response = self.post("/chat/chat", json=data, stream=True, **kwargs)
                    return self._httpx_stream2generator(response, as_json=True)
        return [{
            "chat_history_id": "123",
            "text": "internal error!"
        }]

    def _httpx_stream2generator(
        self,
        response: contextlib._GeneratorContextManager,
        as_json: bool = False,
    ):
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
                                msg = f"json failed: '{chunk}'. error: {e}."
                                print(f'{e.__class__.__name__}: {msg}')
                        else:
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"Can't connect to API Server, Please confirm 'api.py' starting。({e})"
                print(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API communication timeout. Please ensure that FastChat and API service have been started (refer to Wiki '5. Start API Service or Web UI'). ({e})"
                print(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API communication error: {e}"
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
                                msg = f"json failed: '{chunk}'. error: {e}."
                                print(f'{e.__class__.__name__}: {msg}')
                        else:
                            yield chunk
            except httpx.ConnectError as e:
                msg = f"Can't connect to API Server, Please confirm 'api.py' starting。({e})"
                print(msg)
                yield {"code": 500, "msg": msg}
            except httpx.ReadTimeout as e:
                msg = f"API communication timeout. Please ensure that FastChat and API service have been started (refer to Wiki '5. Start API Service or Web UI'). ({e})"
                print(msg)
                yield {"code": 500, "msg": msg}
            except Exception as e:
                msg = f"API communication error: {e}"
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
    
    def ret_sync(self, response):
        return self._get_response_value(response, as_json=True)

    async def ret_async(self, response):
        return self._get_response_value(response, as_json=True)
    
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
        running_models = self.get_running_models()
        if new_model_name == model_name or new_model_name in running_models:
            return {
                "code": 200,
                "msg": "Not necessary to switch models."
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

        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def eject_llm_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": f"name for the new model is None."
            }
        
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
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def save_model_config(self,
        modelconfig: Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "mspecial": "", "config": dict},
        controller_address: str = None,
    ):
        if modelconfig is None or all(key in modelconfig for key in ["mtype", "msize", "msubtype", "mname", "config"]) is False:
            return {
                "code": 500,
                "msg": f"modelconfig is None."
            }
        
        data = {
            "mtype": modelconfig["mtype"].value,
            "msize": modelconfig["msize"].value,
            "msubtype": modelconfig["msubtype"].value,
            "model_name": modelconfig["mname"],
            "config": modelconfig["config"],
            "controller_address": controller_address,
        }

        response = self.post(
            "/llm_model/save_model_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def download_llm_model(self,
        model_name: str = "",
        hugg_path: str = "",
        local_path: str = "",
        controller_address: str = None,              
    ):
        if model_name == "" or hugg_path == "" or local_path == "":
            return {
                "code": 500,
                "msg": f"Parameter is incorrect."
            }
        
        data = {
            "model_name": model_name,
            "hugg_path": hugg_path,
            "local_path": local_path,
            "controller_address": controller_address,
        }

        response = self.post(
            "/llm_model/download_llm_model",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)
        
    def save_vtot_model_config(self,
        model_name: str = "",
        modelconfig: dict = {},
        controller_address: str = None,
    ):
        if model_name == "" or modelconfig is None:
            return {
                "code": 500,
                "msg": f"modelconfig is None."
            }
        data = {
            "model_name": model_name,
            "config": modelconfig,
            "controller_address": controller_address,
        }

        response = self.post(
            "/voice_model/save_voice_model_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_vtot_model(self, controller_address: str = None):
        data = {
            "controller_address": controller_address,
        }
        response = self.post(
            "/voice_model/get_vtot_model",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", []))
    
    def eject_voice_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": f"name for the new model is None."
            }
        
        running_model = self.get_vtot_model()
        if model_name != running_model:
            return {
                "code": 200,
                "msg": f"the model '{model_name}' is not running."
            }

        data = {
            "model_name": model_name,
            "controller_address": controller_address,
        }

        response = self.post(
            "/voice_model/stop",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    def change_voice_model(self,
        model_name: str,
        new_model_name: str,
        controller_address: str = None,
    ):
        if not new_model_name:
            return {
                "code": 500,
                "msg": f"name for the new model is None."
            }
        running_model = self.get_vtot_model()
        if new_model_name == model_name or new_model_name == running_model:
            return {
                "code": 200,
                "msg": "Not necessary to switch models."
            }

        data = {
            "model_name": model_name,
            "new_model_name": new_model_name,
            "controller_address": controller_address,
        }

        response = self.post(
            "/voice_model/change",
            json=data,
        )

        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_vtot_data(self,
        voice_data: bytes,
        controller_address: str = None
    ):
        if voice_data is None or len(voice_data) == 0:
            return ""
        base64_data = base64.b64encode(voice_data).decode('utf-8')
        data = {
            "voice_data": base64_data,
            "controller_address": controller_address,
        }
        response = self.post(
            "/voice_model/get_vtot_data",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", ""))
    
    def get_ttov_model(self, controller_address: str = None):
        data = {
            "controller_address": controller_address,
        }
        response = self.post(
            "/speech_model/get_ttov_model",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", {"model": "", "speaker": ""}))
    
    def eject_speech_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": f"name for the new model is None."
            }
        
        running_model = self.get_ttov_model()
        if model_name != running_model.get("model", ""):
            return {
                "code": 200,
                "msg": f"the model '{model_name}' is not running."
            }

        data = {
            "model_name": model_name,
            "controller_address": controller_address,
        }

        response = self.post(
            "/speech_model/stop",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    def change_speech_model(self,
        model_name: str,
        new_model_name: str,
        speaker: str,
        controller_address: str = None,
    ):
        if not new_model_name or not speaker:
            return {
                "code": 500,
                "msg": f"name or speaker for the new model is None."
            }
        running_model = self.get_ttov_model()
        if new_model_name == model_name or new_model_name == running_model.get("model", ""):
            return {
                "code": 200,
                "msg": "Not necessary to switch models."
            }

        data = {
            "model_name": model_name,
            "new_model_name": new_model_name,
            "speaker": speaker,
            "controller_address": controller_address,
        }

        response = self.post(
            "/speech_model/change",
            json=data,
        )

        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_ttov_data(self,
        text_data: str = None,
        controller_address: str = None
    ):
        if text_data is None or len(text_data) == 0:
            return ""
        data = {
            "text_data": text_data,
            "controller_address": controller_address,
        }
        response = self.post(
            "/speech_model/get_ttov_data",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", ""))
    
    def save_speech_model_config(self,
        model_name: str = "",
        modelconfig: dict = {},
        controller_address: str = None,
    ):
        if model_name == "" or modelconfig is None:
            return {
                "code": 500,
                "msg": f"modelconfig is None."
            }
        data = {
            "model_name": model_name,
            "config": modelconfig,
            "controller_address": controller_address,
        }

        response = self.post(
            "/speech_model/save_speech_model_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    def save_chat_config(self,
        chatconfig: dict,
        controller_address: str = None,
    ):
        if chatconfig is None:
            return {
                "code": 500,
                "msg": f"chatconfig is None."
            }
        
        data = {
            "config": chatconfig,
            "controller_address": controller_address,
        }

        response = self.post(
            "/llm_model/save_chat_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    
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

