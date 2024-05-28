import os
import httpx
import json
import base64
import contextlib
from pathlib import Path
from pprint import pprint
from typing import List, Dict, Any, Union, Tuple, Iterator, Callable
from WebUI.configs.basicconfig import (ModelType, ModelSize, ModelSubType, GetModelInfoByName, GetProviderByName, GetSpeechModelInfo)
from WebUI.Server.utils import get_httpx_client
from WebUI.configs.serverconfig import API_SERVER
from WebUI.configs import HTTPX_DEFAULT_TIMEOUT
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.Server.knowledge_base.utils import (CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE, SCORE_THRESHOLD)

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
        timeout: Union[None, int] = None,
        **kwargs: Any
    ) -> Union[httpx.Response, Iterator[httpx.Response], None]:
        while retry > 0:
            try:
                if stream:
                    if timeout == 0:
                        return self.client.stream("POST", url, data=data, json=json, timeout=None, **kwargs)
                    elif timeout is None:
                        return self.client.stream("POST", url, data=data, json=json, **kwargs)
                else:
                    if timeout == 0:
                        return self.client.post(url, data=data, json=json, timeout=None, **kwargs)
                    elif timeout is None:
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
        audiosdata: List[bytes] = [],
        videosdata: List[bytes] = [],
        imagesprompt: List[str] = [],
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

        imageslist = []
        audioslist = []
        videoslist = []
        if len(imagesdata):
            for imagedata in imagesdata:
                imageslist.append(base64.b64encode(imagedata).decode('utf-8'))
        if len(audiosdata):
            for audiodata in audiosdata:
                audioslist.append(base64.b64encode(audiodata).decode('utf-8'))
        if len(videosdata):
            for videodata in videosdata:
                videoslist.append(base64.b64encode(videodata).decode('utf-8'))
        data = {
            "query": query,
            "imagesdata": imageslist,
            "audiosdata": audioslist,
            "videosdata": videoslist,
            "imagesprompt": imagesprompt,
            "history": history,
            "stream": stream,
            "model_name": model,
            "speechmodel": speechmodel,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        print("received input message:")
        pprint(data)

        if modelinfo["mtype"] == ModelType.Local:
            response = self.post("/chat/chat", json=data, stream=True, **kwargs)
            return self._httpx_stream2generator(response, as_json=True)
        
        elif modelinfo["mtype"] == ModelType.Special or modelinfo["mtype"] == ModelType.Code or modelinfo["mtype"] == ModelType.Multimodal:
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
                if provider == "openai-api" or provider == "kimi-cloud-api" or provider == "yi-01ai-api":
                    response = self.post("/chat/chat", json=data, stream=True, **kwargs)
                    return self._httpx_stream2generator(response, as_json=True)
                else:
                    response = self.post(
                        "/llm_model/chat",
                        json=data,
                        stream=True
                    )
                    return self._httpx_stream2generator(response, as_json=True)
        return [{
            "chat_history_id": "123",
            "text": "internal error!"
        }]
    
    def knowledge_base_chat(
        self,
        query: str,
        knowledge_base_name: str,
        top_k: int,
        score_threshold: float,
        history: List[Dict] = [],
        stream: bool = True,
        model: str = "",
        imagesdata: List[bytes] = [],
        speechmodel: dict = {"model": "", "speaker": ""},
        temperature: float = 0.7,
        max_tokens: int = None,
        prompt_name: str = "default",
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
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "history": history,
            "stream": stream,
            "model_name": model,
            "imagesdata": dataslist,
            "speechmodel": speechmodel,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        print("received input message:")
        pprint(data)

        if modelinfo["mtype"] == ModelType.Local:
            response = self.post(
                "/chat/knowledge_base_chat",
                json=data,
                stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)
        elif modelinfo["mtype"] == ModelType.Special:
            return [{
                "code": 500,
                "msg": "Unsupport RAG feature for Special Model!"
            }]
            # response = self.post(
            #    "/llm_model/knowledge_base_chat",
            #    json=data,
            #    stream=True,
            # )
            # return self._httpx_stream2generator(response, as_json=True)
        elif modelinfo["mtype"] == ModelType.Online:
            provider = GetProviderByName(webui_config, model)
            if provider is not None:
                if provider == "google-api":
                    return [{
                        "code": 500,
                        "msg": f"Unsupport RAG feature for {provider}",
                    }]
                    # response = self.post(
                    #     "/llm_model/knowledge_base_chat",
                    #     json=data,
                    #     stream=True
                    # )
                    # return self._httpx_stream2generator(response, as_json=True)
                else:
                    response = self.post("/chat/knowledge_base_chat", json=data, stream=True,)
                    return self._httpx_stream2generator(response, as_json=True)
        elif modelinfo["mtype"] == ModelType.Multimodal:
            return [{
                "code": 500,
                "msg": "Unsupport RAG feature for Multimodal Model!"
            }]
        return [{
                "code": 500,
                "msg": "internal error!",
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

    # llm model api    

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
    
    def get_aigenerator_config(
            self,
        ) -> Dict:

        response = self.post(
            "/server/get_aigenerator_config",
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
                "msg": "name for the new model is None."
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
                "msg": "name for the new model is None."
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
                "msg": "modelconfig is None."
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
                "msg": "Parameter is incorrect."
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
            retry=1,
            timeout=0,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)
    
    # voice & speech model api
        
    def save_vtot_model_config(self,
        model_name: str = "",
        modelconfig: dict = {},
        controller_address: str = None,
    ):
        if model_name == "" or modelconfig is None:
            return {
                "code": 500,
                "msg": "modelconfig is None."
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
                "msg": "name for the new model is None."
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
                "msg": "name for the new model is None."
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
                "msg": "name for the new model is None."
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
                "msg": "name or speaker for the new model is None."
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
                "msg": "modelconfig is None."
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
        
    # image recognition & generation model api
        
    def save_image_recognition_model_config(self,
        model_name: str = "",
        modelconfig: dict = {},
        controller_address: str = None,
    ):
        if model_name == "" or modelconfig is None:
            return {
                "code": 500,
                "msg": "modelconfig is None."
            }
        data = {
            "model_name": model_name,
            "config": modelconfig,
            "controller_address": controller_address,
        }

        response = self.post(
            "/image_model/save_image_recognition_model_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_image_recognition_model(self, controller_address: str = None):
        data = {
            "controller_address": controller_address,
        }
        response = self.post(
            "/image_model/get_image_recognition_model",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", []))
    
    def eject_image_recognition_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": "name for the new model is None."
            }
        
        running_model = self.get_image_recognition_model()
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
            "/image_model/eject_image_recognition_model",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    def change_image_recognition_model(self,
        model_name: str,
        new_model_name: str,
        controller_address: str = None,
    ):
        if not new_model_name:
            return {
                "code": 500,
                "msg": "name for the new model is None."
            }
        running_model = self.get_image_recognition_model()
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
            "/image_model/change_image_recognition_model",
            json=data,
        )

        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_image_recognition_data(self,
        imagedata: bytes,
        controller_address: str = None
    ):
        if imagedata is None or len(imagedata) == 0:
            return ""
        imagedata = base64.b64encode(imagedata).decode('utf-8')
        data = {
            "imagedata": imagedata,
            "controller_address": controller_address,
        }
        response = self.post(
            "/image_model/get_image_recognition_data",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", ""))
    
    def save_image_generation_model_config(self,
        model_name: str = "",
        modelconfig: dict = {},
        controller_address: str = None,
    ):
        if model_name == "" or modelconfig is None:
            return {
                "code": 500,
                "msg": "modelconfig is None."
            }
        data = {
            "model_name": model_name,
            "config": modelconfig,
            "controller_address": controller_address,
        }

        response = self.post(
            "/image_model/save_image_generation_model_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_image_generation_model(self, controller_address: str = None):
        data = {
            "controller_address": controller_address,
        }
        response = self.post(
            "/image_model/get_image_generation_model",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", []))
    
    def eject_image_generation_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": "name for the new model is None."
            }
        
        running_model = self.get_image_generation_model()
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
            "/image_model/eject_image_generation_model",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    def change_image_generation_model(self,
        model_name: str,
        new_model_name: str,
        controller_address: str = None,
    ):
        if not new_model_name:
            return {
                "code": 500,
                "msg": "name for the new model is None."
            }
        running_model = self.get_image_generation_model()
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
            "/image_model/change_image_generation_model",
            json=data,
        )

        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_image_generation_data(self,
        prompt_data: str,
        negative_prompt: str,
        btranslate_prompt: bool,
        controller_address: str = None
    ):
        if prompt_data is None or len(prompt_data) == 0:
            return ""
        data = {
            "prompt_data": prompt_data,
            "negative_prompt": negative_prompt,
            "btranslate_prompt": btranslate_prompt,
            "controller_address": controller_address,
        }
        response = self.post(
            "/image_model/get_image_generation_data",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", ""))
    
    # music generation model api

    def save_music_generation_model_config(self,
        model_name: str = "",
        modelconfig: dict = {},
        controller_address: str = None,
    ):
        if model_name == "" or modelconfig is None:
            return {
                "code": 500,
                "msg": "modelconfig is None."
            }
        data = {
            "model_name": model_name,
            "config": modelconfig,
            "controller_address": controller_address,
        }

        response = self.post(
            "/music_model/save_music_generation_model_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_music_generation_model(self, controller_address: str = None):
        data = {
            "controller_address": controller_address,
        }
        response = self.post(
            "/music_model/get_music_generation_model",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", []))
    
    def eject_music_generation_model(self,
        model_name: str,
        controller_address: str = None,
    ):
        if not model_name:
            return {
                "code": 500,
                "msg": "name for the new model is None."
            }
        
        running_model = self.get_music_generation_model()
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
            "/music_model/eject_music_generation_model",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)

    def change_music_generation_model(self,
        model_name: str,
        new_model_name: str,
        controller_address: str = None,
    ):
        if not new_model_name:
            return {
                "code": 500,
                "msg": "name for the new model is None."
            }
        running_model = self.get_music_generation_model()
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
            "/music_model/change_music_generation_model",
            json=data,
        )

        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def get_music_generation_data(self,
        prompt_data: str,
        btranslate_prompt: bool,
        controller_address: str = None
    ):
        if prompt_data is None or len(prompt_data) == 0:
            return ""
        data = {
            "prompt_data": prompt_data,
            "btranslate_prompt": btranslate_prompt,
            "controller_address": controller_address,
        }
        response = self.post(
            "/music_model/get_music_generation_data",
            json=data,
        )
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", ""))

    # chat & knowledge base api

    def save_chat_config(self,
        chatconfig: dict,
        controller_address: str = None,
    ):
        if chatconfig is None:
            return {
                "code": 500,
                "msg": "chatconfig is None."
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
        
    # Knowledge base api
    def list_knowledge_bases(
        self,
    ):
        response = self.get("/knowledge_base/list_knowledge_bases")
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    def create_knowledge_base(
        self,
        knowledge_base_name: str = "",
        knowledge_base_info: str = "",
        vector_store_type: str = "",
        embed_model: str = "",
    ):
        data = {
            "knowledge_base_name": knowledge_base_name,
            "knowledge_base_info": knowledge_base_info,
            "vector_store_type": vector_store_type,
            "embed_model": embed_model,
        }

        response = self.post(
            "/knowledge_base/create_knowledge_base",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def delete_knowledge_base(
        self,
        knowledge_base_name: str,
    ):
        response = self.post(
            "/knowledge_base/delete_knowledge_base",
            json=f"{knowledge_base_name}",
        )
        return self._get_response_value(response, as_json=True)
    
    def list_kb_docs(
        self,
        knowledge_base_name: str,
    ):
        response = self.get(
            "/knowledge_base/list_files",
            params={"knowledge_base_name": knowledge_base_name}
        )
        return self._get_response_value(response,
                                        as_json=True,
                                        value_func=lambda r: r.get("data", []))

    def search_kb_docs(
        self,
        knowledge_base_name: str,
        query: str = "",
        top_k: int = 3,
        score_threshold: float = SCORE_THRESHOLD,
        file_name: str = "",
        metadata: dict = {},
    ) -> List:
        data = {
            "query": query,
            "knowledge_base_name": knowledge_base_name,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "file_name": file_name,
            "metadata": metadata,
        }
        response = self.post(
            "/knowledge_base/search_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_docs_by_id(
        self,
        knowledge_base_name: str,
        docs: Dict[str, Dict],
    ) -> bool:
        data = {
            "knowledge_base_name": knowledge_base_name,
            "docs": docs,
        }
        response = self.post(
            "/knowledge_base/update_docs_by_id",
            json=data
        )
        return self._get_response_value(response)
    
    def upload_kb_docs(
        self,
        files: List[Union[str, Path, bytes]],
        knowledge_base_name: str,
        override: bool = False,
        to_vector_store: bool = True,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        zh_title_enhance=ZH_TITLE_ENHANCE,
        docs: Dict = {},
        not_refresh_vs_cache: bool = False,
    ):
        def convert_file(file, filename=None):
            from io import BytesIO
            if isinstance(file, bytes): # raw bytes
                file = BytesIO(file)
            elif hasattr(file, "read"): # a file io like object
                filename = filename or file.name
            else: # a local path
                file = Path(file).absolute().open("rb")
                filename = filename or os.path.split(file.name)[-1]
            return filename, file

        files = [convert_file(file) for file in files]
        data={
            "knowledge_base_name": knowledge_base_name,
            "override": override,
            "to_vector_store": to_vector_store,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)
        response = self.post(
            "/knowledge_base/upload_docs",
            data=data,
            retry=1,
            timeout=0,
            files=[("files", (filename, file)) for filename, file in files],
        )
        return self._get_response_value(response, as_json=True)

    def delete_kb_docs(
        self,
        knowledge_base_name: str,
        file_names: List[str],
        delete_content: bool = False,
        not_refresh_vs_cache: bool = False,
    ):
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "delete_content": delete_content,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        response = self.post(
            "/knowledge_base/delete_docs",
            json=data,
        )
        return self._get_response_value(response, as_json=True)
    
    def update_kb_info(self,knowledge_base_name,kb_info):
        data = {
            "knowledge_base_name": knowledge_base_name,
            "kb_info": kb_info,
        }

        response = self.post(
            "/knowledge_base/update_info",
            json=data,
        )
        return self._get_response_value(response, as_json=True)

    def update_kb_docs(
        self,
        knowledge_base_name: str,
        file_names: List[str],
        override_custom_docs: bool = False,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        zh_title_enhance=ZH_TITLE_ENHANCE,
        docs: Dict = {},
        not_refresh_vs_cache: bool = False,
    ):
        data = {
            "knowledge_base_name": knowledge_base_name,
            "file_names": file_names,
            "override_custom_docs": override_custom_docs,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "zh_title_enhance": zh_title_enhance,
            "docs": docs,
            "not_refresh_vs_cache": not_refresh_vs_cache,
        }

        if isinstance(data["docs"], dict):
            data["docs"] = json.dumps(data["docs"], ensure_ascii=False)

        response = self.post(
            "/knowledge_base/update_docs",
            retry=1,
            timeout=0,
            json=data,
        )
        return self._get_response_value(response, as_json=True)
    
    def save_search_engine_config(self,
        searchengine: dict = {},
        controller_address: str = None,
    ):
        data = {
            "config": searchengine,
            "controller_address": controller_address,
        }

        response = self.post(
            "/search_engine/save_search_engine_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def search_engine_chat(
            self,
            query: str,
            search_engine_name: str,
            history: List[Dict] = [],
            stream: bool = True,
            model: str = "",
            temperature: float = 0.7,
            max_tokens: int = None,
            prompt_name: str = "default",
    ):
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
        modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model)

        data = {
            "query": query,
            "search_engine_name": search_engine_name,
            "history": history,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_name": prompt_name,
        }

        print("received input message:")
        pprint(data)

        if modelinfo["mtype"] == ModelType.Local:
            response = self.post(
                "/search_engine/search_engine_chat",
                json=data,
                stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)
        
        elif modelinfo["mtype"] == ModelType.Special or modelinfo["mtype"] == ModelType.Code or modelinfo["mtype"] == ModelType.Multimodal:
            response = self.post(
               "/llm_model/search_engine_chat",
               json=data,
               stream=True,
            )
            return self._httpx_stream2generator(response, as_json=True)
        elif modelinfo["mtype"] == ModelType.Online:
            provider = GetProviderByName(webui_config, model)
            if provider is not None:
                if provider == "google-api":
                    response = self.post(
                        "/llm_model/search_engine_chat",
                        json=data,
                        stream=True
                    )
                    return self._httpx_stream2generator(response, as_json=True)
                else:
                    response = self.post("/search_engine/search_engine_chat", json=data, stream=True,)
                    return self._httpx_stream2generator(response, as_json=True)

        return self._httpx_stream2generator(response, as_json=True)

    def save_code_interpreter_config(self,
        codeinterpreter: dict = {},
        controller_address: str = None,
    ):
        data = {
            "config": codeinterpreter,
            "controller_address": controller_address,
        }

        response = self.post(
            "/code_interpreter/save_code_interpreter_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def code_interpreter_chat(
            self,
            query: str,
            interpreter_id: str = "",
            stream: bool = True,
            model: str = "",
            temperature: float = 0.7,
    ):
        data = {
            "query": query,
            "interpreter_id": interpreter_id,
            "stream": stream,
            "model_name": model,
            "temperature": temperature,
        }

        print("received input message:")
        pprint(data)

        response = self.post(
            "/code_interpreter/code_interpreter_chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)
    
    def save_function_calling_config(self,
        function_calling: dict = {},
        controller_address: str = None,
    ):
        data = {
            "function_calling": function_calling,
            "controller_address": controller_address,
        }

        response = self.post(
            "/function_calling/save_function_calling_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def save_google_toolboxes_config(self,
        google_toolboxes: dict = {},
        controller_address: str = None,
    ):
        data = {
            "google_toolboxes": google_toolboxes,
            "controller_address": controller_address,
        }

        response = self.post(
            "/google_toolboxes/save_google_toolboxes_config",
            json=data,
        )
        
        if self._use_async:
            return self.ret_async(response)
        else:
            return self.ret_sync(response)
        
    def is_calling_enable(self,
        controller_address: str = None,
    ):
        data = {
            "controller_address": controller_address,
        }

        response = self.post(
            "/function_calling/is_calling_enable",
            json=data,
        )
        
        return self._get_response_value(response, as_json=True, value_func=lambda r:r.get("data", False))
        
    def chat_solution_chat(
        self,
        query: str,
        prompt_language: str = "",
        imagesdata: List[bytes] = [],
        audiosdata: List[bytes] = [],
        videosdata: List[bytes] = [],
        history: List[dict] = [],
        stream: bool = True,
        chat_solution: dict = {},
        temperature: float = 0.7,
        max_tokens: int = None,
        **kwargs,
    ):
        imageslist = []
        audioslist = []
        videoslist = []
        if len(imagesdata):
            for imagedata in imagesdata:
                imageslist.append(base64.b64encode(imagedata).decode('utf-8'))
        if len(audiosdata):
            for audiodata in audiosdata:
                audioslist.append(base64.b64encode(audiodata).decode('utf-8'))
        if len(videosdata):
            for videodata in videosdata:
                videoslist.append(base64.b64encode(videodata).decode('utf-8'))
        data = {
            "query": query,
            "prompt_language": prompt_language,
            "imagesdata": imageslist,
            "audiosdata": audioslist,
            "videosdata": videoslist,
            "history": history,
            "stream": stream,
            "chat_solution": chat_solution,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        print("received input message:")
        pprint(data)

        response = self.post(
            "/chat_solution/chat",
            json=data,
            stream=True,
        )
        return self._httpx_stream2generator(response, as_json=True)
    
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

