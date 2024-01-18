import pydantic
from pydantic import BaseModel
from typing import Dict, Union
import httpx, os
from WebUI.configs.serverconfig import (FSCHAT_CONTROLLER, FSCHAT_OPENAI_API, FSCHAT_MODEL_WORKERS, HTTPX_DEFAULT_TIMEOUT)
import asyncio
from pathlib import Path
from WebUI import workers
import urllib.request
from fastapi import FastAPI
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.llms import OpenAI, AzureOpenAI, Anthropic
from typing import Dict, Union, Optional, Literal, Any, List, Callable, Awaitable
from WebUI.configs.webuiconfig import *
from WebUI.configs.basicconfig import *

async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        # TODO: handle exception
        msg = f"Caught exception: {e}"
        print(f'{e.__class__.__name__}: {msg}')
    finally:
        # Signal the aiter to stop.
        event.set()

def fschat_controller_address() -> str:
        host = FSCHAT_CONTROLLER["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = FSCHAT_CONTROLLER["port"]
        return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = "") -> str:
    if model := get_model_worker_config(model_name):
        host = model["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


def fschat_openai_api_address() -> str:
    host = FSCHAT_OPENAI_API["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"

def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        **kwargs,
    ) -> Union[httpx.Client, httpx.AsyncClient]:

    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            default_proxies.update({host: None})

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)
    print(kwargs)
    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)
    
def set_httpx_config(timeout: float = HTTPX_DEFAULT_TIMEOUT, proxy: Union[str, Dict] = None):
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        "http://127.0.0.1",
        "http://localhost",
    ]

    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    def _get_proxies():
        return proxies
        
    urllib.request.getproxies = _get_proxies

def get_model_path(modelinfo: dict) -> Optional[str]:
    local_path = modelinfo.get("path", "")
    hugg_path = modelinfo.get("Huggingface", "")
    if Path(local_path).is_dir():
        return local_path
    return hugg_path

def get_embed_model_config(model: str) -> Optional[dict]:
    if model is None or model == "":
        return {}
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    embeddingmodel = webui_config.get("ModelConfig").get("EmbeddingModel")
    embed_list = list(embeddingmodel)
    if model in embed_list:
        local_path = embeddingmodel.get(model).get("path", "")
        hugg_path = embeddingmodel.get(model).get("Huggingface", "")
        api_key = embeddingmodel.get(model).get("apikey", "")
        provider = embeddingmodel.get(model).get("provider", "")
        return {
                "local_path": local_path,
                "hugg_path": hugg_path,
                "api_key": api_key,
                "provider": provider,
               }
    return {}
    
def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    return "cpu"
        
def llm_device(modelinfo: dict) -> Literal["cuda", "mps", "cpu"]:
    device = modelinfo.get("device", "un")
    if device == "gpu":
        device = "cuda"
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device

def load_8bit(modelinfo: dict) -> bool:
    bits = modelinfo.get("loadbits", 16)
    if bits == 8:
        return True
    return False

def get_max_gpumem(modelinfo: dict) -> str:
    memory = modelinfo.get("maxmemory", 20)
    memory_str = f"{memory}GiB"
    return memory_str

def get_model_worker_config(model_name: str = None) -> dict:
    config = {}
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    server_config = webui_config.get("ServerConfig")
    
    config["host"] = server_config.get("default_host_ip")
    config["port"] = server_config["fastchat_model_worker"]["default"].get("port")
    config["vllm_enable"] = server_config["fastchat_model_worker"]["default"].get("vllm_enable")

    if model_name is None or model_name == "":
        return config

    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    onlinemodel = webui_config.get("ModelConfig").get("OnlineModel")
    for _, value in onlinemodel.items():
        modellist = value["modellist"]
        if model_name in modellist:
            config["online_model"] = True 
            config["api_base_url"] = value["baseurl"]
            config["api_key"] = value["apikey"]
            config["api_version"] = value["apiversion"]
            config["api_proxy"] = value["apiproxy"]
            return config

    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
    if modelinfo["mtype"] == ModelType.Local or modelinfo["mtype"] == ModelType.Multimodal or modelinfo["mtype"] == ModelType.Special:
        modelinfo["mname"] = model_name
        modelinfo["config"] = GetModelConfig(webui_config, modelinfo)
        if modelinfo["config"]:
            config["model_path"] = get_model_path(modelinfo["config"])
            config["device"] = llm_device(modelinfo["config"])
            config["load_8bit"] = load_8bit(modelinfo["config"])
            config["max_gpu_memory"] = get_max_gpumem(modelinfo["config"])
        if modelinfo["mtype"] == ModelType.Special:
            config["special_model"] = True
    return config

def get_vtot_worker_config(model_name: str = None) -> dict:
    config = {}
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    server_config = webui_config.get("ServerConfig")
    config["host"] = server_config.get("default_host_ip")
    config["port"] = server_config["vtot_model_worker"].get("port")
    
    if model_name is None or model_name == "":
        return config
    vtot_model = webui_config.get("ModelConfig").get("VtoTModel")
    if model_name in vtot_model:
        if vtot_model[model_name].get("type") == "local":
            config["model_type"] = "local"
            config["model_path"] = vtot_model[model_name].get("path")
            config["device"] = vtot_model[model_name].get("device")
            config["loadbits"] = vtot_model[model_name].get("loadbits")
            config["Huggingface"] = vtot_model[model_name].get("Huggingface")
        elif vtot_model[model_name].get("type") == "cloud":
            config["model_type"] = "cloud"
            config["model_path"] = ""
            config["device"] = "cloud"
            config["loadbits"] = ""
            config["Huggingface"] = ""
        else:
            config["model_type"] = ""
            config["model_path"] = ""
            config["device"] = ""
            config["loadbits"] = ""
            config["Huggingface"] = ""
    return config

def get_speech_worker_config(model_name: str = None) -> dict:
    config = {}
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    server_config = webui_config.get("ServerConfig")
    config["host"] = server_config.get("default_host_ip")
    config["port"] = server_config["ttov_model_worker"].get("port")
    
    if model_name is None or model_name == "":
        return config
    ttov_model = webui_config.get("ModelConfig").get("TtoVModel")
    if model_name in ttov_model:
        if ttov_model[model_name].get("type") == "local":
            config["model_type"] = "local"
            config["model_path"] = ttov_model[model_name].get("path")
            config["device"] = ttov_model[model_name].get("device")
            config["loadbits"] = ttov_model[model_name].get("loadbits")
            config["Huggingface"] = ttov_model[model_name].get("Huggingface")
        elif ttov_model[model_name].get("type") == "cloud":
            config["model_type"] = "cloud"
            config["model_path"] = ""
            config["device"] = "cloud"
            config["loadbits"] = ""
            config["Huggingface"] = ""
        else:
            config["model_type"] = ""
            config["model_path"] = ""
            config["device"] = ""
            config["loadbits"] = ""
            config["Huggingface"] = ""
    return config

def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    from starlette.responses import HTMLResponse

    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        if isinstance(index, int):
            app.routes.pop(index)

    # Set up static file mount
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # Define the doc and redoc pages, pointing at the right files
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()

    if redoc_url is not None:
        remove_route(redoc_url)

        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"

            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - ReDoc",
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",
                with_google_fonts=False,
                redoc_favicon_url=favicon,
            )
        
class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")
    msg: str = pydantic.Field("success", description="API status message")
    data: Any = pydantic.Field(None, description="API data")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


class ListResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of names")

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }

def get_prompt_template(type: str, name: str) -> Optional[str]:
    from WebUI.configs import prompttemplates
    import importlib
    importlib.reload(prompttemplates)
    return prompttemplates.PROMPT_TEMPLATES[type].get(name)

def get_OpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        echo: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> OpenAI:
    ## langchain model
    config_models = list_config_llm_models()
    if model_name in config_models.get("langchain", {}):
        config = config_models["langchain"][model_name]
        if model_name == "Azure-OpenAI":
            model = AzureOpenAI(
                streaming=streaming,
                verbose=verbose,
                callbacks=callbacks,
                deployment_name=config.get("deployment_name"),
                model_version=config.get("model_version"),
                openai_api_type=config.get("openai_api_type"),
                openai_api_base=config.get("api_base_url"),
                openai_api_version=config.get("api_version"),
                openai_api_key=config.get("api_key"),
                openai_proxy=config.get("openai_proxy"),
                temperature=temperature,
                max_tokens=max_tokens,
                echo=echo,
            )

        elif model_name == "OpenAI":
            model = OpenAI(
                streaming=streaming,
                verbose=verbose,
                callbacks=callbacks,
                model_name=config.get("model_name"),
                openai_api_base=config.get("api_base_url"),
                openai_api_key=config.get("api_key"),
                openai_proxy=config.get("openai_proxy"),
                temperature=temperature,
                max_tokens=max_tokens,
                echo=echo,
            )
        elif model_name == "Anthropic":
            model = Anthropic(
                streaming=streaming,
                verbose=verbose,
                callbacks=callbacks,
                model_name=config.get("model_name"),
                anthropic_api_key=config.get("api_key"),
                echo=echo,
            )
    else:
        ## fastchat model
        config = get_model_worker_config(model_name)
        model = OpenAI(
            streaming=streaming,
            verbose=verbose,
            callbacks=callbacks,
            openai_api_key=config.get("api_key", "EMPTY"),
            openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_proxy=config.get("openai_proxy"),
            echo=echo,
            **kwargs
        )

    return model

def list_embed_models() -> List[str]:
    '''
    get names of configured embedding models
    '''
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    embeddingmodel = webui_config.get("ModelConfig").get("EmbeddingModel")
    return list(embeddingmodel)


def list_config_llm_models() -> Dict[str, Dict]:
    '''
    get configured llm models with different types.
    return [(model_name, config_type), ...]
    '''
    workers = list(FSCHAT_MODEL_WORKERS)
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    localmodel = webui_config.get("ModelConfig").get("LocalModel")
    onlinemodel = webui_config.get("ModelConfig").get("OnlineModel")

    return {
        "local": localmodel,
        "online": onlinemodel,
        "worker": workers,
    }

def list_online_embed_models() -> List[str]:
    from WebUI.Server import model_workers

    ret = []
    for k, v in list_config_llm_models()["online"].items():
        if provider := v.get("provider"):
            worker_class = getattr(model_workers, provider, None)
            if worker_class is not None and worker_class.can_embedding():
                ret.append(k)
    return ret

def get_ChatOpenAI(
        model_name: str,
        temperature: float,
        provider: str = None,
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:
    config = get_model_worker_config(model_name)
    apikey = None
    if provider != None and provider == "openai-api":
        apikey = config.get("api_key", "[Your Key]")
        if apikey == "[Your Key]":
            apikey = os.environ.get('OPENAI_API_KEY')
    if apikey == None:
        apikey = "EMPTY"
    proxy = config.get("api_proxy", "[Private Proxy]")
    if proxy == "[Private Proxy]":
        proxy = ""
    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=apikey,
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=proxy,
        **kwargs
    )

    return model

def get_ChatGoogleAI(
        model_name: str,
        temperature: float,
        provider: str = None,
        max_tokens: int = None,
        callbacks: List[Callable] = [],
        verbose: bool = True,
) -> ChatOpenAI:
    config = get_model_worker_config(model_name)
    apikey = None
    if provider != None and provider == "google-api":
        apikey = config.get("api_key", "[Your Key]")
        if apikey == "[Your Key]":
            apikey = os.environ.get('GOOGLE_API_KEY')
    if apikey == None:
        apikey = "EMPTY"
    proxy = config.get("api_proxy", "[Private Proxy]")
    if proxy == "[Private Proxy]":
        proxy = ""
    model = ChatGoogleGenerativeAI(
        verbose=verbose,
        callbacks=callbacks,
        google_api_key=apikey,
        model=model_name,
        temperature=temperature,
    )

    return model

def torch_gc():
    try:
        import torch
        if torch.cuda.is_available():
            # with torch.cuda.device(DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif torch.backends.mps.is_available():
            try:
                from torch.mps import empty_cache
                empty_cache()
            except Exception as e:
                msg = ("Please upgrade pytorch to 2.0.0 when platform is MacOS.")
                print(msg)
    except Exception:
        ...

def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    tasks = []
    with ThreadPoolExecutor() as pool:
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        for obj in as_completed(tasks):  # TODO: Ctrl+c can not stop
            yield obj.result()

def get_server_configs() -> Dict:
    pass

def list_online_embed_models() -> List[str]:
    from WebUI.Server import model_workers

    ret = []
    for k, v in list_config_llm_models()["online"].items():
        if provider := v.get("provider"):
            worker_class = getattr(model_workers, provider, None)
            if worker_class is not None and worker_class.can_embedding():
                ret.append(k)
    return ret


def load_local_embeddings(model: str = None, device: str = detect_device()):
    from WebUI.Server.knowledge_base.kb_cache.base import embeddings_pool
    if model is None:
        return None
    return embeddings_pool.load_embeddings(model=model, device=device)

def get_temp_dir(id: str = None) -> Tuple[str, str]:
    import tempfile
    BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "keras_tempdir")

    if id is not None:
        path = os.path.join(BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    path = tempfile.mkdtemp(dir=BASE_TEMP_DIR)
    return path, os.path.basename(path)