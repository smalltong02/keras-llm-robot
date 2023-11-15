import pydantic
from pydantic import BaseModel
from typing import Dict, Union
import httpx, os
from WebUI.configs.serverconfig import (FSCHAT_CONTROLLER, FSCHAT_OPENAI_API, FSCHAT_MODEL_WORKERS, HTTPX_DEFAULT_TIMEOUT)
import sys
import asyncio
from pathlib import Path
from WebUI import workers
import urllib.request
from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatAnthropic
from langchain.llms import OpenAI, AzureOpenAI, Anthropic
from typing import Dict, Union, Optional, Literal, Any, List, Callable, Awaitable
from WebUI.configs.modelconfig import (LLM_MODELS, ONLINE_LLM_MODEL, MODEL_PATH, MODEL_ROOT_PATH, LLM_DEVICE)

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

def get_model_worker_config(self, model_name: str = None) -> dict:        
    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(ONLINE_LLM_MODEL.get(model_name, {}).copy())
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    if model_name in ONLINE_LLM_MODEL:
        config["online_api"] = True
        if provider := config.get("provider"):
            try:
                config["worker_class"] = getattr(workers, provider)
            except Exception as e:
                msg = f"Online Model ‘{model_name}’'s provider configuration error."
                print(f'{e.__class__.__name__}: {msg}')
        
    if model_name in MODEL_PATH["llm_model"]:
        config["model_path"] = self.get_model_path(model_name)
        config["device"] = self.llm_device(config.get("device"))
    return config

def fschat_controller_address() -> str:
        host = FSCHAT_CONTROLLER["host"]
        if host == "0.0.0.0":
            host = "127.0.0.1"
        port = FSCHAT_CONTROLLER["port"]
        return f"http://{host}:{port}"


def fschat_model_worker_address(model_name: str = LLM_MODELS[0]) -> str:
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

def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    if type in MODEL_PATH:
        paths = MODEL_PATH[type]
    else:
        paths = {}
        for v in MODEL_PATH.values():
            paths.update(v)

    if path_str := paths.get(model_name):
        path = Path(path_str)
        if path.is_dir():
            return str(path)

        root_path = Path(MODEL_ROOT_PATH)
        if root_path.is_dir():
            path = root_path / model_name
            if path.is_dir():  # use key, {MODEL_ROOT_PATH}/chatglm-6b
                return str(path)
            path = root_path / path_str
            if path.is_dir():  # use value, {MODEL_ROOT_PATH}/THUDM/chatglm-6b-new
                return str(path)
            path = root_path / path_str.split("/")[-1]
            if path.is_dir():  # use value split by "/", {MODEL_ROOT_PATH}/chatglm-6b-new
                return str(path)
        return path_str
        
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
        
def llm_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or LLM_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device

def get_model_worker_config(model_name: str = None) -> dict:        
    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    config.update(ONLINE_LLM_MODEL.get(model_name, {}).copy())
    config.update(FSCHAT_MODEL_WORKERS.get(model_name, {}).copy())

    if model_name in ONLINE_LLM_MODEL:
        config["online_api"] = True
        if provider := config.get("provider"):
            try:
                config["worker_class"] = getattr(workers, provider)
            except Exception as e:
                msg = f"Online Model ‘{model_name}’'s provider configuration error."
                print(f'{e.__class__.__name__}: {msg}')
        
    if model_name in MODEL_PATH["llm_model"]:
        config["model_path"] = get_model_path(model_name)
        config["device"] = llm_device(config.get("device"))
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

def list_config_llm_models() -> Dict[str, Dict]:
    '''
    get configured llm models with different types.
    return [(model_name, config_type), ...]
    '''
    workers = list(FSCHAT_MODEL_WORKERS)

    return {
        "local": MODEL_PATH["llm_model"],
        "online": ONLINE_LLM_MODEL,
        "worker": workers,
    }

def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    从prompt_config中加载模板内容
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''

    from WebUI.configs import prompttemplates
    import importlib
    importlib.reload(prompttemplates)  # TODO: 检查configs/prompt_config.py文件有修改再重新加载
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
    ## 以下模型是Langchain原生支持的模型，这些模型不会走Fschat封装
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
    ## TODO 支持其他的Langchain原生支持的模型
    else:
        ## 非Langchain原生支持的模型，走Fschat封装
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
    return list(MODEL_PATH["embedding_model"])


def list_config_llm_models() -> Dict[str, Dict]:
    '''
    get configured llm models with different types.
    return [(model_name, config_type), ...]
    '''
    workers = list(FSCHAT_MODEL_WORKERS)

    return {
        "local": MODEL_PATH["llm_model"],
        "online": ONLINE_LLM_MODEL,
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
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:
    ## 以下模型是Langchain原生支持的模型，这些模型不会走Fschat封装
    config_models = list_config_llm_models()

    ## 非Langchain原生支持的模型，走Fschat封装
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        **kwargs
    )

    return model
