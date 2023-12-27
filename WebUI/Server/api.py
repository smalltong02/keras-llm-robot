import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import Body
from WebUI.Server.chat.completion import completion
from WebUI.Server.utils import (FastAPI, MakeFastAPIOffline, BaseResponse)
from WebUI.configs.serverconfig import OPEN_CROSS_DOMAIN
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from WebUI.Server.chat.chat import chat
from WebUI.Server.chat.feedback import chat_feedback
from WebUI.Server.embeddings_api import embed_texts_endpoint
from WebUI.Server.chat.openai_chat import openai_chat
from WebUI.Server.llm_api import (list_running_models, get_running_models, list_config_models,
                            change_llm_model, stop_llm_model, chat_llm_model,
                            get_model_config, save_chat_config, save_model_config, get_webui_configs,
                            get_vtot_model, get_vtot_data, stop_vtot_model, change_vtot_model, save_voice_model_config,
                            get_speech_model, get_speech_data, save_speech_model_config, stop_speech_model, change_speech_model,
                            list_search_engines)
from WebUI.Server.utils import(get_prompt_template)
from typing import List, Literal
from __about__ import __version__

async def document():
    return RedirectResponse(url="/docs")

def create_app(run_mode: str = None):
    app = FastAPI(
        title="Langchain-Chatchat API Server",
        version=__version__
    )
    MakeFastAPIOffline(app)
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    mount_app_routes(app, run_mode=run_mode)
    return app

def mount_app_routes(app: FastAPI, run_mode: str = None):
    app.get("/",
            response_model=BaseResponse,
            summary="swagger document")(document)

    # Tag: Chat
    app.post("/chat/fastchat",
             tags=["Chat"],
             summary="Conversing with a llm model.(through the FastChat API)",
             )(openai_chat)

    app.post("/chat/chat",
             tags=["Chat"],
             summary="Conversing with a llm model.(through the LLMChain)",
             )(chat)
    
    app.post("/llm_model/save_chat_config",
             tags=["Chat"],
             summary="Save chat configration information",
             )(save_chat_config)

    #app.post("/chat/search_engine_chat",
    #         tags=["Chat"],
    #         summary="Chat with search engine.",
    #         )(search_engine_chat)

    app.post("/chat/feedback",
             tags=["Chat"],
             summary="Return dialogue scores.",
             )(chat_feedback)

    #knowledge interface
    #mount_knowledge_routes(app)

    # LLM Model interface
    app.post("/llm_model/list_running_models",
             tags=["LLM Model Management"],
             summary="List current running Model",
             )(list_running_models)
    
    app.post("/llm_model/get_running_models",
             tags=["LLM Model Management"],
             summary="Get current running Model",
             )(get_running_models)

    app.post("/llm_model/list_config_models",
             tags=["LLM Model Management"],
             summary="List Model configration information",
             )(list_config_models)

    app.post("/llm_model/get_model_config",
             tags=["LLM Model Management"],
             summary="Get Model configration information",
             )(get_model_config)
    
    app.post("/llm_model/save_model_config",
             tags=["LLM Model Management"],
             summary="Save Model configration information",
             )(save_model_config)

    app.post("/llm_model/stop",
             tags=["LLM Model Management"],
             summary="Stop LLM Model (Model Worker)",
             )(stop_llm_model)

    app.post("/llm_model/change",
             tags=["LLM Model Management"],
             summary="Switch to new LLM Model (Model Worker)",
             )(change_llm_model)
    
    app.post("/llm_model/chat",
             tags=["LLM Model Management"],
             summary="Chat with LLM Model (Model Worker)",
             )(chat_llm_model)
    
    # Voice Model interface
    app.post("/voice_model/get_vtot_model",
             tags=["Voice Model Management"],
             summary="Get current running Voice Model",
             )(get_vtot_model)
    
    app.post("/voice_model/get_vtot_data",
             tags=["Voice Model Management"],
             summary="Translate voice to text",
             )(get_vtot_data)
    
    app.post("/voice_model/save_voice_model_config",
             tags=["Voice Model Management"],
             summary="Save Voice Model configration information",
             )(save_voice_model_config)
    
    app.post("/voice_model/stop",
             tags=["Voice Model Management"],
             summary="Stop Voice Model",
             )(stop_vtot_model)
    
    app.post("/voice_model/change",
             tags=["Voice Model Management"],
             summary="Switch to new Voice Model",
             )(change_vtot_model)
    
    # Speech Model interface
    app.post("/speech_model/get_ttov_model",
             tags=["Speech Model Management"],
             summary="Get current running Speech Model",
             )(get_speech_model)
    
    app.post("/speech_model/get_ttov_data",
             tags=["Speech Model Management"],
             summary="Translate text to speech",
             )(get_speech_data)
    
    app.post("/speech_model/save_speech_model_config",
             tags=["Speech Model Management"],
             summary="Save Speech Model configration information",
             )(save_speech_model_config)
    
    app.post("/speech_model/stop",
             tags=["Voice Model Management"],
             summary="Stop Voice Model",
             )(stop_speech_model)
    
    app.post("/speech_model/change",
             tags=["Speech Model Management"],
             summary="Switch to new Speech Model",
             )(change_speech_model)

    # Server interface
    app.post("/server/get_webui_config",
             tags=["Server State"],
             summary="get webui config",
             )(get_webui_configs)
    #app.post("/server/configs",
    #         tags=["Server State"],
    #         summary="Get server configration info.",
    #         )(get_server_configs)

    #app.post("/server/list_search_engines",
    #         tags=["Server State"],
    #         summary="Get all search engine info.",
    #         )(list_search_engines)

    @app.post("/server/get_prompt_template",
             tags=["Server State"],
             summary="Get prompt template")
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="Template Type"),
        name: str = Body("default", description="Template Name"),
    ) -> str:
        return get_prompt_template(type=type, name=name)

    # 其它接口
    app.post("/other/completion",
             tags=["Other"],
             summary="Request LLM model completion (LLMChain)",
             )(completion)

    app.post("/other/embed_texts",
            tags=["Other"],
            summary="Vectorize text, supporting both local and online models.",
            )(embed_texts_endpoint)