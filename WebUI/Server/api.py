import sys
import os
import argparse
import uvicorn
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import Body
from WebUI.Server.chat.completion import completion
from WebUI.configs.serverconfig import OPEN_CROSS_DOMAIN
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from WebUI.Server.chat.chat import chat
from WebUI.Server.chat.feedback import chat_feedback
from WebUI.Server.embeddings_api import embed_texts_endpoint
from WebUI.Server.chat.openai_chat import openai_chat
from WebUI.Server.chat.search_engine_chat import search_engine_chat
from WebUI.Server.chat.code_interpreter_chat import code_interpreter_chat
from WebUI.Server.chat.chat_solution_chat import chat_solution_chat
from WebUI.Server.llm_api import (list_running_models, get_running_models, list_config_models,
                            change_llm_model, stop_llm_model, chat_llm_model, download_llm_model,
                            get_model_config, save_chat_config, save_model_config, get_webui_configs, get_aigenerator_configs,
                            get_vtot_model, get_vtot_data, stop_vtot_model, change_vtot_model, save_voice_model_config,
                            get_speech_model, get_speech_data, save_speech_model_config, stop_speech_model, change_speech_model,
                            get_image_recognition_model, save_image_recognition_model_config, eject_image_recognition_model, change_image_recognition_model, get_image_recognition_data,
                            get_image_generation_model, save_image_generation_model_config, eject_image_generation_model, change_image_generation_model, get_image_generation_data,
                            get_music_generation_model, save_music_generation_model_config, eject_music_generation_model, change_music_generation_model, get_music_generation_data,
                            save_search_engine_config, llm_knowledge_base_chat, llm_search_engine_chat, save_code_interpreter_config, save_function_calling_config, save_google_toolboxes_config,
                            is_calling_enable)
from WebUI.Server.utils import(BaseResponse, ListResponse, FastAPI, MakeFastAPIOffline,
                          get_prompt_template)
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
             summary="Save chat configuration information",
             )(save_chat_config)

    app.post("/chat/feedback",
             tags=["Chat"],
             summary="Return dialogue scores.",
             )(chat_feedback)

    #knowledge interface
    mount_knowledge_routes(app)

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
             summary="List Model configuration information",
             )(list_config_models)

    app.post("/llm_model/get_model_config",
             tags=["LLM Model Management"],
             summary="Get Model configuration information",
             )(get_model_config)
    
    app.post("/llm_model/save_model_config",
             tags=["LLM Model Management"],
             summary="Save Model configuration information",
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
    
    app.post("/llm_model/download_llm_model",
             tags=["LLM Model Management"],
             summary="Download LLM Model (Model Worker)",
             )(download_llm_model)
    
    # Search Engine interface
    app.post("/search_engine/save_search_engine_config",
             tags=["Search Engine Management"],
             summary="Save config for search engine",
             )(save_search_engine_config)
    
    app.post("/search_engine/search_engine_chat",
            tags=["Search Engine Management"],
            summary="Chat with search engine.",
            )(search_engine_chat)
    
    app.post("/llm_model/search_engine_chat",
            tags=["Search Engine Management"],
            summary="Chat with search engine.",
            )(llm_search_engine_chat)
    
    # code interpreter interface
    app.post("/code_interpreter/save_code_interpreter_config",
             tags=["Code Interpreter Management"],
             summary="Save config for code interpreter",
             )(save_code_interpreter_config)
    
    app.post("/code_interpreter/code_interpreter_chat",
            tags=["Code Interpreter Management"],
            summary="Chat with code interpreter.",
            )(code_interpreter_chat)
    
    # function calling interface
    app.post("/function_calling/save_function_calling_config",
             tags=["Function Calling Management"],
             summary="Save config for function calling",
             )(save_function_calling_config)
    
    app.post("/function_calling/is_calling_enable",
             tags=["Function Calling Management"],
             summary="check function calling enable flag.",
             )(is_calling_enable)
    
    # google toolboxes interface
    app.post("/google_toolboxes/save_google_toolboxes_config",
             tags=["Google ToolBoxes"],
             summary="save config for google toolboxes",
             )(save_google_toolboxes_config)
    
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
             summary="Save Voice Model configuration information",
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
             summary="Save Speech Model configuration information",
             )(save_speech_model_config)
    
    app.post("/speech_model/stop",
             tags=["Voice Model Management"],
             summary="Stop Voice Model",
             )(stop_speech_model)
    
    app.post("/speech_model/change",
             tags=["Speech Model Management"],
             summary="Switch to new Speech Model",
             )(change_speech_model)
    
    # Image Recognition Model interface
    app.post("/image_model/get_image_recognition_model",
             tags=["Image Recognition Model Management"],
             summary="Get current running Image Recognition Model",
             )(get_image_recognition_model)
    
    app.post("/image_model/save_image_recognition_model_config",
             tags=["Image Recognition Model Management"],
             summary="Save Image Recognition Model configuration information",
             )(save_image_recognition_model_config)
    
    app.post("/image_model/eject_image_recognition_model",
             tags=["Image Recognition Model Management"],
             summary="Stop Image Recognition Model",
             )(eject_image_recognition_model)
    
    app.post("/image_model/change_image_recognition_model",
             tags=["Image Recognition Model Management"],
             summary="Switch to new Image Recognition Model",
             )(change_image_recognition_model)
    
    app.post("/image_model/get_image_recognition_data",
             tags=["Image Recognition Model Management"],
             summary="Translate image to text",
             )(get_image_recognition_data)

    # Image Generation Model interface
    app.post("/image_model/get_image_generation_model",
             tags=["Image Generation Model Management"],
             summary="Get current running Image Generation Model",
             )(get_image_generation_model)
    
    app.post("/image_model/save_image_generation_model_config",
             tags=["Image Generation Model Management"],
             summary="Save Image Generation Model configuration information",
             )(save_image_generation_model_config)
    
    app.post("/image_model/eject_image_generation_model",
             tags=["Image Generation Model Management"],
             summary="Stop Image Generation Model",
             )(eject_image_generation_model)
    
    app.post("/image_model/change_image_generation_model",
             tags=["Image Generation Model Management"],
             summary="Switch to new Image Generation Model",
             )(change_image_generation_model)
    
    app.post("/image_model/get_image_generation_data",
             tags=["Image Generation Model Management"],
             summary="Generate images based on text",
             )(get_image_generation_data)
    
    # Music Generation Model interface
    app.post("/music_model/get_music_generation_model",
             tags=["Music Generation Model Management"],
             summary="Get current running Music Generation Model",
             )(get_music_generation_model)
    
    app.post("/music_model/save_music_generation_model_config",
             tags=["Music Generation Model Management"],
             summary="Save Music Generation Model configuration information",
             )(save_music_generation_model_config)
    
    app.post("/music_model/eject_music_generation_model",
             tags=["Music Generation Model Management"],
             summary="Stop Music Generation Model",
             )(eject_music_generation_model)
    
    app.post("/music_model/change_music_generation_model",
             tags=["Music Generation Model Management"],
             summary="Switch to new Music Generation Model",
             )(change_music_generation_model)
    
    app.post("/music_model/get_music_generation_data",
             tags=["Music Generation Model Management"],
             summary="Generate Music based on text",
             )(get_music_generation_data)

    # Server interface
    app.post("/server/get_webui_config",
             tags=["Server State"],
             summary="get webui config",
             )(get_webui_configs)
    
    app.post("/server/get_aigenerator_config",
             tags=["Server State"],
             summary="get AI generator config",
             )(get_aigenerator_configs)

    @app.post("/server/get_prompt_template",
             tags=["Server State"],
             summary="Get prompt template")
    def get_server_prompt_template(
        type: Literal["llm_chat", "knowledge_base_chat", "search_engine_chat", "agent_chat"]=Body("llm_chat", description="Template Type"),
        name: str = Body("default", description="Template Name"),
    ) -> str:
        return get_prompt_template(type=type, name=name)
    
    # chat solution interface
    app.post("/chat_solution/chat",
            tags=["Chat Solution Management"],
            summary="Chat with Chat Solution.",
            )(chat_solution_chat)

    # other interface
    app.post("/other/completion",
             tags=["Other"],
             summary="Request LLM model completion (LLMChain)",
             )(completion)

    app.post("/other/embed_texts",
            tags=["Other"],
            summary="Vectorize text, supporting both local and online models.",
            )(embed_texts_endpoint)
    
def mount_knowledge_routes(app: FastAPI):
    from WebUI.Server.chat.knowledge_base_chat import knowledge_base_chat
    from WebUI.Server.chat.file_chat import upload_temp_docs, file_chat
    from WebUI.Server.chat.agent_chat import agent_chat
    from WebUI.Server.knowledge_base.kb_api import list_kbs, create_kb, delete_kb
    from WebUI.Server.knowledge_base.kb_doc_api import (list_files, upload_docs, delete_docs,
                                                update_docs, download_doc, recreate_vector_store,
                                                search_docs, DocumentWithVSId, update_info,
                                                update_docs_by_id,)
    app.post("/chat/knowledge_base_chat",
            tags=["Chat"],
            summary="chat with Knowledge base")(knowledge_base_chat)
    
    app.post("/llm_model/knowledge_base_chat",
            tags=["Chat"],
            summary="chat with Knowledge base")(llm_knowledge_base_chat)

    app.post("/chat/file_chat",
            tags=["Knowledge Base Management"],
            summary="chat with file"
            )(file_chat)

    app.post("/chat/agent_chat",
            tags=["Chat"],
            summary="chat with agent")(agent_chat)

    # Tag: Knowledge Base Management
    app.get("/knowledge_base/list_knowledge_bases",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="get knowledge base list")(list_kbs)

    app.post("/knowledge_base/create_knowledge_base",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="create knowledge base"
            )(create_kb)

    app.post("/knowledge_base/delete_knowledge_base",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="delete knowledge base"
            )(delete_kb)

    app.get("/knowledge_base/list_files",
            tags=["Knowledge Base Management"],
            response_model=ListResponse,
            summary="get file list from knowledge base"
            )(list_files)

    app.post("/knowledge_base/search_docs",
            tags=["Knowledge Base Management"],
            response_model=List[DocumentWithVSId],
            summary="search from knowledge base"
            )(search_docs)

    app.post("/knowledge_base/update_docs_by_id",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="update doc for knowledge base"
            )(update_docs_by_id)

    app.post("/knowledge_base/upload_docs",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="upload docs to knowledge base"
            )(upload_docs)

    app.post("/knowledge_base/delete_docs",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="delete docs from knowledge base"
            )(delete_docs)

    app.post("/knowledge_base/update_info",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="update knowledge base information"
            )(update_info)
    app.post("/knowledge_base/update_docs",
            tags=["Knowledge Base Management"],
            response_model=BaseResponse,
            summary="update docs to knowledge base"
            )(update_docs)

    app.get("/knowledge_base/download_doc",
            tags=["Knowledge Base Management"],
            summary="download doc file")(download_doc)

    app.post("/knowledge_base/recreate_vector_store",
            tags=["Knowledge Base Management"],
            summary="recreate vector store"
            )(recreate_vector_store)

    app.post("/knowledge_base/upload_temp_docs",
            tags=["Knowledge Base Management"],
            summary="upload docs to temp folder for chat"
            )(upload_temp_docs)
        

def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(app,
                    host=host,
                    port=port,
                    ssl_keyfile=kwargs.get("ssl_keyfile"),
                    ssl_certfile=kwargs.get("ssl_certfile"),
                    )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Keras-llm-Robot',
                                    description='About Keras-llm-Robot, local knowledge based LLM Model with langchain')
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    args = parser.parse_args()
    args_dict = vars(args)

    app = create_app()

    run_api(host=args.host,
            port=args.port,
            ssl_keyfile=args.ssl_keyfile,
            ssl_certfile=args.ssl_certfile,
            )
