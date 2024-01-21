import urllib
from WebUI.Server.utils import BaseResponse, ListResponse
from WebUI.Server.knowledge_base.utils import validate_kb_name
from WebUI.Server.knowledge_base.kb_service.base import KBServiceFactory
from fastapi import Body
from WebUI.configs.basicconfig import EmbeddingModelExist
from WebUI.Server.db.repository.knowledge_base_repository import list_kbs_from_db

def list_kbs():
    # Get List of Knowledge Base
    return ListResponse(data=list_kbs_from_db())

def create_kb(knowledge_base_name: str = Body(..., examples=["samples"]),
            knowledge_base_info: str = Body(""),
            vector_store_type: str = Body("faiss"),
            embed_model: str = Body(""),
            ) -> BaseResponse:
    # Create selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me.")
    if not EmbeddingModelExist(embed_model):
        return BaseResponse(code=403, msg="Embedding model not found, please download it in advance.")
    if knowledge_base_name is None or knowledge_base_name.strip() == "":
        return BaseResponse(code=404, msg="Knowledge base name is NULL.")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"The Knowledge base '{knowledge_base_name}' has existed!")

    kb = KBServiceFactory.get_service(knowledge_base_name, vector_store_type, knowledge_base_info, embed_model)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"Create Knowledge base error: {e}"
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=200, msg=f"Add New Knowledge base '{knowledge_base_name}' success!")

def delete_kb(
    knowledge_base_name: str = Body(..., examples=["samples"])
    ) -> BaseResponse:
    # Delete selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")
    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)

    if kb is None:
        return BaseResponse(code=404, msg=f"Not find Knowledge base '{knowledge_base_name}'")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"delete Knowledge base '{knowledge_base_name}' success!")
    except Exception as e:
        msg = f"delete Knowledge base error: {e}"
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"delete Knowledge base '{knowledge_base_name}' failed!")