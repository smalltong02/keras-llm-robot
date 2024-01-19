import os
import urllib
from pydantic import Json
import json
from sse_starlette import EventSourceResponse
from fastapi.responses import FileResponse
from fastapi import File, Form, Body, Query, UploadFile
from WebUI.Server.utils import BaseResponse, ListResponse, run_in_thread_pool
from WebUI.Server.knowledge_base.utils import (validate_kb_name, get_file_path, list_files_from_folder, files2docs_in_thread, KnowledgeFile)
from WebUI.Server.db.repository.knowledge_file_repository import get_file_detail
from WebUI.Server.knowledge_base.kb_service.base import KBServiceFactory
from WebUI.Server.knowledge_base.utils import (CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
from WebUI.Server.knowledge_base.model.kb_document_model import DocumentWithVSId
from langchain.docstore.document import Document
from typing import List, Dict

def update_docs_by_id(
        knowledge_base_name: str = Body(..., description="Knowledge base name", examples=["samples"]),
        docs: Dict[str, Document] = Body(..., description="such as: {id: Document, ...}")
) -> BaseResponse:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=500, msg=f"The Knowledge base '{knowledge_base_name}' not exist!")
    if kb.update_doc_by_ids(docs=docs):
        return BaseResponse(msg=f"update doc success!")
    else:
        return BaseResponse(msg=f"update doc failed.")

def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"Not find Knowledge base '{knowledge_base_name}'", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)
    
def update_info(
        knowledge_base_name: str = Body(..., description="Knowledge base name", examples=["samples"]),
        kb_info: str = Body(..., description="Knowledge base information", examples=[""]),
):
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Not found Knowledge base '{knowledge_base_name}'")
    kb.update_info(kb_info)

    return BaseResponse(code=200, msg=f"Knowledge base information has been changed.", data={"kb_info": kb_info})
    
def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"The file '{filename}' has existed."
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"upload file '{filename}' success!", data=data)
        except Exception as e:
            msg = f"The file '{filename}' upload failed, error: {e}"
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result

def search_docs(
        query: str = Body("", description="User input: ", examples=["chat"]),
        knowledge_base_name: str = Body(..., description="Knowledge base name", examples=["samples"]),
        top_k: int = Body(3, description="Vector count"),
        score_threshold: float = Body(0.6,
                                      description="Knowledge base matching relevance threshold, with a range between 0 and 1. A smaller SCORE indicates higher relevance, and setting it to 1 is equivalent to no filtering. It is recommended to set it around 0.5",
                                      ge=0, le=1),
        file_name: str = Body("", description="file name"),
        metadata: dict = Body({}, description=""),
) -> List[DocumentWithVSId]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k, score_threshold)
            data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
    return data

def update_docs(
        knowledge_base_name: str = Body(..., description="Knowledge base name", examples=["samples"]),
        file_names: List[str] = Body(..., description="file name, support multiple files", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="Chunk size"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="Overlap size"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="zh title enhance"),
        override_custom_docs: bool = Body(False, description="override docs"),
        docs: Json = Body({}, description="",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Body(False, description=""),
) -> BaseResponse:
    print("update docs: ", file_names)
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Not found '{knowledge_base_name}'")

    failed_files = {}
    kb_files = []

    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
            except Exception as e:
                msg = f"load doc '{file_name}' failed: {e}"
                failed_files[file_name] = msg

    for status, result in files2docs_in_thread(kb_files,
                                               chunk_size=chunk_size,
                                               chunk_overlap=chunk_overlap,
                                               zh_title_enhance=zh_title_enhance):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"for {file_name} add docs error: {e}"
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"update docs finished!", data={"failed_files": failed_files})
    
def upload_docs(
        files: List[UploadFile] = File(..., description="upload file, support multiple files"),
        knowledge_base_name: str = Form(..., description="Knowledge base name", examples=["samples"]),
        override: bool = Form(False, description="Override exist file"),
        to_vector_store: bool = Form(True, description="Vector to upload file"),
        chunk_size: int = Form(CHUNK_SIZE, description="Chunk size"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="Overlap size"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="zh title enhance"),
        docs: Json = Form({}, description="",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Form(False, description=""),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Not find Knowledge base '{knowledge_base_name}'")

    failed_files = {}
    file_names = list(docs.keys())

    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="upload and vector finished!", data={"failed_files": failed_files})

def delete_docs(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
        delete_content: bool = Body(False),
        not_refresh_vs_cache: bool = Body(False, description=""),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Not found Knowledge base '{knowledge_base_name}'")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"Not found file '{file_name}'"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"The file '{file_name}' delete failed, error: {e}"
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"Delete file finished!", data={"failed_files": failed_files})

def download_doc(
        knowledge_base_name: str = Query(..., description="Knowledge base name", examples=["samples"]),
        file_name: str = Query(..., description="file name", examples=["test.txt"]),
        preview: bool = Query(False, description=""),
):
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"Not found Knowledge base '{knowledge_base_name}'")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"read file '{kb_file.filename}' failed, error: {e}"
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"read file {kb_file.filename} failed.")

def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(""),
        embed_model: str = Body(""),
        chunk_size: int = Body(CHUNK_SIZE, description="Chunk size"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="Overlap size"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="zh title enhance"),
        not_refresh_vs_cache: bool = Body(False, description=""),
):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
    """

    def output():
        kb = KBServiceFactory.get_service(kb_name=knowledge_base_name, vector_store_type=vs_type, embed_model=embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"Not found Knowledge base '{knowledge_base_name}'"}
        else:
            if kb.exists():
                kb.clear_vs()
            kb.create_kb()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            for status, result in files2docs_in_thread(kb_files,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    kb.add_doc(kb_file, not_refresh_vs_cache=True)
                else:
                    kb_name, file_name, error = result
                    msg = f"add file '{file_name}' to knowledge base '{knowledge_base_name}' error: {error}. skip."
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            if not not_refresh_vs_cache:
                kb.save_vector_store()

    return EventSourceResponse(output())