import os
import json
import asyncio
from fastapi.responses import StreamingResponse
from fastapi import Body, File, Form, UploadFile
from WebUI.Server.chat.utils import History
from WebUI.configs import (DEF_TOKENS, GetProviderByName)
from WebUI.Server.utils import BaseResponse, GetModelApiBaseAddress, run_in_thread_pool, wrap_done, get_ChatOpenAI, get_prompt_template
from WebUI.Server.knowledge_base.utils import KnowledgeFile
from WebUI.configs.basicconfig import GetKbTempFolder, ModelType, ModelSize, ModelSubType, GetModelInfoByName
from WebUI.Server.knowledge_base.kb_cache.faiss_cache import memo_faiss_pool
from WebUI.Server.knowledge_base.utils import (CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE, SCORE_THRESHOLD)
from typing import List, Optional
from fastapi import Request
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from WebUI.Server.knowledge_base.kb_service.base import EmbeddingsFunAdapter
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler
from langchain.callbacks import AsyncIteratorCallbackHandler
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from langchain.prompts.chat import ChatPromptTemplate
from typing import AsyncIterable, Dict

def _parse_files_in_thread(
    files: List[UploadFile],
    dir: str,
    zh_title_enhance: bool,
    chunk_size: int,
    chunk_overlap: int,
):
    """
    Save the uploaded file to the corresponding folder using multithreading. The generator returns the save result: [success or error, filename, msg, docs].
    """
    def parse_file(file: UploadFile) -> dict:
        '''
        save to file.
        '''
        try:
            filename = file.filename
            file_path = os.path.join(dir, filename)
            file_content = file.file.read()

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            kb_file = KnowledgeFile(filename=filename, knowledge_base_name="temp")
            kb_file.filepath = file_path
            docs = kb_file.file2text(zh_title_enhance=zh_title_enhance,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap)
            return True, filename, f"upload {filename} success.", docs
        except Exception as e:
            msg = f"{filename} upload failed, error: {e}"
            return False, filename, msg, []

    params = [{"file": file} for file in files]
    for result in run_in_thread_pool(parse_file, params=params):
        yield result

def upload_temp_docs(
    files: List[UploadFile] = File(..., description="upload fiele, support multiple files."),
    prev_id: str = Form(None, description="Prev knowledge ID"),
    chunk_size: int = Form(CHUNK_SIZE, description="chunk size"),
    chunk_overlap: int = Form(OVERLAP_SIZE, description="overlap size"),
    zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="zh title enhance"),
) -> BaseResponse:
    '''
    Save the file to a temporary directory and perform vectorization. Return the name of the temporary directory as the ID, which will also serve as the ID for the temporary vector library.
    '''
    if prev_id is not None:
        memo_faiss_pool.pop(prev_id)

    failed_files = []
    documents = []
    path, id = GetKbTempFolder(prev_id)
    for success, file, msg, docs in _parse_files_in_thread(files=files,
                                                        dir=path,
                                                        zh_title_enhance=zh_title_enhance,
                                                        chunk_size=chunk_size,
                                                        chunk_overlap=chunk_overlap):
        if success:
            documents += docs
        else:
            failed_files.append({file: msg})

    with memo_faiss_pool.load_vector_store(id).acquire() as vs:
        vs.add_documents(documents)
    return BaseResponse(data={"id": id, "failed_files": failed_files})


async def file_chat(query: str = Body(..., description="User input: ", examples=["chat"]),
        knowledge_id: str = Body(..., description="Knowledge ID", examples=["samples"]),
        top_k: int = Body(3, description="matching vector count"),
        score_threshold: float = Body(
            SCORE_THRESHOLD,
            description="Knowledge base matching relevance threshold, with a range between 0 and 2. A smaller SCORE indicates higher relevance, and setting it to 2 is equivalent to no filtering. It is recommended to set it around 1.5"),
        history: List[History] = Body(
            [],
            description="History chat",
            examples=[[
                {"role": "user",
                "content": "Who are you?"},
                {"role": "assistant",
                "content": "I am AI."}]]
        ),
        stream: bool = Body(False, description="stream output"),
        model_name: str = Body("", description="LLM Model"),
        imagesdata: List[str] = Body([], description="image data", examples=["image"]),
        speechmodel: dict = Body({}, description="speech model config"),
        temperature: float = Body(0.7, description="LLM Temperature", ge=0.0, le=1.0),
        max_tokens: Optional[int] = Body(
            None,
            description="max tokens."
        ),
        prompt_name: str = Body(
            "default",
            description=""
        ),
        request: Request = None,
    ):
    if knowledge_id not in memo_faiss_pool.keys():
        return BaseResponse(code=404, msg=f"Can not found file knowledge base: {knowledge_id}, Please upload the file first.")
    history = [History.from_data(h) for h in history]

    async def file_chat_iterator(
            query: str,
            knowledge_id: str,
            top_k: int,
            score_threshold: float,
            history: Optional[List[History]],
            stream: bool,
            model_name: str = model_name,
            imagesdata: List[str] = [],
            speechmodel: dict = {},
            temperature: float = 0.7,
            max_tokens: Optional[int] = None,
            prompt_name: str = prompt_name,
            request: Request = None,
    ) -> AsyncIterable[str]:
        
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
        modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
        modelinfo["mname"] = model_name
        async_callback = AsyncIteratorCallbackHandler()
        callbackslist = [async_callback]
        if len(speechmodel):
            modeltype = speechmodel.get("type", "")
            provider = speechmodel.get("provider", "")
            #spmodel = speechmodel.get("model", "")
            spspeaker = speechmodel.get("speaker", "")
            speechkey = speechmodel.get("speech_key", "")
            speechregion = speechmodel.get("speech_region", "")
            if modeltype == "local" or modeltype == "cloud":
                speak_handler = StreamSpeakHandler(run_place=modeltype, provider=provider, synthesis=spspeaker, subscription=speechkey, region=speechregion)
                callbackslist.append(speak_handler)
        if len(imagesdata):
            if max_tokens is None:
                max_tokens = DEF_TOKENS
        if modelinfo["mtype"] == ModelType.Local:
            provider = GetProviderByName(webui_config, model_name)
            model = get_ChatOpenAI(
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbackslist,
            )
        else:
            api_base = GetModelApiBaseAddress(modelinfo)
            model = ChatOpenAI(
                streaming=True,
                verbose=False,
                openai_api_key="EMPTY",
                openai_api_base=api_base,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                callbacks=callbackslist,
            )
        embed_func = EmbeddingsFunAdapter()
        embeddings = await embed_func.aembed_query(query)
        with memo_faiss_pool.acquire(knowledge_id) as vs:
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [x[0] for x in docs]

        context = "\n".join([doc.page_content for doc in docs])
        if len(docs) == 0:
            prompt_template = get_prompt_template("knowledge_base_chat", "Empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            async_callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = doc.metadata.get("source")
            text = f"""In [{inum + 1}] [{filename}] \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:
            source_documents.append(f"""<span style='color:red'>No relevant documents were found. This response is generated based on the LLM Model '{model_name}' itself!</span>""")

        if stream:
            async for token in async_callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in async_callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)
        await task

    return StreamingResponse(file_chat_iterator(query=query, 
                                                knowledge_id=knowledge_id, 
                                                top_k=top_k,
                                                score_threshold=score_threshold,
                                                history=history,
                                                stream=stream,
                                                model_name=model_name,
                                                imagesdata=imagesdata,
                                                speechmodel=speechmodel,
                                                temperature=temperature,
                                                max_tokens=max_tokens,
                                                prompt_name=prompt_name,
                                                request=request))