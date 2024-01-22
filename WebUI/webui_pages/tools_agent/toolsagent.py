import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from WebUI.webui_pages.utils import *
from WebUI.Server.knowledge_base.kb_service.base import (get_kb_details, get_kb_file_details)
from WebUI.Server.knowledge_base.utils import (get_file_path, LOADER_DICT, CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)

training_devices_list = ["auto","cpu","gpu","mps"]
loadbits_list = ["32 bits","16 bits","8 bits"]

KB_CREATE_NEW = "[Create New...]"

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        # pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=10
    )
    return gb

def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""

def tools_agent_page(api: ApiRequest, is_lite: bool = False):
    
    running_model = ""
    models_list = list(api.get_running_models())
    if len(models_list):
        running_model = models_list[0]
    webui_config = api.get_webui_config()
    current_voice_model = api.get_vtot_model()
    current_speech_model = api.get_ttov_model()
    voicemodel = None
    
    if running_model == "":
        running_model = "None"
    st.caption(
            f"""<h1 style="font-size: 1.5em; text-align: center; color: #3498db;">Running LLM Model: {running_model}</h1>""",
            unsafe_allow_html=True,
        )
    tabretrieval, tabinterpreter, tabspeech, tabvoice, tabimager, tabimageg, tabfunctions = st.tabs(["Retrieval", "Code Interpreter", "Text-to-Voice", "Voice-to-Text", "Image Recognition", "Image Generation", "Functions"])
    with tabretrieval:
        kb_list = {}
        try:
            kb_details = get_kb_details()
            if len(kb_details):
                kb_list = {x["kb_name"]: x for x in kb_details}
        except Exception as e:
            st.error("Get Knowledge Base failed!")
            st.stop()
        kb_names = [KB_CREATE_NEW]
        if len(kb_list):
            kb_names.extend(list(kb_list.keys()))
        print("kb_names: ", kb_names)
        kb_index = 0
        if st.session_state.get("selected_kb_name"):
            kb_index = kb_names.index(st.session_state.get("selected_kb_name"))
        selected_kb = st.selectbox(
            "Knowledge Base:",
            kb_names,
            index=kb_index
        )

        if selected_kb == KB_CREATE_NEW:
            st.session_state["selected_kb_name"] = ""
            embeddingmodel = webui_config.get("ModelConfig").get("EmbeddingModel")

            with st.form("Create Knowledge Base"):
                kb_name = st.text_input(
                    "Knowledge Base Name:",
                    key="kb_name",
                )
                kb_info = st.text_input(
                    "Introduction to the Knowledge Base",
                    key="kb_info",
                )

                cols = st.columns(2)
                #vs_types = GetKbsList()
                vs_types = ["faiss", "milvus", "pg"]  # current just support faiss.
                vs_type = cols[0].selectbox(
                    "Vector Store Type",
                    vs_types,
                    index=0,
                    key="vs_type",
                )

                embed_models = [f"{key}" for key in embeddingmodel]

                embed_model = cols[1].selectbox(
                    "Embedding Model",
                    embed_models,
                    index=0,
                    key="embed_model",
                )
                submit_create_kb = st.form_submit_button(
                    "Create",
                    use_container_width=True,
                )

            if submit_create_kb:
                with st.spinner(f"Create new Knowledge Base `{kb_name}`, Please do not perform any actions or refresh the page."):
                    if not kb_name or not kb_name.strip():
                        st.error(f"Knowledge Base Name is None!")
                    elif kb_name in kb_list:
                        st.error(f"The {kb_name} exist!")
                    else:
                        ret = api.create_knowledge_base(
                            knowledge_base_name=kb_name,
                            knowledge_base_info=kb_info,
                            vector_store_type=vs_type,
                            embed_model=embed_model,
                        )
                        if msg := check_success_msg(ret):
                            st.toast(msg, icon="✔")
                            st.session_state["selected_kb_name"] = kb_name
                            st.session_state["selected_kb_info"] = kb_info
                            st.session_state["need_rerun"] = True
                        elif msg := check_error_msg(ret):
                            st.error(msg)
                            st.toast(msg, icon="✖")
        
        elif selected_kb:
            kb = selected_kb
            st.session_state["selected_kb_name"] = kb
            st.session_state["selected_kb_info"] = kb_list[kb]['kb_info']

            # upload documents
            docs = st.file_uploader("upload files: ",
                                    [i for ls in LOADER_DICT.values() for i in ls],
                                    accept_multiple_files=True,
                                    )
            kb_info = st.text_area("Introduction to the Knowledge Base:", value=st.session_state["selected_kb_info"], max_chars=None, key=None,
                                help=None, on_change=None, args=None, kwargs=None)

            if kb_info != st.session_state["selected_kb_info"]:
                st.session_state["selected_kb_info"] = kb_info
                api.update_kb_info(kb, kb_info)

            # with st.sidebar:
            with st.expander(
                    "Embedding Configuration",
                    expanded=True,
            ):
                cols = st.columns(3)
                chunk_size = cols[0].number_input("Chunk Size", 1, 1000, CHUNK_SIZE, help="Maximum length of a single piece of text")
                chunk_overlap = cols[1].number_input("Overlap Size", 0, chunk_size, OVERLAP_SIZE, help="Length of overlap between adjacent texts")
                cols[2].write("")
                cols[2].write("")
                zh_title_enhance = cols[2].checkbox("Title Enh.", ZH_TITLE_ENHANCE, help="Enable Chinese title enhancement")
            kb_details = get_kb_file_details(kb)
            brepeat = False
            if len(docs) and len(kb_details):
                doc_names = [doc.name for doc in docs]
                docs_lower = [name.lower() for name in doc_names]
                file_name_list_lower = [entry["file_name"].lower() for entry in kb_details]
                duplicate_names = set(docs_lower) & set(file_name_list_lower)
                if len(duplicate_names):
                    print("duplicate_names: ", duplicate_names)
                    brepeat = True
            if st.button(
                    "Add Documents to Knowledge Base",
                    # use_container_width=True,
                    disabled=len(docs) == 0,
            ):
                with st.spinner(f"Add docs to `{kb}`, Please do not perform any actions or refresh the page."):
                    if brepeat:
                        st.toast("There are duplicate documents, please extract the duplicate ones first.", icon="✖")
                    else:
                        ret = api.upload_kb_docs(docs,
                                                knowledge_base_name=kb,
                                                override=True,
                                                chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                zh_title_enhance=zh_title_enhance)
                        if msg := check_success_msg(ret):
                            st.toast(msg, icon="✔")
                            st.rerun()
                        elif msg := check_error_msg(ret):
                            st.toast(msg, icon="✖")
            
            doc_details = pd.DataFrame(kb_details)
            selected_rows = []
            if not len(doc_details):
                pass
                #st.write(f"Knowledge Base `{kb}` no any files.")
            else:
                st.divider()
                st.write(f"Knowledge Base `{kb}` have files:")
                st.info("The knowledge base includes source files and a vector database. Please select a file from the table below for further actions.")
                doc_details.drop(columns=["kb_name"], inplace=True)
                doc_details = doc_details[[
                    "No", "file_name", "document_loader", "text_splitter", "docs_count", "in_folder", "in_db",
                ]]
                gb = config_aggrid(
                    doc_details,
                    {
                        ("No", "Serial Number"): {},
                        ("file_name", "Document Name"): {},
                        # ("file_ext", "Document Type"): {},
                        # ("file_version", "File Version"): {},
                        ("document_loader", "Document Loader"): {},
                        ("docs_count", "Document Count"): {},
                        ("text_splitter", "Text Splitter"): {},
                        # ("create_time", "Create Time"): {},
                        ("in_folder", "Source Files"): {"cellRenderer": cell_renderer},
                        ("in_db", "Vector Database"): {"cellRenderer": cell_renderer},
                    },
                    "multiple",
                )

                doc_grid = AgGrid(
                    doc_details,
                    gb.build(),
                    columns_auto_size_mode="FIT_CONTENTS",
                    theme="alpine",
                    custom_css={
                        "#gridToolBar": {"display": "none"},
                    },
                    allow_unsafe_jscode=True,
                    enable_enterprise_modules=False
                )

                selected_rows = doc_grid.get("selected_rows", [])

                cols = st.columns(4)
                file_name, file_path = file_exists(kb, selected_rows)
                if file_path:
                    with open(file_path, "rb") as fp:
                        cols[0].download_button(
                            "Download Document",
                            fp,
                            file_name=file_name,
                            use_container_width=True, )
                else:
                    cols[0].download_button(
                        "Download Document",
                        "",
                        disabled=True,
                        use_container_width=True, )

                st.write()
                # Tokenize the file and load it into the vector database
                if cols[1].button(
                        "Reload to Vector Database" if selected_rows and (pd.DataFrame(selected_rows)["in_db"]).any() else "Add to Vector Database",
                        disabled=not file_exists(kb, selected_rows)[0],
                        use_container_width=True,
                ):
                    file_names = [row["file_name"] for row in selected_rows]
                    with st.spinner(f"Reload `{file_names[0]}` to Vector Database, Please do not perform any actions or refresh the page."):
                        ret = api.update_kb_docs(kb,
                                        file_names=file_names,
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap,
                                        zh_title_enhance=zh_title_enhance)
                        if msg := check_success_msg(ret):
                            st.toast(msg, icon="✔")
                            #st.rerun()
                        elif msg := check_error_msg(ret):
                            st.toast(msg, icon="✖")

                if cols[2].button(
                        "Delete from Vector Database",
                        disabled=not (selected_rows and selected_rows[0]["in_db"]),
                        use_container_width=True,
                ):
                    file_names = [row["file_name"] for row in selected_rows]
                    with st.spinner(f"Delete `{file_names[0]}` from Vector Database, Please do not perform any actions or refresh the page."):
                        ret = api.delete_kb_docs(kb, file_names=file_names)
                        if msg := check_success_msg(ret):
                            st.toast(msg, icon="✔")
                            #st.rerun()
                        elif msg := check_error_msg(ret):
                            st.toast(msg, icon="✖")

                if cols[3].button(
                        "Delete from Knowledge Base",
                        type="primary",
                        use_container_width=True,
                ):
                    file_names = [row["file_name"] for row in selected_rows]
                    if len(file_names):
                        with st.spinner(f"Delete `{file_names[0]}` from Knowledge Base, Please do not perform any actions or refresh the page."):
                            ret = api.delete_kb_docs(kb, file_names=file_names, delete_content=True)
                            if msg := check_success_msg(ret):
                                st.toast(msg, icon="✔")
                                st.rerun()
                            elif msg := check_error_msg(ret):
                                st.toast(msg, icon="✖")
                    else:
                        st.toast("Please select delete files.", icon="✖")

            docs = []
            df = pd.DataFrame([], columns=["seq", "id", "content", "source"])
            if selected_rows:
                st.divider()
                file_name = selected_rows[0]["file_name"]
                st.write(f'Document View in the `{file_name}`:') # Document list in the file. Double-click to modify, enter Y in the delete column to remove the corresponding row.
                docs = api.search_kb_docs(knowledge_base_name=selected_kb, file_name=file_name)
                #print("docs: ", docs)
                data = [{"seq": i+1, "id": x["id"], "page_content": x["page_content"], "source": x["metadata"].get("source"),
                        "type": x["type"],
                        "metadata": json.dumps(x["metadata"], ensure_ascii=False),
                        "to_del": "",
                        } for i, x in enumerate(docs)]
                df = pd.DataFrame(data)

                gb = GridOptionsBuilder.from_dataframe(df)
                gb.configure_columns(["id", "source", "type", "metadata"], hide=True)
                gb.configure_column("seq", "No.", width=50)
                gb.configure_column("page_content", "Content", editable=False, autoHeight=True, wrapText=True, flex=1,
                                    cellEditor="agLargeTextCellEditor", cellEditorPopup=True)
                gb.configure_column("to_del", "Delete", editable=False, width=50, wrapHeaderText=True,
                                    cellEditor="agCheckboxCellEditor", cellRender="agCheckboxCellRenderer")
                gb.configure_selection()
                edit_docs = AgGrid(df, gb.build())

                # if st.button("Save"):
                #     # origin_docs = {x["id"]: {"page_content": x["page_content"], "type": x["type"], "metadata": x["metadata"]} for x in docs}
                #     changed_docs = []
                #     for index, row in edit_docs.data.iterrows():
                #         # origin_doc = origin_docs[row["id"]]
                #         # if row["page_content"] != origin_doc["page_content"]:
                #         if row["to_del"] not in ["Y", "y", 1]:
                #             changed_docs.append({
                #                 "page_content": row["page_content"],
                #                 "type": row["type"],
                #                 "metadata": json.loads(row["metadata"]),
                #             })

                #     if changed_docs:
                #         if api.update_kb_docs(knowledge_base_name=selected_kb,
                #                             file_names=[file_name],
                #                             docs={file_name: changed_docs}):
                #             st.toast("Update Document Success!")
                #         else:
                #             st.toast("Update Document Failed!")

    with tabinterpreter:
        pass

    with tabspeech:
        ttovmodel = webui_config.get("ModelConfig").get("TtoVModel")
        ttovmodel_lists = [f"{key}" for key in ttovmodel]
        col1, col2 = st.columns(2)
        spmodel = current_speech_model.get("model", "")
        with col1:
            if len(spmodel) == 0:
                index = 0
            else:
                index = ttovmodel_lists.index(spmodel)
            speechmodel = st.selectbox(
                    "Please Select Speech Model",
                    ttovmodel_lists,
                    index=index,
                )
            sple_button = st.button(
                "Load & Eject",
                key="sple_btn",
                use_container_width=True,
            )
            if sple_button:
                if speechmodel == spmodel:
                    with st.spinner(f"Release Model: `{speechmodel}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_speech_model(speechmodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_speech_model = {"model": "", "speaker": ""}
                else:
                    with st.spinner(f"Loading Model: `{speechmodel}`, Please do not perform any actions or refresh the page."):
                        speaker = st.session_state["speaker"]
                        print("speaker: ", speaker)
                        r = api.change_speech_model(spmodel, speechmodel, speaker)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_speech_model = {"model": speechmodel, "speaker": speaker}
        modelconfig = ttovmodel[speechmodel]
        synthesisconfig = modelconfig["synthesis"]
        #print(modelconfig)
        with col2:
            if modelconfig["type"] == "local":
                pathstr = modelconfig.get("path")
                st.text_input("Local Path", pathstr, key="sp_local_path")
                spsave_path = st.button(
                    "Save Path",
                    key="spsave_btn",
                    use_container_width=True,
                )
                if spsave_path:
                    with st.spinner(f"Saving Path, Please do not perform any actions or refresh the page."):
                        modelconfig["path"] = spsave_path
                        r = api.save_speech_model_config(speechmodel, modelconfig)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="sp_cloud_path", disabled=True)
                spsave_path = st.button(
                    "Save Path",
                    key="spsave_btn",
                    use_container_width=True,
                    disabled=True
                )

        st.divider()
        if modelconfig["type"] == "local":
            with st.form("speech_model"):
                devcol, bitcol = st.columns(2)
                with devcol:
                    if modelconfig["type"] == "local":
                        sdevice = modelconfig.get("device").lower()
                        if sdevice in training_devices_list:
                            index = training_devices_list.index(sdevice)
                        else:
                            index = 0
                        predict_dev = st.selectbox(
                                "Please select Device",
                                training_devices_list,
                                index=index
                            )
                
                with bitcol:
                    nloadbits = modelconfig.get("loadbits")
                    index = 0 if nloadbits == 32 else (1 if nloadbits == 16 else (2 if nloadbits == 8 else 16))
                    nloadbits = st.selectbox(
                        "Load Bits",
                        loadbits_list,
                        index=index
                    )
                save_parameters = st.form_submit_button(
                    "Save Parameters",
                    use_container_width=True
                )
                if save_parameters:
                    modelconfig["device"] = predict_dev
                    if nloadbits == "32 bits":
                        modelconfig["loadbits"] = 32
                    elif nloadbits == "16 bits":
                        modelconfig["loadbits"] = 16
                    else:
                        modelconfig["loadbits"] = 8
                    with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_speech_model_config(speechmodel, modelconfig)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)

        elif modelconfig["type"] == "cloud":
            with st.form("speech_cloud_model"):
                keycol, regcol = st.columns(2)
                with keycol:
                    speechkey = modelconfig.get("speech_key")
                    speechkey = st.text_input("Speech Key", speechkey, key="speech_key")
                with regcol:
                    speechregion = modelconfig.get("speech_region")
                    speechregion = st.text_input("Speech Region", speechregion, key="speech_region")

                save_parameters = st.form_submit_button(
                    "Save Parameters",
                    use_container_width=True
                )
                if save_parameters:
                    with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                        if speechkey == "" or speechkey == "[Your Key]" or speechregion == "" or speechregion == "[Your Region]":
                            st.error("Please enter the correct key and region, save failed!")
                        else:
                            modelconfig["speech_key"] = speechkey
                            modelconfig["speech_region"] = speechregion
                            r = api.save_speech_model_config(speechmodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            templates_list = [modelconfig.get("CloudTemplates", "")]
            templates = st.selectbox(
                "Please Select templates",
                templates_list,
                index=0,
                disabled=True
            )
            lang_index = 0
            language_list = list(synthesisconfig.keys())
            language = st.selectbox(
                "Please Select language",
                language_list,
                index=lang_index,
            )
            
        with col2:
            sex_index = 0
            sex_list = list(synthesisconfig[language].keys())
            sex = st.selectbox(
                "Please Select Sex",
                sex_list,
                index=sex_index
            )
            speaker_index = 0
            speaker_list = synthesisconfig[language][sex]
            speaker = st.selectbox(
                "Please Select Speaker",
                speaker_list,
                index=speaker_index,
            )
            if speaker:
                st.session_state["speaker"] = speaker

    with tabvoice:
        vtotmodel = webui_config.get("ModelConfig").get("VtoTModel")
        vtotmodel_lists = [f"{key}" for key in vtotmodel]
        col1, col2 = st.columns(2)
        with col1:
            if current_voice_model == "":
                index = 0
            else:
                index = vtotmodel_lists.index(current_voice_model)
            voicemodel = st.selectbox(
                    "Please Select Voice Model",
                    vtotmodel_lists,
                    index=index,
                )
            vole_button = st.button(
                "Load & Eject",
                key="vole_btn",
                use_container_width=True,
            )
            if vole_button:
                if voicemodel == current_voice_model:
                    with st.spinner(f"Release Model: `{voicemodel}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_voice_model(voicemodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_voice_model = ""
                else:
                    with st.spinner(f"Loading Model: `{voicemodel}`, Please do not perform any actions or refresh the page."):
                        r = api.change_voice_model(current_voice_model, voicemodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_voice_model = voicemodel
            modelconfig = vtotmodel[voicemodel]
        with col2:
            if modelconfig["type"] == "local":
                if voicemodel is not None:
                    pathstr = vtotmodel[voicemodel].get("path")
                else:
                    pathstr = ""
                st.text_input("Local Path", pathstr, key="vo_local_path")
                vosave_path = st.button(
                    "Save Path",
                    key="vosave_btn",
                    use_container_width=True,
                )
                if vosave_path:
                    with st.spinner(f"Saving Path, Please do not perform any actions or refresh the page."):
                            vtotmodel[voicemodel]["path"] = vosave_path
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save path for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save path for model {voicemodel}.")
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="vo_cloud_path", disabled=True)
                spsave_path = st.button(
                    "Save Path",
                    key="vosave_btn",
                    use_container_width=True,
                    disabled=True
                )

        st.divider()
        if modelconfig["type"] == "local":
            if voicemodel == "whisper-large-v3" or voicemodel == "whisper-base" or voicemodel == "whisper-medium":
                with st.form("whisper_model"):
                    devcol, bitcol = st.columns(2)
                    with devcol:
                        sdevice = modelconfig.get("device").lower()
                        if sdevice in training_devices_list:
                            index = training_devices_list.index(sdevice)
                        else:
                            index = 0
                        predict_dev = st.selectbox(
                                "Please select Device",
                                training_devices_list,
                                index=index
                            )
                    with bitcol:
                        nloadbits = modelconfig.get("loadbits")
                        index = 0 if nloadbits == 32 else (1 if nloadbits == 16 else (2 if nloadbits == 8 else 16))
                        nloadbits = st.selectbox(
                            "Load Bits",
                            loadbits_list,
                            index=index
                        )
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        modelconfig["device"] = predict_dev
                        if nloadbits == "32 bits":
                            modelconfig["loadbits"] = 32
                        elif nloadbits == "16 bits":
                            modelconfig["loadbits"] = 16
                        else:
                            modelconfig["loadbits"] = 8
                        with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save configuration for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save configuration for model {voicemodel}.")

            elif voicemodel == "faster-whisper-large-v3":
                with st.form("faster_whisper_model"):
                    devcol, bitcol = st.columns(2)
                    with devcol:
                        sdevice = modelconfig.get("device").lower()
                        if sdevice in training_devices_list:
                            index = training_devices_list.index(sdevice)
                        else:
                            index = 0
                        predict_dev = st.selectbox(
                                "Please select Device",
                                training_devices_list,
                                index=index
                            )
                    with bitcol:
                        nloadbits = modelconfig.get("loadbits")
                        index = 0 if nloadbits == 32 else (1 if nloadbits == 16 else (2 if nloadbits == 8 else 16))
                        nloadbits = st.selectbox(
                            "Load Bits",
                            loadbits_list,
                            index=index
                        )
                    save_parameters = st.form_submit_button(
                            "Save Parameters",
                            use_container_width=True
                        )
                    if save_parameters:
                        modelconfig["device"] = predict_dev
                        if nloadbits == "32 bits":
                            modelconfig["loadbits"] = 32
                        elif nloadbits == "16 bits":
                            modelconfig["loadbits"] = 16
                        else:
                            modelconfig["loadbits"] = 8
                        with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save configuration for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save configuration for model {voicemodel}.")
            else:
                pass
        elif modelconfig["type"] == "cloud":
            with st.form("voice_cloud_model"):
                keycol, regcol = st.columns(2)
                with keycol:
                    voicekey = modelconfig.get("voice_key")
                    voicekey = st.text_input("Voice Key", voicekey, key="voice_key")
                with regcol:
                    voiceregion = modelconfig.get("voice_region")
                    voiceregion = st.text_input("Voice Region", voiceregion, key="voice_region")

                save_parameters = st.form_submit_button(
                    "Save Parameters",
                    use_container_width=True
                )
                if save_parameters:
                    with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                        if voicekey == "" or voicekey == "[Your Key]" or voiceregion == "" or voiceregion == "[Your Region]":
                            st.error("Please enter the correct key and region, save failed!")
                        else:
                            modelconfig["voice_key"] = voicekey
                            modelconfig["voice_region"] = voiceregion
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save configuration for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save configuration for model {voicemodel}.")

    with tabimager:
        pass

    with tabimageg:
        pass

    with tabfunctions:
        pass

    st.session_state["current_page"] = "retrieval_agent_page"

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()