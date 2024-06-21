import os
import json
import streamlit as st
import pandas as pd
from WebUI.configs.basicconfig import LocalModelExist, ImageModelExist, MusicModelExist
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from WebUI.webui_pages.utils import ApiRequest, check_success_msg, check_error_msg
from typing import List, Dict, Tuple, Literal
from WebUI.Server.knowledge_base.kb_service.base import (get_kb_details, get_kb_file_details)
from WebUI.Server.knowledge_base.utils import (get_file_path, LOADER_DICT, CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE)
from WebUI.configs import training_devices_list, loadbits_list

KB_CREATE_NEW = "[Create New...]"

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return 'X'}}""")


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
    current_imagere_model = api.get_image_recognition_model()
    current_imagegen_model = api.get_image_generation_model()
    current_musicgen_model = api.get_music_generation_model()
    current_running_config = api.get_current_running_config()
    voicemodel = None
    
    if running_model == "":
        running_model = "None"

    tabretrieval, tabinterpreter, tabspeech, tabvoice, tabimager, tabimageg, tabmusic, tabextra = st.tabs(["Retrieval", "Code Interpreter", "Text-to-Voice", "Voice-to-Text", "Image Recognition", "Image Generation", "Music", "Extra"])
    with tabretrieval:
        kb_list = {}
        try:
            kb_details = get_kb_details()
            if len(kb_details):
                kb_list = {x["kb_name"]: x for x in kb_details}
        except Exception as _:
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
                        st.error("Knowledge Base Name is None!")
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
        current_running_config["knowledge_base"]["name"] = st.session_state["selected_kb_name"]
        api.save_current_running_config(current_running_config)
    with tabinterpreter:
        codeinterpreter = webui_config.get("CodeInterpreter")
        codeinterpreter_lists = []
        for key, value in codeinterpreter.items():
            if isinstance(value, dict):
                codeinterpreter_lists.append(key)
        if current_running_config["code_interpreter"]["name"]:
            interpreter_enable = True
            index = codeinterpreter_lists.index(current_running_config["code_interpreter"]["name"])
        else:
            index = 0
            interpreter_enable = False
        current_interpreter = st.selectbox(
            "Please select Code Interpreter",
            codeinterpreter_lists,
            index=index
        )
        with st.form("CodeInterpreter"):
            col1, col2 = st.columns(2)
            offline = codeinterpreter.get("offline", True)
            auto_run = codeinterpreter.get("auto_run", True)
            safe_mode = codeinterpreter.get("safe_mode", True)
            custom_instructions = codeinterpreter.get(current_interpreter).get("custom_instructions", "[default]")
            system_message = codeinterpreter.get(current_interpreter).get("system_message", "[default]")
            with col1:
                custom_instructions = st.text_input("Custom Instructions", custom_instructions)
                interpreter_enable = st.checkbox("Enable", value=interpreter_enable, help="After enabling, The code interpreter feature will activate.")
                #auto_run = st.checkbox('Autorun', value=auto_run, help="After enabling, The code will run without asking the user.")
            with col2:
                system_message = st.text_input("System Message", system_message)
                #offline = st.checkbox("Offline", value=offline, help="After enabling, The code interpreter will work offline.")
                #safe_mode = st.checkbox('Safe Mode', value=safe_mode, help="After enabling, The running code will be checked for security.")

            save_parameters = st.form_submit_button(
                "Save Parameters",
                use_container_width=True,
            )
            if save_parameters:
                with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                    codeinterpreter["offline"] = offline
                    codeinterpreter["auto_run"] = auto_run
                    codeinterpreter["safe_mode"] = safe_mode
                    codeinterpreter.get(current_interpreter)["custom_instructions"] = custom_instructions
                    codeinterpreter.get(current_interpreter)["system_message"] = system_message
                    r = api.save_code_interpreter_config(codeinterpreter)
                    if msg := check_error_msg(r):
                        st.toast(msg, icon="✖")
                    elif msg := check_success_msg(r):
                        st.toast("success save configuration for Code Interpreter.", icon="✔")
                    if interpreter_enable:
                        current_running_config["code_interpreter"]["name"] = current_interpreter
                    else:
                        current_running_config["code_interpreter"]["name"] = ""
                    api.save_current_running_config(current_running_config)

    pathstr = ""
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
                            current_running_config["speech"]["name"] = ""
                            current_running_config["speech"]["speaker"] = ""
                else:
                    with st.spinner(f"Loading Model: `{speechmodel}`, Please do not perform any actions or refresh the page."):
                        provider = ttovmodel[speechmodel].get("provider", "")
                        if provider != "" or LocalModelExist(pathstr):
                            speaker = st.session_state["speaker"]
                            print("speaker: ", speaker)
                            r = api.change_speech_model(spmodel, speechmodel, speaker)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                current_speech_model = {"model": speechmodel, "speaker": speaker}
                                current_running_config["speech"]["name"] = speechmodel
                                current_running_config["speech"]["speaker"] = speaker
                        else:
                            st.error("Please download the model to your local machine first.")
                api.save_current_running_config(current_running_config)
                
        modelconfig = ttovmodel[speechmodel]
        synthesisconfig = modelconfig["synthesis"]
        #print(modelconfig)
        with col2:
            if modelconfig["type"] == "local":
                pathstr = modelconfig.get("path")
                st.text_input("Local Path", pathstr, key="sp_local_path", disabled=True)
                spdownload_btn = st.button(
                    "Download",
                    key="spdownload_btn",
                    use_container_width=True,
                )
                if spdownload_btn:
                    with st.spinner("Model downloading..., Please do not perform any actions or refresh the page."):
                        if LocalModelExist(pathstr):
                            st.error(f'The model {speechmodel} already exists in the folder {pathstr}')
                        else:
                            huggingface_path = modelconfig["Huggingface"]
                            r = api.download_llm_model(speechmodel, huggingface_path, pathstr)
                            download_error = False
                            progress_bar = st.progress(0)
                            for t in r:
                                if _ := check_error_msg(t):  # check whether error occured
                                    download_error = True
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    break
                                tqdm = t.get("percentage", 0.0) / 100
                                progress_bar.progress(tqdm)
                            if download_error is False:
                                progress_bar.progress(1.0)
                                st.success("downloading success!")
                                st.toast("downloading success!", icon="✔")
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="sp_cloud_path", disabled=True)
                spdownload_btn = st.button(
                    "Download",
                    key="spdownload_btn",
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
                    with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_speech_model_config(speechmodel, modelconfig)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)

        elif modelconfig["type"] == "cloud":
            if speechmodel == "AzureSpeechService" or speechmodel == "OpenAISpeechService":
                with st.form("speech_cloud_model"):
                    keycol, regcol = st.columns(2)
                    with keycol:
                        speechkey = modelconfig.get("speech_key")
                        speechkey = st.text_input("Speech Key", speechkey, key="speech_key", type="password")
                    with regcol:
                        speechregion = modelconfig.get("speech_region")
                        speechregion = st.text_input("Speech Region", speechregion, key="speech_region")

                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            modelconfig["speech_key"] = speechkey
                            modelconfig["speech_region"] = speechregion
                            r = api.save_speech_model_config(speechmodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
            elif speechmodel == "GoogleSpeechService":
                with st.form("speech_cloud_model"):
                    keycol, regcol = st.columns(2)
                    with keycol:
                        key_json_path = modelconfig.get("speech_key")
                        key_json_path = st.text_input("Key Json Path", key_json_path, key="key_json_path-2")
                    with regcol:
                        speechregion = modelconfig.get("speech_region")
                        speechregion = st.text_input("Speech Region", speechregion, key="speech_region", disabled=True)
                    
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            modelconfig["speech_key"] = key_json_path
                            modelconfig["speech_region"] = speechregion
                            r = api.save_speech_model_config(speechmodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
            else:
                pass

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            templates_list = [modelconfig.get("CloudTemplates", "")]
            _ = st.selectbox(
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
            if not current_voice_model:
                index = 0
            else:
                index = vtotmodel_lists.index(current_voice_model)
            voicemodel = st.selectbox(
                    "Please Select Voice Model",
                    vtotmodel_lists,
                    index=index,
                )
            modelconfig = vtotmodel[voicemodel]
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
                            current_running_config["voice"]["name"] = ""
                            current_running_config["voice"]["language"] = ""
                else:
                    with st.spinner(f"Loading Model: `{voicemodel}`, Please do not perform any actions or refresh the page."):
                        provider = vtotmodel[voicemodel].get("provider", "")
                        if provider != "" or LocalModelExist(pathstr):
                            r = api.change_voice_model(current_voice_model, voicemodel)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                                current_voice_model = voicemodel
                                current_running_config["voice"]["name"] = current_voice_model
                                language = modelconfig.get("language", [])
                                current_running_config["voice"]["language"] = language[0] if language else "en-US"
                        else:
                            st.error("Please download the model to your local machine first.")
                api.save_current_running_config(current_running_config)
        with col2:
            if modelconfig["type"] == "local":
                if voicemodel is not None:
                    pathstr = vtotmodel[voicemodel].get("path")
                else:
                    pathstr = ""
                st.text_input("Local Path", pathstr, key="vo_local_path", disabled=True)
                vodownload_btn = st.button(
                    "Download",
                    key="vodownload_btn",
                    use_container_width=True,
                )
                if vodownload_btn:
                    with st.spinner("Model downloading..., Please do not perform any actions or refresh the page."):
                        if LocalModelExist(pathstr):
                            st.error(f'The model {voicemodel} already exists in the folder {pathstr}')
                        else:
                            huggingface_path = modelconfig["Huggingface"]
                            r = api.download_llm_model(voicemodel, huggingface_path, pathstr)
                            download_error = False
                            progress_bar = st.progress(0)
                            for t in r:
                                if _ := check_error_msg(t):  # check whether error occured
                                    download_error = True
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    break
                                tqdm = t.get("percentage", 0.0) / 100
                                progress_bar.progress(tqdm)
                            if download_error is False:
                                progress_bar.progress(1.0)
                                st.success("downloading success!")
                                st.toast("downloading success!", icon="✔")
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="vo_cloud_path", disabled=True)
                vodownload_btn = st.button(
                    "Download",
                    key="vodownload_btn",
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
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
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
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save configuration for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save configuration for model {voicemodel}.")
            else:
                pass
        elif modelconfig["type"] == "cloud":
            if voicemodel == "GoogleVoiceService":
                with st.form("voice_cloud_model"):
                    keycol, langcol = st.columns(2)
                    with keycol:
                        key_json_path = modelconfig.get("key_json_path")
                        key_json_path = st.text_input("Key Json Path", key_json_path, key="key_json_path-1")
                    with langcol:
                        language_code = modelconfig.get("language", "")
                        if language_code:
                            language_code = language_code[0]
                        language_code_list = modelconfig.get("language_code", [])
                        if language_code in language_code_list:
                            index = language_code_list.index(language_code)
                        else:
                            index = 0
                        language_code = st.selectbox(
                            "Language",
                            language_code_list,
                            index=index
                        )
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            modelconfig["key_json_path"] = key_json_path
                            modelconfig["language"] = [language_code]
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save configuration for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save configuration for model {voicemodel}.")

            elif voicemodel == "AzureVoiceService":
                with st.form("voice_cloud_model"):
                    keycol, regcol = st.columns(2)
                    with keycol:
                        voicekey = modelconfig.get("voice_key")
                        voicekey = st.text_input("Voice Key", voicekey, key="voice_key", type="password")
                        language_code = modelconfig.get("language", "")
                        if language_code:
                            language_code = language_code[0]
                        language_code_list = modelconfig.get("language_code", [])
                        if language_code in language_code_list:
                            index = language_code_list.index(language_code)
                        else:
                            index = 0
                        language_code = st.selectbox(
                            "Language",
                            language_code_list,
                            index=index
                        )
                    with regcol:
                        voiceregion = modelconfig.get("voice_region")
                        voiceregion = st.text_input("Voice Region", voiceregion, key="voice_region")

                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            modelconfig["voice_key"] = voicekey
                            modelconfig["voice_region"] = voiceregion
                            modelconfig["language"] = [language_code]
                            r = api.save_vtot_model_config(voicemodel, modelconfig)
                            if msg := check_error_msg(r):
                                st.error(f"failed to save configuration for model {voicemodel}.")
                            elif msg := check_success_msg(r):
                                st.success(f"success save configuration for model {voicemodel}.")
            else:
                pass

    with tabimager:
        imageremodels = webui_config.get("ModelConfig").get("ImageRecognition")
        imageremodels_lists = [f"{key}" for key in imageremodels]
        col1, col2 = st.columns(2)
        with col1:
            if current_imagere_model == "":
                index = 0
            else:
                index = imageremodels_lists.index(current_imagere_model)
            imageremodel = st.selectbox(
                    "Please Select Image Recognition Model",
                    imageremodels_lists,
                    index=index,
                )
            imagerele_button = st.button(
                "Load & Eject",
                key="imagere_btn",
                use_container_width=True,
            )
            if imagerele_button:
                if imageremodel == current_imagere_model:
                    with st.spinner(f"Release Model: `{imageremodel}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_image_recognition_model(imageremodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_imagere_model = ""
                else:
                    with st.spinner(f"Loading Model: `{imageremodel}`, Please do not perform any actions or refresh the page."):
                        provider = imageremodels[imageremodel].get("provider", "")
                        pathstr = imageremodels[imageremodel].get("path")
                        if provider != "" or LocalModelExist(pathstr):
                            r = api.change_image_recognition_model(current_imagere_model, imageremodel)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                current_imagere_model = imageremodel
                        else:
                            st.error("Please download the model to your local machine first.")
            modelconfig = imageremodels[imageremodel]
        with col2:
            if modelconfig["type"] == "local":
                if imageremodel is not None:
                    pathstr = imageremodels[imageremodel].get("path")
                else:
                    pathstr = ""
                st.text_input("Local Path", pathstr, key="re_local_path", disabled=True)
                redownload_btn = st.button(
                    "Download",
                    key="redownload_btn",
                    use_container_width=True,
                )
                if redownload_btn:
                    with st.spinner("Model downloading..., Please do not perform any actions or refresh the page."):
                        pathstr = modelconfig.get("path")
                        if ImageModelExist(pathstr):
                            st.error(f'The model {imageremodel} already exists in the folder {pathstr}')
                        else:
                            huggingface_path = modelconfig["Huggingface"]
                            r = api.download_llm_model(imageremodel, huggingface_path, pathstr)
                            download_error = False
                            progress_bar = st.progress(0)
                            for t in r:
                                if __ := check_error_msg(t):  # check whether error occured
                                    download_error = True
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    break
                                tqdm = t.get("percentage", 0.0) / 100
                                progress_bar.progress(tqdm)
                            if download_error is False:
                                progress_bar.progress(1.0)
                                st.success("downloading success!")
                                st.toast("downloading success!", icon="✔")
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="re_cloud_path", disabled=True)
                redownload_btn = st.button(
                    "Download",
                    key="redownload_btn",
                    use_container_width=True,
                    disabled=True
                )

        st.divider()
        if modelconfig["type"] == "local":
            with st.form("image_recognition_model"):
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
                    with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_image_recognition_model_config(imageremodel, modelconfig)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)

        elif modelconfig["type"] == "cloud":
            pass

    with tabimageg:
        imagegenmodels = webui_config.get("ModelConfig").get("ImageGeneration")
        imagegenmodels_lists = [f"{key}" for key in imagegenmodels]
        col1, col2 = st.columns(2)
        with col1:
            if current_imagegen_model == "":
                index = 0
            else:
                index = imagegenmodels_lists.index(current_imagegen_model)
            imagegenmodel = st.selectbox(
                    "Please Select Image Generation Model",
                    imagegenmodels_lists,
                    index=index,
                )
            imagegenle_button = st.button(
                "Load & Eject",
                key="imagegen_btn",
                use_container_width=True,
            )
            if imagegenle_button:
                if imagegenmodel == current_imagegen_model:
                    with st.spinner(f"Release Model: `{imagegenmodel}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_image_generation_model(imagegenmodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_imagegen_model = ""
                else:
                    with st.spinner(f"Loading Model: `{imagegenmodel}`, Please do not perform any actions or refresh the page."):
                        provider = imagegenmodels[imagegenmodel].get("provider", "")
                        pathstr = imagegenmodels[imagegenmodel].get("path")
                        if provider != "" or ImageModelExist(pathstr):
                            if (imagegenmodel == "OpenDalleV1.1" or imagegenmodel == "ProteusV0.2") and ImageModelExist("models/imagegeneration/sdxl-vae-fp16-fix") is False:
                                st.error("Please first download the sdxl-vae-fp16-fix model from Hugginface.")
                            else:
                                r = api.change_image_generation_model(current_imagegen_model, imagegenmodel)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                elif msg := check_success_msg(r):
                                    st.success(msg)
                                    current_imagegen_model = imagegenmodel
                        else:
                            st.error("Please download the model to your local machine first.")
            modelconfig = imagegenmodels[imagegenmodel]
        with col2:
            if modelconfig["type"] == "local":
                if imagegenmodel is not None:
                    pathstr = imagegenmodels[imagegenmodel].get("path")
                else:
                    pathstr = ""
                st.text_input("Local Path", pathstr, key="gen_local_path", disabled=True)
                gendownload_btn = st.button(
                    "Download",
                    key="gendownload_btn",
                    use_container_width=True,
                )
                if gendownload_btn:
                    with st.spinner("Model downloading..., Please do not perform any actions or refresh the page."):
                        if LocalModelExist(pathstr):
                            st.error(f'The model {imagegenmodel} already exists in the folder {pathstr}')
                        else:
                            huggingface_path = modelconfig["Huggingface"]
                            r = api.download_llm_model(imagegenmodel, huggingface_path, pathstr)
                            download_error = False
                            progress_bar = st.progress(0)
                            for t in r:
                                if _ := check_error_msg(t):  # check whether error occured
                                    download_error = True
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    break
                                tqdm = t.get("percentage", 0.0) / 100
                                progress_bar.progress(tqdm)
                            if download_error is False:
                                progress_bar.progress(1.0)
                                st.success("downloading success!")
                                st.toast("downloading success!", icon="✔")
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="gen_cloud_path", disabled=True)
                gendownload_btn = st.button(
                    "Download",
                    key="gendownload_btn",
                    use_container_width=True,
                    disabled=True
                )

        st.divider()
        if modelconfig["type"] == "local":
            with st.form("image_generation_model"):
                devcol, bitcol = st.columns(2)
                with devcol:
                    subconfig = modelconfig.get("config", {})
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
                    if subconfig:
                        seed = subconfig.get("seed", 0)
                        seed = st.number_input("Seed (-1 for random)", value = seed)
                
                with bitcol:
                    if modelconfig["type"] == "local":
                        nloadbits = modelconfig.get("loadbits")
                        index = 0 if nloadbits == 32 else (1 if nloadbits == 16 else (2 if nloadbits == 8 else 16))
                        nloadbits = st.selectbox(
                            "Load Bits",
                            loadbits_list,
                            index=index
                        )
                    if subconfig:
                        torch_compile = subconfig.get("torch_compile", False)
                        torch_compile = st.checkbox('Torch Compile', value=torch_compile, help="Note: Not support torch compile on windows.")
                        cpu_offload = subconfig.get("cpu_offload", False)
                        cpu_offload = st.checkbox('Cpu Offload', value=cpu_offload)
                        refiner = subconfig.get("refiner", False)
                        refiner = st.checkbox('Refiner', value=refiner, help="Please download the model stable-diffusion-xl-refiner-1.0 in advance.")
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
                    if subconfig:
                        modelconfig["config"]["seed"] = seed
                        modelconfig["config"]["torch_compile"] = torch_compile
                        modelconfig["config"]["cpu_offload"] = cpu_offload
                        modelconfig["config"]["refiner"] = refiner
                    with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_image_generation_model_config(imagegenmodel, modelconfig)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)

        elif modelconfig["type"] == "cloud":
            pass

    with tabmusic:
        musicgenmodels = webui_config.get("ModelConfig").get("MusicGeneration")
        musicgenmodels_lists = [f"{key}" for key in musicgenmodels]
        col1, col2 = st.columns(2)
        with col1:
            if current_musicgen_model == "":
                index = 0
            else:
                index = musicgenmodels_lists.index(current_musicgen_model)
            musicgenmodel = st.selectbox(
                    "Please Select Music Generation Model",
                    musicgenmodels_lists,
                    index=index,
                )
            musicgenle_button = st.button(
                "Load & Eject",
                key="musicgen_btn",
                use_container_width=True,
            )
            if musicgenle_button:
                if musicgenmodel == current_musicgen_model:
                    with st.spinner(f"Release Model: `{musicgenmodel}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_music_generation_model(musicgenmodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_musicgen_model = ""
                else:
                    with st.spinner(f"Loading Model: `{musicgenmodel}`, Please do not perform any actions or refresh the page."):
                        provider = musicgenmodels[musicgenmodel].get("provider", "")
                        pathstr = musicgenmodels[musicgenmodel].get("path")
                        if provider != "" or MusicModelExist(pathstr):
                            r = api.change_music_generation_model(current_musicgen_model, musicgenmodel)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                current_musicgen_model = musicgenmodel
                        else:
                            st.error("Please download the model to your local machine first.")
            modelconfig = musicgenmodels[musicgenmodel]
        with col2:
            if modelconfig["type"] == "local":
                if musicgenmodel is not None:
                    pathstr = musicgenmodels[musicgenmodel].get("path")
                else:
                    pathstr = ""
                st.text_input("Local Path", pathstr, key="music_local_path", disabled=True)
                musicdownload_btn = st.button(
                    "Download",
                    key="musicdownload_btn",
                    use_container_width=True,
                )
                if musicdownload_btn:
                    with st.spinner("Model downloading..., Please do not perform any actions or refresh the page."):
                        if LocalModelExist(pathstr):
                            st.error(f'The model {musicgenmodel} already exists in the folder {pathstr}')
                        else:
                            huggingface_path = modelconfig["Huggingface"]
                            r = api.download_llm_model(musicgenmodel, huggingface_path, pathstr)
                            download_error = False
                            progress_bar = st.progress(0)
                            for t in r:
                                if _ := check_error_msg(t):  # check whether error occured
                                    download_error = True
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    break
                                tqdm = t.get("percentage", 0.0) / 100
                                progress_bar.progress(tqdm)
                            if download_error is False:
                                progress_bar.progress(1.0)
                                st.success("downloading success!")
                                st.toast("downloading success!", icon="✔")
            elif modelconfig["type"] == "cloud":
                pathstr = modelconfig.get("path")
                st.text_input("Cloud Path", pathstr, key="gen_cloud_path", disabled=True)
                musicdownload_btn = st.button(
                    "Download",
                    key="musicdownload_btn",
                    use_container_width=True,
                    disabled=True
                )
            
        st.divider()
        if modelconfig["type"] == "local":
            with st.form("music_generation_model"):
                devcol, bitcol = st.columns(2)
                with devcol:
                    subconfig = modelconfig.get("config", {})
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
                    if subconfig:
                        max_tokens = subconfig.get("max_new_tokens", 256)
                        max_tokens = st.slider("Max Tokens", 1, 1500, max_tokens, 1)
                        do_sample = subconfig.get("do_sample", False)
                        do_sample = st.checkbox('Do Sample', value=do_sample)

                with bitcol:
                    if modelconfig["type"] == "local":
                        nloadbits = modelconfig.get("loadbits")
                        index = 0 if nloadbits == 32 else (1 if nloadbits == 16 else (2 if nloadbits == 8 else 16))
                        nloadbits = st.selectbox(
                            "Load Bits",
                            loadbits_list,
                            index=index
                        )
                    if subconfig:
                        guiding_scale = subconfig.get("guiding_scale", 3)
                        guiding_scale = st.slider("Guiding Scale", 1, 20, guiding_scale, 1)
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
                    if subconfig:
                        modelconfig["config"]["seed"] = seed
                        modelconfig["config"]["guiding_scale"] = guiding_scale
                        modelconfig["config"]["max_new_tokens"] = max_tokens
                        modelconfig["config"]["do_sample"] = do_sample
                    with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_music_generation_model_config(musicgenmodel, modelconfig)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)

        elif modelconfig["type"] == "cloud":
            pass

    with tabextra:
        toolboxes = webui_config.get("ToolBoxes")
        google_toolboxes = {}
        if toolboxes:
            google_toolboxes = toolboxes.get("Google ToolBoxes")

        if google_toolboxes:
            from WebUI.Server.funcall.google_toolboxes.gmail_funcall import GetMailFuncallList, GetMailFuncallDescription
            from WebUI.Server.funcall.google_toolboxes.gmap_funcall import GetMapFuncallList, GetMapFuncallDescription
            from WebUI.Server.funcall.google_toolboxes.calendar_funcall import GetCalendarFuncallList, GetCalendarFuncallDescription
            from WebUI.Server.funcall.google_toolboxes.gcloud_funcall import GetStorageFuncallList, GetStorageFuncallDescription
            from WebUI.Server.funcall.google_toolboxes.youtube_funcall import GetYoutubeFuncallList, GetYoutubeFuncallDescription
            from WebUI.Server.funcall.google_toolboxes.photo_funcall import GetPhotoFuncallList, GetPhotoFuncallDescription
            toolboxes_lists = google_toolboxes.get("Tools")
            google_credential = google_toolboxes.get("credential")
            print("credential: ", google_credential)
            search_key = google_toolboxes.get("search_key")
            current_function = ""
            function_name_list = []
            col1, col2 = st.columns(2)
            with col1:
                google_tool = st.selectbox(
                    "Google ToolBoxes:",
                    toolboxes_lists,
                    index=0
                )
                if google_tool == "Google Mail":
                    function_name_list = GetMailFuncallList()
                elif google_tool == "Google Maps":
                    function_name_list = GetMapFuncallList()
                elif google_tool == "Google Calendar":
                    function_name_list = GetCalendarFuncallList()
                elif google_tool == "Google Drive":
                    function_name_list = GetStorageFuncallList()
                elif google_tool == "Google Youtube":
                    function_name_list = GetYoutubeFuncallList()
                elif google_tool == "Google Photos":
                    function_name_list = GetPhotoFuncallList()
                calling_enable = current_running_config["ToolBoxes"]["Google ToolBoxes"]["Tools"][google_tool]["enable"]
                calling_enable = st.checkbox("Enable", key="funcall_box", value=calling_enable, help="After enabling, The function will be called automatically.")
                current_running_config["ToolBoxes"]["Google ToolBoxes"]["Tools"][google_tool]["enable"] = calling_enable
            with col2:
                current_function = st.selectbox(
                    "Please Check Function",
                    function_name_list,
                    index=0,
                )
            if current_function:
                with st.form("ToolBoxes"):
                    google_credential = st.text_input("Google Credential", google_credential, key="google_credential")
                    search_key = st.text_input("Search Key", search_key, type="password", key="search_key")
                    function_description = ""
                    if google_tool == "Google Mail":
                        function_description = GetMailFuncallDescription(current_function)
                    elif google_tool == "Google Maps":
                        function_description = GetMapFuncallDescription(current_function)
                    elif google_tool == "Google Calendar":
                        function_description = GetCalendarFuncallDescription(current_function)
                    elif google_tool == "Google Drive":
                        function_description = GetStorageFuncallDescription(current_function)
                    elif google_tool == "Google Youtube":
                        function_description = GetYoutubeFuncallDescription(current_function)
                    elif google_tool == "Google Photo":
                        function_description = GetPhotoFuncallDescription(current_function)
                    st.text_input("Description", function_description, disabled=True)
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            toolboxes["Google ToolBoxes"]["credential"] = google_credential
                            toolboxes["Google ToolBoxes"]["search_key"] = search_key
                            r = api.save_google_toolboxes_config(google_toolboxes)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                api.save_current_running_config(current_running_config)
                                st.success("success save configuration for google toolboxes.")
                                st.toast("success save configuration for google toolboxes.", icon="✔")
            st.divider()

    st.session_state["current_page"] = "retrieval_agent_page"

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()