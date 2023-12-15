import streamlit as st
from WebUI.webui_pages.utils import *

training_devices_list = ["auto","cpu","gpu","mps"]
loadbits_list = ["32 bits","16 bits","8 bits"]

def tools_agent_page(api: ApiRequest, is_lite: bool = False):
    
    running_model = ""
    models_list = list(api.get_running_models())
    if len(models_list):
        running_model = models_list[0]
    webui_config = api.get_webui_config()
    current_vtot_model = api.get_vtot_model()
    current_ttov_model = ""#api.get_ttov_model()
    voicemodel = None
    
    if running_model == "":
        running_model = "None"
    st.caption(
            f"""<h1 style="font-size: 1.5em; text-align: center; color: #3498db;">Running LLM Model: {running_model}</h1>""",
            unsafe_allow_html=True,
        )
    tabretrieval, tabinterpreter, tabttov, tabvtot, tabimager, tabimageg, tabfunctions = st.tabs(["Retrieval", "Code Interpreter", "Text-to-Voice", "Voice-to-Text", "Image Recognition", "Image Generation", "Functions"])
    with tabretrieval:
        selected_kb = st.selectbox(
            "Knowledge Base:",
            ["[Create New...]"],
            index=0
        )

        if selected_kb == "[Create New...]":
            with st.form("Create Knowledge Base"):

                kb_name = st.text_input(
                    "Knowledge Base Name:",
                    key="kb_name",
                )
                kb_info = st.text_input(
                    "Introduction to the Knowledge Base",
                    key="kb_info",
                )

            #     cols = st.columns(2)
            #     vs_types = list(kbs_config.keys())
            #     vs_type = cols[0].selectbox(
            #         "向量库类型",
            #         vs_types,
            #         index=vs_types.index(DEFAULT_VS_TYPE),
            #         key="vs_type",
            #     )

            #     embed_models = list_embed_models()

            #     embed_model = cols[1].selectbox(
            #         "Embedding 模型",
            #         embed_models,
            #         index=embed_models.index(EMBEDDING_MODEL),
            #         key="embed_model",
            #     )

                submit_create_kb = st.form_submit_button(
                    "Create",
                    # disabled=not bool(kb_name),
                    use_container_width=True,
                )

            if submit_create_kb:
                pass
                # if not kb_name or not kb_name.strip():
                #     st.error(f"知识库名称不能为空！")
                # elif kb_name in kb_list:
                #     st.error(f"名为 {kb_name} 的知识库已经存在！")
                # else:
                #     ret = api.create_knowledge_base(
                #         knowledge_base_name=kb_name,
                #         vector_store_type=vs_type,
                #         embed_model=embed_model,
                #     )
                #     st.toast(ret.get("msg", " "))
                #     st.session_state["selected_kb_name"] = kb_name
                #     st.session_state["selected_kb_info"] = kb_info
                #     st.experimental_rerun()

    with tabinterpreter:
        pass

    with tabttov:
        ttovmodel = webui_config.get("ModelConfig").get("TtoVModel")
        ttovmodel_lists = [f"{key}" for key in ttovmodel]
        col1, col2 = st.columns(2)
        with col1:
            if current_ttov_model == "":
                index = 0
            else:
                index = ttovmodel_lists.index(current_vtot_model)
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
                if speechmodel == current_ttov_model:
                    with st.spinner(f"Release Model: {speechmodel}, Please do not perform any actions or refresh the page."):
                        pass
                else:
                    with st.spinner(f"Loading Model: {speechmodel}, Please do not perform any actions or refresh the page."):
                        pass
        with col2:
            if speechmodel is not None:
                pathstr = ttovmodel[speechmodel].get("path")
            else:
                pathstr = ""
            st.text_input("Local Path", pathstr)
            spsave_path = st.button(
                "Save Path",
                key="spsave_btn",
                use_container_width=True,
            )

        st.divider()
        config = ttovmodel[speechmodel]
        with st.form("speech_model"):
            devcol, bitcol = st.columns(2)
            with devcol:
                sdevice = config.get("device").lower()
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
                nloadbits = config.get("loadbits")
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
                config["device"] = predict_dev
                if nloadbits == "32 bits":
                    config["loadbits"] = 32
                elif nloadbits == "16 bits":
                    config["loadbits"] = 16
                else:
                    config["loadbits"] = 8
                with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                    pass

    with tabvtot:
        vtotmodel = webui_config.get("ModelConfig").get("VtoTModel")
        vtotmodel_lists = [f"{key}" for key in vtotmodel]
        col1, col2 = st.columns(2)
        with col1:
            if current_vtot_model == "":
                index = 0
            else:
                index = vtotmodel_lists.index(current_vtot_model)
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
                if voicemodel == current_vtot_model:
                    with st.spinner(f"Release Model: {voicemodel}, Please do not perform any actions or refresh the page."):
                        r = api.eject_voice_model(voicemodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_vtot_model = ""
                else:
                    with st.spinner(f"Loading Model: {voicemodel}, Please do not perform any actions or refresh the page."):
                        r = api.change_voice_model(current_vtot_model, voicemodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_vtot_model = voicemodel
        with col2:
            if voicemodel is not None:
                pathstr = vtotmodel[voicemodel].get("path")
            else:
                pathstr = ""
            st.text_input("Local Path", pathstr)
            vosave_path = st.button(
                "Save Path",
                key="vosave_btn",
                use_container_width=True,
            )

        st.divider()
        config = vtotmodel[voicemodel]
        if voicemodel == "whisper-large-v3" or voicemodel == "whisper-base" or voicemodel == "whisper-medium":
            with st.form("whisper_model"):
                devcol, bitcol = st.columns(2)
                with devcol:
                    sdevice = config.get("device").lower()
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
                    nloadbits = config.get("loadbits")
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
                    config["device"] = predict_dev
                    if nloadbits == "32 bits":
                        config["loadbits"] = 32
                    elif nloadbits == "16 bits":
                        config["loadbits"] = 16
                    else:
                        config["loadbits"] = 8
                    with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_vtot_model_config(voicemodel, config)
                        if msg := check_error_msg(r):
                            st.error(f"failed to save configuration for model {voicemodel}.")
                        elif msg := check_success_msg(r):
                            st.success(f"success save configuration for model {voicemodel}.")

        elif voicemodel == "faster-whisper-large-v3":
            with st.form("faster_whisper_model"):
                devcol, bitcol = st.columns(2)
                with devcol:
                    sdevice = config.get("device").lower()
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
                    nloadbits = config.get("loadbits")
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
                    config["device"] = predict_dev
                    if nloadbits == "32 bits":
                        config["loadbits"] = 32
                    elif nloadbits == "16 bits":
                        config["loadbits"] = 16
                    else:
                        config["loadbits"] = 8
                    with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                        r = api.save_vtot_model_config(voicemodel, config)
                        if msg := check_error_msg(r):
                            st.error(f"failed to save configuration for model {voicemodel}.")
                        elif msg := check_success_msg(r):
                            st.success(f"success save configuration for model {voicemodel}.")
        else:
            pass

    with tabimager:
        pass

    with tabimageg:
        pass

    with tabfunctions:
        pass

    st.session_state["current_page"] = "retrieval_agent_page"