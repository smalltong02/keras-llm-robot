import streamlit as st
from WebUI.webui_pages.utils import *
from WebUI.Server.knowledge_base.kb_service.base import get_kb_details

training_devices_list = ["auto","cpu","gpu","mps"]
loadbits_list = ["32 bits","16 bits","8 bits"]

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
        try:
            pass
            #kb_list = {x["kb_name"]: x for x in get_kb_details()}
        except Exception as e:
            st.error("Get Knowledge Base failed!")
            st.stop()
        #kb_names = list(kb_list.keys())

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
            #         "Vector Store Type",
            #         vs_types,
            #         index=vs_types.index(DEFAULT_VS_TYPE),
            #         key="vs_type",
            #     )

            #     embed_models = list_embed_models()

            #     embed_model = cols[1].selectbox(
            #         "Embedding Model",
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
                #     st.error(f"KnowledgeBase Name is None!")
                # elif kb_name in kb_list:
                #     st.error(f"The {kb_name} exist!")
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
                    with st.spinner(f"Release Model: {speechmodel}, Please do not perform any actions or refresh the page."):
                        r = api.eject_speech_model(speechmodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_speech_model = {"model": "", "speaker": ""}
                else:
                    with st.spinner(f"Loading Model: {speechmodel}, Please do not perform any actions or refresh the page."):
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
                    with st.spinner(f"Release Model: {voicemodel}, Please do not perform any actions or refresh the page."):
                        r = api.eject_voice_model(voicemodel)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            current_voice_model = ""
                else:
                    with st.spinner(f"Loading Model: {voicemodel}, Please do not perform any actions or refresh the page."):
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