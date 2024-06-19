from typing import Dict
import streamlit as st
from WebUI.webui_pages.utils import ApiRequest
from WebUI.Server.funcall.funcall import GetFuncallList, GetFuncallDescription
from WebUI.webui_pages.utils import check_error_msg, check_success_msg
from WebUI.configs import (ModelType, ModelSize, ModelSubType, GetModelType, GetModelConfig, GetModelSubType, GetOnlineProvider, GetOnlineModelList, GetModeList,
    glob_model_type_list, glob_model_size_list, glob_model_subtype_list, glob_assistant_name)
from WebUI.Server.knowledge_base.kb_service.base import get_kb_details

def last_stage_chat_solution(chat_solution: str, stage: int) -> bool:
    if chat_solution:
        if chat_solution == "Intelligent Customer Support":
            if stage == 4:
                return True
        elif chat_solution == "Language Translation and Localization":
            if stage == 3:
                return True
        elif chat_solution == "Virtual Personal Assistant":
            if stage == 4:
                return True
        else:
            return True
    return False

def ai_generator_page(api: ApiRequest, is_lite: bool = False):
    running_model = ""
    models_list = list(api.get_running_models())
    if len(models_list):
        running_model = models_list[0]
    webui_config = api.get_webui_config()
    current_voice_model = api.get_vtot_model()
    current_speech_model = api.get_ttov_model()
    #current_imagere_model = api.get_image_recognition_model()
    #current_imagegen_model = api.get_image_generation_model()
    #current_musicgen_model = api.get_music_generation_model()
    aigenerator_config = api.get_aigenerator_config()
    current_running_config = api.get_current_running_config()

    if running_model == "":
        running_model = "None"

    tabchatsolu, tabcreateivesolu = st.tabs(["Chat Solutions", "Creative Solutions"])
    with tabchatsolu:
        chat_solutions = aigenerator_config.get("Chat Solutions", {})
        running_chat_solution = st.session_state.get("current_chat_solution", {})
        chat_solutions_list = []
        for key, value in chat_solutions.items():
            if isinstance(value, dict):
                chat_solutions_list.append(key)
        if not running_chat_solution:
            index = 0
            running_chat_solution = {
                "stage": 0,
                "name": chat_solutions_list[0],
                "enable": False,
                "config": chat_solutions.get(chat_solutions_list[0], {}),
            }
        else:
            index = chat_solutions_list.index(running_chat_solution["name"])
        current_chat_solution = st.selectbox(
            "Please select Chat Solutions",
            chat_solutions_list,
            index=index,
            key="sel_chat_solutions",
        )
        print("current_chat_solution: ", current_chat_solution)
        if current_chat_solution != running_chat_solution["name"]:
            running_chat_solution = {
                "stage": 0,
                "name": current_chat_solution,
                "enable": False,
                "config": chat_solutions[current_chat_solution],
            }
        col1, col2, col3 = st.columns(3)
        with col1:
            prev_button = st.button(
            "Prev",
            use_container_width=True,
            )
            if prev_button and running_chat_solution["stage"] > 0:
                running_chat_solution["stage"] -= 1
                running_chat_solution["enable"] = False
        with col2:
            next_button = st.button(
            "Next",
            use_container_width=True,
            )
            print("last_stage_chat_solution: ", last_stage_chat_solution(current_chat_solution, running_chat_solution["stage"]))
            if next_button and not last_stage_chat_solution(current_chat_solution, running_chat_solution["stage"]):
                print("next: ", running_chat_solution["stage"], "  +1: ", running_chat_solution["stage"] + 1)
                running_chat_solution["stage"] += 1
        with col3:
            clear_button = st.button(
            "Clear",
            use_container_width=True,
            )
            if clear_button:
                running_chat_solution = {
                "stage": 0,
                "name": current_chat_solution,
                "enable": False,
                "config": chat_solutions[current_chat_solution],
                }
        st.session_state["current_chat_solution"] = running_chat_solution
        st.divider()
        description = chat_solutions[current_chat_solution]["descriptions"]
        st.text_input("Solution Description", current_chat_solution + "....", help=description, disabled=True)
        if running_chat_solution["name"] == "Intelligent Customer Support":
            if running_chat_solution["enable"]:
                st.markdown("**Step 4: Release Model and Configuration**")
                eject_error = False
                eject_cfg_button = st.button(
                    "Eject Configuration",
                    use_container_width=True,
                )
                if eject_cfg_button:
                    running_chat_solution["enable"] = False
                    llm_model = running_chat_solution["config"]["llm_model"]
                    with st.spinner(f"Release LLM Model: `{llm_model}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_llm_model(llm_model)
                        if msg := check_error_msg(r):
                            st.error(msg)
                            st.toast(msg, icon="✖")
                            eject_error = True
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["voice"]["enable"]:
                        voice_model = running_chat_solution["config"]["voice"]["model"]
                        with st.spinner(f"Release Voice Model: `{voice_model}`, Please do not perform any actions or refresh the page."):
                            r = api.eject_voice_model(voice_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                                load_error = True
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["speech"]["enable"]:
                        speech_model = running_chat_solution["config"]["speech"]["model"]
                        with st.spinner(f"Release Speech Model: `{speech_model}`, Please do not perform any actions or refresh the page."):
                            r = api.eject_speech_model(speech_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                                load_error = True
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["function_calling"]:
                        with st.spinner("Release Function Call, Please do not perform any actions or refresh the page."):
                            st.success("success release function calling.")
                            st.toast("success release function calling.", icon="✔")
                    if not eject_error:
                        api.save_current_running_config()
                        st.success("Release all configurations successfully!")
                        st.toast("Release all configurations successfully!", icon="✔")
                        st.session_state["current_chat_solution"] = {}

            else:   
                if running_chat_solution["stage"] == 1:
                    st.divider()
                    st.markdown("**Step 1:  Please Select the LLM Model and add product description**")
                    col1, col2 = st.columns(2)
                    current_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
                    with col1:
                        modeltype = st.selectbox(
                            "Please Select Model Type",
                            glob_model_type_list,
                            index=0,
                            key="sel_model_type",
                        )
                        current_model["mtype"] = GetModelType(modeltype)
                        type_index = current_model["mtype"].value - 1
                    with col2:
                        if type_index == ModelType.Local.value - 1 or type_index == ModelType.Special.value - 1 or type_index == ModelType.Code.value - 1:
                            modelsize = st.selectbox(
                                "Please Select Model Size",
                                glob_model_size_list,
                                index=0,
                                key="sel_model_size",
                            )
                            size_index = glob_model_size_list.index(modelsize)
                            current_model["msubtype"] = ModelSubType.Unknown
                            current_model["msize"] = ModelSize(size_index + 1)
                        elif type_index == ModelType.Multimodal.value - 1:
                            submodel = st.selectbox(
                                    "Please Select Sub Model",
                                    glob_model_subtype_list,
                                    index=0,
                                    key="sel_sub_model",
                                )
                            current_model["msubtype"] = GetModelSubType(submodel)
                            current_model["msize"] = ModelSize.Unknown
                        elif type_index == ModelType.Online.value - 1:
                            online_model_list = GetOnlineProvider(webui_config)
                            onlinemodel = st.selectbox(
                                    "Please Select Online Provider",
                                    online_model_list,
                                    index=0,
                                    key="sel_online_provider",
                                )
                            current_model["msubtype"] = ModelSubType.Unknown
                            current_model["msize"] = ModelSize.Unknown
                            size_index = online_model_list.index(onlinemodel)
                            online_model_list = GetOnlineModelList(webui_config, onlinemodel)
                    if type_index != ModelType.Online.value - 1:
                        model_list = GetModeList(webui_config, current_model)
                    else:
                        model_list = online_model_list
                    st.divider()
                    col1, col2 = st.columns(2)
                    model_name = ""
                    with col1:
                        model_name = st.selectbox(
                            "Please Select Model Name",
                            model_list,
                            index=0,
                            key="sel_model_name",
                        )
                        current_model["mname"] = model_name
                        if current_model["mname"] is not None:
                            current_model["config"] = GetModelConfig(webui_config, current_model)
                        else:
                            current_model["config"] = {}

                        if current_model["mtype"] != ModelType.Online:
                            pathstr = current_model["config"].get("path")
                        else:
                            pathstr = ""

                    with col2:
                        pathstr = st.text_input("Local Path", pathstr, disabled=True)
                    st.divider()
                    product_description = st.text_input("Product Description", value="", placeholder="[Please describe your product]")
                    running_chat_solution["config"]["llm_model"] = model_name
                    running_chat_solution["config"]["description"] = product_description
                    print("chat_solution-1: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution

                if running_chat_solution["stage"] == 2:
                    st.markdown("**Step 2: To decide whether to use Voice and Speech Model**")
                    voice_enable = st.checkbox('Enable Voice', value=False)
                    running_chat_solution["config"]["voice"]["enable"] = False
                    running_chat_solution["config"]["voice"]["model"] = ""
                    current_running_config["voice"]["name"] = ""
                    current_running_config["voice"]["language"] = ""
                    if voice_enable:
                        vtotmodel = webui_config.get("ModelConfig").get("VtoTModel")
                        vtotmodel_lists = [f"{key}" for key in vtotmodel]
                        col1, col2 = st.columns(2)
                        with col1:
                            voicemodel = st.selectbox(
                                    "Please Select Voice Model",
                                    vtotmodel_lists,
                                    index=0,
                                    key="sel_voice_model",
                                )
                            voice_modelconfig = vtotmodel[voicemodel]
                            language_code_list = voice_modelconfig.get("language_code", [])
                            language_code = st.selectbox(
                                "Please Select language",
                                language_code_list,
                                index=0,
                            )
                        with col2:
                            if voice_modelconfig["type"] == "local":
                                if voicemodel is not None:
                                    pathstr = vtotmodel[voicemodel].get("path")
                                else:
                                    pathstr = ""
                                st.text_input("Local Path", pathstr, key="vo_local_path", disabled=True)
                            elif voice_modelconfig["type"] == "cloud":
                                pathstr = voice_modelconfig.get("path")
                                st.text_input("Cloud Path", pathstr, key="vo_cloud_path", disabled=True)
                        running_chat_solution["config"]["voice"]["enable"] = True
                        running_chat_solution["config"]["voice"]["model"] = voicemodel
                        running_chat_solution["config"]["voice"]["language"] = [language_code]
                        current_running_config["voice"]["name"] = voicemodel
                        current_running_config["voice"]["language"] = language_code
                        voice_modelconfig["language"] = [language_code]
                        api.save_vtot_model_config(voicemodel, voice_modelconfig)
                    
                    st.divider()
                    speech_enable = st.checkbox('Enable Speech', value=False)
                    running_chat_solution["config"]["speech"]["enable"] = False
                    running_chat_solution["config"]["speech"]["model"] = ""
                    running_chat_solution["config"]["speech"]["speaker"] = ""
                    current_running_config["speech"]["name"] = ""
                    current_running_config["speech"]["speaker"] = ""
                    if speech_enable:
                        ttovmodel = webui_config.get("ModelConfig").get("TtoVModel")
                        ttovmodel_lists = [f"{key}" for key in ttovmodel]
                        col1, col2 = st.columns(2)
                        with col1:
                            speechmodel = st.selectbox(
                                    "Please Select Speech Model",
                                    ttovmodel_lists,
                                    index=0,
                                    key="sel_speech_model",
                                )
                        modelconfig = ttovmodel[speechmodel]
                        synthesisconfig = modelconfig["synthesis"]
                        with col2:
                            if modelconfig["type"] == "local":
                                pathstr = modelconfig.get("path")
                                st.text_input("Local Path", pathstr, key="sp_local_path", disabled=True)
                            elif modelconfig["type"] == "cloud":
                                pathstr = modelconfig.get("path")
                                st.text_input("Cloud Path", pathstr, key="sp_cloud_path", disabled=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            templates_list = [modelconfig.get("CloudTemplates", "")]
                            _ = st.selectbox(
                                "Please Select templates",
                                templates_list,
                                index=0,
                                key="sel_templates",
                                disabled=True
                            )
                            lang_index = 0
                            language_list = list(synthesisconfig.keys())
                            language = st.selectbox(
                                "Please Select language",
                                language_list,
                                index=lang_index,
                                key="sel_language",
                            )
                            
                        with col2:
                            sex_index = 0
                            sex_list = list(synthesisconfig[language].keys())
                            sex = st.selectbox(
                                "Please Select Sex",
                                sex_list,
                                index=sex_index,
                                key="sel_sex",
                            )
                            speaker_index = 0
                            speaker_list = synthesisconfig[language][sex]
                            speaker = st.selectbox(
                                "Please Select Speaker",
                                speaker_list,
                                index=speaker_index,
                                key="sel_speaker",
                            )
                            if speaker:
                                st.session_state["speaker"] = speaker
                        running_chat_solution["config"]["speech"]["enable"] = True
                        running_chat_solution["config"]["speech"]["model"] = speechmodel
                        running_chat_solution["config"]["speech"]["speaker"] = speaker
                        current_running_config["speech"]["name"] = speechmodel
                        current_running_config["speech"]["speaker"] = speaker

                    st.divider()
                    roleplayer = running_chat_solution["config"]["roleplayer"]
                    st.text_input("Roleplayer:", roleplayer, key="roleplayer", disabled=True)
                    print("chat_solution-2: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution
                if running_chat_solution["stage"] == 3:
                    st.markdown("**Step 3: To decide whether to use Knowledge Base, Search Engine and Function Calling**")
                    knowledge_enable = st.checkbox('Knowledge Base', value=False)
                    running_chat_solution["config"]["knowledge_base"]["enable"] = False
                    running_chat_solution["config"]["knowledge_base"]["name"] = ""
                    if knowledge_enable:
                        kb_list = {}
                        try:
                            kb_details = get_kb_details()
                            if len(kb_details):
                                kb_list = {x["kb_name"]: x for x in kb_details}
                        except Exception as _:
                            st.error("Get Knowledge Base failed!")
                            st.stop()
                        kb_names = list(kb_list.keys())
                        selected_kb = st.selectbox(
                            "Knowledge Base:",
                            kb_names,
                            index=0
                        )
                        running_chat_solution["config"]["knowledge_base"]["enable"] = True
                        running_chat_solution["config"]["knowledge_base"]["name"] = selected_kb
                    st.divider()
                    search_enable = st.checkbox('Search Engine', value=False)
                    running_chat_solution["config"]["search_engine"]["enable"] = False
                    running_chat_solution["config"]["search_engine"]["name"] = ""
                    if search_enable:
                        search_engine_list = []
                        searchengine = webui_config.get("SearchEngine")
                        for key, value in searchengine.items():
                            if isinstance(value, dict):
                                search_engine_list.append(key)

                        search_engine = st.selectbox(
                            "Please select Search Engine",
                            search_engine_list,
                            index=0,
                        )
                        if search_engine:
                            running_chat_solution["config"]["search_engine"]["enable"] = True
                            running_chat_solution["config"]["search_engine"]["name"] = search_engine
                    st.divider()
                    function_calling_enable = st.checkbox('Function Calling', value=False)
                    running_chat_solution["config"]["function_calling"] = False
                    if function_calling_enable:
                        function_name_list = GetFuncallList()
                        current_function = st.selectbox(
                            "Please Check Function",
                            function_name_list,
                            index=0,
                        )
                        description = GetFuncallDescription(current_function)
                        st.text_input("Description", description, disabled=True)
                        running_chat_solution["config"]["function_calling"] = True
                    print("chat_solution-3: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution
                if running_chat_solution["stage"] == 4:
                    st.markdown("**Step 4: Loading Model and Configuration**")
                    load_cfg_button = st.button(
                        "Load Configuration",
                        use_container_width=True,
                    )
                    load_error = False
                    if load_cfg_button:
                        with st.spinner("Checking all configurations...."):
                            api.save_current_running_config()
                            # first check configuration
                            if not running_chat_solution["config"]["llm_model"]:
                                st.error("LLM Model is not configured!")
                                st.toast("LLM Model is not configured!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["voice"]["enable"] and not running_chat_solution["config"]["voice"]["model"]:
                                st.error("Voice model configuration error!")
                                st.toast("Voice model configuration error!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["speech"]["enable"] and not running_chat_solution["config"]["speech"]["model"]:
                                st.error("Speech model configuration error!")
                                st.toast("Speech model configuration error!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["knowledge_base"]["enable"] and not running_chat_solution["config"]["knowledge_base"]["name"]:
                                st.error("Knowledge Base configuration error!")
                                st.toast("Knowledge Base configuration error!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["search_engine"]["enable"] and not running_chat_solution["config"]["search_engine"]["name"]:
                                st.error("Search Engine configuration error!")
                                st.toast("Search Engine configuration error!", icon="✖")
                                load_error = True
                        if not load_error:
                            llm_model = running_chat_solution["config"]["llm_model"]
                            with st.spinner(f"Loading LLM Model: `{llm_model}`, Please do not perform any actions or refresh the page."):
                                models_list = list(api.get_running_models())
                                running_model = ""
                                if len(models_list):
                                    running_model = models_list[0]
                                r = api.change_llm_model(running_model, llm_model)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["voice"]["enable"]:
                            voice_model = running_chat_solution["config"]["voice"]["model"]
                            language = running_chat_solution["config"]["voice"]["language"]
                            language = language[0] if language else "en-US"
                            with st.spinner(f"Loading Voice Model: `{voice_model}`, Please do not perform any actions or refresh the page."):
                                current_voice_model = api.get_vtot_model()
                                r = api.change_voice_model(current_voice_model, voice_model)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    current_running_config["voice"]["name"] = voice_model
                                    current_running_config["voice"]["language"] = language
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["speech"]["enable"]:
                            speech_model = running_chat_solution["config"]["speech"]["model"]
                            speaker = running_chat_solution["config"]["speech"]["speaker"]
                            with st.spinner(f"Loading Speech Model: `{speech_model}`, Please do not perform any actions or refresh the page."):
                                current_speech_model = api.get_ttov_model()
                                current_speech_model = current_speech_model.get("model", "")
                                r = api.change_speech_model(current_speech_model, speech_model, speaker)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    current_running_config["speech"]["name"] = speech_model
                                    current_running_config["speech"]["speaker"] = speaker
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["knowledge_base"]["enable"]:
                            knowledge_base = running_chat_solution["config"]["knowledge_base"]["name"]
                            current_running_config["knowledge_base"]["name"] = knowledge_base
                            with st.spinner(f"Loading Knowledge Base: `{knowledge_base}`, Please do not perform any actions or refresh the page."):
                                st.success(f"Load Knowledge Base `{knowledge_base}` successfully!")
                                st.toast(f"Load Knowledge Base `{knowledge_base}` successfully!", icon="✔")
                        if not load_error and running_chat_solution["config"]["search_engine"]["enable"]:
                            search_engine = running_chat_solution["config"]["search_engine"]["name"]
                            current_running_config["search_engine"]["name"] = search_engine
                            with st.spinner(f"Enabling Search Engine: `{search_engine}`, Please do not perform any actions or refresh the page."):
                                st.success(f"Enable Search Engine `{search_engine}` successfully!")
                                st.toast(f"Enable Search Engine `{search_engine}` successfully!", icon="✔")
                        if not load_error and running_chat_solution["config"]["function_calling"]:
                            with st.spinner("Enabling Function Calling, Please do not perform any actions or refresh the page."):
                                current_running_config["normal_calling"]["enable"] = running_chat_solution["config"]["function_calling"]
                                api.save_current_running_config(current_running_config)
                                st.success("Enable Function Calling successfully!")
                                st.toast("success save configuration for function calling.", icon="✔")
                        if not load_error:
                            name = running_chat_solution["name"]
                            description = running_chat_solution["config"]["description"]
                            roleplayer = running_chat_solution["config"]["roleplayer"]
                            current_running_config["role_player"] = {"name": roleplayer, "language": "english"}
                            current_running_config["chat_solution"] = {"name": name, "description": description}
                            api.save_current_running_config(current_running_config)
                            st.success("Load all configurations successfully!")
                            st.toast("Load all configurations successfully!", icon="✔")
                            running_chat_solution["enable"] = True
                    print("chat_solution-4: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution

        elif running_chat_solution["name"] == "Language Translation and Localization":
            if running_chat_solution["enable"]:
                st.markdown("**Step 3: Release Model and Configuration**")
                eject_error = False
                eject_cfg_button = st.button(
                    "Eject Configuration",
                    use_container_width=True,
                )
                if eject_cfg_button:
                    running_chat_solution["enable"] = False
                    llm_model = running_chat_solution["config"]["llm_model"]
                    with st.spinner(f"Release LLM Model: `{llm_model}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_llm_model(llm_model)
                        if msg := check_error_msg(r):
                            st.error(msg)
                            st.toast(msg, icon="✖")
                            eject_error = True
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["voice"]["enable"]:
                        voice_model = running_chat_solution["config"]["voice"]["model"]
                        with st.spinner(f"Release Voice Model: `{voice_model}`, Please do not perform any actions or refresh the page."):
                            r = api.eject_voice_model(voice_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                                load_error = True
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["speech"]["enable"]:
                        speech_model = running_chat_solution["config"]["speech"]["model"]
                        with st.spinner(f"Release Speech Model: `{speech_model}`, Please do not perform any actions or refresh the page."):
                            r = api.eject_speech_model(speech_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                                load_error = True
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["function_calling"]:
                        with st.spinner("Release Function Call, Please do not perform any actions or refresh the page."):
                            st.success("success release function calling.")
                            st.toast("success release function calling.", icon="✔")
                    if not eject_error:
                        api.save_current_running_config()
                        st.success("Release all configurations successfully!")
                        st.toast("Release all configurations successfully!", icon="✔")
                        st.session_state["current_chat_solution"] = {}

            else:   
                if running_chat_solution["stage"] == 1:
                    st.divider()
                    st.markdown("**Step 1:  Please Select the LLM Model**")
                    col1, col2 = st.columns(2)
                    current_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
                    with col1:
                        modeltype = st.selectbox(
                            "Please Select Model Type",
                            glob_model_type_list,
                            index=0,
                            key="sel_model_type",
                        )
                        current_model["mtype"] = GetModelType(modeltype)
                        type_index = current_model["mtype"].value - 1
                    with col2:
                        if type_index == ModelType.Local.value - 1 or type_index == ModelType.Special.value - 1 or type_index == ModelType.Code.value - 1:
                            modelsize = st.selectbox(
                                "Please Select Model Size",
                                glob_model_size_list,
                                index=0,
                                key="sel_model_size",
                            )
                            size_index = glob_model_size_list.index(modelsize)
                            current_model["msubtype"] = ModelSubType.Unknown
                            current_model["msize"] = ModelSize(size_index + 1)
                        elif type_index == ModelType.Multimodal.value - 1:
                            submodel = st.selectbox(
                                    "Please Select Sub Model",
                                    glob_model_subtype_list,
                                    index=0,
                                    key="sel_sub_model",
                                )
                            current_model["msubtype"] = GetModelSubType(submodel)
                            current_model["msize"] = ModelSize.Unknown
                        elif type_index == ModelType.Online.value - 1:
                            online_model_list = GetOnlineProvider(webui_config)
                            onlinemodel = st.selectbox(
                                    "Please Select Online Provider",
                                    online_model_list,
                                    index=0,
                                    key="sel_online_provider",
                                )
                            current_model["msubtype"] = ModelSubType.Unknown
                            current_model["msize"] = ModelSize.Unknown
                            size_index = online_model_list.index(onlinemodel)
                            online_model_list = GetOnlineModelList(webui_config, onlinemodel)
                    if type_index != ModelType.Online.value - 1:
                        model_list = GetModeList(webui_config, current_model)
                    else:
                        model_list = online_model_list
                    st.divider()
                    col1, col2 = st.columns(2)
                    model_name = ""
                    with col1:
                        model_name = st.selectbox(
                            "Please Select Model Name",
                            model_list,
                            index=0,
                            key="sel_model_name",
                        )
                        current_model["mname"] = model_name
                        if current_model["mname"] is not None:
                            current_model["config"] = GetModelConfig(webui_config, current_model)
                        else:
                            current_model["config"] = {}

                        if current_model["mtype"] != ModelType.Online:
                            pathstr = current_model["config"].get("path")
                        else:
                            pathstr = ""

                    with col2:
                        pathstr = st.text_input("Local Path", pathstr, disabled=True)
                    running_chat_solution["config"]["llm_model"] = model_name
                    print("chat_solution-1: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution

                if running_chat_solution["stage"] == 2:
                    st.markdown("**Step 2: To decide Voice and Speech Model**")
                    voice_enable = st.checkbox('Enable Voice', value=True)
                    running_chat_solution["config"]["voice"]["enable"] = False
                    running_chat_solution["config"]["voice"]["model"] = ""
                    current_running_config["voice"]["name"] = ""
                    current_running_config["voice"]["language"] = ""
                    if voice_enable:
                        vtotmodel = webui_config.get("ModelConfig").get("VtoTModel")
                        vtotmodel_lists = [f"{key}" for key in vtotmodel]
                        col1, col2 = st.columns(2)
                        with col1:
                            voicemodel = st.selectbox(
                                    "Please Select Voice Model",
                                    vtotmodel_lists,
                                    index=0,
                                    key="sel_voice_model",
                                )
                            voice_modelconfig = vtotmodel[voicemodel]
                            language_code_list = voice_modelconfig.get("language_code", [])
                            language_code = st.selectbox(
                                "Please Select language",
                                language_code_list,
                                index=0,
                            )
                        with col2:
                            if voice_modelconfig["type"] == "local":
                                if voicemodel is not None:
                                    pathstr = vtotmodel[voicemodel].get("path")
                                else:
                                    pathstr = ""
                                st.text_input("Local Path", pathstr, key="vo_local_path", disabled=True)
                            elif voice_modelconfig["type"] == "cloud":
                                pathstr = voice_modelconfig.get("path")
                                st.text_input("Cloud Path", pathstr, key="vo_cloud_path", disabled=True)
                        running_chat_solution["config"]["voice"]["enable"] = True
                        running_chat_solution["config"]["voice"]["model"] = voicemodel
                        running_chat_solution["config"]["voice"]["language"] = [language_code]
                        current_running_config["voice"]["name"] = voicemodel
                        current_running_config["voice"]["language"] = language_code
                        voice_modelconfig["language"] = [language_code]
                        api.save_vtot_model_config(voicemodel, voice_modelconfig)
                    
                    st.divider()
                    speech_enable = st.checkbox('Enable Speech', value=True)
                    running_chat_solution["config"]["speech"]["enable"] = False
                    running_chat_solution["config"]["speech"]["model"] = ""
                    running_chat_solution["config"]["speech"]["speaker"] = ""
                    current_running_config["speech"]["name"] = ""
                    current_running_config["speech"]["speaker"] = ""
                    if speech_enable:
                        ttovmodel = webui_config.get("ModelConfig").get("TtoVModel")
                        ttovmodel_lists = [f"{key}" for key in ttovmodel]
                        col1, col2 = st.columns(2)
                        with col1:
                            speechmodel = st.selectbox(
                                    "Please Select Speech Model",
                                    ttovmodel_lists,
                                    index=0,
                                    key="sel_speech_model",
                                )
                        modelconfig = ttovmodel[speechmodel]
                        synthesisconfig = modelconfig["synthesis"]
                        with col2:
                            if modelconfig["type"] == "local":
                                pathstr = modelconfig.get("path")
                                st.text_input("Local Path", pathstr, key="sp_local_path", disabled=True)
                            elif modelconfig["type"] == "cloud":
                                pathstr = modelconfig.get("path")
                                st.text_input("Cloud Path", pathstr, key="sp_cloud_path", disabled=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            templates_list = [modelconfig.get("CloudTemplates", "")]
                            _ = st.selectbox(
                                "Please Select templates",
                                templates_list,
                                index=0,
                                key="sel_templates",
                                disabled=True
                            )
                            lang_index = 0
                            language_list = list(synthesisconfig.keys())
                            language = st.selectbox(
                                "Please Select language",
                                language_list,
                                index=lang_index,
                                key="sel_language",
                            )
                            
                        with col2:
                            sex_index = 0
                            sex_list = list(synthesisconfig[language].keys())
                            sex = st.selectbox(
                                "Please Select Sex",
                                sex_list,
                                index=sex_index,
                                key="sel_sex",
                            )
                            speaker_index = 0
                            speaker_list = synthesisconfig[language][sex]
                            speaker = st.selectbox(
                                "Please Select Speaker",
                                speaker_list,
                                index=speaker_index,
                                key="sel_speaker",
                            )
                            if speaker:
                                st.session_state["speaker"] = speaker
                        running_chat_solution["config"]["speech"]["enable"] = True
                        running_chat_solution["config"]["speech"]["model"] = speechmodel
                        running_chat_solution["config"]["speech"]["speaker"] = speaker
                        current_running_config["speech"]["name"] = speechmodel
                        current_running_config["speech"]["speaker"] = speaker
                        if running_chat_solution["config"]["voice"]["enable"] and speaker:
                            if speechmodel == "OpenAISpeechService":
                                language_code = "en-US"
                            else:
                                result = speaker.split('-', 2)[:2]
                                language_code = '-'.join(result)
                            if language_code not in running_chat_solution["config"]["voice"]["language"]:
                                running_chat_solution["config"]["voice"]["language"].append(language_code)
                            if language_code not in voice_modelconfig["language"]:
                                voice_modelconfig["language"].append(language_code)
                                api.save_vtot_model_config(voicemodel, voice_modelconfig)

                    st.divider()
                    roleplayer = running_chat_solution["config"]["roleplayer"]
                    st.text_input("Roleplayer:", roleplayer, key="roleplayer", disabled=True)
                    print("chat_solution-2: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution
                if running_chat_solution["stage"] == 3:
                    st.markdown("**Step 3: Loading Model and Configuration**")
                    load_cfg_button = st.button(
                        "Load Configuration",
                        use_container_width=True,
                    )
                    load_error = False
                    if load_cfg_button:
                        with st.spinner("Checking all configurations...."):
                            api.save_current_running_config()
                            # first check configuration
                            if not running_chat_solution["config"]["llm_model"]:
                                st.error("LLM Model is not configured!")
                                st.toast("LLM Model is not configured!", icon="✖")
                                load_error = True
                            if not running_chat_solution["config"]["voice"]["enable"]:
                                st.error("Voice model must be enabled!")
                                st.toast("Voice model must be enabled!", icon="✖")
                                load_error = True
                            if not running_chat_solution["config"]["speech"]["enable"]:
                                st.error("Speech model must be enabled!")
                                st.toast("Speech model must be enabled!", icon="✖")
                                load_error = True
                        if not load_error:
                            llm_model = running_chat_solution["config"]["llm_model"]
                            with st.spinner(f"Loading LLM Model: `{llm_model}`, Please do not perform any actions or refresh the page."):
                                models_list = list(api.get_running_models())
                                running_model = ""
                                if len(models_list):
                                    running_model = models_list[0]
                                r = api.change_llm_model(running_model, llm_model)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["voice"]["enable"]:
                            voice_model = running_chat_solution["config"]["voice"]["model"]
                            language = running_chat_solution["config"]["voice"]["language"]
                            language = language[0] if language else "en-US"
                            with st.spinner(f"Loading Voice Model: `{voice_model}`, Please do not perform any actions or refresh the page."):
                                current_voice_model = api.get_vtot_model()
                                r = api.change_voice_model(current_voice_model, voice_model)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    current_running_config["voice"]["name"] = voice_model
                                    current_running_config["voice"]["language"] = language
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["speech"]["enable"]:
                            speech_model = running_chat_solution["config"]["speech"]["model"]
                            speaker = running_chat_solution["config"]["speech"]["speaker"]
                            with st.spinner(f"Loading Speech Model: `{speech_model}`, Please do not perform any actions or refresh the page."):
                                current_speech_model = api.get_ttov_model()
                                current_speech_model = current_speech_model.get("model", "")
                                r = api.change_speech_model(current_speech_model, speech_model, speaker)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    current_running_config["speech"]["name"] = speech_model
                                    current_running_config["speech"]["speaker"] = speaker
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error:
                            current_running_config["chat_solution"]["name"] = running_chat_solution["name"]
                            roleplayer = running_chat_solution["config"]["roleplayer"]
                            current_running_config["role_player"] = {"name": roleplayer, "language": "english"}
                            api.save_current_running_config(current_running_config)
                            st.success("Load all configurations successfully!")
                            st.toast("Load all configurations successfully!", icon="✔")
                            running_chat_solution["enable"] = True
                    print("chat_solution-4: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution

        elif running_chat_solution["name"] == "Virtual Personal Assistant":
            if running_chat_solution["enable"]:
                st.markdown("**Step 4: Release Model and Configuration**")
                eject_error = False
                eject_cfg_button = st.button(
                    "Eject Configuration",
                    use_container_width=True,
                )
                if eject_cfg_button:
                    running_chat_solution["enable"] = False
                    llm_model = running_chat_solution["config"]["llm_model"]
                    with st.spinner(f"Release LLM Model: `{llm_model}`, Please do not perform any actions or refresh the page."):
                        r = api.eject_llm_model(llm_model)
                        if msg := check_error_msg(r):
                            st.error(msg)
                            st.toast(msg, icon="✖")
                            eject_error = True
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["voice"]["enable"]:
                        voice_model = running_chat_solution["config"]["voice"]["model"]
                        with st.spinner(f"Release Voice Model: `{voice_model}`, Please do not perform any actions or refresh the page."):
                            r = api.eject_voice_model(voice_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                                load_error = True
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["speech"]["enable"]:
                        speech_model = running_chat_solution["config"]["speech"]["model"]
                        with st.spinner(f"Release Speech Model: `{speech_model}`, Please do not perform any actions or refresh the page."):
                            r = api.eject_speech_model(speech_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                                st.toast(msg, icon="✖")
                                load_error = True
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                st.toast(msg, icon="✔")
                    if not eject_error and running_chat_solution["config"]["toolboxes"]:
                        with st.spinner("Release ToolBoxes, Please do not perform any actions or refresh the page."):
                            st.success("Release ToolBoxes successfully!")
                            st.toast("Release ToolBoxes successfully!", icon="✔")
                    if not eject_error and running_chat_solution["config"]["function_calling"]:
                        with st.spinner("Release Function Call, Please do not perform any actions or refresh the page."):
                            st.success("success release function calling.")
                            st.toast("success release function calling.", icon="✔")
                    if not eject_error:
                        api.save_current_running_config()
                        st.success("Release all configurations successfully!")
                        st.toast("Release all configurations successfully!", icon="✔")
                        st.session_state["current_chat_solution"] = {}

            else:   
                if running_chat_solution["stage"] == 1:
                    st.divider()
                    st.markdown("**Step 1:  Please Select the LLM Model and Assistant name**")
                    col1, col2 = st.columns(2)
                    current_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
                    with col1:
                        modeltype = st.selectbox(
                            "Please Select Model Type",
                            glob_model_type_list,
                            index=0,
                            key="sel_model_type",
                        )
                        current_model["mtype"] = GetModelType(modeltype)
                        type_index = current_model["mtype"].value - 1
                    with col2:
                        if type_index == ModelType.Local.value - 1 or type_index == ModelType.Special.value - 1 or type_index == ModelType.Code.value - 1:
                            modelsize = st.selectbox(
                                "Please Select Model Size",
                                glob_model_size_list,
                                index=0,
                                key="sel_model_size",
                            )
                            size_index = glob_model_size_list.index(modelsize)
                            current_model["msubtype"] = ModelSubType.Unknown
                            current_model["msize"] = ModelSize(size_index + 1)
                        elif type_index == ModelType.Multimodal.value - 1:
                            submodel = st.selectbox(
                                    "Please Select Sub Model",
                                    glob_model_subtype_list,
                                    index=0,
                                    key="sel_sub_model",
                                )
                            current_model["msubtype"] = GetModelSubType(submodel)
                            current_model["msize"] = ModelSize.Unknown
                        elif type_index == ModelType.Online.value - 1:
                            online_model_list = GetOnlineProvider(webui_config)
                            onlinemodel = st.selectbox(
                                    "Please Select Online Provider",
                                    online_model_list,
                                    index=0,
                                    key="sel_online_provider",
                                )
                            current_model["msubtype"] = ModelSubType.Unknown
                            current_model["msize"] = ModelSize.Unknown
                            size_index = online_model_list.index(onlinemodel)
                            online_model_list = GetOnlineModelList(webui_config, onlinemodel)
                    if type_index != ModelType.Online.value - 1:
                        model_list = GetModeList(webui_config, current_model)
                    else:
                        model_list = online_model_list
                    st.divider()
                    col1, col2 = st.columns(2)
                    model_name = ""
                    with col1:
                        model_name = st.selectbox(
                            "Please Select Model Name",
                            model_list,
                            index=0,
                            key="sel_model_name",
                        )
                        current_model["mname"] = model_name
                        if current_model["mname"] is not None:
                            current_model["config"] = GetModelConfig(webui_config, current_model)
                        else:
                            current_model["config"] = {}

                        if current_model["mtype"] != ModelType.Online:
                            pathstr = current_model["config"].get("path")
                        else:
                            pathstr = ""

                    with col2:
                        pathstr = st.text_input("Local Path", pathstr, disabled=True)
                    st.divider()
                    assistant_name = st.selectbox(
                            "Please Select Assistant Name",
                            glob_assistant_name,
                            index=0,
                            key="sel_assistant_name",
                        )
                    running_chat_solution["config"]["llm_model"] = model_name
                    running_chat_solution["config"]["assistant_name"] = assistant_name
                    print("chat_solution-1: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution

                if running_chat_solution["stage"] == 2:
                    st.markdown("**Step 2: To decide whether to use Voice and Speech Model**")
                    voice_enable = st.checkbox('Enable Voice', value=False)
                    running_chat_solution["config"]["voice"]["enable"] = False
                    running_chat_solution["config"]["voice"]["model"] = ""
                    if voice_enable:
                        vtotmodel = webui_config.get("ModelConfig").get("VtoTModel")
                        vtotmodel_lists = [f"{key}" for key in vtotmodel]
                        col1, col2 = st.columns(2)
                        with col1:
                            voicemodel = st.selectbox(
                                    "Please Select Voice Model",
                                    vtotmodel_lists,
                                    index=0,
                                    key="sel_voice_model",
                                )
                            voice_modelconfig = vtotmodel[voicemodel]
                            language_code_list = voice_modelconfig.get("language_code", [])
                            language_code = st.selectbox(
                                "Please Select language",
                                language_code_list,
                                index=0,
                            )
                        with col2:
                            if voice_modelconfig["type"] == "local":
                                if voicemodel is not None:
                                    pathstr = vtotmodel[voicemodel].get("path")
                                else:
                                    pathstr = ""
                                st.text_input("Local Path", pathstr, key="vo_local_path", disabled=True)
                            elif voice_modelconfig["type"] == "cloud":
                                pathstr = voice_modelconfig.get("path")
                                st.text_input("Cloud Path", pathstr, key="vo_cloud_path", disabled=True)
                        running_chat_solution["config"]["voice"]["enable"] = True
                        running_chat_solution["config"]["voice"]["model"] = voicemodel
                        running_chat_solution["config"]["voice"]["language"] = [language_code]
                        current_running_config["voice"]["name"] = voicemodel
                        current_running_config["voice"]["language"] = language_code
                        voice_modelconfig["language"] = [language_code]
                        api.save_vtot_model_config(voicemodel, voice_modelconfig)
                    
                    st.divider()
                    speech_enable = st.checkbox('Enable Speech', value=False)
                    running_chat_solution["config"]["speech"]["enable"] = False
                    running_chat_solution["config"]["speech"]["model"] = ""
                    running_chat_solution["config"]["speech"]["speaker"] = ""
                    current_running_config["speech"]["name"] = ""
                    current_running_config["speech"]["speaker"] = ""
                    if speech_enable:
                        ttovmodel = webui_config.get("ModelConfig").get("TtoVModel")
                        ttovmodel_lists = [f"{key}" for key in ttovmodel]
                        col1, col2 = st.columns(2)
                        with col1:
                            speechmodel = st.selectbox(
                                    "Please Select Speech Model",
                                    ttovmodel_lists,
                                    index=0,
                                    key="sel_speech_model",
                                )
                        modelconfig = ttovmodel[speechmodel]
                        synthesisconfig = modelconfig["synthesis"]
                        with col2:
                            if modelconfig["type"] == "local":
                                pathstr = modelconfig.get("path")
                                st.text_input("Local Path", pathstr, key="sp_local_path", disabled=True)
                            elif modelconfig["type"] == "cloud":
                                pathstr = modelconfig.get("path")
                                st.text_input("Cloud Path", pathstr, key="sp_cloud_path", disabled=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            templates_list = [modelconfig.get("CloudTemplates", "")]
                            _ = st.selectbox(
                                "Please Select templates",
                                templates_list,
                                index=0,
                                key="sel_templates",
                                disabled=True
                            )
                            lang_index = 0
                            language_list = list(synthesisconfig.keys())
                            language = st.selectbox(
                                "Please Select language",
                                language_list,
                                index=lang_index,
                                key="sel_language",
                            )
                            
                        with col2:
                            sex_index = 0
                            sex_list = list(synthesisconfig[language].keys())
                            sex = st.selectbox(
                                "Please Select Sex",
                                sex_list,
                                index=sex_index,
                                key="sel_sex",
                            )
                            speaker_index = 0
                            speaker_list = synthesisconfig[language][sex]
                            speaker = st.selectbox(
                                "Please Select Speaker",
                                speaker_list,
                                index=speaker_index,
                                key="sel_speaker",
                            )
                            if speaker:
                                st.session_state["speaker"] = speaker
                        running_chat_solution["config"]["speech"]["enable"] = True
                        running_chat_solution["config"]["speech"]["model"] = speechmodel
                        running_chat_solution["config"]["speech"]["speaker"] = speaker

                    st.divider()
                    roleplayer = running_chat_solution["config"]["roleplayer"]
                    st.text_input("Roleplayer:", roleplayer, key="roleplayer", disabled=True)
                    print("chat_solution-2: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution
                if running_chat_solution["stage"] == 3:
                    st.markdown("**Step 3: To decide whether to use Knowledge Base, Search Engine and ToolBoxes**")
                    knowledge_enable = st.checkbox('Knowledge Base', value=False)
                    running_chat_solution["config"]["knowledge_base"]["enable"] = False
                    running_chat_solution["config"]["knowledge_base"]["name"] = ""
                    if knowledge_enable:
                        kb_list = {}
                        try:
                            kb_details = get_kb_details()
                            if len(kb_details):
                                kb_list = {x["kb_name"]: x for x in kb_details}
                        except Exception as _:
                            st.error("Get Knowledge Base failed!")
                            st.stop()
                        kb_names = list(kb_list.keys())
                        selected_kb = st.selectbox(
                            "Knowledge Base:",
                            kb_names,
                            index=0
                        )
                        running_chat_solution["config"]["knowledge_base"]["enable"] = True
                        running_chat_solution["config"]["knowledge_base"]["name"] = selected_kb
                    st.divider()
                    search_enable = st.checkbox('Search Engine', value=False)
                    running_chat_solution["config"]["search_engine"]["enable"] = False
                    running_chat_solution["config"]["search_engine"]["name"] = ""
                    if search_enable:
                        search_engine_list = []
                        searchengine = webui_config.get("SearchEngine")
                        for key, value in searchengine.items():
                            if isinstance(value, dict):
                                search_engine_list.append(key)

                        search_engine = st.selectbox(
                            "Please select Search Engine",
                            search_engine_list,
                            index=0,
                        )
                        if search_engine:
                            running_chat_solution["config"]["search_engine"]["enable"] = True
                            running_chat_solution["config"]["search_engine"]["name"] = search_engine
                    st.divider()
                    toolboxes_enable = st.checkbox('ToolBoxes', value=False)
                    running_chat_solution["config"]["toolboxes"] = ""
                    if toolboxes_enable:
                        toolboxes = webui_config.get("ToolBoxes")
                        toolboxes_list = list(toolboxes.keys())

                        current_toolboxes = st.selectbox(
                            "Please Select ToolBoxes",
                            toolboxes_list,
                            index=0,
                        )
                        running_chat_solution["config"]["toolboxes"] = current_toolboxes
                    print("chat_solution-3: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution
                    st.divider()
                    function_calling_enable = st.checkbox('Function Calling', value=False)
                    running_chat_solution["config"]["function_calling"] = False
                    if function_calling_enable:
                        function_name_list = GetFuncallList()
                        current_function = st.selectbox(
                            "Please Check Function",
                            function_name_list,
                            index=0,
                        )
                        description = GetFuncallDescription(current_function)
                        st.text_input("Description", description, disabled=True)
                        running_chat_solution["config"]["function_calling"] = True

                if running_chat_solution["stage"] == 4:
                    st.markdown("**Step 4: Loading Model and Configuration**")
                    load_cfg_button = st.button(
                        "Load Configuration",
                        use_container_width=True,
                    )
                    load_error = False
                    if load_cfg_button:
                        with st.spinner("Checking all configurations...."):
                            api.save_current_running_config()
                            # first check configuration
                            if not running_chat_solution["config"]["llm_model"]:
                                st.error("LLM Model is not configured!")
                                st.toast("LLM Model is not configured!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["voice"]["enable"] and not running_chat_solution["config"]["voice"]["model"]:
                                st.error("Voice model configuration error!")
                                st.toast("Voice model configuration error!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["speech"]["enable"] and not running_chat_solution["config"]["speech"]["model"]:
                                st.error("Speech model configuration error!")
                                st.toast("Speech model configuration error!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["knowledge_base"]["enable"] and not running_chat_solution["config"]["knowledge_base"]["name"]:
                                st.error("Knowledge Base configuration error!")
                                st.toast("Knowledge Base configuration error!", icon="✖")
                                load_error = True
                            if running_chat_solution["config"]["search_engine"]["enable"] and not running_chat_solution["config"]["search_engine"]["name"]:
                                st.error("Search Engine configuration error!")
                                st.toast("Search Engine configuration error!", icon="✖")
                                load_error = True
                        if not load_error:
                            llm_model = running_chat_solution["config"]["llm_model"]
                            with st.spinner(f"Loading LLM Model: `{llm_model}`, Please do not perform any actions or refresh the page."):
                                models_list = list(api.get_running_models())
                                running_model = ""
                                if len(models_list):
                                    running_model = models_list[0]
                                r = api.change_llm_model(running_model, llm_model)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["voice"]["enable"]:
                            voice_model = running_chat_solution["config"]["voice"]["model"]
                            language = running_chat_solution["config"]["voice"]["language"]
                            language = language[0] if language else "en-US"
                            with st.spinner(f"Loading Voice Model: `{voice_model}`, Please do not perform any actions or refresh the page."):
                                current_voice_model = api.get_vtot_model()
                                r = api.change_voice_model(current_voice_model, voice_model)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    current_running_config["voice"]["name"] = voice_model
                                    current_running_config["voice"]["language"] = language
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["speech"]["enable"]:
                            speech_model = running_chat_solution["config"]["speech"]["model"]
                            speaker = running_chat_solution["config"]["speech"]["speaker"]
                            with st.spinner(f"Loading Speech Model: `{speech_model}`, Please do not perform any actions or refresh the page."):
                                current_speech_model = api.get_ttov_model()
                                current_speech_model = current_speech_model.get("model", "")
                                r = api.change_speech_model(current_speech_model, speech_model, speaker)
                                if msg := check_error_msg(r):
                                    st.error(msg)
                                    st.toast(msg, icon="✖")
                                    load_error = True
                                elif msg := check_success_msg(r):
                                    current_running_config["speech"]["name"] = speech_model
                                    current_running_config["speech"]["speaker"] = speaker
                                    st.success(msg)
                                    st.toast(msg, icon="✔")
                        if not load_error and running_chat_solution["config"]["knowledge_base"]["enable"]:
                            knowledge_base = running_chat_solution["config"]["knowledge_base"]["name"]
                            current_running_config["knowledge_base"]["name"] = knowledge_base
                            with st.spinner(f"Loading Knowledge Base: `{knowledge_base}`, Please do not perform any actions or refresh the page."):
                                st.success(f"Load Knowledge Base `{knowledge_base}` successfully!")
                                st.toast(f"Load Knowledge Base `{knowledge_base}` successfully!", icon="✔")
                        if not load_error and running_chat_solution["config"]["search_engine"]["enable"]:
                            search_engine = running_chat_solution["config"]["search_engine"]["name"]
                            current_running_config["search_engine"]["name"] = search_engine
                            with st.spinner(f"Enabling Search Engine: `{search_engine}`, Please do not perform any actions or refresh the page."):
                                st.success(f"Enable Search Engine `{search_engine}` successfully!")
                                st.toast(f"Enable Search Engine `{search_engine}` successfully!", icon="✔")
                        if not load_error and running_chat_solution["config"]["toolboxes"]:
                            with st.spinner("Enabling ToolBoxes, Please do not perform any actions or refresh the page."):
                                google_toolboxes = current_running_config["ToolBoxes"]["Google ToolBoxes"]
                                google_toolboxes["Tools"]["Google Maps"]["enable"] = True
                                google_toolboxes["Tools"]["Google Mail"]["enable"] = True
                                google_toolboxes["Tools"]["Google Youtube"]["enable"] = True
                                google_toolboxes["Tools"]["Google Calendar"]["enable"] = True
                                google_toolboxes["Tools"]["Google Drive"]["enable"] = True
                                google_toolboxes["Tools"]["Google Photos"]["enable"] = True
                                google_toolboxes["Tools"]["Google Docs"]["enable"] = True
                                google_toolboxes["Tools"]["Google Sheets"]["enable"] = True
                                google_toolboxes["Tools"]["Google Forms"]["enable"] = True
                                current_running_config["ToolBoxes"]["Google ToolBoxes"] = google_toolboxes
                                api.save_current_running_config(current_running_config)
                                st.success("Enable ToolBoxes successfully!")
                                st.toast("Enable ToolBoxes successfully!", icon="✔")
                        if not load_error and running_chat_solution["config"]["function_calling"]:
                            with st.spinner("Enabling Function Calling, Please do not perform any actions or refresh the page."):
                                current_running_config["normal_calling"]["enable"] = running_chat_solution["config"]["function_calling"]
                                api.save_current_running_config(current_running_config)
                                st.success("Enable Function Calling successfully!")
                                st.toast("success save configuration for function calling.", icon="✔")
                        if not load_error:
                            name = running_chat_solution["name"]
                            assistant_name = running_chat_solution["config"]["assistant_name"]
                            roleplayer = running_chat_solution["config"]["roleplayer"]
                            current_running_config["role_player"] = {"name": roleplayer, "language": "english"}
                            current_running_config["chat_solution"] = {"name": name, "assistant_name": assistant_name}
                            api.save_current_running_config(current_running_config)
                            st.success("Load all configurations successfully!")
                            st.toast("Load all configurations successfully!", icon="✔")
                            running_chat_solution["enable"] = True
                    print("chat_solution-4: ", running_chat_solution)
                    st.session_state["current_chat_solution"] = running_chat_solution

    with tabcreateivesolu:
        creative_solutions = aigenerator_config.get("Creative Solutions", {})
        creative_solutions_list = []
        for key, value in creative_solutions.items():
            if isinstance(value, dict):
                creative_solutions_list.append(key)
        current_creative_solution = st.selectbox(
            "Please select Creative Solutions",
            creative_solutions_list,
            index=0,
        )

    st.session_state["current_page"] = "ai_generator_page"

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()