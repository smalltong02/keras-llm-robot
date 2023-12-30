import streamlit as st
from WebUI.webui_pages.utils import *
from WebUI.configs import *
from WebUI.webui_pages import *

training_devices_list = ["auto","cpu","gpu","mps"]
loadbits_list = ["16 bits","8 bits","4 bits"]
quantization_list = ["16 bits", "8 bits", "6 bits", "5 bits", "4 bits"]

def configuration_page(api: ApiRequest, is_lite: bool = False):
    running_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
    current_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    webui_config = api.get_webui_config()
    localmodel = webui_config.get("ModelConfig").get("LocalModel")
    commonmodel = localmodel.get("LLM Model")
    multimodalmodel = localmodel.get("Multimodal Model")
    specialmodel = localmodel.get("Special Model")
    onlinemodel = webui_config.get("ModelConfig").get("OnlineModel")
    embeddingmodel = webui_config.get("ModelConfig").get("EmbeddingModel")
    chatconfig = webui_config.get("ChatConfiguration")
    quantconfig = webui_config.get("QuantizationConfiguration")
    finetunning = webui_config.get("Fine-Tunning")
    prompttemplates = webui_config.get("PromptTemplates")

    running_model["mname"] = ""
    models_list = list(api.get_running_models())
    print("models_list: ", models_list)
    if len(models_list):
        running_model["mname"] = models_list[0]
        running_model["mtype"], running_model["msize"], running_model["msubtype"] = GetModelInfoByName(webui_config, running_model["mname"])

    if running_model["mtype"] != ModelType.Unknown:
        type_index = int(running_model["mtype"].value) - 1
    else:
        type_index = 0
    if running_model["msize"] != ModelSize.Unknown:
        size_index = int(running_model["msize"].value) - 1
    else:
        if running_model["msubtype"] != ModelSubType.Unknown:
            size_index = int(running_model["msubtype"].value) - 1
        else:
            size_index = 0

    col1, col2 = st.columns(2)
    with col1:
        modeltype = st.selectbox(
                "Please Select Model Type",
                glob_model_type_list,
                index=type_index,
            )
        current_model["mtype"] = GetModelType(modeltype)
        type_index = current_model["mtype"].value - 1
    with col2:
        if type_index == ModelType.Local.value - 1 or type_index == ModelType.Special.value - 1:
            modelsize = st.selectbox(
                    "Please Select Model Size",
                    glob_model_size_list,
                    index=size_index,
                )
            size_index = glob_model_size_list.index(modelsize)
            current_model["msubtype"] = ModelSubType.Unknown
            current_model["msize"] = ModelSize(size_index + 1)
        elif type_index == ModelType.Multimodal.value - 1:
            submodel = st.selectbox(
                    "Please Select Sub Model",
                    glob_model_subtype_list,
                    index=size_index,
                )
            size_index = glob_model_subtype_list.index(submodel)
            current_model["msubtype"] = ModelSubType(size_index + 1)
            current_model["msize"] = ModelSize.Unknown
        elif type_index == ModelType.Online.value - 1:
            online_model_list = GetOnlineProvider(webui_config)
            onlinemodel = st.selectbox(
                    "Please Select Online Provider",
                    online_model_list,
                    index=size_index,
                )
            current_model["msubtype"] = ModelSubType.Unknown
            current_model["msize"] = ModelSize.Unknown
            size_index = online_model_list.index(onlinemodel)
            online_model_list = GetOnlineModelList(webui_config, onlinemodel)

    if type_index != ModelType.Online.value - 1:
        model_list = GetModeList(webui_config, current_model)
    else:
        model_list = online_model_list
    model_index = 0
    if running_model["mname"]:
        try:
            model_index = model_list.index(running_model["mname"])
            model_list[model_index] += " (running)"
        except ValueError:
            model_index = 0
    
    st.divider()
    col1, col2 = st.columns(2)
    disabled = False
    with col1:
        model_name = st.selectbox(
            "Please Select Model Name",
            model_list,
            index=model_index,
        )
        current_model["mname"] = model_name
        if current_model["mname"] != None:
            if current_model["mname"].endswith("(running)"):
                current_model["mname"] = current_model["mname"][:-10].strip()
            current_model["config"] = GetModelConfig(webui_config, current_model)
        else:
            disabled = True
            current_model["config"] = {}

        le_button = st.button(
            "Load & Eject",
            use_container_width=True,
            disabled=disabled
        )
        if le_button:
            if current_model["mname"] == running_model["mname"]:
                with st.spinner(f"Release Model: {current_model['mname']}, Please do not perform any actions or refresh the page."):
                    r = api.eject_llm_model(current_model["mname"])
                    if msg := check_error_msg(r):
                        st.error(msg)
                    elif msg := check_success_msg(r):
                        st.success(msg)
            else:
                with st.spinner(f"Loading Model: {current_model['mname']}, Please do not perform any actions or refresh the page."):
                    r = api.change_llm_model(running_model["mname"], current_model["mname"])
                    if msg := check_error_msg(r):
                        st.error(msg)
                    elif msg := check_success_msg(r):
                        st.success(msg)
    with col2:
        if disabled != True and current_model["mtype"] != ModelType.Online:
            pathstr = current_model["config"].get("path")
        else:
            pathstr = ""
        pathstr = st.text_input("Local Path", pathstr, disabled=True)
        save_path = st.button(
            "Download",
            use_container_width=True,
            disabled=disabled
        )
        if save_path:
            pass
            # with st.spinner(f"Saving path, Please do not perform any actions or refresh the page."):
            #     if current_model["mname"] == None or current_model["mtype"] == ModelType.Online:
            #         st.error("Save path failed!")
            #     else:
            #         current_model["config"]["path"] = pathstr
            #         r = api.save_model_config(current_model)
            #         if msg := check_error_msg(r):
            #             st.error(msg)
            #         elif msg := check_success_msg(r):
            #             st.success(msg)

    st.divider()
    if current_model["config"]:
        preset_list = GetPresetPromptList()
        if current_model["mtype"] == ModelType.Local or current_model["mtype"] == ModelType.Multimodal or current_model["mtype"] == ModelType.Special:
            tabparams, tabquant, tabembedding, tabtunning, tabprompt = st.tabs(["Parameters", "Quantization", "Embedding Model", "Fine-Tunning", "Prompt Templates"])
            with tabparams:
                with st.form("Parameter"):
                    col1, col2 = st.columns(2)
                    with col1:
                        sdevice = current_model["config"].get("device").lower()
                        if sdevice in training_devices_list:
                            index = training_devices_list.index(sdevice)
                        else:
                            index = 0
                        predict_dev = st.selectbox(
                            "Please select Device",
                            training_devices_list,
                            index=index,
                            disabled=disabled
                        )
                        nthreads = current_model["config"].get("cputhreads")
                        nthreads = st.number_input("CPU Threads", value = nthreads, min_value=1, max_value=32, disabled=disabled)
                        st.text("")
                        spreset = current_model["config"].get("preset", "default")
                        index = preset_list.index(spreset)
                        preset_dev = st.selectbox(
                            "Please select Preset",
                            preset_list,
                            index=index,
                            disabled=disabled
                        )
                        maxtokens = chatconfig.get("tokens_length")
                        min = maxtokens.get("min")
                        max = maxtokens.get("max")
                        cur = maxtokens.get("cur")
                        step = maxtokens.get("step")
                        maxtokens = st.slider("Max tokens", min, max, cur, step, disabled=disabled)
                        temperature = chatconfig.get("temperature")
                        temperature = st.slider("Temperature", 0.0, 1.0, temperature, 0.05, disabled=disabled)
                        epsilon_cutoff = chatconfig.get("epsilon_cutoff")
                        epsilon_cutoff = st.slider("epsilon_cutoff", 0.0, 1.0, epsilon_cutoff, 0.1, disabled=disabled)
                        eta_cutoff = chatconfig.get("eta_cutoff")
                        eta_cutoff = st.slider("eta_cutoff", 0.0, 1.0, eta_cutoff, 0.1, disabled=disabled)
                        diversity_penalty = chatconfig.get("diversity_penalty")
                        diversity_penalty = st.slider("diversity_penalty", 0.0, 1.0, diversity_penalty, 0.1, disabled=disabled)
                        repetition_penalty = chatconfig.get("repetition_penalty")
                        min = repetition_penalty.get("min")
                        max = repetition_penalty.get("max")
                        cur = repetition_penalty.get("cur")
                        step = repetition_penalty.get("step")
                        repetition_penalty = st.slider("repetition_penalty", min, max, cur, step, disabled=disabled)
                        length_penalty = chatconfig.get("length_penalty")
                        min = length_penalty.get("min")
                        max = length_penalty.get("max")
                        cur = length_penalty.get("cur")
                        step = length_penalty.get("step")
                        length_penalty = st.slider("length_penalty", min, max, cur, step, disabled=disabled)
                        do_samples = chatconfig.get("do_samples")
                        do_samples = st.checkbox('do samples', value=do_samples, disabled=disabled)
                    with col2:
                        seed = chatconfig.get("seed")
                        min = seed.get("min")
                        max = seed.get("max")
                        cur = seed.get("cur")
                        seed = st.number_input("Seed (-1 for random)", value = cur, min_value=min, max_value=max, disabled=disabled)
                        nloadbits = current_model["config"].get("loadbits")
                        index = 0 if nloadbits == 16 else (1 if nloadbits == 8 else 2)
                        nloadbits = st.selectbox(
                            "Load Bits",
                            loadbits_list,
                            index=index,
                            disabled=disabled
                        )
                        top_p = chatconfig.get("top_p")
                        top_p = st.slider("Top_p", 0.0, 1.0, top_p, 0.1, disabled=disabled)
                        top_k = chatconfig.get("top_k")
                        min = top_k.get("min")
                        max = top_k.get("max")
                        cur = top_k.get("cur")
                        step = top_k.get("step")
                        top_k = st.slider("Top_k", min, max, cur, step, disabled=disabled)
                        typical_p = chatconfig.get("typical_p")
                        typical_p = st.slider("Typical_p", 0.0, 1.0, typical_p, 0.1, disabled=disabled)
                        top_a = chatconfig.get("top_a")
                        top_a = st.slider("Top_a", 0.0, 1.0, top_a, 0.1, disabled=disabled)
                        tfs = chatconfig.get("tfs")
                        tfs = st.slider("tfs", 0.0, 1.0, tfs, 0.1, disabled=disabled)
                        no_repeat_ngram_size = chatconfig.get("no_repeat_ngram_size")
                        no_repeat_ngram_size = st.slider("no_repeat_ngram_size", 0, 1, no_repeat_ngram_size, 1, disabled=disabled)
                        guidance_scale = chatconfig.get("guidance_scale")
                        min = guidance_scale.get("min")
                        max = guidance_scale.get("max")
                        cur = guidance_scale.get("cur")
                        step = guidance_scale.get("step")
                        guidance_scale = st.slider("guidance_scale", min, max, cur, step, disabled=disabled)
                        encoder_repetition_penalty = chatconfig.get("encoder_repetition_penalty")
                        min = encoder_repetition_penalty.get("min")
                        max = encoder_repetition_penalty.get("max")
                        cur = encoder_repetition_penalty.get("cur")
                        step = encoder_repetition_penalty.get("step")
                        encoder_repetition_penalty = st.slider("encoder_repetition_penalty", min, max, cur, step, disabled=disabled)
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True,
                        disabled=disabled
                    )
                    if save_parameters:
                        current_model["config"]["device"] = predict_dev
                        current_model["config"]["cputhreads"] = nthreads
                        if nloadbits == "16 bits":
                            current_model["config"]["loadbits"] = 16
                        else:
                            current_model["config"]["loadbits"] = 8
                        current_model["config"]["preset"] = preset_dev
                        chatconfig["seed"]["cur"] = seed
                        chatconfig["tokens_length"]["cur"] = maxtokens
                        chatconfig["temperature"] = temperature
                        chatconfig["epsilon_cutoff"] = epsilon_cutoff
                        chatconfig["eta_cutoff"] = eta_cutoff
                        chatconfig["diversity_penalty"] = diversity_penalty
                        chatconfig["repetition_penalty"]["cur"] = repetition_penalty
                        chatconfig["length_penalty"]["cur"] = length_penalty
                        chatconfig["encoder_repetition_penalty"]["cur"] = encoder_repetition_penalty
                        chatconfig["top_p"] = top_p
                        chatconfig["top_k"]["cur"] = top_k
                        chatconfig["typical_p"] = typical_p
                        chatconfig["top_a"] = top_a
                        chatconfig["tfs"] = tfs
                        chatconfig["no_repeat_ngram_size"] = no_repeat_ngram_size
                        chatconfig["guidance_scale"]["cur"] = guidance_scale
                        if do_samples:
                            chatconfig["do_samples"] = True
                        else:
                            chatconfig["do_samples"] = False

                        with st.spinner(f"Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_model_config(current_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                r = api.save_chat_config(chatconfig)
                                if msg := check_error_msg(r):
                                    st.error("failed to save configuration for model and chat.")
                                elif msg := check_success_msg(r):
                                    st.success("success save configuration for model and chat.")

            with tabquant:
                with st.form("Quantization"):
                    methods_lists = ["AutoGPTQ", "ExllamaV2", "Llamacpp"]
                    st.selectbox(
                        "Methods",
                        methods_lists,
                        index=0,
                        disabled=disabled
                    )
                    st.selectbox(
                        "Quantization Bits",
                        quantization_list,
                        index=0,
                        disabled=disabled
                    )
                    format_lists = ["GPTQ", "GGUF", "AWQ"]
                    st.selectbox(
                        "Format",
                        format_lists,
                        index=0,
                        disabled=disabled
                    )
                    submit_quantization = st.form_submit_button(
                        "Launch",
                        use_container_width=True,
                        disabled=disabled
                    )
                    if submit_quantization:
                        st.success("The model quantization has been successful, and the quantized file path is model/llama-2-7b-hf-16bit.bin.")

            with tabembedding:
                embedding_lists = [f"{key}" for key in embeddingmodel]
                st.selectbox(
                    "Please Select Embedding Model",
                    embedding_lists,
                    index=0
                )

            with tabtunning:
                st.selectbox(
                    "Please select Device",
                    training_devices_list,
                    index=0,
                    disabled=disabled
                )
            with tabprompt:
                pass
        
        elif current_model["mtype"] == ModelType.Online:
            tabparams, tabapiconfig, tabprompt = st.tabs(["Parameters", "API Config", "Prompt Templates"])
            with tabparams:
                with st.form("Parameters"):
                    col1, col2 = st.columns(2)
                    with col1:
                        spreset = current_model["config"].get("preset", "default")
                        index = preset_list.index(spreset)
                        preset_dev = st.selectbox(
                            "Please select Preset",
                            preset_list,
                            index=index,
                            disabled=disabled
                        )
                    with col2:
                        pass
                    submit_params = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if submit_params:
                        st.error("Not Support Now!")

            with tabapiconfig:
                with st.form("ApiConfig"):
                    col1, col2 = st.columns(2)
                    with col1:
                        baseurl = st.text_input("Base URL", current_model["config"].get("baseurl", ""), disabled=disabled)
                        apikey = st.text_input("API Key", current_model["config"].get("apikey", ""), disabled=disabled)
                        provider = st.text_input("Provider", current_model["config"].get("provider", ""), disabled=disabled)

                    with col2:
                        apiversion = st.text_input("API Version", current_model["config"].get("apiversion", ""), disabled=disabled)
                        apiproxy = st.text_input("API Proxy", current_model["config"].get("apiproxy", ""), disabled=disabled)
                    submit_config = st.form_submit_button(
                        "Save API Config",
                        use_container_width=True
                    )
                    if submit_config:
                        current_model["config"]["baseurl"] = baseurl
                        current_model["config"]["apikey"] = apikey
                        current_model["config"]["provider"] = provider
                        current_model["config"]["apiversion"] = apiversion
                        current_model["config"]["apiproxy"] = apiproxy
                        savename = current_model["mname"]
                        current_model["mname"] = onlinemodel
                        with st.spinner(f"Saving online config, Please do not perform any actions or refresh the page."):
                            print("current_model: ", current_model)
                            r = api.save_model_config(current_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
                        current_model["mname"] = savename
            
            with tabprompt:
                pass

    st.session_state["current_page"] = "configuration_page"
