import streamlit as st
from WebUI.webui_pages.utils import ApiRequest
from WebUI.configs import (ROLEPLAY_TEMPLATES, ModelType, ModelSize, ModelSubType, GetModelType, GetModelInfoByName, GetModelConfig, GetModelSubType, GetOnlineProvider, GetOnlineModelList, GetModeList, LocalModelExist, GetPresetPromptList,
                           glob_model_type_list, glob_model_size_list, glob_model_subtype_list)
from WebUI.webui_pages.utils import check_error_msg, check_success_msg
from typing import Dict

training_devices_list = ["auto","cpu","gpu","mps"]
loadbits_list = ["16 bits","8 bits","4 bits"]
quantization_list = ["16 bits", "8 bits", "6 bits", "5 bits", "4 bits"]
player_language_list = ["english","chinese"]

def configuration_page(api: ApiRequest, is_lite: bool = False):
    running_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
    current_model : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    webui_config = api.get_webui_config()
    onlinemodel = webui_config.get("ModelConfig").get("OnlineModel")
    chatconfig = webui_config.get("ChatConfiguration")
    finetuning = webui_config.get("Fine-Tuning")
    searchengine = webui_config.get("SearchEngine")
    functioncalling = webui_config.get("FunctionCalling")
    current_running_config = api.get_current_running_config()
    calling_enable = False

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
        if type_index == ModelType.Local.value - 1 or type_index == ModelType.Special.value - 1 or type_index == ModelType.Code.value - 1:
            if size_index >= len(glob_model_size_list):
                size_index = 0
            modelsize = st.selectbox(
                    "Please Select Model Size",
                    glob_model_size_list,
                    index=size_index,
                )
            size_index = glob_model_size_list.index(modelsize)
            current_model["msubtype"] = ModelSubType.Unknown
            current_model["msize"] = ModelSize(size_index + 1)
        elif type_index == ModelType.Multimodal.value - 1:
            if size_index >= len(glob_model_subtype_list):
                size_index = 0
            submodel = st.selectbox(
                    "Please Select Sub Model",
                    glob_model_subtype_list,
                    index=size_index,
                )
            current_model["msubtype"] = GetModelSubType(submodel)
            current_model["msize"] = ModelSize.Unknown
        elif type_index == ModelType.Online.value - 1:
            online_model_list = GetOnlineProvider(webui_config)
            if size_index >= len(online_model_list):
                size_index = 0
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
        if current_model["mname"] is not None:
            if current_model["mname"].endswith("(running)"):
                current_model["mname"] = current_model["mname"][:-10].strip()
            current_model["config"] = GetModelConfig(webui_config, current_model)
        else:
            disabled = True
            current_model["config"] = {}

        if disabled is not True and current_model["mtype"] != ModelType.Online:
            pathstr = current_model["config"].get("path")
        else:
            pathstr = ""

        le_button = st.button(
            "Load & Eject",
            use_container_width=True,
            disabled=disabled
        )
        if le_button:
            if current_model["mname"] == running_model["mname"]:
                with st.spinner(f"Release Model: `{current_model['mname']}`, Please do not perform any actions or refresh the page."):
                    r = api.eject_llm_model(current_model["mname"])
                    if msg := check_error_msg(r):
                        st.error(msg)
                        st.toast(msg, icon="✖")
                    elif msg := check_success_msg(r):
                        st.success(msg)
                        st.toast(msg, icon="✔")
            else:
                with st.spinner(f"Loading Model: `{current_model['mname']}`, Please do not perform any actions or refresh the page."):
                    if current_model["mtype"] == ModelType.Online or LocalModelExist(pathstr):
                        r = api.change_llm_model(running_model["mname"], current_model["mname"])
                        if msg := check_error_msg(r):
                            st.error(msg)
                            st.toast(msg, icon="✖")
                        elif msg := check_success_msg(r):
                            st.success(msg)
                            st.toast(msg, icon="✔")
                    else:
                        st.error("Please download the model to your local machine first.")
    with col2:
        pathstr = st.text_input("Local Path", pathstr, disabled=True)
        download_path = st.button(
            "Download",
            use_container_width=True,
            disabled=disabled
        )
        if download_path:
            with st.spinner("Model downloading..., Please do not perform any actions or refresh the page."):
                if current_model["mname"] is None or current_model["mtype"] == ModelType.Online:
                    st.error("Download failed!")
                else:
                    if LocalModelExist(pathstr):
                        st.error(f'The model {current_model["mname"]} already exists in the folder {pathstr}')
                    else:
                        huggingface_path = current_model["config"]["Huggingface"]
                        r = api.download_llm_model(current_model["mname"], huggingface_path, pathstr)
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

    st.divider()
    if current_model["config"]:
        preset_list = GetPresetPromptList()
        if current_model["mtype"] == ModelType.Local or current_model["mtype"] == ModelType.Multimodal or current_model["mtype"] == ModelType.Special or current_model["mtype"] == ModelType.Code:
            tabparams, tabsearch, tabfuncall, tabroleplay, tabquant, tabtuning = st.tabs(["Parameters", "Search Engine", "Function Calling", "Role Player", "Quantization", "Fine-Tuning"])
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

                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_model_config(current_model)
                            if msg := check_error_msg(r):
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                r = api.save_chat_config(chatconfig)
                                if msg := check_error_msg(r):
                                    st.toast("failed to save configuration for model and chat.", icon="✖")
                                elif msg := check_success_msg(r):
                                    st.toast("success save configuration for model and chat.", icon="✔")

            with tabquant:
                with st.form("Quantization"):
                    methods_lists = ["AutoGPTQ", "ExllamaV2", "Llamacpp"]
                    st.selectbox(
                        "Methods",
                        methods_lists,
                        index=0,
                        disabled=True
                    )
                    st.selectbox(
                        "Quantization Bits",
                        quantization_list,
                        index=0,
                        disabled=True
                    )
                    format_lists = ["GPTQ", "GGUF", "AWQ"]
                    st.selectbox(
                        "Format",
                        format_lists,
                        index=0,
                        disabled=True
                    )
                    submit_quantization = st.form_submit_button(
                        "Launch",
                        use_container_width=True,
                        disabled=True
                    )
                    if submit_quantization:
                        st.toast("The model quantization has been successful, and the quantized file path is model/llama-2-7b-hf-16bit.bin.", icon="✔")

            with tabtuning:
                from WebUI.configs.basicconfig import glob_compute_type_list, glob_save_strategy_list, glob_optimizer_list, glob_lr_scheduler_list, glob_Lora_rank_list, glob_save_model_list, glob_save_method_list, glob_quantization_method_list
                finetuning_list = []
                for key, value in finetuning.items():
                    if isinstance(value, dict):
                        finetuning_list.append(key)
                current_finetuning_library = st.selectbox(
                    "Please select Fine-Tuning Library",
                    finetuning_list,
                    index=0,
                    disabled=disabled
                )
                tun_basic_config = finetuning[current_finetuning_library].get("basic_config", {})
                tun_train_config = finetuning[current_finetuning_library].get("train_config", {})
                tun_lora_config = finetuning[current_finetuning_library].get("lora_config", {})
                tun_dataset_config = finetuning[current_finetuning_library].get("dataset_config", {})
                tun_output_config = finetuning[current_finetuning_library].get("output_config", {})

                with st.form("Fine-Tuning"):
                    col1, col2 = st.columns(2)
                    with col1:
                        compute_type = tun_basic_config.get("compute_type", "fp16")
                        compute_index = glob_compute_type_list.index(compute_type)
                        compute_type = st.selectbox(
                            "Compute type",
                            glob_compute_type_list,
                            index=compute_index,
                            disabled=disabled
                        )
                        load_in_4bit = tun_basic_config.get("load_in_4bit", False)
                        load_in_4bit = st.checkbox('load_in_4bit', value=load_in_4bit, disabled=disabled)
                    with col2:
                        seq_length = tun_basic_config.get("seq_length", 512)
                        seq_length = st.slider("Sentence length", 16, 32000, seq_length, 1, disabled=disabled)
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        learning_rate = tun_train_config.get("lr", 5e-5)
                        learning_rate = st.text_input("Learning rate", learning_rate, disabled=disabled)
                        batch_size = tun_train_config.get("batch_size", 5)
                        batch_size = st.number_input("Batch size", value = batch_size, min_value=1, max_value=512, disabled=disabled)
                        gradient_steps = tun_train_config.get("gradient_steps", 2)
                        gradient_steps = st.number_input("Gradient steps", value = gradient_steps, min_value=1, max_value=64, disabled=disabled)
                        logging_steps = tun_train_config.get("logging_steps", 10)
                        logging_steps = st.number_input("Logging steps", value = logging_steps, min_value=1, max_value=64, disabled=disabled)
                        save_steps = tun_train_config.get("save_steps", 10)
                        save_steps = st.number_input("Save steps", value = save_steps, min_value=1, max_value=10000, disabled=disabled)
                        weight_decay = tun_train_config.get("weight_decay", 0.0)
                        weight_decay = st.slider("Weight decay", 0.0, 1.0, weight_decay, 0.1, disabled=disabled)
                        seed = tun_train_config.get("seed", -1)
                        seed = st.number_input("Seed (-1 for random)", value = seed, min_value=-1, max_value=10000, disabled=disabled)
                        packing = tun_train_config.get("packing", False)
                        packing = st.checkbox("Packing", value = packing, disabled=disabled)
                    with col2:
                        epochs = tun_train_config.get("epochs", 10)
                        epochs = st.number_input("Epochs", value = epochs, min_value=1, max_value=10000, disabled=disabled)
                        num_proc = tun_train_config.get("num_proc", 2)
                        num_proc = st.number_input("Process Nums", value = num_proc, min_value=1, max_value=32, disabled=disabled)
                        warmup_steps = tun_train_config.get("warmup_steps", 2)
                        warmup_steps = st.number_input("Warmup steps", value = warmup_steps, min_value=1, max_value=64, disabled=disabled)
                        save_strategy = tun_basic_config.get("save_strategy", "steps")
                        save_strategy_index = glob_save_strategy_list.index(save_strategy)
                        save_strategy = st.selectbox(
                            "Save strategy",
                            glob_save_strategy_list,
                            index=save_strategy_index,
                            disabled=disabled
                        )
                        optim = tun_train_config.get("optim", "adamw_torch")
                        optim_index = glob_optimizer_list.index(optim)
                        optim = st.selectbox(
                            "Optimizer",
                            glob_optimizer_list,
                            index=optim_index,
                            disabled=disabled
                        )
                        lr_scheduler = tun_train_config.get("lr_scheduler", "linear")
                        lr_scheduler_index = glob_lr_scheduler_list.index(lr_scheduler)
                        lr_scheduler = st.selectbox(
                            "LR Scheduler",
                            glob_lr_scheduler_list,
                            index=lr_scheduler_index,
                            disabled=disabled
                        )
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        lora_rank = tun_lora_config.get("lora_rank", 8)
                        lora_rank_index = glob_Lora_rank_list.index(lora_rank)
                        lora_rank = st.selectbox(
                            "Lora Rank",
                            glob_Lora_rank_list,
                            index=lora_rank_index,
                            disabled=disabled
                        )
                        lora_alpha = tun_lora_config.get("lora_alpha", 32)
                        lora_alpha = st.number_input("Lora Alpha", value = lora_alpha, min_value=1, max_value=512, disabled=disabled)
                        use_gradient_checkpointing = tun_lora_config.get("use_gradient_checkpointing", "unsloth")
                        use_gradient_checkpointing = st.text_input("Gradient Checkpointing", use_gradient_checkpointing, disabled=True)
                        use_rslora = tun_lora_config.get("use_rslora", False)
                        use_rslora = st.checkbox("Use rslora", use_rslora, disabled=disabled)
                    with col2:
                        target_modules_list = tun_lora_config.get("target_modules", "q_proj")
                        target_modules = ','.join(target_modules_list)
                        target_modules = st.text_input("Target Modules", target_modules, disabled=disabled)
                        lora_dropout = tun_lora_config.get("lora_dropout", 0)
                        lora_dropout = st.number_input("Lora Alpha", value = lora_dropout, min_value=0.0, max_value=1.0, disabled=disabled)
                        random_state = tun_lora_config.get("random_state", 128)
                        random_state = st.number_input("Random State", value = random_state, min_value=0, max_value=10000, disabled=disabled)
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        dataset_path = tun_dataset_config.get("dataset_path", "")
                        dataset_path = st.text_input("Dataset Path", dataset_path, disabled=disabled)
                    with col2:
                        checkpoint_path = tun_dataset_config.get("checkpoint_path", "")
                        checkpoint_path = st.text_input("Checkpoint Path", checkpoint_path, disabled=disabled)
                    st.divider()
                    col1, col2 = st.columns(2)
                    with col1:
                        output_path = tun_output_config.get("output_path", "")
                        output_path = st.text_input("Output Path", output_path, disabled=disabled)
                        save_method = tun_output_config.get("save_method", "lora")
                        save_method_index = glob_save_method_list.index(save_method)
                        save_method = st.selectbox(
                            "Save Method",
                            glob_save_method_list,
                            index=save_method_index,
                            disabled=disabled
                        )
                    with col2:
                        output_format = tun_output_config.get("output_format", "full")
                        output_format_index = glob_save_model_list.index(output_format)
                        output_format = st.selectbox(
                            "Output Format",
                            glob_save_model_list,
                            index=output_format_index,
                            disabled=disabled
                        )
                        quantization_method = tun_output_config.get("quantization_method", "f16")
                        quantization_method_index = glob_quantization_method_list.index(quantization_method)
                        quantization_method = st.selectbox(
                            "Quantization Method",
                            glob_quantization_method_list,
                            index=quantization_method_index,
                            disabled=disabled
                        )

                    fine_tuning_btn = st.form_submit_button(
                        "Fine-Tuning",
                        use_container_width=True,
                        disabled=disabled
                    )
                    if fine_tuning_btn:
                        pass

            with tabsearch:
                search_enable = False
                search_engine_list = []
                for key, value in searchengine.items():
                    if isinstance(value, dict):
                        search_engine_list.append(key)
                if current_running_config["search_engine"]["name"]:
                    search_enable = True
                    index = search_engine_list.index(current_running_config["search_engine"]["name"])
                else:
                    index = 0

                current_search_engine = st.selectbox(
                    "Please select Search Engine",
                    search_engine_list,
                    index=index,
                    disabled=disabled
                )
                with st.form("SearchEngine"):
                    col1, col2 = st.columns(2)
                    top_k = searchengine.get("top_k", 3)
                    cse_id = searchengine.get(current_search_engine).get("cse_id", None)
                    api_key = searchengine.get(current_search_engine).get("api_key", "")
                    with col1:
                        api_key = st.text_input("API Key", api_key, type="password", disabled=disabled)
                        search_enable = st.checkbox('Enable', value=search_enable, help="After enabling, parameters need to be saved for the configuration to take effect.", disabled=disabled)
                    with col2:
                        if cse_id is not None:
                            cse_id = st.text_input("Google CSE ID", cse_id, type="password", disabled=disabled)
                        top_k = st.slider("Top_k", 1, 10, top_k, 1, disabled=disabled)
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True,
                        disabled=disabled
                    )
                    if save_parameters:
                        searchengine.get(current_search_engine)["api_key"] = api_key
                        if cse_id is not None:
                            searchengine.get(current_search_engine)["cse_id"] = cse_id
                        searchengine["top_k"] = top_k
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_search_engine_config(searchengine)
                            if msg := check_error_msg(r):
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                st.toast("success save configuration for search engine.", icon="✔")
                if search_enable:
                    current_running_config["search_engine"]["name"] = current_search_engine
                else:
                    current_running_config["search_engine"]["name"] = ""
                api.save_current_running_config(current_running_config)
            with tabroleplay:
                roleplay_list = list(ROLEPLAY_TEMPLATES.keys())
                if current_running_config["role_player"]["name"] and current_running_config["role_player"]["language"]:
                    role_enable = True
                    role_index = roleplay_list.index(current_running_config["role_player"]["name"])
                    lang_index = player_language_list.index(current_running_config["role_player"]["language"])
                else:
                    role_index = 0
                    lang_index = 0
                    role_enable = False
                with st.form("Roleplay"):
                    col1, col2 = st.columns(2)
                    with col1:
                        current_roleplayer = st.selectbox(
                            "Please Select Role Player",
                            roleplay_list,
                            index=role_index,
                        )
                        role_enable = st.checkbox("Enable", value=role_enable, help="After enabling, The Role Play feature will activate.")
                    with col2:
                        roleplayer_language = st.selectbox(
                            "Please Select Language",
                            player_language_list,
                            index=lang_index,
                        )
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            if role_enable:
                                current_running_config["role_player"]["name"] = current_roleplayer
                                current_running_config["role_player"]["language"] = roleplayer_language
                            else:
                                current_running_config["role_player"]["name"] = ""
                                current_running_config["role_player"]["language"] = ""
                            api.save_current_running_config(current_running_config)
                            st.toast("success save configuration for Code Interpreter.", icon="✔")
            with tabfuncall:
                from WebUI.Server.funcall.funcall import GetFuncallList, GetFuncallDescription
                calling_enable = functioncalling.get("calling_enable", False)
                current_function = ""
                function_name_list = GetFuncallList()
                current_function = st.selectbox(
                    "Please Check Function",
                    function_name_list,
                    index=0,
                )
                with st.form("Funcall"):
                    description = GetFuncallDescription(current_function)
                    st.text_input("Description", description, disabled=True)
                    calling_enable = st.checkbox("Enable", key="funcall_box", value=calling_enable, help="After enabling, The function will be called automatically.")
                    print("calling_enable", calling_enable)
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            functioncalling["calling_enable"] = calling_enable
                            r = api.save_function_calling_config(functioncalling)
                            if msg := check_error_msg(r):
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                st.toast("success save configuration for function calling.", icon="✔")
        
        elif current_model["mtype"] == ModelType.Online:
            tabparams, tabapiconfig, tabsearch, tabfuncall, tabroleplay = st.tabs(["Parameters", "API Config", "Search Engine", "Function Calling", "Role Player"])
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
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        st.toast("Not Support Now!", icon="✖")

            with tabapiconfig:
                with st.form("ApiConfig"):
                    col1, col2 = st.columns(2)
                    with col1:
                        baseurl = st.text_input("Base URL", current_model["config"].get("baseurl", ""), disabled=disabled)
                        apikey = st.text_input("API Key", current_model["config"].get("apikey", ""), type="password", disabled=disabled)
                        apiproxy = st.text_input("API Proxy", current_model["config"].get("apiproxy", ""), disabled=disabled)

                    with col2:
                        apiversion = st.text_input("API Version", current_model["config"].get("apiversion", ""), disabled=disabled)
                        secretkey = current_model["config"].get("secretkey", None)
                        if secretkey is not None:
                            secretkey = st.text_input("Secret Key", secretkey, type="password", disabled=disabled)
                        else:
                            st.text_input("Secret Key", "None", disabled=True)
                        provider = st.text_input("Provider", current_model["config"].get("provider", ""), disabled=True)

                    submit_config = st.form_submit_button(
                        "Save API Config",
                        use_container_width=True
                    )
                    if submit_config:
                        current_model["config"]["baseurl"] = baseurl
                        current_model["config"]["apikey"] = apikey
                        if secretkey is not None:
                            current_model["config"]["secretkey"] = secretkey
                        current_model["config"]["provider"] = provider
                        current_model["config"]["apiversion"] = apiversion
                        current_model["config"]["apiproxy"] = apiproxy
                        savename = current_model["mname"]
                        current_model["mname"] = onlinemodel
                        with st.spinner("Saving online config, Please do not perform any actions or refresh the page."):
                            r = api.save_model_config(current_model)
                            if msg := check_error_msg(r):
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                st.toast(msg, icon="✔")
                        current_model["mname"] = savename

            with tabsearch:
                search_enable = False
                search_engine_list = []
                for key, value in searchengine.items():
                    if isinstance(value, dict):
                        search_engine_list.append(key)
                if current_running_config["search_engine"]["name"]:
                    search_enable = True
                    index = search_engine_list.index(current_running_config["search_engine"]["name"])
                else:
                    index = 0

                current_search_engine = st.selectbox(
                    "Please select Search Engine",
                    search_engine_list,
                    index=index,
                    disabled=disabled
                )
                with st.form("SearchEngine"):
                    col1, col2 = st.columns(2)
                    top_k = searchengine.get("top_k", 3)
                    cse_id = searchengine.get(current_search_engine).get("cse_id", None)
                    api_key = searchengine.get(current_search_engine).get("api_key", "")
                    with col1:
                        api_key = st.text_input("API Key", api_key, type="password", disabled=disabled)
                        search_enable = st.checkbox('Enable', value=search_enable, help="After enabling, parameters need to be saved for the configuration to take effect.", disabled=disabled)
                    with col2:
                        if cse_id is not None:
                            cse_id = st.text_input("Google CSE ID", cse_id, type="password", disabled=disabled)
                        top_k = st.slider("Top_k", 1, 10, top_k, 1, disabled=disabled)
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True,
                        disabled=disabled
                    )
                    if save_parameters:
                        searchengine.get(current_search_engine)["api_key"] = api_key
                        if cse_id is not None:
                            searchengine.get(current_search_engine)["cse_id"] = cse_id
                        searchengine["top_k"] = top_k
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            r = api.save_search_engine_config(searchengine)
                            if msg := check_error_msg(r):
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                st.toast("success save configuration for search engine.", icon="✔")
                if search_enable:
                    current_running_config["search_engine"]["name"] = current_search_engine
                else:
                    current_running_config["search_engine"]["name"] = ""
                api.save_current_running_config(current_running_config)
            with tabroleplay:
                roleplay_list = list(ROLEPLAY_TEMPLATES.keys())
                if current_running_config["role_player"]["name"] and current_running_config["role_player"]["language"]:
                    if current_running_config["role_player"]["name"] in roleplay_list:
                        role_enable = True
                        role_index = roleplay_list.index(current_running_config["role_player"]["name"])
                        lang_index = player_language_list.index(current_running_config["role_player"]["language"])
                    else:
                        role_index = 0
                        lang_index = 0
                        role_enable = False
                else:
                    role_index = 0
                    lang_index = 0
                    role_enable = False
                with st.form("Roleplay"):
                    col1, col2 = st.columns(2)
                    with col1:
                        current_roleplayer = st.selectbox(
                            "Please Select Role Player",
                            roleplay_list,
                            index=role_index,
                        )
                        role_enable = st.checkbox("Enable", value=role_enable, help="After enabling, The Role Play feature will activate.")
                    with col2:
                        roleplayer_language = st.selectbox(
                            "Please Select Language",
                            player_language_list,
                            index=lang_index,
                        )
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            if role_enable:
                                current_running_config["role_player"]["name"] = current_roleplayer
                                current_running_config["role_player"]["language"] = roleplayer_language
                            else:
                                current_running_config["role_player"]["name"] = ""
                                current_running_config["role_player"]["language"] = ""
                            api.save_current_running_config(current_running_config)
                            st.toast("success save configuration for Code Interpreter.", icon="✔")
            with tabfuncall:
                from WebUI.Server.funcall.funcall import GetFuncallList, GetFuncallDescription
                calling_enable = functioncalling.get("calling_enable", False)
                current_function = ""
                function_name_list = GetFuncallList()
                current_function = st.selectbox(
                    "Please Check Function",
                    function_name_list,
                    index=0,
                )
                with st.form("Funcall"):
                    description = GetFuncallDescription(current_function)
                    st.text_input("Description", description, disabled=True)
                    calling_enable = st.checkbox("Enable", key="funcall_box", value=calling_enable, help="After enabling, The function will be called automatically.")
                    print("calling_enable", calling_enable)
                    save_parameters = st.form_submit_button(
                        "Save Parameters",
                        use_container_width=True
                    )
                    if save_parameters:
                        with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
                            functioncalling["calling_enable"] = calling_enable
                            r = api.save_function_calling_config(functioncalling)
                            if msg := check_error_msg(r):
                                st.toast(msg, icon="✖")
                            elif msg := check_success_msg(r):
                                st.toast("success save configuration for function calling.", icon="✔")
    
    st.session_state["current_page"] = "configuration_page"

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()
            
