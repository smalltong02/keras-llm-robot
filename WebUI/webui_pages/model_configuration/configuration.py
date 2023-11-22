import streamlit as st
from WebUI.webui_pages.utils import *
from WebUI.configs import *
from streamlit_modal import Modal
import streamlit.components.v1 as components
from WebUI.webui_pages import *
import time

def configuration_page(api: ApiRequest, is_lite: bool = False):
    running_model = ""
    webui_config = api.get_webui_config()

    models_list = list(api.get_running_models())
    if len(models_list):
        running_model = models_list[0]
    
    localmodel_lists = [f"{key}" for key in MODEL_PATH["llm_model"]]
    onlinemodel_lists = [config.get("model_list", []) for config in ONLINE_LLM_MODEL.values()]
    onlinemodel_lists = [model for sublist in onlinemodel_lists for model in sublist]
    onlinemodel_lists = [model for model in onlinemodel_lists if model]
    all_model_lists = ["None"] + localmodel_lists + onlinemodel_lists

    index = 0
    if running_model != "":
        try:
            index = all_model_lists.index(running_model)
            all_model_lists[index] += " (running)"
        except ValueError:
            index = 0

    col1, col2 = st.columns(2)
    disabled = False
    with col1:
        with st.form("Select Model"):
            new_model = st.selectbox(
                    "Please Select LLM Model",
                    all_model_lists,
                    index=index,
                )
            #if new_model == "None":
            #    disabled = True
            le_button = st.form_submit_button(
                "Load & Eject",
                use_container_width=True,
            )
            if le_button:
                if new_model != "None":
                    if new_model.endswith("(running)"):
                        with st.spinner(f"Release Model: {new_model}, Please do not perform any actions or refresh the page."):
                            r = api.eject_llm_model(running_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
                                new_model = "None"
                    else:
                        with st.spinner(f"Loading Model: {new_model}, Please do not perform any actions or refresh the page."):
                            r = api.change_llm_model(running_model, new_model)
                            if msg := check_error_msg(r):
                                st.error(msg)
                            elif msg := check_success_msg(r):
                                st.success(msg)
    with col2:
        with st.form("Local Path"):
            pathstr = st.text_input("Local Path")
            save_path = st.form_submit_button(
                "Save Path",
                use_container_width=True,
            )
            if save_path:
                with st.spinner(f"Saving path, Please do not perform any actions or refresh the page."):
                    time.sleep(1)
                    if new_model != "None":
                        st.success("Save path success!")
                    else:
                        st.error("Save path failed!")
    st.divider()
    tabparams, tabquant, tabembedding, tabtunning, tabprompt = st.tabs(["Parameters", "Quantization", "Embedding Mode", "Fine-Tunning", "Prompt Templates"])

    with tabparams:
        with st.form("Parameter"):
            col1, col2 = st.columns(2)
            with col1:
                predict_dev = st.selectbox(
                    "Please select Device",
                    ["Auto","CPU","GPU","MPS"],
                    index=0,
                    disabled=disabled
                )
                threads_input = st.number_input("CPU Thread", value = 4, min_value=1, max_value=32, disabled=disabled)
                st.slider("Max tokens", 0, 5000, 1000, 10, disabled=disabled)
                st.slider("Temperature", 0.0, 1.0, TEMPERATURE, 0.05, disabled=disabled)
                st.slider("epsilon_cutoff", 0.0, 1.0, 0.0, 0.1, disabled=disabled)
                st.slider("eta_cutoff", 0.0, 1.0, 0.0, 0.1, disabled=disabled)
                st.slider("diversity_penalty", 0.0, 1.0, 0.0, 0.1, disabled=disabled)
                st.slider("repetition_penalty", 0.0, 3.0, 1.15, 0.05, disabled=disabled)
                st.slider("length_penalty", 0, 2, 1, 1, disabled=disabled)
                st.slider("encoder_repetition_penalty", 0.0, 3.0, 1.15, 0.05, disabled=disabled)
            with col2:
                seed_input = st.number_input("Seed (-1 for random)", value = -1, min_value=-1, max_value=100, disabled=disabled)
                predict_dev = st.selectbox(
                    "Load Bits",
                    ["32 bits","16 bits","8 bits","4 bits"],
                    index=0,
                    disabled=disabled
                )
                st.slider("Top_p", 0.0, 1.0, 0.9, 0.1, disabled=disabled)
                st.slider("Top_k", 0, 200, 50, 1, disabled=disabled)
                st.slider("Typical_p", 0.0, 1.0, 1.0, 0.1, disabled=disabled)
                st.slider("Top_a", 0.0, 1.0, 1.0, 0.1, disabled=disabled)
                st.slider("tfs", 0.0, 1.0, 1.0, 0.1, disabled=disabled)
                st.slider("no_repeat_ngram_size", 0, 1, 0, 1, disabled=disabled)
                st.slider("guidance_scale", 0, 2, 1, 1, disabled=disabled)
                st.text("")
                st.text("")
                st.text("")
                st.checkbox('do samples', disabled=disabled)
            submit_create_kb = st.form_submit_button(
                "Save Parameters",
                use_container_width=True,
                disabled=disabled
            )
    with tabquant:
        with st.form("Quantization"):
            methods_lists = ["AutoGPTQ", "ExllamaV2", "Llamacpp"]
            st.selectbox(
                "Methods",
                methods_lists,
                index=0,
                disabled=disabled
            )
            quantization_lists = ["16 bits", "8 bits", "6 bits", "5 bits", "4 bits"]
            st.selectbox(
                "Quantization Bits",
                quantization_lists,
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
        embedding_lists = [f"{key}" for key in MODEL_PATH["embedding_model"]]

        st.selectbox(
            "Please Select Embedding Model",
            embedding_lists,
            index=0
        )

    with tabtunning:
        device_lists = ["auto", "CPU", "GPU", "MPS"]
        st.selectbox(
            "Please select Device",
            device_lists,
            index=0,
            disabled=disabled
        )
    with tabprompt:
        pass

    st.session_state["current_page"] = "configuration_page"
