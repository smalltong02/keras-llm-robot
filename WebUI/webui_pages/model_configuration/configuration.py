import streamlit as st
from WebUI.webui_pages.utils import *
from WebUI.configs import *
from streamlit_modal import Modal
import streamlit.components.v1 as components

def configuration_page(api: ApiRequest, is_lite: bool = False):
    running_model = ""
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
        new_model = st.selectbox(
                "Please Select LLM Model:",
                all_model_lists,
                index=index,
            )
        if new_model == "None":
            disabled = True
    with col2:
        st.markdown("<style> .custom-space { height: 14px; } </style>", unsafe_allow_html=True)
        st.markdown("<div class='custom-space'></div>", unsafe_allow_html=True)
        loadbutton = st.button("Load", disabled=disabled)
        loadmodal = Modal(title="", key="modal_key", max_width=400)
        if loadbutton:
            if new_model.endswith("(running)"):
                with loadmodal.container():
                    st.write("Error")
                    st.markdown("""
                        The model is currently running!
                    """)
            else:
                with loadmodal.container():
                    with st.spinner(f"Loading Model: {new_model}, Please do not perform any actions or refresh the page."):
                        r = api.change_llm_model(running_model, new_model)
                        if msg := check_error_msg(r):
                            st.error(msg)
                        elif msg := check_success_msg(r):
                            st.success(msg)
    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["Parameter", "Quantization", "Fine-Tunning", "Prompt Templates"])

    with tab1:
        st.slider("Temperature: ", 0.0, 1.0, TEMPERATURE, 0.05, disabled=disabled)
    with tab2:
        quantization_lists = ["16-bit", "8-bit", "6-bit", "5-bit", "4-bit"]
        st.selectbox(
            "Quantization: ",
            quantization_lists,
            index=0,
            disabled=disabled
        )
    with tab3:
        device_lists = ["auto", "CPU", "GPU", "MPS"]
        st.selectbox(
            "Device: ",
            device_lists,
            index=0,
            disabled=disabled
        )
    with tab4:
        pass
