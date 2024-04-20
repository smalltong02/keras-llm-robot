import streamlit as st
from WebUI.webui_pages.utils import ApiRequest


def ai_generator_page(api: ApiRequest, is_lite: bool = False):
    
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
    voicemodel = None

    if running_model == "":
        running_model = "None"

    tabtexttask, tabspetask, tabimgtask, tabvidtask, tabcodtask = st.tabs(["Text Creation", "Voice Creation", "Image Creation", "Video Creation", "Code Creation"])
    with tabtexttask:
        pass

    with tabspetask:
        pass

    with tabimgtask:
        pass

    with tabvidtask:
        pass

    with tabcodtask:
        pass

    st.session_state["current_page"] = "ai_generator_page"

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()