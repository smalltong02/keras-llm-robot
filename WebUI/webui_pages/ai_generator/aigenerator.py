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
    aigenerator_config = api.get_aigenerator_config()

    if running_model == "":
        running_model = "None"

    tabchatsolu, tabcreateivesolu = st.tabs(["Chat Solutions", "Creative Solutions"])
    with tabchatsolu:
        chat_solutions = aigenerator_config.get("Chat Solutions", {})
        chat_solutions_list = []
        for key, value in chat_solutions.items():
            if isinstance(value, dict):
                chat_solutions_list.append(key)
        current_chat_solution = st.selectbox(
            "Please select Chat Solutions",
            chat_solutions_list,
            index=0,
        )
        print("current_chat_solution: ", current_chat_solution)
        description = chat_solutions[current_chat_solution]["descriptions"]
        print("description: ", description)
        with st.form("ChatSolutions"):
            st.text_input("Description", description[0:10]+"....", help=description, disabled=True)
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("System Prompt", placeholder="[Please describe your product]")
            with col2:
                pass
            running = st.form_submit_button(
                        "Running",
                        use_container_width=True,
                    )
            if running:
                pass

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