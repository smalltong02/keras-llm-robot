from typing import Dict
import streamlit as st
from WebUI.webui_pages.utils import ApiRequest
from WebUI.webui_pages.utils import check_error_msg, check_success_msg

def last_stage_chat_solution(solution: str, stage: int) -> bool:
    pass

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
        if not chat_solutions or chat_solutions.get("stage", 0) == 0:
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
            running_chat_solution = st.session_state.get("current_chat_solution", {})
            if not running_chat_solution:
                running_chat_solution = {
                    "stage": 0,
                    "name": current_chat_solution,
                    "enable": False,
                    "config": chat_solutions[current_chat_solution],
                }
        print("running_chat_solution: ", running_chat_solution)
        col1, col2, col3 = st.columns(3)
        with col1:
            if running_chat_solution["stage"] == 0:
                prev_disabled = True
            else:
                prev_disabled = False
            prev_button = st.button(
            "Prev",
            use_container_width=True,
            disabled=prev_disabled
            )
            if prev_button:
                running_chat_solution["stage"] -= 1
        with col2:
            if last_stage_chat_solution(current_chat_solution, running_chat_solution["stage"]):
                next_disabled = True
            else:
                next_disabled = False
            next_button = st.button(
            "Next",
            use_container_width=True,
            disabled=next_disabled
            )
            if next_button:
                running_chat_solution["stage"] += 1
        with col3:
            clear_button = st.button(
            "Clear",
            use_container_width=True,
            disabled=False
            )
            if clear_button:
                running_chat_solution["stage"] = 0

        st.session_state["current_chat_solution"] = running_chat_solution
        # description = chat_solutions[current_chat_solution]["descriptions"]
        # print("description: ", description)
        # st.text_input("Solution Description", description[0:10]+"....", help=description, disabled=True)
        # st.divider()
        # if current_chat_solution == "Intelligent Customer Support":
        #     product_description = st.text_input("Product Description", placeholder="[Please describe your product]", help="Step 1: Please describe your product.")
        #     st.divider()
        #     search_enable = st.checkbox('Search Engine', value=False, help="Step 2: Intelligent selection from network search results when enable Search Engine.")
        #     if search_enable:
        #         searchengine = webui_config.get("SearchEngine")
        #         search_engine_list = []
        #         for key, value in searchengine.items():
        #             if isinstance(value, dict):
        #                 search_engine_list.append(key)
        #         current_search_engine = st.session_state.get("current_search_engine", {})
        #         if current_search_engine:
        #             search_enable = True
        #             smart_search = current_search_engine["smart"]
        #             index = search_engine_list.index(current_search_engine["engine"])
        #         else:
        #             index = 0
        #             smart_search = False

        #         current_search_engine = st.selectbox(
        #             "Please select Search Engine",
        #             search_engine_list,
        #             index=index,
        #         )
        #         with st.form("SearchEngine"):
        #             col1, col2 = st.columns(2)
        #             top_k = searchengine.get("top_k", 3)
        #             #search_url = searchengine.get(current_search_engine).get("search_url", "")
        #             api_key = searchengine.get(current_search_engine).get("api_key", "")
        #             with col1:
        #                 api_key = st.text_input("API Key", api_key, type="password")
        #                 smart_search = st.checkbox('Smart Search', value=smart_search, help="Let the model handle the question first, and let the model decide whether to invoke the search engine.")
        #             with col2:
        #                 top_k = st.slider("Top_k", 1, 10, top_k, 1)
        #             save_parameters = st.form_submit_button(
        #                 "Save Parameters",
        #                 use_container_width=True,
        #             )
        #             if save_parameters:
        #                 searchengine.get(current_search_engine)["api_key"] = api_key
        #                 searchengine["top_k"] = top_k
        #                 with st.spinner("Saving Parameters, Please do not perform any actions or refresh the page."):
        #                     r = api.save_search_engine_config(searchengine)
        #                     if msg := check_error_msg(r):
        #                         st.toast(msg, icon="✖")
        #                     elif msg := check_success_msg(r):
        #                         st.toast("success save configuration for search engine.", icon="✔")
        #         if search_enable:
        #             st.session_state["current_search_engine"] = {"engine": current_search_engine, "smart": smart_search}
        #         else:
        #             st.session_state["current_search_engine"] = {}
        #     st.divider()
        #     knowledge_base_enable = st.checkbox('Knowledge Base', value=False, help="Step 3: Firstly retrieve content from the Knowledge Base when enable Knowledge Base.")
        #     with st.form("ChatSolutions"):
        #         col1, col2 = st.columns(2)
        #         with col1:
        #             pass
        #         with col2:
        #             pass
        #         save_parameters = st.form_submit_button(
        #                     "Save Parameters",
        #                     use_container_width=True,
        #                 )
        #         if save_parameters:
        #             pass

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