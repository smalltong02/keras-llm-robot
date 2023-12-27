import streamlit as st
import os
import sys
from streamlit_option_menu import option_menu
from WebUI.configs.modelconfig import KERAS_LLM_VERSION
from WebUI.webui_pages.utils import *
from WebUI.configs.serverconfig import API_SERVER
from WebUI.configs import HTTPX_DEFAULT_TIMEOUT
from WebUI.webui_pages.dialogue.dialogue import dialogue_page, chat_box
from WebUI.webui_pages.model_configuration.configuration import configuration_page
from WebUI.webui_pages.tools_agent.toolsagent import tools_agent_page

def api_address() -> str:
    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"

api = ApiRequest(base_url=api_address(), timeout=HTTPX_DEFAULT_TIMEOUT)

if __name__ == "__main__":
    is_lite = "lite" in sys.argv

    st.set_page_config(
        "Langchain-Keras-llm-robot WebUI",
        os.path.join("img", "keras_llm_robot_webui_logo.jfif"),
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/smalltong02/keras-llm-robot',
            'Report a bug': "https://github.com/smalltong02/keras-llm-robot/issues",
            'About': f"""Langchain-Keras llm Robot WebUI {KERAS_LLM_VERSION}!"""
        }
    )

    pages = {
        "Dialogue": {
            "icon": "chat-left-text",
            "func": dialogue_page,
        },
        "Model Configuration": {
            "icon": "magic",
            "func": configuration_page,
        },
        "Tools & Agent": {
            "icon": "archive-fill",
            "func": tools_agent_page,
        },
    }

    with st.sidebar:
        st.caption(
            f"""<h1 style="font-size: 2.5em; text-align: center; color: #3498db;">KERAS LLM Robot</h1>""",
            unsafe_allow_html=True,
        )
        st.image(
            os.path.join(
                "img",
                "keras_llm_robot_webui_gui.jfif"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p style="text-align: right; color: #3498db;">Current Version: {KERAS_LLM_VERSION}</p>""",
            unsafe_allow_html=True,
        )

        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
           #menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api=api, is_lite=is_lite)
