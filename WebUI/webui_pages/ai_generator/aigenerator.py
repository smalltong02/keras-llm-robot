import streamlit as st
from WebUI.webui_pages.utils import ApiRequest


def ai_generator_page(api: ApiRequest, is_lite: bool = False):
    
    st.session_state["current_page"] = "ai_generator_page"

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()