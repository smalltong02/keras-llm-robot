import streamlit as st
from WebUI.webui_pages.utils import *

def knowledge_base_page(api: ApiRequest, is_lite: bool = False):
    st.session_state["current_page"] = "knowledge_base_page"