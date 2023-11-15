import streamlit as st
from WebUI.webui_pages.utils import *

def configuration_page(api: ApiRequest, is_lite: bool = False):
    st.text_input(
        "新建知识库名称",
        placeholder="新知识库名称，不支持中文命名",
        key="kb_name",
    )