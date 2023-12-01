import streamlit as st
from WebUI.webui_pages.utils import *

def tools_agent_page(api: ApiRequest, is_lite: bool = False):
    
    running_model = ""
    models_list = list(api.get_running_models())
    if len(models_list):
        running_model = models_list[0]
    if running_model == "":
        running_model = "None"
    st.caption(
            f"""<h1 style="font-size: 1.5em; text-align: center; color: #3498db;">Running Model: {running_model}</h1>""",
            unsafe_allow_html=True,
        )
    tabretrieval, tabinterpreter, tabttov, tabvtot, tabimager, tabimageg, tabfunctions = st.tabs(["Retrieval", "Code Interpreter", "Text-to-Voice", "Voice-to-Text", "Image Recognition", "Image Generation", "Functions"])
    with tabretrieval:
        selected_kb = st.selectbox(
            "Knowledge Base:",
            ["[Create New...]"],
            index=0
        )

        if selected_kb == "[Create New...]":
            with st.form("Create Knowledge Base"):

                kb_name = st.text_input(
                    "Knowledge Base Name:",
                    key="kb_name",
                )
                kb_info = st.text_input(
                    "Introduction to the Knowledge Base",
                    key="kb_info",
                )

            #     cols = st.columns(2)
            #     vs_types = list(kbs_config.keys())
            #     vs_type = cols[0].selectbox(
            #         "向量库类型",
            #         vs_types,
            #         index=vs_types.index(DEFAULT_VS_TYPE),
            #         key="vs_type",
            #     )

            #     embed_models = list_embed_models()

            #     embed_model = cols[1].selectbox(
            #         "Embedding 模型",
            #         embed_models,
            #         index=embed_models.index(EMBEDDING_MODEL),
            #         key="embed_model",
            #     )

                submit_create_kb = st.form_submit_button(
                    "Create",
                    # disabled=not bool(kb_name),
                    use_container_width=True,
                )

            if submit_create_kb:
                pass
                # if not kb_name or not kb_name.strip():
                #     st.error(f"知识库名称不能为空！")
                # elif kb_name in kb_list:
                #     st.error(f"名为 {kb_name} 的知识库已经存在！")
                # else:
                #     ret = api.create_knowledge_base(
                #         knowledge_base_name=kb_name,
                #         vector_store_type=vs_type,
                #         embed_model=embed_model,
                #     )
                #     st.toast(ret.get("msg", " "))
                #     st.session_state["selected_kb_name"] = kb_name
                #     st.session_state["selected_kb_info"] = kb_info
                #     st.experimental_rerun()

    with tabinterpreter:
        pass

    with tabttov:
        pass

    with tabvtot:
        pass

    with tabimager:
        pass

    with tabimageg:
        pass

    with tabfunctions:
        pass

    st.session_state["current_page"] = "retrieval_agent_page"