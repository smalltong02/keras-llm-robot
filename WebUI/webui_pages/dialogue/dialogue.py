import streamlit as st
from WebUI.webui_pages.utils import *
from streamlit_chatbox import *
from WebUI.configs.prompttemplates import PROMPT_TEMPLATES
from WebUI.configs.modelconfig import (TEMPERATURE, HISTORY_LEN)
import os, platform
import time
import psutil
import GPUtil
import pynvml

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "keras_llm_robot_webui_logo.jfif"
    )
)

def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)

def get_default_llm_model(api: ApiRequest) -> str:
    running_models = api.get_running_models()
    if not running_models:
        return ""
    return running_models

gal_cpu_usage = 0.0

def get_cpu_info(busage = True):
    global gal_cpu_usage
    cpu_name = platform.processor()
    if busage:
        gal_cpu_usage = psutil.cpu_percent(interval=1)
    meminfo = psutil.virtual_memory()
    used = meminfo.used
    return cpu_name, gal_cpu_usage, used / (1024 ** 3)

def get_gpu_info():
    pynvml.nvmlInit()
    gpucount = pynvml.nvmlDeviceGetCount()
    if gpucount == 0:
        return "", 0.0, 0.0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpuname = pynvml.nvmlDeviceGetName(handle)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pynvml.nvmlShutdown()
    return gpuname, utilization.gpu, meminfo.used / (1024 ** 3)

def dialogue_page(api: ApiRequest, is_lite: bool = False):
    with st.sidebar:
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"Switched to {mode} mode."
            if mode == "KnowledgeBase Chat":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} Current Knowledge Base： `{cur_kb}`."
            st.toast(text)

        dialogue_modes = ["LLM Chat",
                        "KnowledgeBase Chat",
                        "Search Engine Chat",
                        "Agent Chat",
                        ]
        dialogue_mode = st.selectbox("Please Select Chat Mode：",
                                    dialogue_modes,
                                    index=0,
                                    on_change=on_mode_change,
                                    key="dialogue_mode",
                                    )
        
        index_prompt = {
            "LLM Chat": "llm_chat",
            "Search Engine Chat": "search_engine_chat",
            "KnowledgeBase Chat": "knowledge_base_chat",
            "Agent Chat": "agent_chat",
        }

        prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())
        prompt_template_name = prompt_templates_kb_list[0]
        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():
            text = f"Switch to {prompt_template_name} Template。"
            st.toast(text)

        prompt_template_select = st.selectbox(
            "Please Select Prompt Template：",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            key="prompt_template_select",
        )
        prompt_template_name = st.session_state.prompt_template_select
        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("Dialogue Turns：", 0, 20, HISTORY_LEN)

        st.caption(
            f"""<p style="font-size: 1.5em; text-align: left; color: #3498db;"><b>Running Status:</b></p>""",
            unsafe_allow_html=True,
        )

        running_model = "None"
        models_list = list(api.get_running_models())
        if len(models_list) == 0:
            running_model = "None"
        else:
            running_model = models_list[0]
        st.caption(
            f"""<p style="font-size: 1em; text-align: center; color: #333333;">Load Model：{running_model}</p>""",
            unsafe_allow_html=True,
        )

        cpuname, cpuutil, cpumem = get_cpu_info(False)
        #if cpuname == "":
        #    cpuname = "Unknown"
        #st.caption(
        #    f"""<p style="font-size: 1em; text-align: center; color: #333333;">CPU Name: {cpuname}</p>""",
        #    unsafe_allow_html=True,
        #)
        placeholder_cpu = st.empty()
        cpuutil = gal_cpu_usage
        placeholder_cpu.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">CPU Util：{cpuutil:.2f}%</p>""",
            unsafe_allow_html=True,
            )
        placeholder_ram = st.empty()
        placeholder_ram.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">CPU RAM: {cpumem:.2f} GB</p>""",
            unsafe_allow_html=True,
            )
        gpuname, gpuutil, gpumem = get_gpu_info()
        if gpuname == "":
            gpuname = "Unknown"
        st.caption(
            f"""<p style="font-size: 1em; text-align: center; color: #333333;">GPU Name: {gpuname}</p>""",
            unsafe_allow_html=True,
        )
        placeholder_gpuutil = st.empty()
        placeholder_gpuutil.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">GPU Util: {gpuutil:.2f}%</p>""",
            unsafe_allow_html=True,
            )
        placeholder_gpumem = st.empty()
        placeholder_gpumem.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">GPU RAM: {gpumem:.2f} GB</p>""",
            unsafe_allow_html=True,
            )

    if not chat_box.chat_inited:
        if running_model == "":
            st.toast(
                f"Currently, no models are configured. Please select model on the Model Configuration tab.\n"
                )
        else:
            st.toast(
                f"Welcome to use [`Langchain-KERAS-llm-Robot`](https://github.com/smalltong02/keras-llm-robot).\n"
                f"The model `{running_model}` has been loaded."
            )
        chat_box.init_session()
    # Display chat messages from history on app rerun
    chat_box.output_messages()
    chat_input_placeholder = "Enter a user message, new line: Shift+Enter "

    def on_feedback(
        feedback,
        chat_history_id: str = "",
        history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(chat_history_id=chat_history_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM Chat":
            chat_box.ai_say("Thinking...")
            text = ""
            chat_history_id = ""
            r = api.chat_chat(prompt,
                              history=history,
                              model=running_model,
                              prompt_name=prompt_template_name,
                              temperature=temperature)
            for t in r:
                if error_msg := check_error_msg(t):  # check whether error occured
                    st.error(error_msg)
                    break
                text += t.get("text", "")
                chat_box.update_msg(text)
                chat_history_id = t.get("chat_history_id", "")

            metadata = {
                "chat_history_id": chat_history_id,
                }
            chat_box.update_msg(text, streaming=False, metadata=metadata)
            chat_box.show_feedback(**feedback_kwargs,
                                   key=chat_history_id,
                                   on_submit=on_feedback,
                                   kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
    
    while True:
        _, cpuutil, cpumem = get_cpu_info()
        placeholder_cpu.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">CPU Util：{cpuutil:.2f}%</p>""",
            unsafe_allow_html=True,
            )
        placeholder_ram.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">CPU RAM: {cpumem:.2f} GB</p>""",
            unsafe_allow_html=True,
            )
        _, gpuutil, gpumem = get_gpu_info()
        placeholder_gpuutil.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">GPU Util: {gpuutil:.2f}%</p>""",
            unsafe_allow_html=True,
            )
        placeholder_gpumem.caption(f"""<p style="font-size: 1em; text-align: center; color: #333333;">GPU RAM: {gpumem:.2f} GB</p>""",
            unsafe_allow_html=True,
            )
        time.sleep(1)