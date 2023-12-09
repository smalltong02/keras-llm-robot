import streamlit as st
from WebUI.webui_pages.utils import *
from streamlit_chatbox import *
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from aiortc.contrib.media import MediaRecorder
from WebUI.configs.prompttemplates import PROMPT_TEMPLATES
from WebUI.configs.modelconfig import HISTORY_LEN
import os, platform
from pathlib import Path
from datetime import datetime
import speech_recognition as sr
import torch
import time
import psutil
import pynvml

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "keras_llm_robot_webui_logo.jfif"
    )
)

gal_cpuutil = 0.0
gal_cpumem = 0.0
gal_gpuutil = 0.0
gal_gpumem = 0.0

TMP_DIR = Path('temp')
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)

def update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem, bshow = True, binit = False, bcache = False):
    global gal_cpuutil
    global gal_cpumem
    global gal_gpuutil
    global gal_gpumem

    if bshow:
        return
    
    if binit:
        gal_cpuutil = cpuutil = 0.0
        gal_cpuutil = cpumem = 0.0
        gal_cpuutil = gpuutil = 0.0
        gal_cpuutil = gpumem = 0.0
    else:
        if bcache:
            cpuutil = gal_cpuutil
            cpumem = gal_cpumem
            gpuutil = gal_gpuutil
            gpumem = gal_gpumem
        else:
            _, cpuutil, cpumem = get_cpu_info()
            _, gpuutil, gpumem = get_gpu_info()
    
    placeholder_cpu.caption(f"""<p style="font-size: 1em; text-align: center;">CPU Util: {cpuutil:.2f}%</p>""",
        unsafe_allow_html=True,
        )
    placeholder_ram.caption(f"""<p style="font-size: 1em; text-align: center;">CPU RAM: {cpumem:.2f} GB</p>""",
        unsafe_allow_html=True,
        )
    placeholder_gpuutil.caption(f"""<p style="font-size: 1em; text-align: center;">GPU Util: {gpuutil:.2f}%</p>""",
        unsafe_allow_html=True,
        )
    placeholder_gpumem.caption(f"""<p style="font-size: 1em; text-align: center;">GPU RAM: {gpumem:.2f} GB</p>""",
        unsafe_allow_html=True,
        )
    gal_cpuutil = cpuutil
    gal_cpumem = cpumem
    gal_gpuutil = gpuutil
    gal_gpumem = gpumem

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
    running_model = "None"
    models_list = list(api.get_running_models())
    if len(models_list) == 0:
        running_model = "None"
    else:
        running_model = models_list[0]

    webui_config = api.get_webui_config()
    chatconfig = webui_config.get("ChatConfiguration")
    webconfig = webui_config.get("WebConfig")
    temperature = chatconfig.get("Temperature")
    bshowstatus = webconfig.get("ShowRunningStatus")
    voicemodel = api.get_vtot_model()
    imagemodel = ""

    disabled = False
    voice_prompt = ""
    with st.sidebar:
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"Switched to {mode} mode."
            if mode == "KnowledgeBase Chat":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} Current Knowledge Base: `{cur_kb}`."
            st.toast(text)

        if running_model == "None":
            disabled = True

        dialogue_modes = ["LLM Chat",
                        "KnowledgeBase Chat",
                        "Search Engine Chat",
                        "Agent Chat",
                        ]
        dialogue_mode = st.selectbox("Please Select Chat Mode:",
                                    dialogue_modes,
                                    index=0,
                                    on_change=on_mode_change,
                                    key="dialogue_mode",
                                    disabled = disabled,
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
            "Please Select Prompt Template:",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            key="prompt_template_select",
            disabled=disabled
        )
        prompt_template_name = st.session_state.prompt_template_select
        history_len = st.number_input("Dialogue Turns:", 0, 20, HISTORY_LEN, disabled=disabled)

        now = datetime.now()
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button('New chat', use_container_width=True):
            chat_box.reset_history()
        export_btn.download_button(
            "Export chat",
            "".join(chat_box.export2md()),
            file_name=f"{now:%Y-%m-%d %H.%M}_chatrecord.md",
            mime="text/markdown",
            use_container_width=True,
        )

        voicedisable = True if voicemodel == "" else False
        if voicedisable == False:
            st.divider()
            st.write("Chat by 🎧 and 🎬: ")

            wavpath = TMP_DIR / "record.wav"
            def recorder_factory():
                return MediaRecorder(str(wavpath))

            webrtc_ctx = webrtc_streamer(
                key="sendonly-audio",
                mode=WebRtcMode.SENDRECV,
                in_recorder_factory=recorder_factory,
                # rtc_configuration={
                #     "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}] 
                # },
                client_settings=ClientSettings(
                    media_stream_constraints={
                        "video": False,
                        "audio": {
                            "echoCancellation": False,  # don't turn on else it would reduce wav quality
                            "noiseSuppression": True,
                            "autoGainControl": True,
                        },
                    },
                ),
                audio_html_attrs={"muted": True},
            )
            st.divider()
            voice_data = None
            if wavpath.exists() and not webrtc_ctx.state.playing:
                with open(wavpath, "rb") as file:
                    voice_data = file.read()
                if voice_data is not None:
                    try:
                        voice_prompt = api.get_vtot_data(voice_data)
                        print("voice_prompt: ", voice_prompt)
                        if voice_prompt:
                            st.success("Translation finished!")
                        else:
                            st.error("Translation failed...")
                        if running_model == "" or running_model == "None":
                            voice_prompt = ""
                    except Exception as e:
                        print(e)
                        st.error("Recording failed...")
                try:
                    wavpath.unlink()
                except Exception as e:
                    pass

        # voicechatbtn = st.button(label="🎧", use_container_width=True, disabled=voicedisable)
        # if voicechatbtn:
        #     with st.spinner(f"Voice recording starting..."):
        #         r = sr.Recognizer()
        #         with sr.Microphone() as source:
        #             st.success("Please speaking...")
        #             audio = r.listen(source, timeout=5, phrase_time_limit=5)
        #             try:
        #                 voice_data = audio.get_wav_data(convert_rate=16000)
        #                 voice_prompt = api.get_vtot_data(voice_data)
        #                 if voice_prompt:
        #                     st.success("Translation finished!")
        #                 else:
        #                     st.error("Translation failed...")
        #                 if running_model == "" or running_model == "None":
        #                     voice_prompt = ""
        #             except Exception as e:
        #                 print(e)
        #                 st.error("Recording failed...")

        imagedisable = True if imagemodel == "" else False
        imagechatbtn = st.button(label="🎨", use_container_width=True, disabled=imagedisable)
        if imagechatbtn:
            pass
        if bshowstatus:
            st.caption(
                f"""<p style="font-size: 1.5em; text-align: left; color: #3498db;"><b>Running Status:</b></p>""",
                unsafe_allow_html=True,
            )
            st.caption(
                f"""<p style="font-size: 1em; text-align: center;">Load Model: {running_model}</p>""",
                unsafe_allow_html=True,
            )

            #cpuname, cpuutil, cpumem = get_cpu_info(False)
            #if cpuname == "":
            #    cpuname = "Unknown"
            #st.caption(
            #    f"""<p style="font-size: 1em; text-align: center; color: #333333;">CPU Name: {cpuname}</p>""",
            #    unsafe_allow_html=True,
            #)
            placeholder_cpu = st.empty()
            placeholder_ram = st.empty()
            gpuname, _, _ = get_gpu_info()
            if gpuname == "":
                gpuname = "Unknown"
            st.caption(
                f"""<p style="font-size: 1em; text-align: center;">GPU Name: {gpuname}</p>""",
                unsafe_allow_html=True,
            )
            placeholder_gpuutil = st.empty()
            placeholder_gpumem = st.empty()
            binit = True
            if st.session_state.get("current_page", "") == "dialogue_page":
                binit = False
            update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem, bshowstatus, binit, True)

    if not chat_box.chat_inited:
        if running_model == "" or running_model == "None":
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
        "optional_text_label": "Please provide feedback on the reasons for your rating.",
    }

    
    prompt = st.chat_input(chat_input_placeholder, key="prompt", disabled=disabled)
    if voice_prompt != "":
        prompt = voice_prompt
    
    if prompt != None and prompt != "":
        if bshowstatus:
            update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem)
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
                
            if st.session_state.get("need_rerun"):
                st.session_state["need_rerun"] = False
                st.rerun()
                
            st.session_state["current_page"] = "dialogue_page"
            if bshowstatus:
                while True:
                    update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem)
                    time.sleep(1)
        
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

