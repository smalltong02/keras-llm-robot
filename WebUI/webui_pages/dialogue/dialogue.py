import os
import platform
from datetime import datetime
import time
import base64
import psutil
import pynvml
import streamlit as st
from WebUI.webui_pages.utils import ApiRequest, check_error_msg
from streamlit_chatbox import ChatBox, Image, Audio, Video, Markdown
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from aiortc.contrib.media import MediaRecorder
from WebUI.configs.basicconfig import (TMP_DIR, ModelType, ModelSize, ModelSubType, GetModelInfoByName, GetTypeName, generate_prompt_for_imagegen, generate_prompt_for_smart_search, 
                                       use_search_engine, glob_multimodal_vision_list, glob_multimodal_voice_list, glob_multimodal_video_list)
from WebUI.configs.prompttemplates import PROMPT_TEMPLATES
from WebUI.configs.roleplaytemplates import ROLEPLAY_TEMPLATES
from WebUI.Server.funcall.funcall import use_function_calling
from io import BytesIO
from typing import List, Dict, Any

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
    print("running_model: ", running_model)
    webui_config = api.get_webui_config()
    chatconfig = webui_config.get("ChatConfiguration")
    webconfig = webui_config.get("WebConfig")
    temperature = chatconfig.get("Temperature")
    bshowstatus = webconfig.get("ShowRunningStatus")
    voicemodel = api.get_vtot_model()
    speechmodel = api.get_ttov_model()
    imagerecognition_model = api.get_image_recognition_model()
    imagegeneration_model = api.get_image_generation_model()
    musicgeneration_model = api.get_music_generation_model()
    functioncalling = webui_config.get("FunctionCalling")
    calling_enable = functioncalling.get("calling_enable", False)
    running_chat_solution = st.session_state.get("current_chat_solution", {})
    current_engine_name = ""
    current_smart = False
    current_search_engine = {}
    code_interpreter = {}
    role_player = {}
    if st.session_state.get("current_search_engine"):
        current_search_engine = st.session_state["current_search_engine"]
        current_engine_name = current_search_engine["engine"]
        current_smart = current_search_engine["smart"]
    if st.session_state.get("current_interpreter"):
        current_interpreter = st.session_state["current_interpreter"]
        code_interpreter = current_interpreter["interpreter"]
    if st.session_state.get("current_roleplayer"):
        current_role_player = st.session_state["current_roleplayer"]
        role_player = current_role_player
    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
    print("voicemodel: ", voicemodel)
    print("speechmodel: ", speechmodel)
    print("imagerecognition_model: ", imagerecognition_model)
    print("imagegeneration_model: ", imagegeneration_model)
    print("musicgeneration_model: ", musicgeneration_model)
    print("search_engine: ", current_search_engine)
    print("code_interpreter: ", code_interpreter)
    print("role_player: ", role_player)
    print("running_chat_solution: ", running_chat_solution)
    chat_solution_enable = running_chat_solution.get("enable", False)

    dialogue_turns = chatconfig.get("dialogue_turns", 5)
    disabled = False
    voice_prompt = ""
    binit = True
    if st.session_state.get("current_page", "") == "dialogue_page":
        binit = False
    negative_prompt = ""
    with st.sidebar:
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"Switched to {mode} mode."
            if mode == "KnowledgeBase Chat":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} Current Knowledge Base: `{cur_kb}`."
            #st.toast(text)

        if running_model == "None":
            disabled = True
        else:
            modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, running_model)
            modelinfo["mname"] = running_model
        print("msubtype: ", modelinfo["msubtype"])

        dialogue_modes = ["LLM Chat",
                        "KnowledgeBase Chat",
                        "Agent Chat",
                        ]
        mode_index = 0
        if chat_solution_enable:
            mode_index = 2
        elif st.session_state.get("selected_kb_name"):
            mode_index = 1
        dialogue_mode = st.selectbox("Please Select Chat Mode:",
                                    dialogue_modes,
                                    index=mode_index,
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
            _ = f"Switch to {prompt_template_name} Template。"
            #st.toast(text)

        _ = st.selectbox(
            "Please Select Prompt Template:",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            key="prompt_template_select",
            disabled=disabled
        )
        prompt_template_name = st.session_state.prompt_template_select
        history_len = st.number_input("Dialogue Turns:", 0, 20, dialogue_turns, disabled=disabled)

        kb_top_k = 0
        selected_kb = ""
        score_threshold = 0
        if dialogue_mode == "KnowledgeBase Chat":
            from WebUI.Server.knowledge_base.utils import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
            def on_kb_change():
                st.toast(f"Current Knowledge Base: `{st.session_state.selected_kb}`")
            with st.expander("Knowledge Base", True):
                kb_index = 0
                kb_list = api.list_knowledge_bases()
                if st.session_state.get("selected_kb_name"):
                    kb_index = kb_list.index(st.session_state.get("selected_kb_name"))
                selected_kb = st.selectbox(
                    "Please Select KB:",
                    kb_list,
                    index=kb_index,
                    on_change=on_kb_change,
                    key="selected_kb",
                    disabled=disabled
                )
                kb_top_k = st.number_input("Knowledge Counts:", 1, 20, VECTOR_SEARCH_TOP_K, disabled=disabled)
                score_threshold = st.slider("Score Threshold:", 0.0, 2.0, SCORE_THRESHOLD, 0.01, disabled=disabled)

        now = datetime.now()
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button('New chat', use_container_width=True):
            chat_box.reset_history()

        def md_callback(msg: Any):
            user_avatar : str = "User"
            ai_avatar : str = "AI"
            user_bg_color : str = "#DCFDC8"
            ai_bg_color : str = "#E0F7FA"
            def set_bg_color(text, bg_color):
                text = text.replace("\n", "<br>")
                return f"<div style=\"background-color:{bg_color}\">{text}</div>"
            contents = [e.content for e in msg["elements"]]
            if msg["role"] == "user":
                content = "<br><br>".join(set_bg_color(c, user_bg_color) for c in contents if isinstance(c, str))
                avatar = set_bg_color(user_avatar, user_bg_color)
            else:
                avatar = set_bg_color(ai_avatar, ai_bg_color)
                content = "<br><br>".join(set_bg_color(c, ai_bg_color) for c in contents if isinstance(c, str))
            line = f"|{avatar}|{content}|\n"
            return line
        export_btn.download_button(
            "Export chat",
            "".join(chat_box.export2md(callback=md_callback)),
            file_name=f"{now:%Y-%m-%d %H.%M}_chatrecord.md",
            mime="text/markdown",
            use_container_width=True,
        )
        
        if imagegeneration_model:
            negative_prompt = st.text_input(
                "Negative Prompt:",
                key="negative_prompt",
            )

        voicedisable = False if voicemodel != "" else True
        if voicedisable is False:
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
                except Exception as _:
                    pass

        imagesdata = []
        imagesprompt = []
        audiosdata = []
        videosdata = []
        imagedisable = False if imagerecognition_model != ""  or modelinfo["msubtype"] == ModelSubType.VisionChatModel else True
        audiodisable = False if modelinfo["msubtype"] == ModelSubType.VoiceChatModel else True
        videodisable = False if modelinfo["msubtype"] == ModelSubType.VideoChatModel else True
        
        if imagedisable is False:
            imagefiles = st.file_uploader("Please upload 🎨:",
                glob_multimodal_vision_list,
                accept_multiple_files=True,
                )
            if len(imagefiles):
                for imagefile in imagefiles:
                    print("image_file: ", imagefile)
                    print("image_type: ", imagefile.type)
                    def is_image_type(mime_type):
                        return mime_type.startswith('image/')
                    if is_image_type(imagefile.type):
                        imagesdata.append(imagefile.getvalue())
                    print("imagesdata size: ", len(imagesdata))
                if modelinfo["msubtype"] != ModelSubType.VisionChatModel and imagerecognition_model != "" and len(imagesdata):
                    try:
                        for imagedata in imagesdata:
                            image_prompt = api.get_image_recognition_data(imagedata)
                            imagesprompt.append(image_prompt)
                        print("imagesprompt: ", imagesprompt)
                        if running_model == "" or running_model == "None":
                            image_prompt = ""
                    except Exception as e:
                        print(e)

        elif audiodisable is False:
            audiofiles = st.file_uploader("Please upload 🎹:",
                glob_multimodal_voice_list,
                accept_multiple_files=True,
                )
            if len(audiofiles):
                for audiofile in audiofiles:
                    print("audio_file: ", audiofile)
                    print("audio_type: ", audiofile.type)
                    def is_audio_type(mime_type):
                        return mime_type.startswith('audio/')
                    if is_audio_type(audiofile.type):
                        audiosdata.append(audiofile.getvalue())
                    print("audiosdata size: ", len(audiosdata))
        elif videodisable is False:
            videofiles = st.file_uploader("Please upload 🎬:",
                glob_multimodal_video_list,
                accept_multiple_files=True,
                )
            if len(videofiles):
                for videofile in videofiles:
                    print("video_file: ", videofile)
                    print("video_type: ", videofile.type)
                    def is_video_type(mime_type):
                        return mime_type.startswith('video/')
                    if is_video_type(videofile.type):
                        videosdata.append(videofile.getvalue())
                    print("videosdata size: ", len(videosdata))

        for image in imagesdata:
            st.image(image=BytesIO(image))
        for audio in audiosdata:
            st.audio(data=audio)
        for video in videosdata:
            st.video(data=video)

        if  chat_solution_enable:
            solution_name = running_chat_solution["name"]
            st.caption(
                f"""<p style="font-size: 1.5em; text-align: center; color: #3498db;"><b>{solution_name}</b></p>""",
                unsafe_allow_html=True,
            )
            st.divider()
            model_name = running_chat_solution["config"]["llm_model"]
            st.caption(
                f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ {model_name}</b></p>""",
                unsafe_allow_html=True,
            )
            voice_enable = running_chat_solution["config"]["voice"]["enable"]
            if voice_enable:
                voice_model = running_chat_solution["config"]["voice"]["model"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ {voice_model}</b></p>""",
                    unsafe_allow_html=True,
                )
            speech_enable = running_chat_solution["config"]["speech"]["enable"]
            if speech_enable:
                speech_model = running_chat_solution["config"]["speech"]["model"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ {speech_model}</b></p>""",
                    unsafe_allow_html=True,
                )
            knowledge_enable = running_chat_solution["config"]["knowledge_base"]["enable"]
            if knowledge_enable:
                knowledge_base = running_chat_solution["config"]["knowledge_base"]["name"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ KB: {knowledge_base}</b></p>""",
                    unsafe_allow_html=True,
                )
            search_enable = running_chat_solution["config"]["search_engine"]["enable"]
            if search_enable:
                search_engine = running_chat_solution["config"]["search_engine"]["name"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ Search: {search_engine}</b></p>""",
                    unsafe_allow_html=True,
                )
            calling_enable = running_chat_solution["config"]["function_calling"]
            if calling_enable:
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ Function Calling: {calling_enable}</b></p>""",
                    unsafe_allow_html=True,
                )
        else:
            model_name = running_model
            if model_name == "None":
                model_name = ""
            st.caption(
                f"""<p style="font-size: 1.5em; text-align: center; color: #3498db;"><b>{model_name}</b></p>""",
                unsafe_allow_html=True,
            )
            if bshowstatus:
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
                update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem, bshowstatus, binit, True)

    if not chat_box.chat_inited:
        chat_box.init_session()

    if binit:
        if running_model == "" or running_model == "None":
            st.toast(
                "Currently, no models are configured. Please select model on the Model Configuration tab.\n"
                )
        else:
            type_name = GetTypeName(modelinfo["mtype"])
            st.toast(
                f"Welcome to use [`KERAS-llm-Robot`](https://github.com/smalltong02/keras-llm-robot).\n"
                f"The {type_name} `{running_model}` has been loaded."
            )

    st.session_state["current_page"] = "dialogue_page"   
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
    
    if imagegeneration_model or musicgeneration_model:
        prompt = st.chat_input(chat_input_placeholder, key="prompt")
    else:
        prompt = st.chat_input(chat_input_placeholder, key="prompt", disabled=disabled)
    if voice_prompt != "":
        prompt = voice_prompt

    if modelinfo["mtype"] == ModelType.Code:
        imagesdata = []
        audiosdata = []
        videosdata = []
    
    btranslate_prompt = False
    if prompt is not None and prompt != "":
        print("prompt: ", prompt)
        if bshowstatus:
            update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem)
        history = get_messages_history(history_len)
        prompt_list = [prompt]
        if imagesdata:
            for image in imagesdata:
                prompt_list.append(Image(image))
        if audiosdata:
            for audio in audiosdata:
                prompt_list.append(Audio(audio))
        if videosdata:
            for video in videosdata:
                prompt_list.append(Video(video))
        if imagesprompt:
            imagesdata = []
        chat_box.user_say(prompt_list)
        if dialogue_mode == "LLM Chat":
            if disabled:
                import uuid
                chat_history_id = uuid.uuid4().hex
                metadata = {
                    "chat_history_id": chat_history_id,
                    }
                if imagegeneration_model:
                    with st.spinner("Image generation in progress...."):
                        gen_image = api.get_image_generation_data(prompt, negative_prompt, False)
                        if gen_image:
                            chat_box.ai_say([""])
                            decoded_data = base64.b64decode(gen_image)
                            gen_image=Image(BytesIO(decoded_data))
                            chat_box.update_msg(gen_image, element_index=0, metadata=metadata)
                        chat_box.show_feedback(**feedback_kwargs,
                            key=chat_history_id,
                            on_submit=on_feedback,
                            kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
                if musicgeneration_model:
                    with st.spinner("Music generation in progress...."):
                        gen_music = api.get_music_generation_data(prompt, False)
                        if gen_music:
                            chat_box.ai_say([""])
                            decoded_data = base64.b64decode(gen_music)
                            gen_music=Audio(BytesIO(decoded_data))
                            chat_box.update_msg(gen_music, element_index=0)
                        chat_box.show_feedback(**feedback_kwargs,
                            key=chat_history_id,
                            on_submit=on_feedback,
                            kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
            else:
                if len(code_interpreter):
                    import uuid
                    chat_history_id = uuid.uuid4().hex
                    metadata = {
                        "chat_history_id": chat_history_id,
                        }
                    with st.spinner("Code Interpreter in progress...."):
                        r = api.code_interpreter_chat(
                                    prompt,
                                    interpreter_id=code_interpreter,
                                    model=running_model,
                                    temperature=temperature)
                        text = ""
                        chat_box.ai_say(["Think..."])
                        for t in r:
                            if error_msg := check_error_msg(t):  # check whether error occured
                                st.error(error_msg)
                                break
                            content = t.get("text", "")
                            if content.startswith("image-data:"):
                                chat_box.ai_say([""])
                                decoded_data = base64.b64decode(content[len("image-data:"):])
                                gen_image=Image(BytesIO(decoded_data))
                                chat_box.update_msg(gen_image, element_index=0, metadata=metadata)
                                chat_box.ai_say(["Think..."])
                                text = ""
                            elif content.startswith("image-file:"):
                                with open(content[len("image-file:"):], "rb") as f:
                                    image_bytes = f.read()
                                    gen_image=Image(BytesIO(image_bytes))
                                    chat_box.ai_say([""])
                                    chat_box.update_msg(gen_image, element_index=0, metadata=metadata)
                                    chat_box.ai_say(["Think..."])
                                    text = ""
                            else:
                                text += content
                                chat_box.update_msg(text, element_index=0)

                        chat_box.update_msg(text, element_index=0, streaming=False, metadata=metadata)
                        chat_box.show_feedback(**feedback_kwargs,
                            key=chat_history_id,
                            on_submit=on_feedback,
                            kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
                else:
                    return_video = False
                    if modelinfo["mtype"] == ModelType.Multimodal:
                        if running_model == "stable-video-diffusion-img2vid" or running_model == "stable-video-diffusion-img2vid-xt":
                            return_video = True
                    if current_engine_name == "" or current_smart is True:
                        if return_video:
                            chat_box.ai_say("")
                        else:
                            if current_smart is True:
                                chat_box.ai_say([
                                    "Thinking...",
                                    Markdown("...", in_expander=True, title="Internet search results", state="complete"),
                                ])
                            else:
                                chat_box.ai_say(["Thinking...", ""])
                        text = ""
                        chat_history_id = ""
                        if imagegeneration_model and modelinfo["mtype"] != ModelType.Code:
                            imageprompt = ""
                            if imagesprompt:
                                imageprompt = imagesprompt[0]
                            prompt, btranslate_prompt = generate_prompt_for_imagegen(imagegeneration_model, prompt, imageprompt)
                            imagesprompt = []
                            history = []
                        if return_video:
                            with st.spinner("Video generation in progress...."):
                                r = api.chat_chat(prompt,
                                            imagesdata=imagesdata,
                                            audiosdata=audiosdata,
                                            videosdata=videosdata,
                                            imagesprompt=imagesprompt,
                                            history=history,
                                            model=running_model,
                                            speechmodel=speechmodel,
                                            prompt_name=prompt_template_name,
                                            temperature=temperature)
                                for t in r:
                                    if error_msg := check_error_msg(t):  # check whether error occured
                                        st.error(error_msg)
                                        break
                                    text += t.get("text", "")
                                    chat_history_id = t.get("chat_history_id", "")
                                print("video_path: ", text)
                                with open(text, "rb") as f:
                                    video_bytes = f.read()
                                    gen_video=Video(BytesIO(video_bytes))
                                    chat_box.update_msg(gen_video, streaming=False)
                        else:
                            if current_smart is True:
                                new_prompt = generate_prompt_for_smart_search(prompt)
                            else:
                                new_prompt = prompt
                            if role_player:
                                role_template = ROLEPLAY_TEMPLATES[role_player["roleplayer"]][role_player["language"]]
                                history = [{
                                    'content': f'{role_template}',
                                    'role': 'system'
                                }] + history
                                new_prompt = ROLEPLAY_TEMPLATES[role_player["roleplayer"]][role_player["language"]+'-prompt'].format(prompt=new_prompt)
                                print("ROLEPLAY_TEMPLATES: ", new_prompt)
                            r = api.chat_chat(new_prompt,
                                            imagesdata=imagesdata,
                                            audiosdata=audiosdata,
                                            videosdata=videosdata,
                                            imagesprompt=imagesprompt,
                                            history=history,
                                            model=running_model,
                                            speechmodel=speechmodel,
                                            prompt_name=prompt_template_name,
                                            temperature=temperature)
                            for t in r:
                                if error_msg := check_error_msg(t):  # check whether error occured
                                    st.error(error_msg)
                                    break
                                text += t.get("text", "")
                                chat_box.update_msg(text, element_index=0)
                                chat_history_id = t.get("chat_history_id", "")
                                #print("text: ", text)
                            if calling_enable is True:
                                text = use_function_calling(text)
                                print("calling_text: ", text)
                            if current_smart is True and use_search_engine(text):
                                name = current_engine_name
                                chat_box.update_msg(f"Searching is now being conducted through `{name}`...", element_index=0)
                                chat_box.update_msg(Markdown("...", in_expander=True, title="Internet search results", state="complete"), element_index=1)
                                text = ""
                                for d in api.search_engine_chat(prompt,
                                                            search_engine_name=name,
                                                            history=history,
                                                            model=running_model,
                                                            prompt_name=prompt_template_name,
                                                            temperature=temperature):
                                    if error_msg := check_error_msg(d):
                                        st.error(error_msg)
                                    elif chunk := d.get("answer"):
                                        text += chunk
                                        chat_box.update_msg(text, element_index=0)
                                chat_box.update_msg(text, element_index=0, streaming=False)
                                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

                            metadata = {
                                "chat_history_id": chat_history_id,
                                }
                            chat_box.update_msg(text, element_index=0, streaming=False, metadata=metadata)
                            if imagegeneration_model and modelinfo["mtype"] != ModelType.Code:
                                with st.spinner("Image generation in progress...."):
                                    gen_image = api.get_image_generation_data(text, negative_prompt, btranslate_prompt)
                                    if gen_image:
                                        decoded_data = base64.b64decode(gen_image)
                                        gen_image=Image(BytesIO(decoded_data))
                                        chat_box.update_msg(gen_image, element_index=1, streaming=False)
                        #print("chat_box.history: ", len(chat_box.history))
                        chat_box.show_feedback(**feedback_kwargs,
                                            key=chat_history_id,
                                            on_submit=on_feedback,
                                            kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
                    else:
                        if return_video is False:
                            name = current_engine_name
                            chat_box.ai_say([
                                f"Searching is now being conducted through `{name}`...",
                                Markdown("...", in_expander=True, title="Internet search results", state="complete"),
                            ])
                            text = ""
                            for d in api.search_engine_chat(prompt,
                                                        search_engine_name=name,
                                                        history=history,
                                                        model=running_model,
                                                        prompt_name=prompt_template_name,
                                                        temperature=temperature):
                                if error_msg := check_error_msg(d):
                                    st.error(error_msg)
                                elif chunk := d.get("answer"):
                                    text += chunk
                                    chat_box.update_msg(text, element_index=0)
                            chat_box.update_msg(text, element_index=0, streaming=False)
                            chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

        elif dialogue_mode == "KnowledgeBase Chat":
            if len(selected_kb) and not disabled:
                chat_box.ai_say([
                    f"Querying from knowledge base `{selected_kb}` ...",
                    Markdown("...", in_expander=True, title="Knowledge base match results", state="complete"),
                ])
                text = ""
                for d in api.knowledge_base_chat(prompt,
                                                knowledge_base_name=selected_kb,
                                                top_k=kb_top_k,
                                                score_threshold=score_threshold,
                                                history=history,
                                                model=running_model,
                                                imagesdata=imagesdata,
                                                speechmodel=speechmodel,
                                                prompt_name=prompt_template_name,
                                                temperature=temperature):
                    if error_msg := check_error_msg(d):  # check whether error occured
                        st.error(error_msg)
                    elif chunk := d.get("answer"):
                        text += chunk
                        chat_box.update_msg(text, element_index=0)
                chat_box.update_msg(text, element_index=0, streaming=False)
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)
        elif dialogue_mode == "Agent Chat":
            if chat_solution_enable:
                chat_box.ai_say([
                    "Thinking...",
                ])
                text = ""
                chat_history_id = 0
                for d in api.chat_solution_chat(prompt,
                        imagesdata=imagesdata,
                        audiosdata=audiosdata,
                        videosdata=videosdata,
                        history=history,
                        chat_solution=running_chat_solution,
                        temperature=temperature):
                    print("d: ", d)
                    if error_msg := check_error_msg(d):  # check whether error occured
                            st.error(error_msg)
                    else:
                        chunk = d.get("text", "")
                        if chunk:
                            chat_history_id = d.get("chat_history_id", "")
                            text += chunk
                            chat_box.update_msg(text, element_index=0)
                        elif d.get("cmd", "") == "clear":
                            text = ""
                            chat_box.update_msg(text, element_index=0)
                metadata = {
                    "chat_history_id": chat_history_id,
                    }
                print("asdfadfadqwer")
                chat_box.update_msg(text, element_index=0, streaming=False, metadata=metadata)
                chat_box.show_feedback(**feedback_kwargs,
                    key=chat_history_id,
                    on_submit=on_feedback,
                    kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
            
        if bshowstatus:
            while True:
                update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem)
                time.sleep(1)
        
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

