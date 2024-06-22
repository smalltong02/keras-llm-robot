import os
import platform
from datetime import datetime
import time
import base64
import psutil
import pynvml
import folium
import streamlit as st
from streamlit_folium import st_folium
from WebUI.webui_pages.utils import ApiRequest, check_error_msg
from streamlit_chatbox import ChatBox, Image, Audio, Video, Markdown
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
from aiortc.contrib.media import MediaRecorder
from WebUI.configs.basicconfig import (TMP_DIR, ModelType, ModelSize, ModelSubType, ToolsType, GetModelInfoByName, GetTypeName, generate_prompt_for_imagegen, 
                                       is_toolboxes_enable, glob_multimodal_vision_list, glob_multimodal_voice_list, glob_multimodal_video_list)
from WebUI.configs.prompttemplates import PROMPT_TEMPLATES
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
    current_running_config = api.get_current_running_config()
    modelinfo : Dict[str, any] = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str}
    print("voicemodel: ", voicemodel)
    print("speechmodel: ", speechmodel)
    print("imagerecognition_model: ", imagerecognition_model)
    print("imagegeneration_model: ", imagegeneration_model)
    print("musicgeneration_model: ", musicgeneration_model)
    chat_solution_enable = bool(current_running_config["chat_solution"]["name"])

    dialogue_turns = chatconfig.get("dialogue_turns", 5)
    disabled = False
    voice_language = ""
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
                cur_kb = current_running_config["knowledge_base"]["name"]
                if cur_kb:
                    text = f"{text} Current Knowledge Base: `{cur_kb}`."
            st.toast(text)

        if running_model == "None":
            disabled = True
        else:
            modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, running_model)
            modelinfo["mname"] = running_model
        print("msubtype: ", modelinfo["msubtype"])

        dialogue_modes = ["LLM Chat",
                        "KnowledgeBase Chat",
                        "File Chat",
                        "Agent Chat"
                        ]
        mode_index = 0
        if chat_solution_enable:
            mode_index = 0
        elif current_running_config["knowledge_base"]["name"]:
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
            "KnowledgeBase Chat": "knowledge_base_chat",
            "File Chat": "knowledge_base_chat",
            "Agent Chat": "agent_chat"
        }
        prompt_templates_kb_list = list(PROMPT_TEMPLATES[index_prompt[dialogue_mode]].keys())
        prompt_template_name = prompt_templates_kb_list[0]
        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():
            prompt_template_name = st.session_state.prompt_template_select
            text = f"Switch to {prompt_template_name} Template。"
            st.toast(text)

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
        if dialogue_mode == "LLM Chat":
            current_running_config["enable"] = True
        elif dialogue_mode == "KnowledgeBase Chat":
            from WebUI.Server.knowledge_base.utils import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
            def on_kb_change():
                st.toast(f"Current Knowledge Base: `{st.session_state.selected_kb}`")
                current_running_config["knowledge_base"]["name"] = st.session_state.selected_kb
                api.save_current_running_config(current_running_config)
            with st.expander("Knowledge Base", True):
                kb_index = 0
                kb_list = api.list_knowledge_bases()
                if current_running_config["knowledge_base"]["name"]:
                    kb_index = kb_list.index(current_running_config["knowledge_base"]["name"])
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
            current_running_config["enable"] = False
        elif dialogue_mode == "File Chat":
            from WebUI.Server.knowledge_base.utils import VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, LOADER_DICT
            with st.expander("File Chat Configuration", True):
                files = st.file_uploader("Upload Files:",
                                         [i for ls in LOADER_DICT.values() for i in ls],
                                         accept_multiple_files=True,
                                         disabled=disabled
                                         )
                kb_top_k = st.number_input("Knowledge Counts:", 1, 20, VECTOR_SEARCH_TOP_K, disabled=disabled)
                score_threshold = st.slider("Score Threshold:", 0.0, 2.0, float(SCORE_THRESHOLD), 0.01, disabled=disabled)
                if st.button("Upload Files", disabled=len(files) == 0):
                    with st.spinner("Upload files in progress...."):
                        temp_docs = api.upload_temp_docs(files)
                        if temp_docs:
                            st.session_state["file_chat_id"] = temp_docs.get("data", {}).get("id")
                            st.success("file upload success!")
                            st.toast("file upload success!")
                        else:
                            st.session_state["file_chat_id"] = ""
                            st.error("file upload failed!")
                            st.toast("file upload failed!")
            current_running_config["enable"] = False
        elif dialogue_mode == "Agent Chat":
            current_running_config["enable"] = False
        api.save_current_running_config(current_running_config)
        print("current_running_config: ", current_running_config)
        now = datetime.now()
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button('New chat', use_container_width=True):
            chat_box.reset_history()
            st.session_state["tool_dict"] = {}

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

        imagesdata = []
        imagesprompt = []
        audiosdata = []
        videosdata = []
        imagedisable = False if imagerecognition_model != ""  or modelinfo["msubtype"] == ModelSubType.VisionChatModel else True
        audiodisable = False if modelinfo["msubtype"] == ModelSubType.VoiceChatModel else True
        videodisable = False if modelinfo["msubtype"] == ModelSubType.VideoChatModel else True
        
        if not imagedisable:
            imagefiles = st.file_uploader("Please upload 🎨:",
                glob_multimodal_vision_list,
                accept_multiple_files=True,
                )
            print("imagefiles: ", imagefiles)
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

        if not audiodisable:
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
        if not videodisable:
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
                        voice_language, voice_prompt = voice_prompt.split(":", 1)
                        if voice_language and voice_prompt:
                            st.success("Translation finished!")
                        else:
                            st.error("Translation failed...")
                            voice_language = ""
                            voice_prompt = ""
                        if running_model == "" or running_model == "None":
                            voice_prompt = ""
                    except Exception as e:
                        print(e)
                        st.error("Recording failed...")
                try:
                    wavpath.unlink()
                except Exception as _:
                    pass

        if  chat_solution_enable:
            solution_name = current_running_config["chat_solution"]["name"]
            st.caption(
                f"""<p style="font-size: 1.5em; text-align: center; color: #3498db;"><b>{solution_name}</b></p>""",
                unsafe_allow_html=True,
            )
            st.divider()
            st.caption(
                f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ {running_model}</b></p>""",
                unsafe_allow_html=True,
            )
            voice_enable = bool(voicemodel)
            if voice_enable:
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ {voicemodel}</b></p>""",
                    unsafe_allow_html=True,
                )
            speech_enable = bool(speechmodel["model"])
            if speech_enable:
                speech_name = speechmodel["model"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ {speech_name}</b></p>""",
                    unsafe_allow_html=True,
                )
            knowledge_enable = bool(current_running_config["knowledge_base"]["name"])
            if knowledge_enable:
                knowledge_base = current_running_config["knowledge_base"]["name"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ KB: {knowledge_base}</b></p>""",
                    unsafe_allow_html=True,
                )
            search_enable = bool(current_running_config["search_engine"]["name"])
            if search_enable:
                search_engine = current_running_config["search_engine"]["name"]
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ Search: {search_engine}</b></p>""",
                    unsafe_allow_html=True,
                )
            calling_enable = current_running_config["normal_calling"]["enable"]
            if calling_enable:
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ Function Calling: {calling_enable}</b></p>""",
                    unsafe_allow_html=True,
                )
            toolboxes_enable = is_toolboxes_enable(current_running_config)
            if toolboxes_enable:
                st.caption(
                    f"""<p style="font-size: 1.5em; text-align: left; color: #9932CC;"><b>▪️ ToolBoxes: {toolboxes_enable}</b></p>""",
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

        for image in imagesdata:
            st.image(image=BytesIO(image))
        for audio in audiosdata:
            st.audio(data=audio)
        for video in videosdata:
            st.video(data=video)

    tool_dict = st.session_state.get("tool_dict", {})
    if tool_dict:
        if tool_dict["name"] == "map_directions":
            origin_location = tool_dict.get("start_location", {})
            destination_location = tool_dict.get("end_location", {})
            if origin_location and destination_location:
                start_address = tool_dict.get("start_address", "")
                end_address = tool_dict.get("end_address", "")
                overview_polyline = tool_dict.get("overview_polyline", "")
                distance = tool_dict.get("distance", "")
                duration = tool_dict.get("duration", "")
                start_coordinates = [origin_location['lat'], origin_location['lng']]
                end_coordinates = [destination_location['lat'], destination_location['lng']]
                m = folium.Map(location=start_coordinates, zoom_start=16)
                folium.Marker(start_coordinates, popup=start_address, tooltip="Start").add_to(m)
                folium.Marker(end_coordinates, popup=end_address, tooltip="End").add_to(m)
                if overview_polyline:
                    import polyline
                    coordinates = polyline.decode(overview_polyline)
                    folium.PolyLine(
                        coordinates,
                        color="blue",
                        weight=8,
                        opacity=1,
                        tooltip=distance + ", " + duration
                    ).add_to(m)
                st_folium(m, width=725, use_container_width=True)
        elif tool_dict["name"] == "map_places":
            locations = tool_dict.get("locations", [])
            current_location = tool_dict.get("current_location", {})
            start_name = tool_dict.get("start", "")
            if current_location:
                current_coordinates = [current_location['lat'], current_location['lng']]
                m = folium.Map(location=current_coordinates, zoom_start=16)
                folium.Marker(current_coordinates, icon=folium.Icon(color="blue", prefix='fa'), popup=start_name, tooltip="Start").add_to(m)
                for location in locations:
                    place_name = location["name"]
                    #place_status = location["status"]
                    #place_address = location["address"]
                    #open_now = location["open_now"]
                    place_rating = location["rating"]
                    #place_price = location["price"]
                    #icon = location["icon"]
                    #user_ratings_total = location["user_ratings_total"]
                    place_location = location["location"]
                    place_coordinates = [place_location['lat'], place_location['lng']]
                    folium.Marker(place_coordinates, icon=folium.Icon(color="orange", prefix='fa'), popup=str(place_rating)+" Star", tooltip=place_name).add_to(m)
                st_folium(m, width=725, use_container_width=True)

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

    if prompt:
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
                text = ""
                if running_model == "stable-video-diffusion-img2vid" or running_model == "stable-video-diffusion-img2vid-xt":
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
                        chat_box.ai_say(["Thinking..."])
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
                    chat_box.ai_say(["Thinking..."])
                    chat_history_id = ""
                    tool_dict = {}
                    metadata = {
                        "chat_history_id": chat_history_id,
                    }
                    if imagegeneration_model and modelinfo["mtype"] != ModelType.Code:
                        imageprompt = ""
                        if imagesprompt:
                            imageprompt = imagesprompt[0]
                        prompt = generate_prompt_for_imagegen(imagegeneration_model, prompt, imageprompt)
                        imagesprompt = []
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
                    for d in r:
                        if error_msg := check_error_msg(d):  # check whether error occured
                            st.error(error_msg)
                            break
                        else:
                            chunk = d.get("text", "")
                            if chunk:
                                chat_history_id = d.get("chat_history_id", "")
                                text += chunk
                                metadata["chat_history_id"] = chat_history_id
                                chat_box.update_msg(text, element_index=0)
                            elif d.get("clear", ""):
                                tool_dict = d.get("tool_dict", {})
                                text = d.get("clear", "")
                                chat_box.update_msg(text, element_index=0, streaming=False, metadata=metadata)
                                text = ""
                            elif d.get("user", ""):
                                tooltype = d.get("tooltype", ToolsType.Unknown.value)
                                tooltype = ToolsType(tooltype)
                                if tooltype == ToolsType.ToolSearchEngine:
                                    title = "Additional information by Search Engine"
                                elif tooltype == ToolsType.ToolKnowledgeBase:
                                    title = "Additional information by Knowledge Base"
                                elif tooltype == ToolsType.ToolFunctionCalling:
                                    title = "Additional information by Function Calling"
                                elif tooltype == ToolsType.ToolCodeInterpreter:
                                    title = "Additional information by Code Interpreter"
                                elif tooltype == ToolsType.ToolToolBoxes:
                                    title = "Additional information by ToolBoxes"
                                else:
                                    title = "Additional information by Unknown Tool"
                                text += d.get("user", "")
                                chat_box.user_say(text) 
                                text = ""
                                if tool_dict:
                                    if tool_dict["name"] == "search_photos":
                                        photos = tool_dict.get("photos", [])
                                        for photo in photos:
                                            imgpath = photo.get("imgpath", "")
                                            if imgpath:
                                                chat_box.ai_say([""])
                                                photo_image=Image(imgpath)
                                                chat_box.update_msg(photo_image, element_index=0, metadata=metadata)
                                    elif tool_dict["name"] == "youtube_video_url":
                                        videos = tool_dict.get("videos", [])
                                        for video in videos:
                                            video_url = video.get("video_url", "")
                                            if video_url:
                                                chat_box.ai_say([""])
                                                video_image=Video(video_url)
                                                chat_box.update_msg(video_image, element_index=0, metadata=metadata)
                                chat_box.ai_say(["Thinking...", Markdown("...", in_expander=True, title=title, state="complete")])
                            elif d.get("docs", []):
                                message = "\n\n".join(d.get("docs", []))
                                chat_box.update_msg(Markdown(message, in_expander=True, title=title, state="complete"), element_index=1, streaming=False)

                    chat_box.update_msg(text, element_index=0, streaming=False, metadata=metadata)
                    chat_box.show_feedback(**feedback_kwargs,
                        key=chat_history_id,
                        on_submit=on_feedback,
                        kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})
                    if tool_dict:
                        st.session_state["tool_dict"] = tool_dict
                        tool_dict={}
                        st.rerun()
                    
                    if imagegeneration_model and modelinfo["mtype"] != ModelType.Code:
                        with st.spinner("Image generation in progress...."):
                            import uuid
                            chat_history_id = uuid.uuid4().hex
                            metadata["chat_history_id"] = chat_history_id
                            gen_image = api.get_image_generation_data(text, negative_prompt, True)
                            if gen_image:
                                chat_box.ai_say([""])
                                decoded_data = base64.b64decode(gen_image)
                                gen_image=Image(BytesIO(decoded_data))
                                chat_box.update_msg(gen_image, element_index=0, streaming=False, metadata=metadata)
                                chat_box.show_feedback(**feedback_kwargs,
                                        key=chat_history_id,
                                        on_submit=on_feedback,
                                        kwargs={"chat_history_id": chat_history_id, "history_index": len(chat_box.history) - 1})

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

        elif dialogue_mode == "File Chat":
            if not st.session_state.get("file_chat_id", ""):
                st.error("Please upload the file first before Chat.")
                st.stop()
            chat_box.ai_say([
                f"Querying from files `{st.session_state['file_chat_id']}` ...",
                Markdown("...", in_expander=True, title="Files content match results", state="complete"),
            ])
            text = ""
            for d in api.files_chat(prompt,
                                    knowledge_id=st.session_state["file_chat_id"],
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
            if current_running_config["code_interpreter"]["name"]:
                import uuid
                chat_history_id = uuid.uuid4().hex
                metadata = {
                    "chat_history_id": chat_history_id,
                    }
                with st.spinner("Agent Chat in progress...."):
                    r = api.agent_chat(
                                prompt,
                                interpreter_id=current_running_config["code_interpreter"]["name"],
                                model=running_model,
                                temperature=temperature)
                    text = ""
                    chat_box.ai_say(["Thinking..."])
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
                st.error("No code interpreter selected.")
                st.toast("No code interpreter selected.")
            
        if bshowstatus:
            while True:
                update_running_status(placeholder_cpu, placeholder_ram, placeholder_gpuutil, placeholder_gpumem)
                time.sleep(1)
        
    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

