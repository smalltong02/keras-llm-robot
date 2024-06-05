from enum import Enum
from typing import Dict, List, Union, Optional
from pathlib import Path
import os
import copy
import json
from WebUI.Server.chat.utils import History
from langchain.prompts.chat import ChatPromptTemplate
from WebUI.configs.roleplaytemplates import CATEGORICAL_ROLEPLAY_TEMPLATES
from fastchat.protocol.openai_api_protocol import ChatCompletionRequest

SAVE_CHAT_HISTORY = True
MIN_LLMMODEL_SIZE = 1024**3 # 1G
MIN_IMAGEMODEL_SIZE = 1024**2 # 1MB
MIN_EMBEDMODEL_SIZE = 1024**2 # 1MB

TMP_DIR = Path('temp')
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)

glob_model_type_list = ["LLM Model","Multimodal Model", "Code Model", "Special Model","Online Model"]
glob_model_size_list = ["3B Model","7B Model","13B Model","34B Model","70B Model"]
glob_model_subtype_list = ["Vision Chat Model","Voice Chat Model","Video Chat Model"]
training_devices_list = ["auto","cpu","gpu","mps"]
loadbits_list = ["32 bits","16 bits","8 bits"]
glob_roleplay_list = [""]
glob_assistant_name = ["James","Michael","William","David","John","Emily","Sarah","Jessica","Elizabeth","Jennifer"]
glob_compute_type_list = ["fp16","fp32","fp8"]
glob_save_strategy_list = ["no", "epoch", "steps"]
glob_optimizer_list = ["adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_torch_xla", "adamw_torch_npu_fused", "adamw_apex_fused", "adafactor", "adamw_anyprecision", "sgd", "adagrad", "adamw_bnb_8bit", "adamw_8bit", "lion_8bit",
                       "lion_32bit", "paged_adamw_32bit", "paged_adamw_8bit", "paged_lion_32bit", "paged_lion_8bit", "rmsprop", "rmsprop_bnb", "rmsprop_bnb_8bit", "rmsprop_bnb_32bit", "galore_adamw", "galore_adamw_8bit", "galore_adafactor",
                       "galore_adamw_layerwise", "galore_adamw_8bit_layerwise", "galore_adafactor_layerwise"]
glob_lr_scheduler_list = ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau", "cosine_with_min_lr"]
glob_Lora_rank_list = [1, 2, 4, 8, 16, 32, 64]
glob_save_model_list = ["full", "lora", "gguf"]
glob_save_method_list = ["lora", "merged_16bit", "merged_4bit"]
glob_quantization_method_list = ["not_quantized", "fast_quantized", "quantized", "f32", "f16", "q8_0", "q4_k_m", "q5_k_m", "q2_k", "q3_k_l", "q3_k_m", "q3_k_s", "q4_0", "q4_1", "q4_k_s", "q4_k", "q5_k", "q5_0", "q5_1", "q5_k_s", "q6_k", "iq2_xxs", "iq2_xs", "iq3_xxs", "q3_k_xs"]

class ModelType(Enum):
    Unknown = 0
    Local = 1
    Multimodal = 2
    Code = 3
    Special = 4
    Online = 5

class ModelSize(Enum):
    Unknown = 0
    Mod3B = 1
    Mod7B = 2
    Mod13B = 3
    Mod34B = 4
    Mod70B = 5

glob_multimodal_vision_list = ['.png', '.jpg', '.jpeg', '.bmp']
glob_multimodal_voice_list = ['.mp3', '.flac', '.wav', '.m4a']
glob_multimodal_video_list = ['.mp4', '.wmv', '.m4v', '.mov', '.avi', '.mkv', '.flv']

class ModelSubType(Enum):
    Unknown = 0
    VisionChatModel = 1
    VoiceChatModel = 2
    VideoChatModel = 4

class ModelAbilityType(Enum):
    Unknown = 0


def GetTypeName(type: ModelType) -> str:
    if type == ModelType.Local:
        return "LLM Model"
    if type == ModelType.Multimodal:
        return "Multimodal Model"
    if type == ModelType.Code:
        return "Code Model"
    if type == ModelType.Special:
        return "Special Model"
    if type == ModelType.Online:
        return "Online Model"
    return "Unknown"

def GetSizeName(size: ModelSize) -> str:
    if size == ModelSize.Mod3B:
        return "3B Model"
    if size == ModelSize.Mod7B:
        return "7B Model"
    if size == ModelSize.Mod13B:
        return "13B Model"
    if size == ModelSize.Mod34B:
        return "34B Model"
    if size == ModelSize.Mod70B:
        return "70B Model"
    return "Unknown"

def GetSubTypeName(subtype: ModelSubType) -> str:
    if subtype == ModelSubType.VisionChatModel:
        return "Vision Chat Model"
    if subtype == ModelSubType.VoiceChatModel:
        return "Voice Chat Model"
    if subtype == ModelSubType.VideoChatModel:
        return "Video Chat Model"
    return "Unknown"

def GetModelType(Typestr : str) -> ModelType:
    if Typestr == "LLM Model":
        return ModelType.Local
    if Typestr == "Multimodal Model":
        return ModelType.Multimodal
    if Typestr == "Code Model":
        return ModelType.Code
    if Typestr == "Special Model":
        return ModelType.Special
    if Typestr == "Online Model":
        return ModelType.Online
    return ModelType.Unknown

def GetModelSize(Sizestr : str) -> ModelSize:
    if Sizestr == "3B Model":
        return ModelSize.Mod3B
    if Sizestr == "7B Model":
        return ModelSize.Mod7B
    if Sizestr == "13B Model":
        return ModelSize.Mod13B
    if Sizestr == "34B Model":
        return ModelSize.Mod34B
    if Sizestr == "70B Model":
        return ModelSize.Mod70B
    return ModelSize.Unknown

def GetModelSubType(SubTypestr : str) -> ModelSubType:
    if SubTypestr == "Vision Chat Model":
        return ModelSubType.VisionChatModel
    if SubTypestr == "Voice Chat Model":
        return ModelSubType.VoiceChatModel
    if SubTypestr == "Video Chat Model":
        return ModelSubType.VideoChatModel
    return ModelSubType.Unknown

def GetOnlineProvider(webui_config) -> list:
    provider = []
    onlineprovider = webui_config.get("ModelConfig").get("OnlineModel")
    for key, _ in onlineprovider.items():
        provider.append(key)
    return provider

def GetOnlineModelList(webui_config, provider: str) -> list:
    onlineprovider = webui_config.get("ModelConfig").get("OnlineModel")
    for key, value in onlineprovider.items():
        if key.casefold() == provider.casefold():
            return copy.deepcopy(value["modellist"])
    return []

def GetModelInfoByName(webui_config: Dict, name : str):
    if name:
        localmodel = webui_config.get("ModelConfig").get("LocalModel")
        for typekey, typevalue in localmodel.items():
            for sizekey, sizevalue in typevalue.items():
                for modelkey, _ in sizevalue.items():
                    if modelkey.casefold() == name.casefold():
                        return GetModelType(typekey), GetModelSize(sizekey), GetModelSubType(sizekey)
        onlinemodel = webui_config.get("ModelConfig").get("OnlineModel")
        provider_size = 1
        for _, value in onlinemodel.items():
            modellist = value["modellist"]
            if name in modellist:
                model_size = ModelSize(provider_size % (len(ModelSize) - 1))
                model_subtype = ModelSubType.Unknown
                if name == "gpt-4-vision-preview" or name == "gpt-4-turbo" or name == "gpt-4o":
                    model_subtype = ModelSubType.VisionChatModel
                if name == "gemini-pro-vision" or name == "gemini-1.5-pro-latest":
                    model_subtype = ModelSubType.VisionChatModel
                if name == "yi-vision":
                    model_subtype = ModelSubType.VisionChatModel
                return ModelType.Online, model_size, model_subtype
            provider_size = provider_size+1

    return ModelType.Unknown, ModelSize.Unknown, ModelSubType.Unknown

def GetProviderByName(webui_config: Dict, name : str):
    if name:
        onlinemodel = webui_config.get("ModelConfig").get("OnlineModel")
        for key, value in onlinemodel.items():
            modellist = value["modellist"]
            if name in modellist:
                return key
    return None

def GetModeList(webui_config, current_model  : Dict[str, any]) -> list:
    localmodel = webui_config.get("ModelConfig").get("LocalModel")
    mtype = current_model["mtype"]
    if mtype == ModelType.Local:
        msize = current_model["msize"]
        return [f"{key}" for key in localmodel.get("LLM Model").get(GetSizeName(msize))]
    if mtype == ModelType.Code:
        msize = current_model["msize"]
        return [f"{key}" for key in localmodel.get("Code Model").get(GetSizeName(msize))]
    elif mtype == ModelType.Special:
        msize = current_model["msize"]
        return [f"{key}" for key in localmodel.get("Special Model").get(GetSizeName(msize))]
    elif mtype == ModelType.Multimodal:
        msubtype = current_model["msubtype"]
        return [f"{key}" for key in localmodel.get("Multimodal Model").get(GetSubTypeName(msubtype))]
    elif mtype == ModelType.Online:
        pass
    return []

def GetModelConfig(webui_config, current_model : Dict[str, any]) -> Dict:
    mtype = current_model["mtype"]
    if mtype == ModelType.Local:
        msize = GetSizeName(current_model["msize"])
        provider = "LLM Model"
    elif mtype == ModelType.Code:
        msize = GetSizeName(current_model["msize"])
        provider = "Code Model"
    elif mtype == ModelType.Special:
        msize = GetSizeName(current_model["msize"])
        provider = "Special Model"
    elif mtype == ModelType.Multimodal:
        msize = GetSubTypeName(current_model["msubtype"])
        provider = "Multimodal Model"
    elif mtype == ModelType.Online:
        onlineprovider = webui_config.get("ModelConfig").get("OnlineModel")
        for _, value in onlineprovider.items():
            modellist = value["modellist"]
            if current_model["mname"] in modellist:
                    return value
        return {}
    else:
        return {}
    localmodel = webui_config.get("ModelConfig").get("LocalModel")
    return localmodel.get(provider).get(msize).get(current_model["mname"])

def GetGGUFModelPath(pathstr : str) -> list:
    found_files = []
    if pathstr:
        local_path = Path(pathstr)
        if local_path.is_dir():
            for root, _, files in os.walk(local_path):
                for file in files:
                    if file.endswith(".gguf"):
                        file_path = file
                        found_files.append(file_path)
    return found_files

def GetSpeechModelInfo(webui_config: Dict, name : str):
    speechmodel = webui_config.get("ModelConfig").get("TtoVModel")
    if name:
        for modelkey, configvalue in speechmodel.items():
            if name.casefold() == modelkey.casefold():
                return configvalue
    return {}

def GetPresetConfig(preset_name : str) -> dict:
    if preset_name:
        from WebUI.configs.webuiconfig import InnerJsonConfigPresetTempParse
        presetinst = InnerJsonConfigPresetTempParse()
        preset_config = presetinst.dump()
        tmp_lists = preset_config["templates_lists"]
        for config in tmp_lists:
            name = config.get("name", "")
            if preset_name == name:
                return config
    return {}

def GetPresetPromptList() -> list:
    from WebUI.configs.webuiconfig import InnerJsonConfigPresetTempParse
    presetinst = InnerJsonConfigPresetTempParse()
    preset_config = presetinst.dump()
    tmp_lists = preset_config["templates_lists"]
    preset_list = []
    for config in tmp_lists:
        name = config.get("name", "")
        if len(name):
            preset_list.append(name)
    return preset_list

def GeneratePresetPrompt(preset_name: str) -> dict:
    if preset_name:
        from WebUI.configs.webuiconfig import InnerJsonConfigPresetTempParse
        presetinst = InnerJsonConfigPresetTempParse()
        preset_config = presetinst.dump()
        tmp_lists = preset_config["templates_lists"]
        input_variables = []
        prompt_templates = ""
        for config in tmp_lists:
            name = config.get("name", "")
            if preset_name == name:
                inference_params = config.get("inference_params", "")
                if len(inference_params):
                    pre_prompt = inference_params.get("pre_prompt", "")
                    if pre_prompt:
                        prompt_templates += pre_prompt + '\n'
                    pre_prompt_prefix = inference_params.get("pre_prompt_prefix", "")
                    if pre_prompt_prefix:
                        input_variables.append("system")
                        prompt_templates += pre_prompt_prefix + "{system}"
                    pre_prompt_suffix = inference_params.get("pre_prompt_suffix", "")
                    if pre_prompt_suffix:
                        prompt_templates += pre_prompt_suffix
                    input_prefix = inference_params.get("input_prefix", "")
                    if input_prefix:
                        input_variables.append("input")
                        prompt_templates += input_prefix + "{input}"
                    input_suffix = inference_params.get("input_suffix", "")
                    if input_suffix:
                        prompt_templates += input_suffix
                    anti_prompt = inference_params.get("antiprompt", [])
                    return {
                        "input_variables": input_variables,
                        "prompt_templates": prompt_templates,
                        "anti_prompt": anti_prompt
                    }
    return {}

def GenerateModelPrompt(inputs: list, input: str) -> Union[str, dict]:
    if len(inputs):
        prompt_dict = {key: "" for key in inputs}
        last_key = inputs[-1]
        prompt_dict[last_key] = input
        return prompt_dict
    return input

def GetRerankerModelPath() -> Union[str, None]:
    return "models/reranker/bge-reranker-large"

def LocalModelExist(local_path):
    total_size_bytes = 0
    for dirpath, _, filenames in os.walk(local_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size_bytes += os.path.getsize(filepath)
    total_size_gb = total_size_bytes / (MIN_LLMMODEL_SIZE)
    print("total_size_gb: ", total_size_gb)
    if total_size_gb > 1:
        return True
    return False

def ImageModelExist(local_path):
    total_size_bytes = 0
    for dirpath, _, filenames in os.walk(local_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size_bytes += os.path.getsize(filepath)
    total_size_mb = total_size_bytes / (MIN_IMAGEMODEL_SIZE)
    print("total_size_mb: ", total_size_mb)
    if total_size_mb > 100:
        return True
    return False

def MusicModelExist(local_path):
    return ImageModelExist(local_path)

def EmbeddingModelExist(embed_model: str):
    if embed_model:
        from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
        configinst = InnerJsonConfigWebUIParse()
        webui_config = configinst.dump()
        embeddingmodel = webui_config.get("ModelConfig").get("EmbeddingModel")
        config = None
        for key, value in embeddingmodel.items():
            if embed_model == key:
                    config = value
        if config:
            provider = config.get("provider", "")
            if provider: # online model
                return False
            else:
                local_path = config.get("path", "")
                total_size_bytes = 0
                for dirpath, _, filenames in os.walk(local_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size_bytes += os.path.getsize(filepath)
                total_size_mb = total_size_bytes / (MIN_EMBEDMODEL_SIZE)
                print("total_size_mb: ", total_size_mb)
                if total_size_mb > 50:
                    return True
    return False

def GetKbConfig():
    from WebUI.configs.webuiconfig import InnerJsonConfigKnowledgeBaseParse
    knowledgeinst = InnerJsonConfigKnowledgeBaseParse()
    kb_config = knowledgeinst.dump()
    return kb_config

def GetKbRootPath(kb_config: dict):
    if isinstance(kb_config, dict):
        return kb_config.get("kb_root_path", "")
    return ""

def GetDbUri(kb_config: dict):
    if isinstance(kb_config, dict):
        return kb_config.get("sqlalchemy_db_uri", "")
    return ""

def GetDbRootPath(kb_config: dict):
    if isinstance(kb_config, dict):
        return kb_config.get("db_root_path", "")
    return ""

def GetKbInfo(kb_name: str):
    return f"about '{kb_name}' knowledge base."

def GetKbPath(kb_name: str):
    return os.path.join(GetKbRootPath(GetKbConfig()), kb_name)

def GetDocPath(kb_name: str):
    return os.path.join(GetKbPath(kb_name), "content")

def GetKbsList():
    kb_list = []
    kb_config = GetKbConfig()
    kbs_config = kb_config.get("kbs_config", {})
    for key, _ in kbs_config.items():
        kb_list.append(key)
    return kb_list

def GetKbsConfig(kbs_name: str) -> dict:
    if kbs_name:
        kb_config = GetKbConfig()
        kbs_config = kb_config.get("kbs_config", {})
        kbs_config = kbs_config.get(kbs_name, {})
        return kbs_config
    return {}

def GetTextSplitterDict():
    kb_config = GetKbConfig()
    text_splitter_dict = kb_config.get("text_splitter_dict", {})
    return text_splitter_dict

def generate_new_query(query : str = "", imagesprompt : List[str] = []):
    en_nums = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']

    new_query = ""
    if len(query) == 0 or len(imagesprompt) == 0:
        return query
    index = 0
    for prompt in imagesprompt:
        new_query += f'The content of the {en_nums[index]} picture: ' + prompt + '\n\n'
        index = index + 1
        if index >= 10:
            break
    
    if new_query:
        new_query += "If the user's question involves pictures, please answer the user's question based on the content of the pictures provided above. \n\n"
        new_query += "Please respond to the user using the language in which the user asked the question.\n\n"
    new_query += "The user's question is: " + query
    print("new_query: ", new_query)
    return new_query

def generate_prompt_for_imagegen(model_name : str = "", prompt : str = "", imagesprompt : str = ""):
    new_prompt = ""
    if len(model_name) == 0 or len(prompt) == 0:
        return prompt
    #if model_name == "OpenDalleV1.1" or model_name == "ProteusV0.2":
    new_prompt = """
            You need to create prompts for an image generation model based on the user's question. The format of the prompts is the features of the image, separated by commas, with no any other information outputted, for example:

            1. black fluffy gorgeous dangerous cat animal creature, large orange eyes, big fluffy ears, piercing gaze, full moon, dark ambiance, best quality, extremely detailed
            2. an anime female general laughing, with a military cap, evil smile, sadistic, grim
            3. John Berkey Style page,ral-oilspill, There is no road ahead,no land, Strangely,the river is still flowing,crossing the void into the mysterious unknown, The end of nothingness,a huge ripple,it is a kind of wave,and it is the law of time that lasts forever in that void, At the end of the infinite void,there is a colorful world,very hazy and mysterious,and it cannot be seen clearly,but it is real, And that's where the river goes
            4. (impressionistic realism by csybgh), a 50 something male, working in banking, very short dyed dark curly balding hair, Afro-Asiatic ancestry, talks a lot but listens poorly, stuck in the past, wearing a suit, he has a certain charm, bronze skintone, sitting in a bar at night, he is smoking and feeling cool, drunk on plum wine, masterpiece, 8k, hyper detailed, smokey ambiance, perfect hands AND fingers
            5. Super Closeup Portrait, action shot, Profoundly dark whitish meadow, glass flowers, Stains, space grunge style, Jeanne d'Arc wearing White Olive green used styled Cotton frock, Wielding thin silver sword, Sci-fi vibe, dirty, noisy, Vintage monk style, very detailed, hd
            6. cinematic film still of Kodak Motion Picture Film: (Sharp Detailed Image) An Oscar winning movie for Best Cinematography a woman in a kimono standing on a subway train in Japan Kodak Motion Picture Film Style, shallow depth of field, vignette, highly detailed, high budget, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy
            7. in the style of artgerm, comic style,3D model, mythical seascape, negative space, space quixotic dreams, temporal hallucination, psychedelic, mystical, intricate details, very bright neon colors, (vantablack background:1.5), pointillism, pareidolia, melting, symbolism, very high contrast, chiaroscuro bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image
            8. ((OpenDAlle!)text logo:1), ~*~aesthetic~*~
            \n
            Please generate appropriate prompts for the user's question based on the above example. Please note do not reply to what I say. that the format should only consist of features separated by commas, with no any other information outputted, Just output a prompt.
            \n
            """
    
    new_prompt += "The user's question is: " + prompt
    if imagesprompt:
        new_prompt += f". Contents of this image is '{imagesprompt}'"
    print("new_prompt: ", new_prompt)
    return new_prompt, True
    #return prompt, False

def generate_prompt_for_smart_search(prompt : str = ""):
    new_prompt = "You are an AI assistant, answering questions based on user inquiries. If you are absolutely certain of the answer to the question, please answer it to the best of your ability and refrain from returning the 'search_engine' command. If you don't know how to answer the question or you require real-time information or need to search the internet before answering questions, then please only return the command: 'search_engine'. \n\n User's question: " + prompt
    return new_prompt

def use_search_engine(text : str = ""):
    return "search_engine" in text.lower()

def ConvertCompletionRequestToHistory(request: ChatCompletionRequest):
    messages = request.messages
    if len(messages) == 0:
        return [], None
    historys = []
    query = None
    answer = None
    for message in messages:
        if message['role'] == 'system':
            historys.append({'role': 'user', 'content': message['content']})
            historys.append({'role': 'assistant', 'content': 'I will strictly follow your instructions to carry out the task.'})
        elif message['role'] == 'user':
            if query is not None and answer is not None:
                historys.append({'role': 'user', 'content': query})
                historys.append({'role': 'assistant', 'content': answer})
                query = None
                answer = None
            if query is None:
                query = ""
            query += message['content']
        elif message['role'] == 'assistant':
            if answer is None:
                answer = ""
            answer += message['content']
    if query is None:
        return [], None
    if answer is not None:
        return [], None 
    return historys, query

def ExtractJsonStrings(input_string):
    import json
    json_data_list = []
    stack = []
    start = None

    for i, char in enumerate(input_string):
        if char == '{':
            if not stack:
                start = i
            stack.append(char)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack:
                    json_str = input_string[start:i+1]
                    try:
                        json.loads(json_str)
                        json_data_list.append(json_str)
                    except json.JSONDecodeError:
                        pass
    return json_data_list

def use_new_search_engine(json_lists : list = []) ->bool:
    for item in json_lists:
        it = json.loads(item)
        if it.get("name", "") == "search_engine":
            return True
    return False

def use_knowledge_base(json_lists : list = []) ->bool:
    for item in json_lists:
        it = json.loads(item)
        if it.get("name", "") == "knowledge_base":
            return True
    return False

def use_new_function_calling(json_lists : list = []) ->bool:
    from WebUI.Server.funcall.funcall import tool_names
    for item in json_lists:
        it = json.loads(item)
        if it.get("name", "") in tool_names:
            return True
    return False

def use_new_toolboxes_calling(json_lists : list = []) ->bool:
    from WebUI.Server.funcall.google_toolboxes.calendar_funcall import calendar_tool_names
    from WebUI.Server.funcall.google_toolboxes.gmail_funcall import email_tool_names
    from WebUI.Server.funcall.google_toolboxes.gcloud_funcall import drive_tool_names
    from WebUI.Server.funcall.google_toolboxes.gmap_funcall import map_tool_names
    from WebUI.Server.funcall.google_toolboxes.youtube_funcall import youtube_tool_names
    for item in json_lists:
        it = json.loads(item)
        if it.get("name", "") in calendar_tool_names:
            return True
        if it.get("name", "") in email_tool_names:
            return True
        if it.get("name", "") in drive_tool_names:
            return True
        if it.get("name", "") in map_tool_names:
            return True
        if it.get("name", "") in youtube_tool_names:
            return True
    return False

def GetPromptChatSolution(chat_solution : dict, prompt_name: str) -> str:
    if not chat_solution:
        return "{{prompt}}"
    if not chat_solution["enable"]:
        return "{{prompt}}"
    if chat_solution["name"] == "Intelligent Customer Support" or chat_solution["name"] == "Language Translation and Localization" or chat_solution["name"] == "Virtual Personal Assistant":
        config = chat_solution.get("config", {})
        if not config:
            return "{{prompt}}"
        roleplayer = config.get("roleplayer", {})
        if not roleplayer:
            return "{{prompt}}"
        prompt = CATEGORICAL_ROLEPLAY_TEMPLATES[roleplayer][prompt_name]
        return prompt
    return {}

def GetSpeechForChatSolution(chat_solution : dict, prompt_language : str) ->dict:
    if not chat_solution:
        return {}
    if not chat_solution["enable"]:
        return {}
    if chat_solution["name"] == "Intelligent Customer Support" or chat_solution["name"] == "Language Translation and Localization" or chat_solution["name"] == "Virtual Personal Assistant":
        config = chat_solution.get("config", {})
        if not config:
            return {}
        speech = config.get("speech", {})
        if not speech:
            return {}
        if speech["enable"]:
            if prompt_language:
                speaker = speech["speaker"]
                result = speaker.split('-', 2)[:2]
                language_code = '-'.join(result)
                parts = prompt_language.split('-')
                parts[1] = parts[1].upper()
                prompt_language = '-'.join(parts)
                if prompt_language != language_code:
                    return speech
                else:
                    voice_model = config.get("voice", {}).get("model", "")
                    if voice_model:
                        from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
                        configinst = InnerJsonConfigWebUIParse()
                        webui_config = configinst.dump()
                        vtotmodel = webui_config.get("ModelConfig").get("VtoTModel").get(voice_model)
                        language_list = vtotmodel.get("language")
                        if language_list and prompt_language in language_list:
                            other_language = ""
                            for lang in language_list:
                                if lang != prompt_language:
                                    other_language = lang
                                    break
                            if other_language:
                                spmodel = webui_config.get("ModelConfig").get("TtoVModel").get(speech["model"])
                                language_config = spmodel.get("synthesis", {}).get(other_language, {})
                                if language_config:
                                    male_list = language_config.get("male", [])
                                    female_list = language_config.get("female", [])
                                    if male_list or female_list:
                                        new_speaker = ""
                                        if male_list:
                                            new_speaker = male_list[0]
                                        elif female_list:
                                            new_speaker = female_list[0]
                                        if new_speaker:
                                            speech["speaker"] = new_speaker
            return speech
    return {}

def GetSearchEngineForChatSolution(chat_solution : dict) ->dict:
    if not chat_solution:
        return {}
    if not chat_solution["enable"]:
        return {}
    if chat_solution["name"] == "Intelligent Customer Support" or chat_solution["name"] == "Virtual Personal Assistant":
        config = chat_solution.get("config", {})
        if not config:
            return {}
        search_engine = config.get("search_engine", {})
        if not search_engine:
            return {}
        if search_engine["enable"]:
            return search_engine
    return {}

def GetKnowledgeBaseForChatSolution(chat_solution : dict) ->dict:
    if not chat_solution:
        return {}
    if not chat_solution["enable"]:
        return {}
    if chat_solution["name"] == "Intelligent Customer Support" or chat_solution["name"] == "Virtual Personal Assistant":
        config = chat_solution.get("config", {})
        if not config:
            return {}
        knowledge_base = config.get("knowledge_base", {})
        if not knowledge_base:
            return {}
        if knowledge_base["enable"]:
            return knowledge_base
    return {}

def GetSystemPromptForChatSolution(chat_solution : dict) ->dict:
    if not chat_solution:
        return None
    roleplayer = chat_solution["config"]["roleplayer"]
    role_template = CATEGORICAL_ROLEPLAY_TEMPLATES[roleplayer]["english"]
    if chat_solution["name"] == "Intelligent Customer Support":
        description = chat_solution["config"]["description"]
        search_engine = GetSearchEngineForChatSolution(chat_solution)
        knowledge_base = GetKnowledgeBaseForChatSolution(chat_solution)
        function_calling = chat_solution["config"].get("function_calling", False)
        toolboxes = chat_solution["config"].get("toolboxes", "")
        system_prompt = role_template
        if description:
            system_prompt += f'\nHere is a brief description of the product: "{description}"\n'
        if knowledge_base:
            system_prompt += """\nYou have the ability to query the company's product knowledge base. When there are inquiries related to the product, please first submit the question to the knowledge base for a search. 
            After a successful search, the results will be appended to the end of the question and sent back to you, enabling you to better address the query. Here are the names and descriptions for knowledge base:
            knowledge_base() - Get search result from knowledge base
            If you'd like to use the knowledge base, Return your response as a JSON blob with 'name' and 'arguments' keys.\n"""
        if search_engine:
            system_prompt += """\nYou have the ability to use a web search engine. When a question exceeds your knowledge scope or when it's beyond the timeframe of your training data, please submit the question to a web search engine for a query first. 
            After a successful search, the results will be appended to the end of the question and sent back to you, allowing you to better address the query. Here are the names and descriptions for search engine:
            search_engine() - Get search result from network
            If you'd like to use a web search engine, Return your response as a JSON blob with 'name' and 'arguments' keys.\n"""
        funcall_tools_prompt = ""
        if function_calling:
            from langchain.tools.render import render_text_description
            from WebUI.Server.funcall.funcall import funcall_tools
            funcall_tools_prompt = render_text_description(funcall_tools)
        toolboxes_tools_prompt = ""
        if toolboxes:
            if toolboxes == "Google ToolBoxes":
                from WebUI.Server.funcall.google_toolboxes.gmap_funcall import map_toolboxes
                from WebUI.Server.funcall.google_toolboxes.gmail_funcall import email_toolboxes
                from WebUI.Server.funcall.google_toolboxes.calendar_funcall import calendar_toolboxes
                from WebUI.Server.funcall.google_toolboxes.gcloud_funcall import drive_toolboxes
                from WebUI.Server.funcall.google_toolboxes.youtube_funcall import youtube_toolboxes
                toolboxes_tools_prompt += render_text_description(map_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(email_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(calendar_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(drive_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(youtube_toolboxes) + '\n\n'
        if funcall_tools_prompt or toolboxes_tools_prompt:
            tools_prompt = GenerateToolsPrompt(funcall_tools_prompt + toolboxes_tools_prompt)
            system_prompt += "\n" + tools_prompt
        return system_prompt
    elif chat_solution["name"] == "Language Translation and Localization":
        if not chat_solution["config"]["voice"]["enable"]:
            return ""
        s_language = chat_solution["config"]["voice"]["language"]
        if not chat_solution["config"]["speech"]["enable"]:
            return ""
        speaker = chat_solution["config"]["speech"]["speaker"]
        result = speaker.split('-', 2)[:2]
        d_language = '-'.join(result)
        system_prompt = role_template.format(d_language=d_language, s_language=s_language)
        return system_prompt
    elif chat_solution["name"] == "Virtual Personal Assistant":
        assistant_name = chat_solution["config"]["assistant_name"]
        search_engine = GetSearchEngineForChatSolution(chat_solution)
        knowledge_base = GetKnowledgeBaseForChatSolution(chat_solution)
        function_calling = chat_solution["config"].get("function_calling", False)
        toolboxes = chat_solution["config"].get("toolboxes", "")
        system_prompt = role_template
        if assistant_name:
            system_prompt = system_prompt.format(assistant_name=assistant_name) + '\n'
        if knowledge_base:
            system_prompt += """\nYou have the ability to query the private knowledge base. When requesting to query the knowledge base, please first submit the question to the knowledge base for a search. 
            After a successful search, the results will be appended to the end of the question and sent back to you, enabling you to better address the query. Here are the names and descriptions for knowledge base:
            knowledge_base() - Get search result from knowledge base
            If you'd like to use the knowledge base, Return your response as a JSON blob with 'name' and 'arguments' keys.\n"""
        if search_engine:
            system_prompt += """\nYou have the ability to use a web search engine. When a question exceeds your knowledge scope or when it's beyond the timeframe of your training data, please submit the question to a web search engine for a query first. 
            After a successful search, the results will be appended to the end of the question and sent back to you, allowing you to better address the query. Here are the names and descriptions for search engine:
            search_engine() - Get search result from network
            If you'd like to use a web search engine, Return your response as a JSON blob with 'name' and 'arguments' keys.\n"""
        funcall_tools_prompt = ""
        if function_calling:
            from langchain.tools.render import render_text_description
            from WebUI.Server.funcall.funcall import funcall_tools
            funcall_tools_prompt = render_text_description(funcall_tools)
        toolboxes_tools_prompt = ""
        if toolboxes:
            if toolboxes == "Google ToolBoxes":
                from langchain.tools.render import render_text_description
                from WebUI.Server.funcall.google_toolboxes.gmap_funcall import map_toolboxes
                from WebUI.Server.funcall.google_toolboxes.gmail_funcall import email_toolboxes
                from WebUI.Server.funcall.google_toolboxes.calendar_funcall import calendar_toolboxes
                from WebUI.Server.funcall.google_toolboxes.gcloud_funcall import drive_toolboxes
                from WebUI.Server.funcall.google_toolboxes.youtube_funcall import youtube_toolboxes
                toolboxes_tools_prompt += render_text_description(map_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(email_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(calendar_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(drive_toolboxes) + '\n\n'
                toolboxes_tools_prompt += render_text_description(youtube_toolboxes) + '\n\n'
        if funcall_tools_prompt or toolboxes_tools_prompt:
            tools_prompt = GenerateToolsPrompt(funcall_tools_prompt + toolboxes_tools_prompt)
            system_prompt += "\n" + tools_prompt
        return system_prompt
    else:
        return ""
    
def is_normal_calling_enable() ->bool:
    from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
    configinst = InnerJsonConfigWebUIParse()
    function_calling = configinst.get("FunctionCalling")
    enable = function_calling.get("calling_enable", False)
    return enable

def is_toolboxes_enable() ->bool:
    from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
    configinst = InnerJsonConfigWebUIParse()
    tool_boxes = configinst.get("ToolBoxes")
    for key_boxes, value_boxes in tool_boxes.items():
        for key_tools, value_tools in value_boxes.get("Tools", {}).items():
            if value_tools.get("enable", False):
                return True
    return False

def is_function_calling_enable() ->bool:
    normal_enable = is_normal_calling_enable()
    toolboxes_enable = is_toolboxes_enable()
    toolboxes_enable = is_toolboxes_enable()
    if toolboxes_enable:
        from WebUI.Server.funcall.google_toolboxes.credential import init_credential
        init_credential()
    return (normal_enable or toolboxes_enable)

def GetCredentialsPath() ->str:
    from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
    configinst = InnerJsonConfigWebUIParse()
    tool_boxes = configinst.get("ToolBoxes")
    if not tool_boxes:
        return ""
    google_toolboxes = tool_boxes.get("Google ToolBoxes")
    if not google_toolboxes:
        return False
    credential = google_toolboxes.get("credential", "")
    if credential == "[Your Key]":
        credential = ""
    return credential

def CallingExternalTools(text: str) -> bool:
    if not text:
        return False, ""
    new_answer = text
    json_lists = ExtractJsonStrings(text)
    if not json_lists:
        return False, ""
    for json_str in json_lists:
        new_answer = new_answer.replace(json_str, "")
    if "```json" in new_answer:
        new_answer = new_answer.replace("```json", "")
    if "```" in new_answer:
        new_answer = new_answer.replace("```", "")
    new_answer = new_answer.strip(' \n')
    if use_new_search_engine(json_lists):
        return True, new_answer
    if use_knowledge_base(json_lists):
        return True, new_answer
    if use_new_function_calling(json_lists):
        return True, new_answer
    if use_new_toolboxes_calling(json_lists):
        return True, new_answer
    return False, ""

def GetNormalCallingSystemPrompt() ->str:
    from langchain.tools.render import render_text_description
    from WebUI.Server.funcall.funcall import funcall_tools
    if is_normal_calling_enable():
        render = render_text_description(funcall_tools)
        return render
    return ""

def GetToolBoxesSystemPrompt() ->str:
    from langchain.tools.render import render_text_description
    from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
    from WebUI.Server.funcall.google_toolboxes.gmap_funcall import map_toolboxes
    from WebUI.Server.funcall.google_toolboxes.gmail_funcall import email_toolboxes
    from WebUI.Server.funcall.google_toolboxes.calendar_funcall import calendar_toolboxes
    from WebUI.Server.funcall.google_toolboxes.gcloud_funcall import drive_toolboxes
    from WebUI.Server.funcall.google_toolboxes.youtube_funcall import youtube_toolboxes
    configinst = InnerJsonConfigWebUIParse()
    tool_boxes = configinst.get("ToolBoxes")
    render_message = ""
    for key_boxes, value_boxes in tool_boxes.items():
        for key_tools, value_tools in value_boxes.get("Tools", {}).items():
            if not value_tools.get("enable", False):
                continue
            if key_tools == "Google Maps":
                render_message += render_text_description(map_toolboxes) + '\n\n'
            elif key_tools == "Google Mail":
                render_message += render_text_description(email_toolboxes) + '\n\n'
            elif key_tools == "Google Calendar":
                render_message += render_text_description(calendar_toolboxes) + '\n\n'
            elif key_tools == "Google Drive":
                render_message += render_text_description(drive_toolboxes) + '\n\n'
            elif key_tools == "Google Youtube":
                render_message += render_text_description(youtube_toolboxes) + '\n\n'
    return render_message
                
def GenerateToolsPrompt(rendered_tools: str) ->str:
    tools_system_prompt = f"""You can access to the following set of tools. Here are the names and descriptions for each tool:
    {rendered_tools}
    Given the user input, you need to use your own judgment whether to use tools. If not needed, please answer the questions to the best of your ability.
    If tools are needed, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""
    return tools_system_prompt

def GetToolsSystemPrompt() ->str:
    rendered_tools = ""
    calling_prompt = GetNormalCallingSystemPrompt()
    if calling_prompt:
        rendered_tools += calling_prompt
        rendered_tools += '\n'
    toolboxes_prompt = GetToolBoxesSystemPrompt()
    if toolboxes_prompt:
        rendered_tools += toolboxes_prompt
    tools_system_prompt = GenerateToolsPrompt(rendered_tools)
    return tools_system_prompt
    
def split_with_calling_blocks(orgin_string: str, json_lists: list[str]):
    from WebUI.Server.funcall.funcall import RunNormalFunctionCalling
    from WebUI.Server.funcall.google_toolboxes.credential import RunFunctionCallingInToolBoxes
    normal_calling_enable = is_normal_calling_enable()
    toolboxes_enable = is_toolboxes_enable()
    result_list = []
    start = 0
    index = 0
    if not json_lists:
        return result_list
    for item in json_lists:
        func_name = ""
        result = ""
        index = orgin_string.find(item, start)
        if index != -1:
            if normal_calling_enable:
                func_name, result = RunNormalFunctionCalling(item)
            if not result and toolboxes_enable:
                func_name, result = RunFunctionCallingInToolBoxes(item)
            result_list.append({"func_name": func_name, "content": result})
            start = index + len(item)
        else:
            break
    return result_list

def call_calling(text: str):
    print("calling_text: ", text)
    json_lists = ExtractJsonStrings(text)
    result_list = split_with_calling_blocks(text, json_lists)
    result_text = ""
    for result in result_list:
        func_name = result["func_name"]
        content = result["content"]
        result_text += f'The function "{func_name}" was called. returned the result "{content}"'
        result_text += '\n\n'
    new_answer = text
    for json_str in json_lists:
        new_answer = new_answer.replace(json_str, "")
    if "```json" in new_answer:
        new_answer = new_answer.replace("```json", "")
    if "```" in new_answer:
        new_answer = new_answer.replace("```", "")
    new_answer = new_answer.strip(' \n')
    return result_text, new_answer

class ToolsType(Enum):
    Unknown = 0
    ToolSearchEngine = 1
    ToolKnowledgeBase = 2
    ToolFunctionCalling = 3
    ToolToolBoxes = 4