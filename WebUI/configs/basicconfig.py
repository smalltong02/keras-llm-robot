from enum import Enum
from typing import *
from pathlib import Path
import os
import copy

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
                if name == "gpt-4-vision-preview":
                    model_subtype = ModelSubType.VisionChatModel
                if name == "gemini-pro-vision":
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
        index = len(inputs)
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
    if model_name == "OpenDalleV1.1" or model_name == "ProteusV0.2":        
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
        return new_prompt
    return prompt