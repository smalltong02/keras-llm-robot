from enum import Enum
from typing import *
from pathlib import Path
import os
import copy

SAVE_CHAT_HISTORY = True

TMP_DIR = Path('temp')
if not TMP_DIR.exists():
    TMP_DIR.mkdir(exist_ok=True, parents=True)

glob_model_type_list = ["LLM Model","Multimodal Model","Special Model","Online Model"]
glob_model_size_list = ["3B Model","7B Model","13B Model","34B Model","70B Model"]
glob_model_subtype_list = ["Vision Chat Model","Voice Chat Model","Video Chat Model"]

class ModelType(Enum):
    Unknown = 0
    Local = 1
    Multimodal = 2
    Special = 3
    Online = 4

class ModelSize(Enum):
    Unknown = 0
    Mod3B = 1
    Mod7B = 2
    Mod13B = 3
    Mod34B = 4
    Mod70B = 5

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

def LocalModelExist(local_path):
    total_size_bytes = 0
    for dirpath, _, filenames in os.walk(local_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size_bytes += os.path.getsize(filepath)
    total_size_gb = total_size_bytes / (1024**3)
    print("total_size_gb: ", total_size_gb)
    if total_size_gb > 1:
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
    pass

def GetKbPath(kb_name: str):
    pass

def GetDocPath(kb_name: str):
    pass

def GetKbsList():
    kb_list = []
    kb_config = GetKbConfig()
    kbs_config = kb_config.get("kbs_config", {})
    for key, _ in kbs_config.items():
        kb_list.append(key)
    return kb_list

def GetTextSplitterDict():
    kb_config = GetKbConfig()
    text_splitter_dict = kb_config.get("text_splitter_dict", {})
    return text_splitter_dict
