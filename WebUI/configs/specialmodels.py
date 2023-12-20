import json
import google.generativeai as genai
from WebUI.configs.basicconfig import *
from WebUI.configs.webuiconfig import InnerJsonConfigWebUIParse
from WebUI.Server.db.repository import add_chat_history_to_db, update_chat_history
from WebUI.Server.chat.StreamHandler import StreamSpeakHandler

def clean_special_text(text : str):
  # Split the text at the first occurrence of "<|im_start|>"
  parts = text.split("<|im_start|>", 1)
  # If there are parts, take the first one excluding the special string
  if len(parts) > 0:
    cleaned_text = parts[0]
  else:
     cleaned_text = text

  cleaned_text = cleaned_text.replace('<|im_end|>', '').strip()

  return cleaned_text

def init_cloud_models(model_name):
    if model_name is None:
        return None
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)
    if modelinfo["mtype"] != ModelType.Online:
        return None
    modelinfo["mname"] = model_name
    provider = GetProviderByName(webui_config, model_name)
    if provider is None or model_name is None:
        return None
    if provider != "google-api":
        return None
    model_config = GetModelConfig(webui_config, modelinfo)
    if len(model_config) == 0:
        return None
    apikey = model_config.get("api_key", "[Your Key]")
    if apikey == "[Your Key]":
        apikey = os.environ.get('GOOGLE_API_KEY')
    if apikey == None:
        apikey = "EMPTY"
    genai.configure(api_key=apikey)
    model = genai.GenerativeModel('gemini-pro')
    return model

def special_model_chat(
        model: Any,
        model_name: str,
        query: str,
        history: List[dict],
        stream: bool,
        speechmodel: dict,
        temperature: float,
        max_tokens: Optional[int],
        prompt_name: str,
) -> dict:
    if model == None or model_name == "":
        return {"code": 500, "answer": "error!"} 
    
    configinst = InnerJsonConfigWebUIParse()
    webui_config = configinst.dump()
    modelinfo = {"mtype": ModelType.Unknown, "msize": ModelSize.Unknown, "msubtype": ModelSubType.Unknown, "mname": str, "config": dict}
    modelinfo["mtype"], modelinfo["msize"], modelinfo["msubtype"] = GetModelInfoByName(webui_config, model_name)

    speak_handler = None
    if len(speechmodel):
            modeltype = speechmodel.get("type", "")
            spmodel = speechmodel.get("model", "")
            spspeaker = speechmodel.get("speaker", "")
            speechkey = speechmodel.get("speech_key", "")
            if speechkey == "":
                speechkey = os.environ.get('SPEECH_KEY')
            speechregion = speechmodel.get("speech_region", "")
            if speechregion == "":
                speechregion = os.environ.get('SPEECH_REGION')
            if modeltype == "local" or modeltype == "cloud":
                speak_handler = StreamSpeakHandler(run_place=modeltype, synthesis=spspeaker, subscription=speechkey, region=speechregion)

    answer = ""
    chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)
    if modelinfo["mtype"] == ModelType.Special:
        prompt_template = '''
            <|im_start|>user
            {input}<|im_end|>
            <|im_start|>assistant
            '''
        chat_message = prompt_template.format(input=query)
        answer = model(chat_message)
        answer = clean_special_text(answer)
        if speak_handler: speak_handler.on_llm_new_token(answer)
    elif modelinfo["mtype"] == ModelType.Online:
        provider = GetProviderByName(webui_config, model_name)
        if provider == "google-api":
            updated_history = [
                {'parts': entry['content'], **({'role': 'model'} if entry['role'] == 'assistant' else {'role': entry['role']})}
                for entry in history
            ]
            chat = model.start_chat(history=updated_history)
            generation_config = {'temperature': temperature}
            response = chat.send_message(query, generation_config=generation_config, stream=stream)
            if stream is True:
                for chunk in response:
                    if speak_handler: speak_handler.on_llm_new_token(chunk.text)
                    answer += chunk.text
            else:
                if speak_handler: speak_handler.on_llm_new_token(answer)
                answer = response.text
    if speak_handler: speak_handler.on_llm_end(None)
    update_chat_history(chat_history_id, response=answer)
    
    jsdata = json.dumps(
        {"text": answer, "chat_history_id": chat_history_id},
        ensure_ascii=False)
    return {"code": 200, "answer": jsdata}