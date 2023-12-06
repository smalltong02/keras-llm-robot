import io
import torch
import base64
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def init_voice_models(config):
    if isinstance(config, dict):
        if config["model_name"] == "whisper-large-v3" or config["model_name"] == "whisper-base" or config["model_name"] == "whisper-medium":
            model_id = config["model_path"]
            device =  'cuda' if config["device"] == 'gpu' else config["device"]
            if device == "cpu":
                torch_dtype = torch.float32
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
                    )
            else:
                torch_dtype = torch.float16
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id, torch_dtype=torch_dtype, use_safetensors=True
                    )
            model.to(device)
            return model
        elif config["model_name"] == "faster-whisper-large-v3":
            from faster_whisper import WhisperModel
            model_id = config["model_path"]
            device =  'cuda' if config["device"] == 'gpu' else config["device"]
            compute_type = "float16"
            if device == "cpu":
                compute_type = "float32"
                if config["loadbits"] == 8:
                    compute_type = "int8"
            elif device == "cuda" and config["loadbits"] == 8:
                compute_type = "int8_float16"
            model = WhisperModel(model_id, device=device, compute_type=compute_type)
            return model
        else:
            pass
    return None

def translate_voice_data(model, config, voice_data: str = "") -> str:
    if len(voice_data):
        decoded_data = base64.b64decode(voice_data)
        if isinstance(config, dict):
            if config["model_name"] == "whisper-large-v3" or config["model_name"] == "whisper-base" or config["model_name"] == "whisper-medium":
                model_id = config["model_path"]
                if config["loadbits"] == 16:
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.float32
                device =  'cuda' if config["device"] == 'gpu' else config["device"]
                model.to(device)
                processor = AutoProcessor.from_pretrained(model_id)
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=16,
                    return_timestamps=True,
                    torch_dtype=torch_dtype,
                    device=device,
                )
                result = pipe(decoded_data)
                return result["text"]
            elif config["model_name"] == "faster-whisper-large-v3":
                binary_data = io.BytesIO(decoded_data)
                segments, _ = model.transcribe(binary_data, beam_size=5)
                result = ""
                for segment in segments:
                    result += segment.text
                return result
            else:
                pass
    return ""


    # 
    # 
    # 

    # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # texts = ""
    # for segment in segments:
    #     texts += segment.text
    # print(texts)