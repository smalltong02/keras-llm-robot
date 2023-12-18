import io
import torch
import base64
from WebUI.configs.basicconfig import TMP_DIR
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from TTS.api import TTS
import wave

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

def init_speech_models(config):
    if isinstance(config, dict):
        model_id = config["model_path"]
        config_path = model_id + "/config.json"
        device =  'cuda' if config["device"] == 'gpu' else config["device"]
        tts_model = TTS(model_path=model_id, config_path=config_path, progress_bar=False)
        tts_model.to(device)
        return tts_model
    return None

def translate_speech_data(model, config, text_data: str = "", speech_type: str = "en-us-female-1") -> str:
    if len(text_data):
        if isinstance(config, dict) and speech_type != None:
            parts = speech_type.split('_')
            language = parts[0]
            synthesis = parts[1]
            if synthesis == "female-v1":
                speaker_wav = "WebUI/configs/speech_template/female-1.wav"
            elif synthesis == "female-v2":
                speaker_wav = "WebUI/configs/speech_template/female-2.wav"
            elif synthesis == "male-v1":
                speaker_wav = "WebUI/configs/speech_template/male-1.wav"
            elif synthesis == "male-v2":
                speaker_wav = "WebUI/configs/speech_template/male-2.wav"
            wav_file_path = str(TMP_DIR / "speech.wav")

            model.tts_to_file(text_data, speaker_wav=speaker_wav, language=language, file_path=wav_file_path)

            raw_data = None
            with wave.open(wav_file_path, 'rb') as wave_file:
                channels = wave_file.getnchannels()
                sample_width = wave_file.getsampwidth()
                frame_rate = wave_file.getframerate()
                frames = wave_file.getnframes()
                raw_data = wave_file.readframes(frames)
            if raw_data != None:
                base64_data = base64.b64encode(raw_data).decode('utf-8')
                return channels, sample_width, frame_rate, base64_data
            return 0, 0, 0, 0, ""
