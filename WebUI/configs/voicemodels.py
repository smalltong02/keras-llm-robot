import io, os
import torch
import base64
from WebUI.Server.utils import detect_device
from WebUI.configs.basicconfig import TMP_DIR
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import wave

def init_voice_models(config):
    if isinstance(config, dict):
        if config["model_name"] == "whisper-large-v3" or config["model_name"] == "whisper-base" or config["model_name"] == "whisper-medium":
            model_id = config["model_path"]
            device = config.get("device", "auto")
            device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
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
            device = config.get("device", "auto")
            device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
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
                device = config.get("device", "auto")
                device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
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

def cloud_voice_data(config, voice_data: str="") -> str:
    if len(voice_data):
        decoded_data = base64.b64decode(voice_data)
        if isinstance(config, dict):
            provider = config["provider"]
            if provider == "AzureCloud":
                import azure.cognitiveservices.speech as speechsdk
                voicekey = config.get("voice_key", "[Your Key]")
                if voicekey == "[Your Key]":
                    voicekey = os.environ.get('SPEECH_KEY')
                voiceregion = config.get("voice_region", "[Your Region]")
                if voiceregion == "[Your Region]":
                    voiceregion = os.environ.get('SPEECH_REGION')
                speech_config = speechsdk.SpeechConfig(subscription=voicekey, region=voiceregion)
                auto_detect_source_language_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-US", "zh-CN"])
                file_name = str(TMP_DIR / "recognizer.wav")
                with open(file_name, "wb") as file:
                    file.write(decoded_data)
                audio_config = speechsdk.audio.AudioConfig(filename=file_name)
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config, auto_detect_source_language_config=auto_detect_source_language_config)
                
                speech_recognition_result = speech_recognizer.recognize_once_async().get()
                if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    print("Recognized: {}".format(speech_recognition_result.text))
                    return speech_recognition_result.text
                elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
                    print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
                elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = speech_recognition_result.cancellation_details
                    print("Speech Recognition canceled: {}".format(cancellation_details.reason))
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        print("Error details: {}".format(cancellation_details.error_details))
                        print("Did you set the speech resource key and region values?")
            else:
                pass
    return ""


def init_speech_models(config):
    if isinstance(config, dict):
        from TTS.api import TTS
        model_id = config["model_path"]
        config_path = model_id + "/config.json"
        device = config.get("device", "auto")
        device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
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
