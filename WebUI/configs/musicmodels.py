import io
import base64
import torch
import PIL.Image
import numpy as np
from typing import *
from WebUI.webui_pages.utils import *
from WebUI.Server.utils import detect_device

def init_music_generation_models(config):
    if isinstance(config, dict):
        device = config.get("device", "auto")
        device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
        model_id = config["model_path"]
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        model = MusicgenForConditionalGeneration.from_pretrained(model_id)
        model.to(device)
        return model, processor
    return None, None

def translate_music_generation_data(model, processor, config, text_data: str = "", btranslate_prompt: bool = False) -> str:
    if len(text_data) and model is not None:
        import scipy
        import scipy.io.wavfile as wavfile
        guiding_scale = config["guiding_scale"]
        max_new_tokens = config["max_new_tokens"]
        do_sample = config["do_sample"]
        device = config.get("device", "auto")
        device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
        inputs = processor(
            text=[text_data],
            padding=True,
            return_tensors="pt",
        ).to(device)
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
        if device != "cpu":
            audio_values = audio_values.cpu()
        wavpath = str(TMP_DIR / "musicgen_out.wav")
        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(wavpath, rate=sampling_rate, data=audio_values[0, 0].numpy())
        sampling_rate, wav_data = wavfile.read(wavpath)
        audiodata = io.BytesIO()
        wav_data = wavfile.write(audiodata, sampling_rate, wav_data)
        audiodata = audiodata.getvalue()
        audiodata = base64.b64encode(audiodata).decode('utf-8')
        return audiodata
    return ""