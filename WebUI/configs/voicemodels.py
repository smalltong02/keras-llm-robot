import torch
import threading
from transformers import AutoModelForSpeechSeq2Seq

def init_voice_models():
    model_id = "D:\\MLModel\\Audio-to-Text\\whisper-large-v3" 
    torch_dtype = torch.float16
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
    model.to(device)
    print("Voice model load success!")
    return model
