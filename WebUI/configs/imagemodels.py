import io
import base64
import torch
import PIL.Image
import numpy as np
from typing import *
from WebUI.webui_pages.utils import *
from WebUI.Server.utils import detect_device

def init_image_recognition_models(config):
    if isinstance(config, dict):
        if config["model_name"] == "blip-image-captioning-large":
            from transformers import BlipProcessor, BlipForConditionalGeneration
            model_id = config["model_path"]
            torch_dtype = torch.float32
            if config["loadbits"] != 32:
                torch_dtype = torch.float16
            device = config.get("device", "auto")
            device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
            if device == "cuda":
                processor = BlipProcessor.from_pretrained(model_id)
                model = BlipForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch_dtype)
            else:
                torch_dtype = torch.float16
                processor = BlipProcessor.from_pretrained(model_id)
                model = BlipForConditionalGeneration.from_pretrained(model_id)
            model.to(device)
            return model, processor
    return None, None

def translate_image_recognition_data(model, processor, config, imagedata: str = "") -> str:
    answer = ""
    if len(imagedata) and model is not None:
        decoded_data = base64.b64decode(imagedata)
        if isinstance(config, dict):
            if config["model_name"] == "blip-image-captioning-large":
                from io import BytesIO
                from PIL import Image
                imagedata = BytesIO(decoded_data)
                raw_image = Image.open(imagedata).convert('RGB')

                torch_dtype = torch.float32
                if config["loadbits"] != 32:
                    torch_dtype = torch.float16
                device = config.get("device", "auto")
                device = "cuda" if device == "gpu" else detect_device() if device == "auto" else device
                if device == "cuda":
                    inputs = processor(raw_image, return_tensors="pt").to(device, torch_dtype)
                else:
                    inputs = processor(raw_image, return_tensors="pt")
                output = model.generate(**inputs)
                answer = processor.decode(output[0], skip_special_tokens=True)
    return answer

def generate(
    pipe: Any,
    refiner: Any,
    prompt: str,
    negative_prompt: str = "",
    prompt_2: str = "",
    negative_prompt_2: str = "",
    use_negative_prompt: bool = False,
    use_prompt_2: bool = False,
    use_negative_prompt_2: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale_base: float = 5.0,
    guidance_scale_refiner: float = 5.0,
    num_inference_steps_base: int = 25,
    num_inference_steps_refiner: int = 25,
    apply_refiner: bool = False,
) -> PIL.Image.Image:
    print(f"** Generating image for: \"{prompt}\" **")
    generator = torch.Generator().manual_seed(seed)

    if not use_negative_prompt:
        negative_prompt = None  # type: ignore
    if not use_prompt_2:
        prompt_2 = None  # type: ignore
    if not use_negative_prompt_2:
        negative_prompt_2 = None  # type: ignore

    if not apply_refiner:
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            width=width,
            height=height,
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type="pil",
        ).images[0]
    else:
        latents = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            width=width,
            height=height,
            guidance_scale=guidance_scale_base,
            num_inference_steps=num_inference_steps_base,
            generator=generator,
            output_type="latent",
        ).images
        image = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_2=prompt_2,
            negative_prompt_2=negative_prompt_2,
            guidance_scale=guidance_scale_refiner,
            num_inference_steps=num_inference_steps_refiner,
            image=latents,
            generator=generator,
        ).images[0]
        return image

def init_image_generation_models(config):
    if not torch.cuda.is_available():
        return None
    if isinstance(config, dict):
        if config["model_name"] == "OpenDalleV1.1" or config["model_name"] == "ProteusV0.2":
            from diffusers import AutoencoderKL, DiffusionPipeline
            model_id = config["model_path"]
            enable_torch_compile = config["torch_compile"]
            enable_cpu_offload = config["cpu_offload"]
            enable_refiner = config["refiner"]
            vae_id = "models/imagegeneration/sdxl-vae-fp16-fix"
            refiner_id = "models/imagegeneration/stable-diffusion-xl-refiner-1.0"

            vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=torch.float16)
            pipe = DiffusionPipeline.from_pretrained(
                model_id,
                vae=vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
            refiner = None
            if ImageModelExist("models/imagegeneration/stable-diffusion-xl-refiner-1.0") == False:
                enable_refiner = False
            if enable_refiner:
                refiner = DiffusionPipeline.from_pretrained(
                    refiner_id,
                    vae=vae,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if enable_cpu_offload:
                pipe.enable_model_cpu_offload()
                if enable_refiner:
                    refiner.enable_model_cpu_offload()
            else:
                pipe.to(device)
                if enable_refiner:
                    refiner.to(device)

            if enable_torch_compile:
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
                if enable_refiner:
                    refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
            return pipe, refiner
    return None, None

def translate_image_generation_data(model, refiner, config, text_data: str = "") -> str:
    def split_prompt(prompt):
        split_index = prompt.find(":")
        first_part = prompt[:split_index + 1].strip()
        second_part = prompt[split_index + 1:].strip()
        return first_part, second_part
    def first_prompt(prompt):
        lines = prompt.strip().split('\n')
        return lines[0]

    if len(text_data) and model is not None:
        if isinstance(config, dict):
            if config["model_name"] == "OpenDalleV1.1" or config["model_name"] == "ProteusV0.2":
                seed = config["seed"]
                apply_refiner = False if refiner is None else True
                if seed == -1:
                    import random
                    seed = random.randint(np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                _, prompt = split_prompt(text_data)
                prompt = first_prompt(prompt)
                image = generate(
                    pipe=model,
                    refiner=refiner,
                    prompt=prompt,
                    seed=seed,
                    apply_refiner=apply_refiner,
                )
                if image:
                    imagedata = io.BytesIO()
                    image.save(imagedata, format="jpeg")
                    imagedata = imagedata.getvalue()
                    imagedata = base64.b64encode(imagedata).decode('utf-8')
                return imagedata
    return ""
