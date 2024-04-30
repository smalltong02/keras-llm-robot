---
license: openrail++
library_name: diffusers
inference: false
tags:
  - lora
  - text-to-image
  - stable-diffusion
---

# Hyper-SD
Official Repository of the paper: *[Hyper-SD](https://arxiv.org/abs/2404.13686)*.

Project Page: https://hyper-sd.github.io/

![](./hypersd_tearser.jpg)


## News🔥🔥🔥

* Apr.28, 2024. ComfyUI workflows on 1-Step Unified LoRA 🥰 with TCDScheduler to inference on different steps are [released](https://huggingface.co/ByteDance/Hyper-SD/tree/main/comfyui)! Remember to install ⭕️ [ComfyUI-TCD](https://github.com/JettHu/ComfyUI-TCD) in your `ComfyUI/custom_nodes` folder!!! You're encouraged to adjust the eta parameter to get better results 🌟!
* Apr.26, 2024. 💥💥💥 Our CFG-Preserved Hyper-SD15/SDXL that facilitate negative prompts and larger guidance scales (e.g. 5~10) will be coming soon!!! 💥💥💥
* Apr.26, 2024. Thanks to @[Pete](https://huggingface.co/pngwn) for contributing to our [scribble demo](https://huggingface.co/spaces/ByteDance/Hyper-SD15-Scribble) with larger canvas right now 👏.
* Apr.24, 2024. The ComfyUI [workflow](https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/Hyper-SDXL-1step-Unet-workflow.json) and [checkpoint](https://huggingface.co/ByteDance/Hyper-SD/blob/main/Hyper-SDXL-1step-Unet-Comfyui.fp16.safetensors) on 1-Step SDXL UNet ✨ is also available! Don't forget ⭕️ to install the custom [scheduler](https://huggingface.co/ByteDance/Hyper-SD/tree/main/comfyui/ComfyUI-HyperSDXL1StepUnetScheduler) in your `ComfyUI/custom_nodes` folder!!!
* Apr.23, 2024. ComfyUI workflows on N-Steps LoRAs are [released](https://huggingface.co/ByteDance/Hyper-SD/tree/main/comfyui)! Worth a try for creators 💥!
* Apr.23, 2024. Our technical report 📚 is uploaded to [arXiv](https://arxiv.org/abs/2404.13686)! Many implementation details are provided and we welcome more discussions👏.
* Apr.21, 2024. Hyper-SD ⚡️ is highly compatible and work well with different base models and controlnets. To clarify, we also append the usage example of controlnet [here](https://huggingface.co/ByteDance/Hyper-SD#controlnet-usage).
* Apr.20, 2024. Our checkpoints and two demos 🤗 (i.e. [SD15-Scribble](https://huggingface.co/spaces/ByteDance/Hyper-SD15-Scribble) and [SDXL-T2I](https://huggingface.co/spaces/ByteDance/Hyper-SDXL-1Step-T2I)) are publicly available on [HuggingFace Repo](https://huggingface.co/ByteDance/Hyper-SD).

## Try our Hugging Face demos: 
Hyper-SD Scribble demo host on [🤗 scribble](https://huggingface.co/spaces/ByteDance/Hyper-SD15-Scribble) 

Hyper-SDXL One-step Text-to-Image demo host on [🤗 T2I](https://huggingface.co/spaces/ByteDance/Hyper-SDXL-1Step-T2I)

## Introduction

Hyper-SD is one of the new State-of-the-Art diffusion model acceleration techniques.
In this repository, we release the models distilled from [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [Stable-Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)。

## Checkpoints

* `Hyper-SDXL-Nstep-lora.safetensors`: Lora checkpoint, for SDXL-related models.
* `Hyper-SD15-Nstep-lora.safetensors`: Lora checkpoint, for SD1.5-related models.
* `Hyper-SDXL-1step-unet.safetensors`: Unet checkpoint distilled from SDXL-Base.

## Text-to-Image Usage
### SDXL-related models
#### 2-Steps, 4-Steps, 8-steps LoRA
Take the 2-steps LoRA as an example, you can also use other LoRAs for the corresponding inference steps setting.  
```python
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SDXL-2steps-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
# lower eta results in more detail
prompt="a photo of a cat"
image=pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0]
```

#### Unified LoRA (support 1 to 8 steps inference)
You can flexibly adjust the number of inference steps and eta value to achieve best performance. 
```python
import torch
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-1step-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Use TCD scheduler to achieve better image quality
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# Lower eta results in more detail for multi-steps inference
eta=1.0
prompt="a photo of a cat"
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, eta=eta).images[0]
```

#### 1-step SDXL Unet
Only for the single step inference.
```python
import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, LCMScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SDXL-1step-Unet.safetensors"
# Load model.
unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo_name, ckpt_name), device="cuda"))
pipe = DiffusionPipeline.from_pretrained(base_model_id, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")
# Use LCM scheduler instead of ddim scheduler to support specific timestep number inputs
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
# Set start timesteps to 800 in the one-step inference to get better results
prompt="a photo of a cat"
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, timesteps=[800]).images[0]
```


### SD1.5-related models

#### 2-Steps, 4-Steps, 8-steps LoRA
Take the 2-steps LoRA as an example, you can also use other LoRAs for the corresponding inference steps setting.
```python
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "ByteDance/Hyper-SD"
# Take 2-steps lora as an example
ckpt_name = "Hyper-SD15-2steps-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
prompt="a photo of a cat"
image=pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0).images[0]
```


#### Unified LoRA (support 1 to 8 steps inference)
You can flexibly adjust the number of inference steps and eta value to achieve best performance.
```python
import torch
from diffusers import DiffusionPipeline, TCDScheduler
from huggingface_hub import hf_hub_download
base_model_id = "runwayml/stable-diffusion-v1-5"
repo_name = "ByteDance/Hyper-SD"
ckpt_name = "Hyper-SD15-1step-lora.safetensors"
# Load model.
pipe = DiffusionPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name))
pipe.fuse_lora()
# Use TCD scheduler to achieve better image quality
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# Lower eta results in more detail for multi-steps inference
eta=1.0
prompt="a photo of a cat"
image=pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0, eta=eta).images[0]
```

## ControlNet Usage
### SDXL-related models

#### 2-Steps, 4-Steps, 8-steps LoRA
Take Canny Controlnet and 2-steps inference as an example:
```python
import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, DDIMScheduler
from huggingface_hub import hf_hub_download

# Load original image
image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
image = np.array(image)
# Prepare Canny Control Image
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
control_image.save("control.png")
control_weight = 0.5  # recommended for good generalization

# Initialize pipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", controlnet=controlnet, vae=vae, torch_dtype=torch.float16).to("cuda")

pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-2steps-lora.safetensors"))
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.fuse_lora()
image = pipe("A chocolate cookie", num_inference_steps=2, image=control_image, guidance_scale=0, controlnet_conditioning_scale=control_weight).images[0]
image.save('image_out.png')
```

#### Unified LoRA (support 1 to 8 steps inference)
Take Canny Controlnet as an example:
```python
import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL, TCDScheduler
from huggingface_hub import hf_hub_download

# Load original image
image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png")
image = np.array(image)
# Prepare Canny Control Image
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
control_image.save("control.png")
control_weight = 0.5  # recommended for good generalization

# Initialize pipeline
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet, vae=vae, torch_dtype=torch.float16).to("cuda")

# Load Hyper-SD15-1step lora
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SDXL-1step-lora.safetensors"))
pipe.fuse_lora()
# Use TCD scheduler to achieve better image quality
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# Lower eta results in more detail for multi-steps inference
eta=1.0
image = pipe("A chocolate cookie", num_inference_steps=4, image=control_image, guidance_scale=0, controlnet_conditioning_scale=control_weight, eta=eta).images[0]
image.save('image_out.png')
```

### SD1.5-related models

#### 2-Steps, 4-Steps, 8-steps LoRA
Take Canny Controlnet and 2-steps inference as an example:
```python
import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, DDIMScheduler

from huggingface_hub import hf_hub_download

controlnet_checkpoint = "lllyasviel/control_v11p_sd15_canny"

# Load original image
image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/input.png")
image = np.array(image)
# Prepare Canny Control Image
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
control_image.save("control.png")

# Initialize pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-2steps-lora.safetensors"))
pipe.fuse_lora()
# Ensure ddim scheduler timestep spacing set as trailing !!!
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
image = pipe("a blue paradise bird in the jungle", num_inference_steps=2, image=control_image, guidance_scale=0).images[0]
image.save('image_out.png')
```


#### Unified LoRA (support 1 to 8 steps inference)
Take Canny Controlnet as an example:
```python
import torch
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, TCDScheduler
from huggingface_hub import hf_hub_download

controlnet_checkpoint = "lllyasviel/control_v11p_sd15_canny"

# Load original image
image = load_image("https://huggingface.co/lllyasviel/control_v11p_sd15_canny/resolve/main/images/input.png")
image = np.array(image)
# Prepare Canny Control Image
low_threshold = 100
high_threshold = 200
image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
control_image = Image.fromarray(image)
control_image.save("control.png")

# Initialize pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16).to("cuda")
# Load Hyper-SD15-1step lora
pipe.load_lora_weights(hf_hub_download("ByteDance/Hyper-SD", "Hyper-SD15-1step-lora.safetensors"))
pipe.fuse_lora()
# Use TCD scheduler to achieve better image quality
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
# Lower eta results in more detail for multi-steps inference
eta=1.0
image = pipe("a blue paradise bird in the jungle", num_inference_steps=1, image=control_image, guidance_scale=0, eta=eta).images[0]
image.save('image_out.png')
```
## Comfyui Usage
* `Hyper-SDXL-Nsteps-lora.safetensors`: [text-to-image workflow](https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/Hyper-SDXL-Nsteps-lora-workflow.json)
* `Hyper-SD15-Nsteps-lora.safetensors`: [text-to-image workflow](https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/Hyper-SD15-Nsteps-lora-workflow.json)
* `Hyper-SDXL-1step-Unet-Comfyui.fp16.safetensors`: [text-to-image workflow](https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/Hyper-SDXL-1step-Unet-workflow.json)
  * **REQUIREMENT / INSTALL** for 1-Step SDXL UNet: Please install our [scheduler folder](https://huggingface.co/ByteDance/Hyper-SD/tree/main/comfyui/ComfyUI-HyperSDXL1StepUnetScheduler) into your `ComfyUI/custom_nodes` to enable sampling from 800 timestep instead of 999. 
  * i.e. making sure the `ComfyUI/custom_nodes/ComfyUI-HyperSDXL1StepUnetScheduler` folder exist.
  * For more details, please refer to our [technical report](https://arxiv.org/abs/2404.13686).
* `Hyper-SD15-1step-lora.safetensors`: [text-to-image workflow](https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/Hyper-SD15-1step-unified-lora-workflow.json)
* `Hyper-SDXL-1step-lora.safetensors`: [text-to-image workflow](https://huggingface.co/ByteDance/Hyper-SD/blob/main/comfyui/Hyper-SDXL-1step-unified-lora-workflow.json)
  * **REQUIREMENT / INSTALL** for 1-Step Unified LoRAs: Please install the [ComfyUI-TCD](https://github.com/JettHu/ComfyUI-TCD) into your `ComfyUI/custom_nodes` to enable TCDScheduler with support of different inference steps (1~8) using single checkpoint.
  * i.e. making sure the `ComfyUI/custom_nodes/ComfyUI-TCD` folder exist.
  * You're encouraged to adjust the eta parameter in TCDScheduler to get better results.

## Citation
```bibtex
@misc{ren2024hypersd,
      title={Hyper-SD: Trajectory Segmented Consistency Model for Efficient Image Synthesis}, 
      author={Yuxi Ren and Xin Xia and Yanzuo Lu and Jiacheng Zhang and Jie Wu and Pan Xie and Xing Wang and Xuefeng Xiao},
      year={2024},
      eprint={2404.13686},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```