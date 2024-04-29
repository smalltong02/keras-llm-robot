---
license: openrail++
tags:
- text-to-image
- stable-diffusion
library_name: diffusers
---

# SDXL-Lightning

![Intro Image](sdxl_lightning_samples.jpg)

SDXL-Lightning is a lightning-fast text-to-image generation model. It can generate high-quality 1024px images in a few steps. For more information, please refer to our research paper: [SDXL-Lightning: Progressive Adversarial Diffusion Distillation](https://arxiv.org/abs/2402.13929). We open-source the model as part of the research.

Our models are distilled from [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0). This repository contains checkpoints for 1-step, 2-step, 4-step, and 8-step distilled models. The generation quality of our 2-step, 4-step, and 8-step model is amazing. Our 1-step model is more experimental.

We provide both full UNet and LoRA checkpoints. The full UNet models have the best quality while the LoRA models can be applied to other base models.

## Diffusers Usage

Please always use the correct checkpoint for the corresponding inference steps.

### 2-Step, 4-Step, 8-Step UNet

```python
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")
```

### 2-Step, 4-Step, 8-Step LoRA

```python
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_lora.safetensors" # Use the correct ckpt for your step setting!

# Load model.
pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.load_lora_weights(hf_hub_download(repo, ckpt))
pipe.fuse_lora()

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=4, guidance_scale=0).images[0].save("output.png")
```

### 1-Step UNet
The 1-step model is only experimental and the quality is much less stable. Consider using the 2-step model for much better quality.

The 1-step model uses "sample" prediction instead of "epsilon" prediction! The scheduler needs to be configured correctly.

```python
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_1step_unet_x0.safetensors" # Use the correct ckpt for your step setting!

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16").to("cuda")

# Ensure sampler uses "trailing" timesteps and "sample" prediction type.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample")

# Ensure using the same inference steps as the loaded model and CFG set to 0.
pipe("A girl smiling", num_inference_steps=1, guidance_scale=0).images[0].save("output.png")
```


## ComfyUI Usage

Please always use the correct checkpoint for the corresponding inference steps.
Please use Euler sampler with sgm_uniform scheduler.

### 2-Step, 4-Step, 8-Step UNet

1. Download the full checkpoint (`sdxl_lightning_Nstep.safetensors`) to `/ComfyUI/models/checkpoints`.
1. Download our [ComfyUI full workflow](comfyui/sdxl_lightning_workflow_full.json).

![SDXL-Lightning ComfyUI Full Workflow](comfyui/sdxl_lightning_workflow_full.jpg)

### 2-Step, 4-Step, 8-Step LoRA

1. Prepare your own base model.
1. Download the LoRA checkpoint (`sdxl_lightning_Nstep_lora.safetensors`) to `/ComfyUI/models/loras`
1. Download our [ComfyUI LoRA workflow](comfyui/sdxl_lightning_workflow_lora.json).

![SDXL-Lightning ComfyUI LoRA Workflow](comfyui/sdxl_lightning_workflow_lora.jpg)

### 1-Step UNet

The 1-step model is only experimental and the quality is much less stable. Consider using the 2-step model for much better quality.

1. Update your ComfyUI to the latest version.
1. Download the full checkpoint (`sdxl_lightning_1step_x0.safetensors`) to `/ComfyUI/models/checkpoints`.
1. Download our [ComfyUI full 1-step workflow](comfyui/sdxl_lightning_workflow_full_1step.json).

![SDXL-Lightning ComfyUI Full 1-Step Workflow](comfyui/sdxl_lightning_workflow_full_1step.jpg)


## Cite Our Work
```
@misc{lin2024sdxllightning,
      title={SDXL-Lightning: Progressive Adversarial Diffusion Distillation}, 
      author={Shanchuan Lin and Anran Wang and Xiao Yang},
      year={2024},
      eprint={2402.13929},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```