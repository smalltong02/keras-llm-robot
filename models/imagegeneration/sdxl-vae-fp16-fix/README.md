---
license: mit
tags:
- stable-diffusion
- stable-diffusion-diffusers
inference: false
---
# SDXL-VAE-FP16-Fix

SDXL-VAE-FP16-Fix is the [SDXL VAE](https://huggingface.co/stabilityai/sdxl-vae)*, but modified to run in fp16 precision without generating NaNs.

| VAE                   | Decoding in `float32` / `bfloat16` precision | Decoding in `float16` precision |
| --------------------- | -------------------------------------------- | ------------------------------- |
| SDXL-VAE              | ✅ ![](./images/orig-fp32.png)              | ⚠️ ![](./images/orig-fp16.png)  |
| SDXL-VAE-FP16-Fix     | ✅ ![](./images/fix-fp32.png)               | ✅ ![](./images/fix-fp16.png)   |

## 🧨 Diffusers Usage

Just load this checkpoint via `AutoencoderKL`:

```py
import torch
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
pipe.to("cuda")

refiner = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
refiner.to("cuda")

n_steps = 40
high_noise_frac = 0.7

prompt = "A majestic lion jumping from a big stone at night"

image = pipe(prompt=prompt, num_inference_steps=n_steps, denoising_end=high_noise_frac, output_type="latent").images
image = refiner(prompt=prompt, num_inference_steps=n_steps, denoising_start=high_noise_frac, image=image).images[0]
image
```

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lion_refined.png)

## Details

SDXL-VAE generates NaNs in fp16 because the internal activation values are too big:
![](./images/activation-magnitudes.jpg)

SDXL-VAE-FP16-Fix was created by finetuning the SDXL-VAE to:
1. keep the final output the same, but
2. make the internal activation values smaller, by
3. scaling down weights and biases within the network

There are slight discrepancies between the output of SDXL-VAE-FP16-Fix and SDXL-VAE, but the decoded images should be [close enough for most purposes](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/discussions/7#64c5c0f8e2e5c94bd04eaa80).

---

\* `sdxl-vae-fp16-fix` is specifically based on [SDXL-VAE (0.9)](https://huggingface.co/stabilityai/sdxl-vae/discussions/6#64acea3f7ac35b7de0554490), but it works with SDXL 1.0 too