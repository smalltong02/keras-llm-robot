---
pipeline_tag: text-to-image
widget:
- text: >-
    black fluffy gorgeous dangerous cat animal creature, large orange eyes, big
    fluffy ears, piercing gaze, full moon, dark ambiance, best quality,
    extremely detailed
  output:
    url: ComfyUI_03087_.png
- text: >-
    (impressionistic realism by csybgh), a 50 something male, working in
    banking, very short dyed dark curly balding hair, Afro-Asiatic ancestry,
    talks a lot but listens poorly, stuck in the past, wearing a suit, he has a
    certain charm, bronze skintone, sitting in a bar at night, he is smoking and
    feeling cool, drunk on plum wine, masterpiece, 8k, hyper detailed, smokey
    ambiance, perfect hands AND fingers
  output:
    url: GEN8-iTXcAA-okN.jpeg
- text: >-
    high quality pixel art, a pixel art silhouette of an anime space-themed girl
    in a space-punk steampunk style, lying in her bed by the window of a
    spaceship, smoking, with a rustic feel. The image should embody epic
    portraiture and double exposure, featuring an isolated landscape visible
    through the window. The colors should primarily be dynamic and
    action-packed, with a strong use of negative space. The entire artwork
    should be in pixel art style, emphasizing the characters shape and set
    against a white background. Silhouette
  output:
    url: ComfyUI_03060_.png
- text: >-
    The image features an older man, a long white beard and mustache,  He has a
    stern expression, giving the impression of a wise and experienced
    individual. The mans beard and mustache are prominent, adding to his
    distinguished appearance. The close-up shot of the mans face emphasizes his
    facial features and the intensity of his gaze.
  output:
    url: ComfyUI_03017_.png
- text: >-
    Super Closeup Portrait, action shot, Profoundly dark whitish meadow, glass
    flowers, Stains, space grunge style, Jeanne d'Arc wearing White Olive green
    used styled Cotton frock, Wielding thin silver sword, Sci-fi vibe, dirty,
    noisy, Vintage monk style, very detailed, hd
  output:
    url: ComfyUI_03045.png
- text: >-
    cinematic film still of Kodak Motion Picture Film: (Sharp Detailed Image) An
    Oscar winning movie for Best Cinematography a woman in a kimono standing on
    a subway train in Japan Kodak Motion Picture Film Style, shallow depth of
    field, vignette, highly detailed, high budget, bokeh, cinemascope, moody,
    epic, gorgeous, film grain, grainy
  output:
    url: 3.png
- text: >-
    in the style of artgerm, comic style,3D model, mythical seascape, negative
    space, space quixotic dreams, temporal hallucination, psychedelic, mystical,
    intricate details, very bright neon colors, (vantablack background:1.5),
    pointillism, pareidolia, melting, symbolism, very high contrast, chiaroscuro
  parameters:
    negative_prompt: >-
      bad quality, bad anatomy, worst quality, low quality, low resolutions,
      extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image
      artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image
  output:
    url: ComfyUI_03061_.png
- text: >-
    1980s anime portrait of a character glitching. His face is separated from
    his body by heavy static. His face is deformed by pain. Dream-like, analog
    horror, glitch, terrifying
  output:
    url: ComfyUI_03092_.png
- text: (("Proteus"):text_logo:1)
  output:
    url: ComfyUI_03297_.png
- text: >-
    dan seagrave, dante, Abandon All Hope, Ye Who Enter Here, hell religious art
    purgatory zdzislaw Beksinski, abyss inferno, lost, wanderer
  output:
    url: ComfyUI_03483_.png
---
<Gallery />
## ProteusV0.2

merged with RealCartoonXL to fix issues with inability to understand tags related to anime or cartoon styles at just a weight of 0.5% out of 100% using custom scripts with slerp like methods.

Version 0.2 shows subtle yet significant improvements over Version 0.1. It demonstrates enhanced prompt understanding that surpasses MJ6, while also approaching its stylistic capabilities.

## Proteus

Proteus serves as a sophisticated enhancement over OpenDalleV1.1, leveraging its core functionalities to deliver superior outcomes. Key areas of advancement include heightened responsiveness to prompts and augmented creative capacities. To achieve this, it was fine-tuned using approximately 220,000 GPTV captioned images from copyright-free stock images (with some anime included), which were then normalized. Additionally, DPO (Direct Preference Optimization) was employed through a collection of 10,000 carefully selected high-quality, AI-generated image pairs.

In pursuit of optimal performance, numerous LORA (Low-Rank Adaptation) models are trained independently before being selectively incorporated into the principal model via dynamic application methods. These techniques involve targeting particular segments within the model while avoiding interference with other areas during the learning phase. Consequently, Proteus exhibits marked improvements in portraying intricate facial characteristics and lifelike skin textures, all while sustaining commendable proficiency across various aesthetic domains, notably surrealism, anime, and cartoon-style visualizations.


## Settings for ProteusV0.2

Use these settings for the best results with ProteusV0.2:

CFG Scale: Use a CFG scale of 8 to 7

Steps: 20 to 60 steps for more detail, 20 steps for faster results.

Sampler: DPM++ 2M SDE

Scheduler: Karras

Resolution: 1280x1280 or 1024x1024

please also consider using these keep words to improve your prompts:
best quality, HD, `~*~aesthetic~*~`. 

if you are having trouble coming up with prompts you can use this GPT I put together to help you refine the prompt. https://chat.openai.com/g/g-RziQNoydR-diffusion-master

## Use it with 🧨 diffusers
```python
import torch
from diffusers import (
    StableDiffusionXLPipeline, 
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL
)

# Load VAE component
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float16
)

# Configure the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "dataautogpt3/ProteusV0.2", 
    vae=vae,
    torch_dtype=torch.float16
)
pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.to('cuda')

# Define prompts and generate image
prompt = "black fluffy gorgeous dangerous cat animal creature, large orange eyes, big fluffy ears, piercing gaze, full moon, dark ambiance, best quality, extremely detailed"
negative_prompt = "nsfw, bad quality, bad anatomy, worst quality, low quality, low resolutions, extra fingers, blur, blurry, ugly, wrongs proportions, watermark, image artifacts, lowres, ugly, jpeg artifacts, deformed, noisy image"

image = pipe(
    prompt, 
    negative_prompt=negative_prompt, 
    width=1024,
    height=1024,
    guidance_scale=7,
    num_inference_steps=20
).images[0]
```
You are free to:

Share — copy and redistribute the material in any medium or format for personal use only. Commercial use is not permitted without direct consultation from the author.

Adapt — remix, transform, and build upon the material for personal use only. Commercial use is not permitted without direct consultation from the author.

The licensor cannot revoke these freedoms as long as you follow the license terms.

Under the following terms:

Attribution — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

please support the work I do through donating to me on: 
https://www.buymeacoffee.com/DataVoid
or following me on
https://twitter.com/DataPlusEngine