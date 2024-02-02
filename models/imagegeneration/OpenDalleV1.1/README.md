---
license: cc-by-nc-nd-4.0
pipeline_tag: text-to-image
widget:
- text: >-
    black fluffy gorgeous dangerous cat animal creature, large orange eyes, big
    fluffy ears, piercing gaze, full moon, dark ambiance, best quality,
    extremely detailed
  output:
    url: ComfyUI_01611_.png
- text: >-
    (impressionistic realism by csybgh), a 50 something male, working in
    banking, very short dyed dark curly balding hair, Afro-Asiatic ancestry,
    talks a lot but listens poorly, stuck in the past, wearing a suit, he has a
    certain charm, bronze skintone, sitting in a bar at night, he is smoking and
    feeling cool, drunk on plum wine, masterpiece, 8k, hyper detailed, smokey
    ambiance, perfect hands AND fingers
  output:
    url: ComfyUI_01609_.jpeg
- text: >-
    an anime female general laughing, with a military cap, evil smile, sadistic,
    grim
  output:
    url: ComfyUI_01556_.jpeg
- text: >-
    John Berkey Style page,ral-oilspill, There is no road ahead,no land,
    Strangely,the river is still flowing,crossing the void into the mysterious
    unknown, The end of nothingness,a huge ripple,it is a kind of wave,and it is
    the law of time that lasts forever in that void, At the end of the infinite
    void,there is a colorful world,very hazy and mysterious,and it cannot be
    seen clearly,but it is real, And that's where the river goes
  output:
    url: ComfyUI_01519_.jpeg
- text: >-
    Super Closeup Portrait, action shot, Profoundly dark whitish meadow, glass
    flowers, Stains, space grunge style, Jeanne d'Arc wearing White Olive green
    used styled Cotton frock, Wielding thin silver sword, Sci-fi vibe, dirty,
    noisy, Vintage monk style, very detailed, hd
  output:
    url: ComfyUI_01817_(1).png
- text: >-
    cinematic film still of Kodak Motion Picture Film: (Sharp Detailed Image) An
    Oscar winning movie for Best Cinematography a woman in a kimono standing on
    a subway train in Japan Kodak Motion Picture Film Style, shallow depth of
    field, vignette, highly detailed, high budget, bokeh, cinemascope, moody,
    epic, gorgeous, film grain, grainy
  output:
    url: ComfyUI_01882_.png
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
    url: ComfyUI_01542_.jpeg
- text: ((OpenDAlle!)text logo:1), ~*~aesthetic~*~
  output:
    url: ComfyUI_01528_.jpeg
---


# OpenDalleV1.1

<Gallery />

OpenDalle v1.1 on Hugging Face - It's Here!

Realism & Style: 
improved 

We're talking about a major glow-up in the realism and style department. Expect images that not only hit the bullseye with your prompts but also bring that extra zing of artistic flair. It's like your prompts went to art school!

Prompt Loyalty: Our Heartbeat

The soul of OpenDalle? Sticking to your prompts like glue. v1.1 takes your words and turns them into visual masterpieces that are just what you pictured – maybe even better.

Where We Stand: The Cool Middle Kid

Here's the scoop: OpenDalle v1.1 is proudly strutting a notch above SDXL. While DALLE-3 is still the big cheese, we're hot on its heels. Think of us as the cool, savvy middle sibling, rocking both brains and beauty.

## Settings for OpenDalle v1.1

Use these settings for the best results with OpenDalle v1.1:

CFG Scale: Use a CFG scale of 8 to 7

Steps: 60 to 70 steps for more detail, 35 steps for faster results.

Sampler: DPM2

Scheduler: Normal or Karras


## Use it with 🧨 diffusers
```python
from diffusers import AutoPipelineForText2Image
import torch
        
pipeline = AutoPipelineForText2Image.from_pretrained('dataautogpt3/OpenDalleV1.1', torch_dtype=torch.float16).to('cuda')        
image = pipeline('black fluffy gorgeous dangerous cat animal creature, large orange eyes, big fluffy ears, piercing gaze, full moon, dark ambiance, best quality, extremely detailed').images[0]
```

Non-Commercial Personal Use License Agreement

For dataautogpt3/OpenDalleV1.1

1. Introduction

This Non-Commercial Personal Use License Agreement ("Agreement") is between Alexander Izquierdo ("Licensor") and the individual or entity ("Licensee") using the Stable Diffusion model with unique merging method and tuning ("Model") hosted on the Hugging Face repository named OpenDalleV1.1.

2. Grant of License

a. Licensor hereby grants to Licensee a non-exclusive, non-transferable, non-sublicensable license to use the Model for personal, non-commercial purposes.

b. "Personal, non-commercial purposes" are defined as use that does not involve any form of compensation or monetary gain. This includes, but is not limited to, academic research, educational use, and hobbyist projects.

c. The Licensee is permitted to modify, merge, and use the Model for personal projects, provided that such use adheres to the terms of this Agreement.

3. Ownership and Intellectual Property Rights

a. The Licensor explicitly retains all rights, title, and interest in and to the unique merging method used in the Model. This merging method is the proprietary creation and intellectual property of the Licensor.

b. The Licensee shall not claim ownership, reverse engineer, or attempt to recreate the merging method for any purpose.

c. The Licensor retains all rights, title, and interest in and to the Model, including any modifications or improvements made by the Licensee.

d. The Licensee agrees to attribute the Licensor in any academic or public display of the Model or derivative works.

4. Restrictions

a. The Licensee shall not use the Model or the merging method for any commercial purposes.

b. The Licensee shall not distribute, sublicense, lease, or lend the Model or the merging method to any third party.

c. The Licensee shall not publicly display, perform, or communicate the Model, the merging method, or any derivative works thereof without the prior written consent of the Licensor.

5. Termination

This Agreement will terminate automatically if the Licensee breaches any of its terms and conditions.

6. Disclaimer of Warranties

The Model and the merging method are provided "as is," and the Licensor makes no warranties, express or implied, regarding their performance, reliability, or suitability for any purpose.

7. Limitation of Liability

The Licensor shall not be liable for any damages arising out of or related to the use or inability to use the Model or the merging method.

8. General Provisions

a. This Agreement constitutes the entire agreement between the parties and supersedes all prior agreements and understandings, whether written or oral, relating to its subject matter.

b. Any amendment to this Agreement must be in writing and signed by both parties.

c. This Agreement shall be governed by the laws of Maryland.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.