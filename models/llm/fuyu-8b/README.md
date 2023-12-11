---
license: cc-by-nc-4.0
---
# Fuyu-8B Model Card

We’re releasing Fuyu-8B, a small version of the multimodal model that powers our product. The model is available on HuggingFace. We think Fuyu-8B is exciting because:
1. It has a much simpler architecture and training procedure than other multi-modal models, which makes it easier to understand, scale, and deploy.
2. It’s designed from the ground up for digital agents, so it can support arbitrary image resolutions, answer questions about graphs and diagrams, answer UI-based questions, and do fine-grained localization on screen images.
3. It’s fast - we can get responses for large images in less than 100 milliseconds.
4. Despite being optimized for our use-case, it performs well at standard image understanding benchmarks such as visual question-answering and natural-image-captioning.

Please note that **the model we have released is a base model. We expect you to need to finetune the model for specific use cases like verbose captioning or multimodal chat.** In our experience, the model responds well to few-shotting and fine-tuning for a variety of use-cases. 

## Model

[Fuyu-8B](https://www.adept.ai/blog/fuyu-8b) is a multi-modal text and image transformer trained by [Adept AI](https://www.adept.ai/).

Architecturally, Fuyu is a vanilla decoder-only transformer - there is no image encoder. 
Image patches are instead linearly projected into the first layer of the transformer, bypassing the embedding lookup. 
We simply treat the transformer decoder like an image transformer (albeit with no pooling and causal attention).
See the below diagram for more details.

![architecture](architecture.png)

This simplification allows us to support arbitrary image resolutions. 
To accomplish this, we treat the sequence of image tokens like the sequence of text tokens. 
We remove image-specific position embeddings and feed in as many image tokens as necessary in raster-scan order. 
To tell the model when a line has broken, we simply use a special image-newline character. 
The model can use its existing position embeddings to reason about different image sizes, and we can use images of arbitrary size at training time, removing the need for separate high and low-resolution training stages.

### Model Description

- **Developed by:** Adept-AI
- **Model type:** Decoder-only multi-modal transformer model 
- **License:** [CC-BY-NC](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
- **Model Description:** This is a multi-modal model that can consume images and text and produce text. 
- **Resources for more information:** Check out our [blog post](https://www.adept.ai/blog/fuyu-8b).

## Evaluation
Though not the focus of this model, we did evaluate it on standard image understanding benchmarks:

| Eval Task           | Fuyu-8B | Fuyu-Medium       | LLaVA 1.5 (13.5B) | QWEN-VL (10B) | PALI-X (55B) | PALM-e-12B | PALM-e-562B |
| ------------------- | ------- | ----------------- | ----------------- | ------------- | ------------ | ---------- | ----------- |
| VQAv2               | 74.2    |     77.4          | 80                | 79.5          | 86.1         | 76.2       | 80.0        |
| OKVQA               | 60.6    |     63.1          | n/a               | 58.6          | 66.1         | 55.5       | 66.1        |
| COCO Captions       | 141     |     138           | n/a               | n/a           | 149          | 135        | 138         |
| AI2D                | 64.5    |     73.7          | n/a               | 62.3          | 81.2         | n/a        | n/a         |

## How to Use

You can load the model and perform inference as follows:
```python
from transformers import FuyuProcessor, FuyuForCausalLM
from PIL import Image
import requests

# load model and processor
model_id = "adept/fuyu-8b"
processor = FuyuProcessor.from_pretrained(model_id)
model = FuyuForCausalLM.from_pretrained(model_id, device_map="cuda:0")

# prepare inputs for the model
text_prompt = "Generate a coco-style caption.\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

# autoregressively generate text
generation_output = model.generate(**inputs, max_new_tokens=7)
generation_text = processor.batch_decode(generation_output[:, -7:], skip_special_tokens=True)
assert generation_text == ['A blue bus parked on the side of a road.']
```

N.B.: The token `|SPEAKER|` is a placeholder token for image patch embeddings, so it will show up in the model context (e.g., in the portion of `generation_output` representing the model context).
`|NEWLINE|` is the "image newline" token, denoting new rows in the raster scan order input of the image patches.
`\x04` is the "beginning of answer" token.

Fuyu can also perform some question answering on natural images and charts/diagrams (thought fine-tuning may be required for good performance):
```python
text_prompt = "What color is the bus?\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/bus.png"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

generation_output = model.generate(**inputs, max_new_tokens=6)
generation_text = processor.batch_decode(generation_output[:, -6:], skip_special_tokens=True)
assert generation_text == ["The bus is blue.\n"]


text_prompt = "What is the highest life expectancy at birth of male?\n"
url = "https://huggingface.co/adept/fuyu-8b/resolve/main/chart.png"
image = Image.open(requests.get(url, stream=True).raw)

model_inputs = processor(text=text_prompt, images=image, return_tensors="pt").to("cuda:0")

generation_output = model.generate(**model_inputs, max_new_tokens=16)
generation_text = processor.batch_decode(generation_output[:, -16:], skip_special_tokens=True)
assert generation_text == ["The life expectancy at birth of males in 2018 is 80.7.\n"]
```
For best performance, it's recommended to end questions with `\n`, as shown above!

## Uses

### Direct Use

The model is intended for research purposes only. 
**Because this is a raw model release, we have not added further finetuning, postprocessing or sampling strategies to control for undesirable outputs. You should expect to have to fine-tune the model for your use-case.**

Possible research areas and tasks include

- Applications in computer control or digital agents.
- Research on multi-modal models generally.

Excluded uses are described below.

### Out-of-Scope Use

The model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model.

## Limitations and Bias

### Limitations

- Faces and people in general may not be generated properly.

### Bias
While the capabilities of these models are impressive, they can also reinforce or exacerbate social biases.