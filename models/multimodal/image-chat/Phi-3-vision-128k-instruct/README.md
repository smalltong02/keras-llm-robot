---
license: mit
license_link: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/resolve/main/LICENSE

language:
- multilingual
pipeline_tag: text-generation
tags:
- nlp
- code
- vision
inference:
  parameters:
    temperature: 0.7
widget:
  - messages:
      - role: user
        content: <|image_1|>Can you describe what you see in the image?
---
## Model Summary

The Phi-3-Vision-128K-Instruct is a lightweight, state-of-the-art open multimodal model built upon datasets which include - synthetic data and filtered publicly available websites - with a focus on very high-quality, reasoning dense data both on text and vision.  The model belongs to the Phi-3 model family, and the multimodal version comes with 128K context length (in tokens) it can support. The model underwent a rigorous enhancement process, incorporating both supervised fine-tuning and direct preference optimization to ensure precise instruction adherence and robust safety measures.

Resources and Technical Documentation:

+ [Phi-3 Microsoft Blog](https://aka.ms/Phi-3Build2024)
+ [Phi-3 Technical Report](https://aka.ms/phi3-tech-report)
+ [Phi-3 on Azure AI Studio](https://aka.ms/try-phi3vision)
+ [Phi-3 Cookbook](https://github.com/microsoft/Phi-3CookBook)


|         | Short Context | Long Context |
| ------- | ------------- | ------------ |
| Mini    | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) ; [[GGUF]](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx)|
| Small   | 8K [[HF]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-8k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-small-128k-instruct-onnx-cuda)|
| Medium  | 4K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct-onnx-cuda) | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct) ; [[ONNX]](https://huggingface.co/microsoft/Phi-3-medium-128k-instruct-onnx-cuda)|
| Vision  |  | 128K [[HF]](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)|

## Intended Uses

**Primary use cases**

The model is intended for broad commercial and research use in English. The model provides uses for general purpose AI systems and applications with visual and text input capabilities which require 

1) memory/compute constrained environments;
2) latency bound scenarios;
3) general image understanding;
4) OCR;
5) chart and table understanding.

Our model is designed to accelerate research on efficient language and multimodal models, for use as a building block for generative AI powered features.

**Use case considerations**

Our models are not specifically designed or evaluated for all downstream purposes. Developers should consider common limitations of language models as they select use cases, and evaluate and mitigate for accuracy, safety, and fairness before using within a specific downstream use case, particularly for high-risk scenarios. 
Developers should be aware of and adhere to applicable laws or regulations (including privacy, trade compliance laws, etc.) that are relevant to their use case. 

Nothing contained in this Model Card should be interpreted as or deemed a restriction or modification to the license the model is released under.

## How to Use

Phi-3-Vision-128K-Instruct has been integrated in the development version (4.40.2) of `transformers`. Until the official version is released through `pip`, ensure that you are doing one of the following:
* When loading the model, ensure that `trust_remote_code=True` is passed as an argument of the `from_pretrained()` function.

* Update your local `transformers` to the development version: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`. The previous command is an alternative to cloning and installing from the source.

The current `transformers` version can be verified with: `pip list | grep transformers`.

Examples of required packages:
```
flash_attn==2.5.8
numpy==1.24.4
Pillow==10.3.0
Requests==2.31.0
torch==2.3.0
torchvision==0.18.0
transformers==4.40.2
```

Phi-3-Vision-128K-Instruct is also available in [Azure AI Studio](https://aka.ms/phi3-azure-ai).

### Chat Format

Given the nature of the training data, the Phi-3-Vision-128K-Instruct model is best suited for a single image input wih prompts using the chat format as follows. 
You can provide the prompt as a single image with a generic template as follow:
```markdown
<|user|>\n<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n 
```

where the model generates the text after `<|assistant|>` . In case of multi-turn conversation, the prompt can be formatted as follows:

```markdown
<|user|>\n<|image_1|>\n{prompt_1}<|end|>\n<|assistant|>\n{response_1}<|end|>\n<|user|>\n{prompt_2}<|end|>\n<|assistant|>\n 
```

### Sample inference code

This code snippets show how to get quickly started with running the model on a GPU:

```python
from PIL import Image 
import requests 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 

model_id = "microsoft/Phi-3-vision-128k-instruct" 

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

messages = [ 
    {"role": "user", "content": "<|image_1|>\nWhat is shown in this image?"}, 
    {"role": "assistant", "content": "The chart displays the percentage of respondents who agree with various statements about their preparedness for meetings. It shows five categories: 'Having clear and pre-defined goals for meetings', 'Knowing where to find the information I need for a meeting', 'Understanding my exact role and responsibilities when I'm invited', 'Having tools to manage admin tasks like note-taking or summarization', and 'Having more focus time to sufficiently prepare for meetings'. Each category has an associated bar indicating the level of agreement, measured on a scale from 0% to 100%."}, 
    {"role": "user", "content": "Provide insightful questions to spark discussion."} 
] 

url = "https://assets-c4akfrf5b4d3f4b7.z01.azurefd.net/assets/2024/04/BMDataViz_661fb89f3845e.png" 
image = Image.open(requests.get(url, stream=True).raw) 

prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0") 

generation_args = { 
    "max_new_tokens": 500, 
    "temperature": 0.0, 
    "do_sample": False, 
} 

generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 

# remove input tokens 
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

print(response) 
```

Additional basic examples are provided [here](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/blob/main/sample_inference.py).


## Responsible AI Considerations

Like other models, the Phi family of models can potentially behave in ways that are unfair, unreliable, or offensive. Some of the limiting behaviors to be aware of include:   

+ Quality of Service: The Phi models are trained primarily on English text. Languages other than English will experience worse performance. English language varieties with less representation in the training data might experience worse performance than standard American English.    
+ Representation of Harms & Perpetuation of Stereotypes: These models can over- or under-represent groups of people, erase representation of some groups, or reinforce demeaning or negative stereotypes. Despite safety post-training, these limitations may still be present due to differing levels of representation of different groups or prevalence of examples of negative stereotypes in training data that reflect real-world patterns and societal biases.  
+ Inappropriate or Offensive Content: These models may produce other types of inappropriate or offensive content, which may make it inappropriate to deploy for sensitive contexts without additional mitigations that are specific to the use case.  
+ Information Reliability: Language models can generate nonsensical content or fabricate content that might sound reasonable but is inaccurate or outdated.   
+ Limited Scope for Code: Majority of Phi-3 training data is based in Python and use common packages such as "typing, math, random, collections, datetime, itertools". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.      

Developers should apply responsible AI best practices and are responsible for ensuring that a specific use case complies with relevant laws and regulations (e.g. privacy, trade, etc.). Important areas for consideration include:  

+ Allocation: Models may not be suitable for scenarios that could have consequential impact on legal status or the allocation of resources or life opportunities (ex: housing, employment, credit, etc.) without further assessments and additional debiasing techniques.
+ High-Risk Scenarios: Developers should assess suitability of using models in high-risk scenarios where unfair, unreliable or offensive outputs might be extremely costly or lead to harm. This includes providing advice in sensitive or expert domains where accuracy and reliability are critical (ex: legal or health advice). Additional safeguards should be implemented at the application level according to the deployment context.
+ Misinformation: Models may produce inaccurate information. Developers should follow transparency best practices and inform end-users they are interacting with an AI system. At the application level, developers can build feedback mechanisms and pipelines to ground responses in use-case specific, contextual information, a technique known as Retrieval Augmented Generation (RAG).
+ Generation of Harmful Content: Developers should assess outputs for their context and use available safety classifiers or custom solutions appropriate for their use case.
+ Misuse: Other forms of misuse such as fraud, spam, or malware production may be possible, and developers should ensure that their applications do not violate applicable laws and regulations.
+ Identification of individuals: models with vision capabilities may have the potential to uniquely identify individuals in images. Safety post-training steers the model to refuse such requests, but developers should consider and implement, as appropriate, additional mitigations or user consent flows as required in their respective jurisdiction, (e.g., building measures to blur faces in image inputs before processing.
  
## Training

### Model

* Architecture: Phi-3-Vision-128K-Instruct has 4.2B parameters and contains image encoder, connector, projector, and Phi-3 Mini language model.
* Inputs: Text and Image. It’s best suited for prompts using the chat format. 
* Context length: 128K tokens
* GPUs: 512 H100-80G
* Training time: 1.5 days
* Training data: 500B vision and text tokens
* Outputs: Generated text in response to the input
* Dates: Our models were trained between February and April 2024
* Status: This is a static model trained on an offline text dataset with cutoff date Mar 15, 2024. Future versions of the tuned models may be released as we improve models.
* Release Type: Open weight release
* Release dates: The model weight is released on May 21, 2024.

### Datasets

Our training data includes a wide variety of sources, and is a combination of 

1) publicly available documents filtered rigorously for quality, selected high-quality educational data and code;
2) selected high-quality image-text interleave;
3) newly created synthetic, “textbook-like” data for the purpose of teaching math, coding, common sense reasoning, general knowledge of the world (science, daily activities, theory of mind, etc.), newly created image data, e.g., chart/table/diagram/slides;
4) high quality chat format supervised data covering various topics to reflect human preferences on different aspects such as instruct-following, truthfulness, honesty and helpfulness.

The data collection process involved sourcing information from publicly available documents, with a meticulous approach to filtering out undesirable documents and images. To safeguard privacy, we carefully filtered various image and text data sources to remove or scrub any potentially personal data from the training data.
 
More details can be found in the [Phi-3 Technical Report](https://aka.ms/phi3-tech-report).

## Benchmarks

To understand the capabilities, we compare Phi-3-Vision-128K-Instruct with a set of models over a variety of zero-shot benchmarks using our internal benchmark platform.

|Benchmark|Phi-3 Vision-128K-In|LlaVA-1.6 Vicuna-7B|QWEN-VL Chat|Llama3-Llava-Next-8B|Claude-3 Haiku|Gemini 1.0 Pro V|GPT-4V-Turbo|
|---------|---------------------|------------------|------------|--------------------|--------------|----------------|------------|
|MMMU|40.4|34.2|39.0|36.4|40.7|42.0|55.5| 
|MMBench|80.5|76.3|75.8|79.4|62.4|80.0|86.1|
|ScienceQA|90.8|70.6|67.2|73.7|72.0|79.7|75.7|
|MathVista|44.5|31.5|29.4|34.8|33.2|35.0|47.5|
|InterGPS|38.1|20.5|22.3|24.6|32.1|28.6|41.0|
|AI2D|76.7|63.1|59.8|66.9|60.3|62.8|74.7|
|ChartQA|81.4|55.0|50.9|65.8|59.3|58.0|62.3|
|TextVQA|70.9|64.6|59.4|55.7|62.7|64.7|68.1|
|POPE|85.8|87.2|82.6|87.0|74.4|84.2|83.7|


## Software

* [PyTorch](https://github.com/pytorch/pytorch)
* [Transformers](https://github.com/huggingface/transformers)
* [Flash-Attention](https://github.com/HazyResearch/flash-attention)

## Hardware
Note that by default, the Phi-3-Vision-128K model uses flash attention, which requires certain types of GPU hardware to run. We have tested on the following GPU types:
* NVIDIA A100
* NVIDIA A6000
* NVIDIA H100

### Running on Windows or without flash attention
To enable the model on these enviroment here are steps that you may consider to follow:

Step 1: comment flash attention import code in modeling_phi3_v.py from line 52 to line 56.
```python
# if is_flash_attn_2_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

#     _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
```

Step 2: change _"_attn_implementation"_ from _"flash_attention_2"_ to _"eager"_ in config.json or disable flash attention when you create the model as below.

```python
model = AutoModelForCausalLM.from_pretrained('microsoft/Phi-3-vision-128k-instruct', device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation="eager")
```

## License

The model is licensed under the [MIT license](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/resolve/main/LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
