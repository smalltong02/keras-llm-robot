---
license: apache-2.0
language:
- en
---
# CogVLM

**CogVLM** 是一个强大的开源视觉语言模型（VLM）。CogVLM-17B 拥有 100 亿视觉参数和 70 亿语言参数，在 10 个经典跨模态基准测试上取得了 SOTA 性能，包括 NoCaps、Flicker30k captioning、RefCOCO、RefCOCO+、RefCOCOg、Visual7W、GQA、ScienceQA、VizWiz VQA 和 TDIUC，而在 VQAv2、OKVQA、TextVQA、COCO captioning 等方面则排名第二，超越或与 PaLI-X 55B 持平。您可以通过线上 [demo](http://36.103.203.44:7861/) 体验 CogVLM 多模态对话。

**CogVLM** is a powerful **open-source visual language model** (**VLM**). CogVLM-17B has 10 billion vision parameters and 7 billion language parameters. CogVLM-17B achieves state-of-the-art performance on 10 classic cross-modal benchmarks, including NoCaps, Flicker30k captioning, RefCOCO, RefCOCO+, RefCOCOg, Visual7W, GQA, ScienceQA, VizWiz VQA and TDIUC, and rank the 2nd on VQAv2, OKVQA, TextVQA, COCO captioning, etc., **surpassing or matching PaLI-X 55B**. CogVLM can also [chat with you](http://36.103.203.44:7861/) about images.

<div align="center">
    <img src="https://github.com/THUDM/CogVLM/raw/main/assets/metrics-min.png" alt="img" style="zoom: 50%;" />
</div>

# 快速开始（Qiuckstart）

硬件需求（hardware requirement）

需要近 40GB GPU 显存用于模型推理。如果没有一整块GPU显存超过40GB，则需要使用accelerate的将模型切分到多个有较小显存的GPU设备上。

40GB VRAM for inference. If there is no single GPU with more than 40GB of VRAM, you will need to use the "accelerate" library to dispatch the model into multiple GPUs with smaller VRAM. 

安装依赖（dependencies）

```base
pip install torch==2.1.0 transformers==4.35.0 accelerate==0.24.1 sentencepiece==0.1.99 einops==0.7.0 xformers==0.0.22.post7 triton==2.1.0
```

代码示例（example）

```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()


# chat example
query = 'Describe this image'
image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))

# This image captures a moment from a basketball game. Two players are prominently featured: one wearing a yellow jersey with the number
# 24 and the word 'Lakers' written on it, and the other wearing a navy blue jersey with the word 'Washington' and the number 34. The player
# in yellow is holding a basketball and appears to be dribbling it, while the player in navy blue is reaching out with his arm, possibly
# trying to block or defend. The background shows a filled stadium with spectators, indicating that this is a professional game.</s>



# vqa example
query = 'How many houses are there in this cartoon?'
image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/3.jpg?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image], template_version='vqa')   # vqa mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))

# 4</s>
```

当单卡显存不足时，可以将模型切分到多个小显存GPU上。以下是个当你有两张24GB的GPU，16GBCPU内存的例子。
你可以将`infer_auto_device_map`的参数改成你的配置。注意这里将GPU显存少写了一点，这是为推理时中间状态预留出一部分显存。

dispatch the model into multiple GPUs with smaller VRAM. This is an example for you have two 24GB GPU and 16GB CPU memory.
you can change the arguments of `infer_auto_device_map` with your own setting.

```python
import torch
import requests
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
device_map = infer_auto_device_map(model, max_memory={0:'20GiB',1:'20GiB','cpu':'16GiB'}, no_split_module_classes=['CogVLMDecoderLayer', 'TransformerLayer'])
model = load_checkpoint_and_dispatch(
    model,
    'local/path/to/hf/version/chat/model',   # typical, '~/.cache/huggingface/hub/models--THUDM--cogvlm-chat-hf/snapshots/balabala'
    device_map=device_map,
)
model = model.eval()

# check device for weights if u want to
for n, p in model.named_parameters():
    print(f"{n}: {p.device}")

# chat example
query = 'Describe this image'
image = Image.open(requests.get('https://github.com/THUDM/CogVLM/blob/main/examples/1.png?raw=true', stream=True).raw).convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 2048, "do_sample": False}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))
```



# 方法（Method）

CogVLM 模型包括四个基本组件：视觉变换器（ViT）编码器、MLP适配器、预训练的大型语言模型（GPT）和一个**视觉专家模块**。更多细节请参见[Paper](https://github.com/THUDM/CogVLM/blob/main/assets/cogvlm-paper.pdf)。

CogVLM model comprises four fundamental components: a vision transformer (ViT) encoder, an MLP adapter, a pretrained large language model (GPT), and a **visual expert module**. See [Paper](https://github.com/THUDM/CogVLM/blob/main/assets/cogvlm-paper.pdf) for more details.

<div align="center">
    <img src="https://github.com/THUDM/CogVLM/raw/main/assets/method-min.png" style="zoom:50%;" />
</div>

# 许可（License）

此存储库中的代码是根据 [Apache-2.0 许可](https://github.com/THUDM/CogVLM/raw/main/LICENSE) 开放源码，而使用 CogVLM 模型权重必须遵循 [模型许可](https://github.com/THUDM/CogVLM/raw/main/MODEL_LICENSE)。

The code in this repository is open source under the [Apache-2.0 license](https://github.com/THUDM/CogVLM/raw/main/LICENSE), while the use of the CogVLM model weights must comply with the [Model License](https://github.com/THUDM/CogVLM/raw/main/MODEL_LICENSE).



# 引用（Citation）

If you find our work helpful, please consider citing the following papers
```
@article{wang2023cogvlm,
      title={CogVLM: Visual Expert for Pretrained Language Models}, 
      author={Weihan Wang and Qingsong Lv and Wenmeng Yu and Wenyi Hong and Ji Qi and Yan Wang and Junhui Ji and Zhuoyi Yang and Lei Zhao and Xixuan Song and Jiazheng Xu and Bin Xu and Juanzi Li and Yuxiao Dong and Ming Ding and Jie Tang},
      year={2023},
      eprint={2311.03079},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```