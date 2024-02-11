---
license: other
datasets:
- tiiuae/falcon-refinedweb
- bigcode/the-stack-github-issues
- bigcode/commitpackft
- bigcode/starcoderdata
- EleutherAI/proof-pile-2
- meta-math/MetaMathQA
language:
- en
tags:
- causal-lm
- code
metrics:
- code_eval
library_name: transformers
model-index:
- name: StarCoderBase-3B
  results:
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Python)
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.4
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (C++)
    metrics:
    - name: pass@1
      type: pass@1
      value: 30.9
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Java)
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.1
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (JavaScript)
    metrics:
    - name: pass@1
      type: pass@1
      value: 32.1
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (PHP)
    metrics:
    - name: pass@1
      type: pass@1
      value: 24.2
      verified: false
  - task:
      type: text-generation
    dataset:
      type: nuprl/MultiPL-E
      name: MultiPL-HumanEval (Rust)
    metrics:
    - name: pass@1
      type: pass@1
      value: 23.0
      verified: false
---
# `stable-code-3b`

## Model Description

`stable-code-3b` is a 2.7B billion parameter decoder-only language model pre-trained on 1.3 trillion tokens of diverse textual and code datasets. `stable-code-3b` is trained on 18 programming languages (selected based on the 2023 StackOverflow Developer Survey) and demonstrates state-of-the-art performance (compared to models of similar size) on the MultiPL-E metrics across multiple programming languages tested using [BigCode's Evaluation Harness](https://github.com/bigcode-project/bigcode-evaluation-harness/tree/main).

![spiderchart](stable_code_3b_spiderchart.svg)

| Model            | Size | Python | C++  | Javascript | Java | PHP  | Rust |
|------------------|------|--------|------|------------|------|------|------|
| **Stable Code**  | 3B   | 32.4%  | 30.9%| 32.1%      | 32.1%| 24.2%| 23.0%|
| CodeLLama        | 7B   | 30.0%  | 28.2%| 32.5%      | 31.1%| 25.7%| 26.3%|
| Deepseek Coder   | 1.3B | 28.6%  | 29.2%| 28.7%      | 29.0%| 23.6%| 18.5%|
| Wizard Coder     | 3B   | 31.6%  | 25.6%| 26.2%      | 25.8%| 25.3%| 20.4%|
| StarCoder        | 3B   | 21.6%  | 19.8%| 21.5%      | 20.5%| 19.0%| 16.9%|
| Replit Code V1.5 | 3B   | 23.0%  | 25.9%| 26.2%      | 23.6%| 23.2%| 21.5%|
| Deci Coder       | 1B   | 19.1%  | 6.8% | 18.4%      | 16.7%| 2.1% | 1.7% |

**Key Features**
* Fill in Middle Capability (FIM)
* Supports Long Context, trained with Sequences upto 16,384

## Usage

Get started generating text with `stable-code-3b` by using the following code snippet:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stable-code-3b",
  trust_remote_code=True,
  torch_dtype="auto",
)
model.cuda()
inputs = tokenizer("import torch\nimport torch.nn as nn", return_tensors="pt").to(model.device)
tokens = model.generate(
  **inputs,
  max_new_tokens=48,
  temperature=0.2,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

### Run with Fill in Middle (FIM) ⚡️

<details>
<summary> Click to expand </summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stable-code-3b",
  trust_remote_code=True,
  torch_dtype="auto",
+ attn_implementation="flash_attention_2",
)
model.cuda()
inputs = tokenizer("<fim_prefix>def fib(n):<fim_suffix>    else:\n        return fib(n - 2) + fib(n - 1)<fim_middle>", return_tensors="pt").to(model.device)
tokens = model.generate(
  **inputs,
  max_new_tokens=48,
  temperature=0.2,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

</details>

### Run with Flash Attention 2 ⚡️

<details>
<summary> Click to expand </summary>

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-code-3b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stable-code-3b",
  trust_remote_code=True,
  torch_dtype="auto",
+ attn_implementation="flash_attention_2",
)
model.cuda()
inputs = tokenizer("import torch\nimport torch.nn as nn", return_tensors="pt").to(model.device)
tokens = model.generate(
  **inputs,
  max_new_tokens=48,
  temperature=0.2,
  do_sample=True,
)
print(tokenizer.decode(tokens[0], skip_special_tokens=True))
```

</details>


## Model Details

* **Developed by**: [Stability AI](https://stability.ai/)
* **Model type**: `stable-code-3b` models are auto-regressive language models based on the transformer decoder architecture.
* **Language(s)**: English, Code
* **Library**: [GPT-NeoX](https://github.com/EleutherAI/gpt-neox)
* **License**: License: StabilityAI Non-Commercial Research Community License. If you want to use this model for your commercial products or purposes, please contact us [here](https://stability.ai/membership) to learn more.
* **Contact**: For questions and comments about the model, please email `lm@stability.ai`

### Model Architecture

The model is a decoder-only transformer similar to the LLaMA ([Touvron et al., 2023](https://arxiv.org/abs/2307.09288)) architecture with the following modifications:

| Parameters     | Hidden Size | Layers | Heads | Sequence Length |
|----------------|-------------|--------|-------|-----------------|
| 2,796,431,360  | 2560        | 32     | 32    | 16384            |

* **Position Embeddings**: Rotary Position Embeddings ([Su et al., 2021](https://arxiv.org/abs/2104.09864)) applied to the first 25% of head embedding dimensions for improved throughput following [Black et al. (2022)](https://arxiv.org/pdf/2204.06745.pdf).
* **Tokenizer**: We use a modified version of the GPTNeoX Tokenizer.[`NeoX`](https://github.com/EleutherAI/gpt-neox). We add special tokens to train for Fill in the Middle (FIM) capabilities like `<FIM_PREFIX>` and `<FIM_SUFFIX>` along with other special tokens.

## Training

### Training Dataset

The dataset is comprised of a filtered mixture of open-source large-scale datasets available on the [HuggingFace Hub](https://huggingface.co/datasets): Falcon RefinedWeb extract ([Penedo et al., 2023](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)), along with [CommitPackFT](https://huggingface.co/datasets/bigcode/commitpackft) and [Github Issues](https://huggingface.co/datasets/bigcode/the-stack-github-issues) (BigCode., 2023), and StarCoder ([Li et al., 2023](https://arxiv.org/abs/2305.06161)). We further supplement our training with data from mathematical domains ([Azerbayev, Zhangir, et al., 2023](https://arxiv.org/abs/2310.10631) and, [Yu, Longhui, et al., 2023](https://arxiv.org/abs/2309.12284)). 

Top 18 programming languages trained on:
- C
- CPP
- Java
- JavaScript
- CSS
- Go
- HTML
- Ruby
- Rust
- Markdown
- Shell
- Php
- Sql
- R
- Typescript
- Python
- Jupyter-Clean
- RestructuredText

### Training Procedure

The model is pre-trained on the aforementioned datasets in `bfloat16` precision, optimized with AdamW.

### Training Infrastructure

* **Hardware**: `stable-code-3b` was trained on the Stability AI cluster across 256 NVIDIA A100 40GB GPUs (AWS P4d instances).

* **Software**: We use a fork of `gpt-neox` ([EleutherAI, 2021](https://github.com/EleutherAI/gpt-neox)), train under 2D parallelism (Data and Tensor Parallel) with ZeRO-1 ([Rajbhandari et al., 2019](https://arxiv.org/abs/1910.02054v3)), and rely on flash-attention as well as SwiGLU and Rotary Embedding kernels from FlashAttention-2 ([Dao et al., 2023](https://tridao.me/publications/flash2/flash2.pdf))

## Use and Limitations

### Intended Use

The model is intended to be used as a foundational base model for application-specific fine-tuning. Developers must evaluate and fine-tune the model for safe performance in downstream applications.

### Limitations and Bias
​
As a base model, this model may exhibit unreliable, unsafe, or other undesirable behaviors that must be corrected through evaluation and fine-tuning prior to deployment. The pre-training dataset may have contained offensive or inappropriate content, even after applying data cleansing filters, which can be reflected in the model-generated text. We recommend that users exercise caution when using these models in production systems. Do not use the models if they are unsuitable for your application, or for any applications that may cause deliberate or unintentional harm to others.

## How to Cite

```bibtex
@misc{stable-code-3b,
      url={[https://huggingface.co/stabilityai/stable-code-3b](https://huggingface.co/stabilityai/stable-code-3b)},
      title={Stable Code 3B},
      author={Pinnaparaju, Nikhil and Adithyan, Reshinth and Phung, Duy and Tow, Jonathan and Baicoianu, James and  and Cooper, Nathan}
}
```