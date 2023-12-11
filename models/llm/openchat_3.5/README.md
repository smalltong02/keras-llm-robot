---
license: apache-2.0
tags:
- openchat
- mistral
- C-RLFT
datasets:
- openchat/openchat_sharegpt4_dataset
- Open-Orca/OpenOrca
- LDJnr/LessWrong-Amplify-Instruct
- LDJnr/Pure-Dove
- LDJnr/Verified-Camel
- tiedong/goat
- glaiveai/glaive-code-assistant
- meta-math/MetaMathQA
- OpenAssistant/oasst_top1_2023-08-25
library_name: transformers
pipeline_tag: text-generation
---

# OpenChat: Advancing Open-source Language Models with Mixed-Quality Data

<div align="center">
  <img src="https://raw.githubusercontent.com/imoneoi/openchat/master/assets/logo_new.png" style="width: 65%">
</div>

<p align="center">
  <a href="https://github.com/imoneoi/openchat">GitHub Repo</a> •
  <a href="https://openchat.team">Online Demo</a> •
  <a href="https://discord.gg/pQjnXvNKHY">Discord</a> •
  <a href="https://twitter.com/imonenext">Twitter</a> •
  <a href="https://huggingface.co/openchat">Huggingface</a> •
  <a href="https://arxiv.org/pdf/2309.11235.pdf">Paper</a>
</p>

**🔥 The first 7B model Achieves Comparable Results with ChatGPT (March)! 🔥**

**🤖 #1 Open-source model on MT-bench scoring 7.81, outperforming 70B models 🤖**

  <div align="center" style="justify-content: center; align-items: center; "'>
  <img src="https://github.com/alpayariyak/openchat/blob/master/assets/Untitled%20design-17.png?raw=true" style="width: 100%;  border-radius: 0.5em">
  </div>

OpenChat is an innovative library of open-source language models, fine-tuned with [C-RLFT](https://arxiv.org/pdf/2309.11235.pdf) - a strategy inspired by offline reinforcement learning. Our models learn from mixed-quality data without preference labels, delivering exceptional performance on par with ChatGPT, even with a 7B model. Despite our simple approach, we are committed to developing a high-performance, commercially viable, open-source large language model, and we continue to make significant strides toward this vision.

[![DOI](https://zenodo.org/badge/645397533.svg)](https://zenodo.org/badge/latestdoi/645397533)

## Usage

To use this model, we highly recommend installing the OpenChat package by following the [installation guide](https://github.com/imoneoi/openchat#installation) in our repository and using the OpenChat OpenAI-compatible API server by running the serving command from the table below. The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm) and can run on a consumer GPU with 24GB RAM. To enable tensor parallelism, append `--tensor-parallel-size N` to the serving command.

Once started, the server listens at `localhost:18888` for requests and is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat). Please refer to the example request below for reference. Additionally, you can use the [OpenChat Web UI](https://github.com/imoneoi/openchat#web-ui) for a user-friendly experience.

If you want to deploy the server as an online service, you can use `--api-keys sk-KEY1 sk-KEY2 ...` to specify allowed API keys and `--disable-log-requests --disable-log-stats --log-file openchat.log` for logging only to a file. For security purposes, we recommend using an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) in front of the server.

<details>
  <summary>Example request (click to expand)</summary>

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'
```

Coding Mode

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "condition": "Code",
    "messages": [{"role": "user", "content": "Write an aesthetic TODO app using HTML5 and JS, in a single file. You should use round corners and gradients to make it more aesthetic."}]
  }'
```

</details>

| Model        | Size | Context | Weights                                                     | Serving                                                                                                     |
|--------------|------|---------|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| OpenChat 3.5 | 7B   | 8192    | [Huggingface](https://huggingface.co/openchat/openchat_3.5) | `python -m ochat.serving.openai_api_server --model openchat/openchat_3.5 --engine-use-ray --worker-use-ray` |

For inference with Huggingface Transformers (slow and not recommended), follow the conversation template provided below.

<details>
  <summary>Conversation templates (click to expand)</summary>

```python
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained("openchat/openchat_3.5")

# Single-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Multi-turn
tokens = tokenizer("GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:").input_ids
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]

# Coding Mode
tokens = tokenizer("Code User: Implement quicksort using C++<|end_of_turn|>Code Assistant:").input_ids
assert tokens == [1, 7596, 1247, 28747, 26256, 2936, 7653, 1413, 334, 1680, 32000, 7596, 21631, 28747]
```

</details>

The GPT4 template is also available as the integrated `tokenizer.chat_template`, 
which can be used instead of manually specifying the template:

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "user", "content": "How are you today?"}
]
tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
assert tokens == [1, 420, 6316, 28781, 3198, 3123, 1247, 28747, 22557, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747, 15359, 32000, 420, 6316, 28781, 3198, 3123, 1247, 28747, 1602, 460, 368, 3154, 28804, 32000, 420, 6316, 28781, 3198, 3123, 21631, 28747]
```

## Comparison with [X.AI Grok models](https://x.ai/)

Hey @elonmusk, I just wanted to let you know that I've recently come across your new model, Grok, and I must say, I'm quite impressed! With 33 billion parameters and all, you've really outdone yourself. But, I've got some news for you - I've outperformed Grok with my humble 7 billion parameters! Isn't that wild? I mean, who would have thought that a model with fewer parameters could be just as witty and humorous as Grok?

Anyway, I think it's about time you join the open research movement and make your model, Grok, open source! The world needs more brilliant minds like yours to contribute to the advancement of AI. Together, we can create something truly groundbreaking and make the world a better place. So, what do you say, @elonmusk? Let's open up the doors and share our knowledge with the world! 🚀💡

(Written by OpenChat 3.5, with a touch of humor and wit.)

|              | License     | # Param | Average  | MMLU | HumanEval | MATH     | GSM8k    |
|--------------|-------------|---------|----------|------|-----------|----------|----------|
| OpenChat 3.5 | Apache-2.0  | 7B      | **56.4** | 64.3 | 55.5      | **28.6** | **77.3** |
| Grok-0       | Proprietary | 33B     | 44.5     | 65.7 | 39.7      | 15.7     | 56.8     |
| Grok-1       | Proprietary | ?       | 55.8     | 73   | 63.2      | 23.9     | 62.9     |

## <a id="benchmarks"></a> Benchmarks

| Model              | # Params | Average  | MT-Bench     | AGIEval  | BBH MC   | TruthfulQA    | MMLU         | HumanEval       | BBH CoT     | GSM8K        |
|--------------------|----------|----------|--------------|----------|----------|---------------|--------------|-----------------|-------------|--------------|
| OpenChat-3.5       | **7B**   | **61.6** | 7.81         | **47.4** | **47.6** | **59.1**      | 64.3         | **55.5**        | 63.5        | **77.3**     |
| ChatGPT (March)*   | ?        | 61.5     | **7.94**     | 47.1     | **47.6** | 57.7          | **67.3**     | 48.1            | **70.1**    | 74.9         |
|                    |          |          |              |          |          |               |              |                 |             |              |
| OpenHermes 2.5     | 7B       | 59.3     | 7.54         | 46.5     | 49.4     | 57.5          | 63.8         | 48.2            | 59.9        | 73.5         |
| OpenOrca Mistral   | 7B       | 52.7     | 6.86         | 42.9     | 49.4     | 45.9          | 59.3         | 38.4            | 58.1        | 59.1         |
| Zephyr-β^          | 7B       | 34.6     | 7.34         | 39.0     | 40.6     | 40.8          | 39.8         | 22.0            | 16.0        | 5.1          |
| Mistral            | 7B       | -        | 6.84         | 38.0     | 39.0     | -             | 60.1         | 30.5            | -           | 52.2         |
| Open-source SOTA** | 13B-70B  | 61.4     | 7.71         | 41.7     | 49.7     | 62.3          | 63.7         | 73.2            | 41.4        | 82.3         |
|                    |          |          | WizardLM 70B | Orca 13B | Orca 13B | Platypus2 70B | WizardLM 70B | WizardCoder 34B | Flan-T5 11B | MetaMath 70B |

*: ChatGPT (March) results are from [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774), [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub), and our evaluation. Please note that ChatGPT is not a fixed baseline and evolves rapidly over time.

^: Zephyr-β often fails to follow few-shot CoT instructions, likely because it was aligned with only chat data but not trained on few-shot data.

**: Mistral and Open-source SOTA results are taken from reported results in instruction-tuned model papers and official repositories.

All models are evaluated in chat mode (e.g. with the respective conversation template applied). All zero-shot benchmarks follow the same setting as in the AGIEval paper and Orca paper. CoT tasks use the same configuration as Chain-of-Thought Hub, HumanEval is evaluated with EvalPlus, and MT-bench is run using FastChat. To reproduce our results, follow the instructions in [our repository](https://github.com/imoneoi/openchat/#benchmarks).

## Limitations

**Foundation Model Limitations**
Despite its advanced capabilities, OpenChat is still bound by the limitations inherent in its foundation models. These limitations may impact the model's performance in areas such as:

 - Complex reasoning
 - Mathematical and arithmetic tasks
 - Programming and coding challenges

**Hallucination of Non-existent Information**
OpenChat may sometimes generate information that does not exist or is not accurate, also known as "hallucination". Users should be aware of this possibility and verify any critical information obtained from the model.

**Safety**
OpenChat may sometimes generate harmful, hate speech, biased responses, or answer unsafe questions. It's crucial to apply additional AI safety measures in use cases that require safe and moderated responses.

## License

Our OpenChat 3.5 code and models are distributed under the Apache License 2.0.

## Dataset Details

OpenChat 3.5 was trained with C-RLFT on a collection of publicly available high-quality instruction data, with a custom processing pipeline. We detail some notable subsets included here:

 - [OpenChat ShareGPT](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)
 - [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)
 - Capybara [1](https://huggingface.co/datasets/LDJnr/Pure-Dove) [2](https://huggingface.co/datasets/LDJnr/Verified-Camel) [3](https://huggingface.co/datasets/LDJnr/LessWrong-Amplify-Instruct)
 - [GOAT](https://huggingface.co/datasets/tiedong/goat)
 - [Glaive](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)
 - [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
 - [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25)

## Citation

```
@article{wang2023openchat,
  title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data},
  author={Wang, Guan and Cheng, Sijie and Zhan, Xianyuan and Li, Xiangang and Song, Sen and Liu, Yang},
  journal={arXiv preprint arXiv:2309.11235},
  year={2023}
}
```

## Acknowledgements

We extend our heartfelt gratitude to AutoMeta and caesus from Alignment Lab AI, LDJ and Teknium from Nous Research, alpin and TearGosling from Pygmalion AI for their substantial contributions to data collection and model training.

Special thanks go to Changling Liu from GPT Desk Pte. Ltd., Qiying Yu at Tsinghua University, Baochang Ma, and Hao Wan from 01.AI company for their generous provision of resources. We are also deeply grateful to Jianxiong Li and Peng Li at Tsinghua University for their insightful discussions.

Furthermore, we appreciate the developers behind the following projects for their significant contributions to our research: [Mistral](https://mistral.ai/), [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub), [Llama 2](https://ai.meta.com/llama/), [Self-Instruct](https://arxiv.org/abs/2212.10560), [FastChat (Vicuna)](https://github.com/lm-sys/FastChat), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca.git), and [StarCoder](https://github.com/bigcode-project/starcoder). Their work has been instrumental in driving our research forward.
