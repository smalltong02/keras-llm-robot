---
license: apache-2.0
base_model: mistralai/Mistral-7B-v0.1
tags:
- openchat
- mistral
- C-RLFT
datasets:
- openchat/openchat_sharegpt4_dataset
- kaist-ai/Feedback-Collection
- imone/OpenOrca_FLAN
- LDJnr/Capybara
- tiedong/goat
- glaiveai/glaive-code-assistant
- meta-math/MetaMathQA
- OpenAssistant/oasst_top1_2023-08-25
- TIGER-Lab/MathInstruct
library_name: transformers
pipeline_tag: text-generation
---
<div align="center">
  <img src="https://raw.githubusercontent.com/imoneoi/openchat/master/assets/logo_new.png" style="width: 65%">
  <h1>Advancing Open-source Language Models with Mixed-Quality Data</h1>
</div>

<p align="center" style="margin-top: 0px;">
  <a href="https://openchat.team">
    <img src="https://github.com/alpayariyak/openchat/blob/master/assets/logo_nobg.png?raw=true" alt="OpenChat Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 10px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text" style=" margin-right: 5px;">Online Demo</span>
  </a> |
  <a href="https://github.com/imoneoi/openchat">
    <img src="https://camo.githubusercontent.com/4133dc1cd4511d4a292b84ce10e52e4ed92569fb2a8165381c9c47be5edc2796/68747470733a2f2f6564656e742e6769746875622e696f2f537570657254696e7949636f6e732f696d616765732f706e672f6769746875622e706e67" alt="GitHub Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text" style=" margin-right: 5px;">GitHub</span>
  </a> |
  <a href="https://arxiv.org/pdf/2309.11235.pdf">
    <img src="https://github.com/alpayariyak/openchat/blob/master/assets/arxiv-logomark-small-square-border.png?raw=true" alt="ArXiv Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text" style="margin-right: 5px;">Paper</span>
  </a> |
  <a href="https://discord.gg/pQjnXvNKHY">
    <img src="https://cloud.githubusercontent.com/assets/6291467/26705903/96c2d66e-477c-11e7-9f4e-f3c0efe96c9a.png" alt="Discord Logo" style="width:20px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
    <span class="link-text">Discord</span>
  </a>
</p>

<p align="center" style="margin-top: 0px;">
    <span class="link-text" style=" margin-right: 0px; font-size: 0.8em">Sponsored by RunPod</span>
   <img src="https://styles.redditmedia.com/t5_6075m3/styles/profileIcon_71syco7c5lt81.png?width=256&height=256&frame=1&auto=webp&crop=256:256,smart&s=24bd3c71dc11edc5d4f88d0cbc1da72ed7ae1969" alt="RunPod Logo" style="width:30px; vertical-align: middle; display: inline-block; margin-right: 5px; margin-left: 5px; margin-top: 0px; margin-bottom: 0px;"/>
</p>

<div style="background-color: white; padding: 0.7em; border-radius: 0.5em; color: black; display: flex; flex-direction: column; justify-content: center; text-align: center; ont-size: 0.5em; border: 0.8em solid #3c72db;">
  <a href="https://huggingface.co/openchat/openchat_3.5" style="text-decoration: none; color: black;">
    <span style="font-size: 1.7em; font-family: 'Helvetica'; letter-spacing: 0.1em; font-weight: bold; color: black;">OPENCHAT</span><span style="font-size: 1.8em; font-family: 'Helvetica'; color: #3c72db; ">3.5</span>
        <span style="font-size: 0.7em;  font-family: 'Helvetica'; color:  white; vertical-align: top;  background-color:red;  border-radius: 6em; padding: 0.066em 0.4em; letter-spacing: 0.1em; font-weight: bold;">1210</span>
    <span style="font-size: 0.85em; font-family: 'Helvetica'; color: black;">
      <br> 🏆 The Overall Best Performing Open Source 7B Model 🏆
    <br> 🤖 Outperforms <span style="font-weight: bold;">ChatGPT</span> (March) and <span style="font-weight: bold;">Grok-1</span> 🤖
      <br> 🚀<span style="font-size: 1em; font-family: 'Helvetica'; color: black; font-weight: bold;">15</span>-point improvement in Coding over <span style="font-size: 0.9em;
      font-family: 'Helvetica'; color: black; font-weight: bold;">OpenChat-3.5🚀</span>
      <br><br><span style="font-size: 1em; font-family: 'Helvetica'; color: #3c72db; font-weight: bold;">New Features</span>
      <br> 💡 2 Modes: Coding + Generalist, Mathematical Reasoning 💡
      <br> 🧑‍⚖️ Experimental support for Evaluator and Feedback capabilities 🧑‍⚖️
    </span>
  </a>
</div>

<div style="display: flex; justify-content: center; align-items: center">
  <img src="https://github.com/alpayariyak/openchat/blob/master/assets/1210bench.png?raw=true" style="width: 100%; border-radius: 1em">
</div>


<div>
<h3> Table of Contents</h3>
</div>

1. [Usage](#usage)
2. [Benchmarks](#benchmarks)
3. [Limitations](#limitations)
4. [License](#license)
5. [Dataset Details](#dataset-details)
6. [Citation](#citation)
7. [Acknowledgements](#acknowledgements)


<div align="center">
<h2> Usage </h2>
</div>

To use this model, we highly recommend installing the OpenChat package by following the [installation guide](https://github.com/imoneoi/openchat#installation) in our repository and using the OpenChat OpenAI-compatible API server by running the serving command from the table below. The server is optimized for high-throughput deployment using [vLLM](https://github.com/vllm-project/vllm) and can run on a consumer GPU with 24GB RAM. To enable tensor parallelism, append `--tensor-parallel-size N` to the serving command.

Once started, the server listens at `localhost:18888` for requests and is compatible with the [OpenAI ChatCompletion API specifications](https://platform.openai.com/docs/api-reference/chat). Please refer to the example request below for reference. Additionally, you can use the [OpenChat Web UI](https://github.com/imoneoi/openchat#web-ui) for a user-friendly experience.

If you want to deploy the server as an online service, you can use `--api-keys sk-KEY1 sk-KEY2 ...` to specify allowed API keys and `--disable-log-requests --disable-log-stats --log-file openchat.log` for logging only to a file. For security purposes, we recommend using an [HTTPS gateway](https://fastapi.tiangolo.com/es/deployment/concepts/#security-https) in front of the server.

| Model             | Size | Context | Weights                                                          | Serving                                                                                                          |
|-------------------|------|---------|------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| OpenChat 3.5 1210 | 7B   | 8192    | [Huggingface](https://huggingface.co/openchat/openchat-3.5-1210) | `python -m ochat.serving.openai_api_server --model openchat/openchat-3.5-1210 --engine-use-ray --worker-use-ray` |

<details>
  <summary>Example request (click to expand)</summary>

💡 **Default Mode (GPT4 Correct)**: Best for coding, chat and general tasks

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "messages": [{"role": "user", "content": "You are a large language model named OpenChat. Write a poem to describe yourself"}]
  }'
```

🧮 **Mathematical Reasoning Mode**: Tailored for solving math problems

```bash
curl http://localhost:18888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openchat_3.5",
    "condition": "Math Correct",
    "messages": [{"role": "user", "content": "10.3 − 7988.8133 = "}]
  }'
```

</details>

### Conversation templates

💡 **Default Mode (GPT4 Correct)**: Best for coding, chat and general tasks

```
GPT4 Correct User: Hello<|end_of_turn|>GPT4 Correct Assistant: Hi<|end_of_turn|>GPT4 Correct User: How are you today?<|end_of_turn|>GPT4 Correct Assistant:
```

🧮 **Mathematical Reasoning Mode**: Tailored for solving math problems

```
Math Correct User: 10.3 − 7988.8133=<|end_of_turn|>Math Correct Assistant:
```

⚠️ **Notice:** Remember to set `<|end_of_turn|>` as end of generation token.

The default (GPT4 Correct) template is also available as the integrated `tokenizer.chat_template`,
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

<div align="center">
<h2> (Experimental) Evaluator / Feedback Capabilities </h2>
</div>
We've included evaluator capabilities in this release to advance open-source models as evaluators. You can use `Default Mode (GPT4 Correct)` with the following prompt (same as [Prometheus](https://huggingface.co/datasets/kaist-ai/Feedback-Collection)) to evaluate a response.

```
###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{orig_instruction}

###Response to evaluate:
{orig_response}

###Reference Answer (Score 5):
{orig_reference_answer}

###Score Rubrics:
[{orig_criteria}]
Score 1: {orig_score1_description}
Score 2: {orig_score2_description}
Score 3: {orig_score3_description}
Score 4: {orig_score4_description}
Score 5: {orig_score5_description}

###Feedback: 
```
<div align="center">
<h2> Benchmarks </h2>
</div>

| Model              | # Params | Average  | MT-Bench     | HumanEval       | BBH MC   | AGIEval  | TruthfulQA    | MMLU         | GSM8K        | BBH CoT     |
|--------------------|----------|----------|--------------|-----------------|----------|----------|---------------|--------------|--------------|-------------|
| OpenChat-3.5-1210  | **7B**   | **63.8** | 7.76         | **68.9**        | **49.5** | **48.0** | **61.8**      | 65.3         | **77.3**     | 61.8        |
| OpenChat-3.5       | **7B**   | 61.6     | 7.81         | 55.5            | 47.6     | 47.4     | 59.1          | 64.3         | **77.3**     | 63.5        |
| ChatGPT (March)*   | ?        | 61.5     | **7.94**     | 48.1            | 47.6     | 47.1     | 57.7          | **67.3**     | 74.9         | **70.1**    |
|                    |          |          |              |                 |          |          |               |              |              |             |
| OpenHermes 2.5     | 7B       | 59.3     | 7.54         | 48.2            | 49.4     | 46.5     | 57.5          | 63.8         | 73.5         | 59.9        |
| OpenOrca Mistral   | 7B       | 52.7     | 6.86         | 38.4            | 49.4     | 42.9     | 45.9          | 59.3         | 59.1         | 58.1        |
| Zephyr-β^          | 7B       | 34.6     | 7.34         | 22.0            | 40.6     | 39.0     | 40.8          | 39.8         | 5.1          | 16.0        |
| Mistral            | 7B       | -        | 6.84         | 30.5            | 39.0     | 38.0     | -             | 60.1         | 52.2         | -           |

<details>
  <summary>Evaluation Details(click to expand)</summary>
*: ChatGPT (March) results are from [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774), [Chain-of-Thought Hub](https://github.com/FranxYao/chain-of-thought-hub), and our evaluation. Please note that ChatGPT is not a fixed baseline and evolves rapidly over time.

^: Zephyr-β often fails to follow few-shot CoT instructions, likely because it was aligned with only chat data but not trained on few-shot data.

**: Mistral and Open-source SOTA results are taken from reported results in instruction-tuned model papers and official repositories.

All models are evaluated in chat mode (e.g. with the respective conversation template applied). All zero-shot benchmarks follow the same setting as in the AGIEval paper and Orca paper. CoT tasks use the same configuration as Chain-of-Thought Hub, HumanEval is evaluated with EvalPlus, and MT-bench is run using FastChat. To reproduce our results, follow the instructions in [our repository](https://github.com/imoneoi/openchat/#benchmarks).
</details>
<div>
<h3>HumanEval+</h3>
</div>

| Model                       | Size     | HumanEval+ pass@1 |
|-----------------------------|----------|------------|
| ChatGPT (December 12, 2023) | -        | 64.6       |
| WizardCoder-Python-34B-V1.0 | 34B      | 64.6       |
| **OpenChat 3.5 (Dec 10)**   | **7B**   | **63.4**   |
| OpenHermes 2.5              | 7B       | 41.5       |

<div>
<h3>OpenChat-3.5-1210 vs. Grok</h3>
</div>

|                   | License     | # Param | Average  | MMLU | HumanEval | MATH     | GSM8k    |
|-------------------|-------------|---------|----------|------|-----------|----------|----------|
| OpenChat 3.5 1210 | Apache-2.0  | **7B**  | **60.1** | 65.3 | **68.9**  | **28.9** | **77.3** |
| OpenChat 3.5      | Apache-2.0  | **7B**  | 56.4     | 64.3 | 55.5      | 28.6     | **77.3** |
| Grok-0            | Proprietary | 33B     | 44.5     | 65.7 | 39.7      | 15.7     | 56.8     |
| Grok-1            | Proprietary | ???B    | 55.8     | 73   | 63.2      | 23.9     | 62.9     |

*: Grok results are reported by [X.AI](https://x.ai/).

<div align="center">
<h2> 中文评估结果 / Chinese Evaluations </h2>
</div>

⚠️ Note that this model was not explicitly trained in Chinese (only < 0.1% of the data is in Chinese). 请注意本模型没有针对性训练中文（中文数据占比小于0.1%）。

<div>
<h3>Multi-Level Multi-Discipline Chinese Evaluation Suite (CEVAL)</h3>
<div>

| Model    | Avg   | STEM  | Social Science | Humanities | Others |
|----------|-------|-------|----------------|------------|--------|
| ChatGPT  | 54.4  | 52.9  | 61.8           | 50.9       | 53.6   |
| OpenChat | 47.29 | 45.22 | 52.49          | 48.52      | 45.08  |

<div>
<h3>Massive Multitask Language Understanding in Chinese (CMMLU, 5-shot)</h3>
</div>

| Models   | STEM  | Humanities | SocialSciences | Other | ChinaSpecific | Avg   |
|----------|-------|------------|----------------|-------|---------------|-------|
| ChatGPT  | 47.81 | 55.68      | 56.5           | 62.66 | 50.69         | 55.51 |
| OpenChat | 38.7  | 45.99      | 48.32          | 50.23 | 43.27         | 45.85 |

<div align="center">
<h2> Limitations </h2>
</div>

**Foundation Model Limitations**
Despite its advanced capabilities, OpenChat is still bound by the limitations inherent in its foundation models. These limitations may impact the model's performance in areas such as:

- Complex reasoning
- Mathematical and arithmetic tasks
- Programming and coding challenges

**Hallucination of Non-existent Information**
OpenChat may sometimes generate information that does not exist or is not accurate, also known as "hallucination". Users should be aware of this possibility and verify any critical information obtained from the model.

**Safety**
OpenChat may sometimes generate harmful, hate speech, biased responses, or answer unsafe questions. It's crucial to apply additional AI safety measures in use cases that require safe and moderated responses.

<div align="center">
<h2> License </h2>
</div>

Our OpenChat 3.5 code and models are distributed under the Apache License 2.0.

<div align="center">
<h2> Dataset Details </h2>
</div>

OpenChat 3.5 was trained with C-RLFT on a collection of publicly available high-quality instruction data, with a custom processing pipeline. We detail some notable subsets included here:

- [OpenChat ShareGPT](https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset)
- [Open-Orca with FLAN answers](https://huggingface.co/datasets/imone/OpenOrca_FLAN)
- [Feedback-Collection](https://huggingface.co/datasets/kaist-ai/Feedback-Collection)
- [Capybara](https://huggingface.co/datasets/LDJnr/Capybara) (de-contaminated against MT-bench)
- [GOAT](https://huggingface.co/datasets/tiedong/goat)
- [Glaive](https://huggingface.co/datasets/glaiveai/glaive-code-assistant)
- [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA)
- [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct)
- [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst_top1_2023-08-25)

<div align="center">
<h2> Citation </h2>
</div>

```
@article{wang2023openchat,
  title={OpenChat: Advancing Open-source Language Models with Mixed-Quality Data},
  author={Wang, Guan and Cheng, Sijie and Zhan, Xianyuan and Li, Xiangang and Song, Sen and Liu, Yang},
  journal={arXiv preprint arXiv:2309.11235},
  year={2023}
}
```

<div align="center">
<h2> 💌 Main Contributor </h2>
</div>

* Wang Guan [imonenext@gmail.com], Cheng Sijie [csj23@mails.tsinghua.edu.cn], Alpay Ariyak [aariyak@wpi.edu]
* We look forward to hearing you and collaborating on this exciting project!
