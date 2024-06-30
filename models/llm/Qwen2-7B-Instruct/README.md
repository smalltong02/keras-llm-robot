---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- chat
---

# Qwen2-7B-Instruct

## Introduction

Qwen2 is the new series of Qwen large language models. For Qwen2, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters, including a Mixture-of-Experts model. This repo contains the instruction-tuned 7B Qwen2 model.

Compared with the state-of-the-art opensource language models, including the previous released Qwen1.5, Qwen2 has generally surpassed most opensource models and demonstrated competitiveness against proprietary models across a series of benchmarks targeting for language understanding, language generation, multilingual capability, coding, mathematics, reasoning, etc.

Qwen2-7B-Instruct supports a context length of up to 131,072 tokens, enabling the processing of extensive inputs. Please refer to [this section](#processing-long-texts) for detailed instructions on how to deploy Qwen2 for handling long texts.

For more details, please refer to our [blog](https://qwenlm.github.io/blog/qwen2/), [GitHub](https://github.com/QwenLM/Qwen2), and [Documentation](https://qwen.readthedocs.io/en/latest/).
<br>

## Model Details
Qwen2 is a language model series including decoder language models of different model sizes. For each size, we release the base language model and the aligned chat model. It is based on the Transformer architecture with SwiGLU activation, attention QKV bias, group query attention, etc. Additionally, we have an improved tokenizer adaptive to multiple natural languages and codes.

## Training details
We pretrained the models with a large amount of data, and we post-trained the models with both supervised finetuning and direct preference optimization.


## Requirements
The code of Qwen2 has been in the latest Hugging face transformers and we advise you to install `transformers>=4.37.0`, or you might encounter the following error:
```
KeyError: 'qwen2'
```

## Quickstart

Here provides a code snippet with `apply_chat_template` to show you how to load the tokenizer and model and how to generate contents.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

### Processing Long Texts

To handle extensive inputs exceeding 32,768 tokens, we utilize [YARN](https://arxiv.org/abs/2309.00071), a technique for enhancing model length extrapolation, ensuring optimal performance on lengthy texts.

For deployment, we recommend using vLLM. You can enable the long-context capabilities by following these steps:

1. **Install vLLM**: You can install vLLM by running the following command.

```bash
pip install "vllm>=0.4.3"
```

Or you can install vLLM from [source](https://github.com/vllm-project/vllm/).

2. **Configure Model Settings**: After downloading the model weights, modify the `config.json` file by including the below snippet:
    ```json
        {
            "architectures": [
                "Qwen2ForCausalLM"
            ],
            // ...
            "vocab_size": 152064,

            // adding the following snippets
            "rope_scaling": {
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
                "type": "yarn"
            }
        }
    ```
    This snippet enable YARN to support longer contexts.

3. **Model Deployment**: Utilize vLLM to deploy your model. For instance, you can set up an openAI-like server using the command:

    ```bash
    python -m vllm.entrypoints.openai.api_server --served-model-name Qwen2-7B-Instruct --model path/to/weights
    ```

    Then you can access the Chat API by:

    ```bash
    curl http://localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
        "model": "Qwen2-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Your Long Input Here."}
        ]
        }'
    ```

    For further usage instructions of vLLM, please refer to our [Github](https://github.com/QwenLM/Qwen2).

**Note**: Presently, vLLM only supports static YARN, which means the scaling factor remains constant regardless of input length, **potentially impacting performance on shorter texts**. We advise adding the `rope_scaling` configuration only when processing long contexts is required.

## Evaluation

We briefly compare Qwen2-7B-Instruct with similar-sized instruction-tuned LLMs, including Qwen1.5-7B-Chat. The results are shown below:

| Datasets | Llama-3-8B-Instruct | Yi-1.5-9B-Chat | GLM-4-9B-Chat | Qwen1.5-7B-Chat | Qwen2-7B-Instruct |
| :--- | :---: | :---: | :---: | :---: | :---: |
| _**English**_ |  |  |  |  |  |
| MMLU | 68.4 | 69.5 | **72.4** | 59.5 | 70.5 |
| MMLU-Pro | 41.0 | - | - | 29.1 | **44.1** |
| GPQA | **34.2** | - | **-** | 27.8 | 25.3 |
| TheroemQA | 23.0 | - | - | 14.1 | **25.3** |
| MT-Bench | 8.05 | 8.20 | 8.35 | 7.60 | **8.41** |
| _**Coding**_ |  |  |  |  |  |
| Humaneval | 62.2 | 66.5 | 71.8 | 46.3 | **79.9** |
| MBPP | **67.9** | - | - | 48.9 | 67.2 |
| MultiPL-E | 48.5 | - | - | 27.2 | **59.1** |
| Evalplus | 60.9 | - | - | 44.8 | **70.3** |
| LiveCodeBench | 17.3 | - | - | 6.0 | **26.6** |
| _**Mathematics**_ |  |  |  |  |  |
| GSM8K | 79.6 | **84.8** | 79.6 | 60.3 | 82.3 |
| MATH | 30.0 | 47.7 | **50.6** | 23.2 | 49.6 |
| _**Chinese**_ |  |  |  |  |  |
| C-Eval | 45.9 | - | 75.6 | 67.3 | **77.2** |
| AlignBench | 6.20 | 6.90 | 7.01 | 6.20 | **7.21** |

## Citation

If you find our work helpful, feel free to give us a cite.

```
@article{qwen2,
  title={Qwen2 Technical Report},
  year={2024}
}
```