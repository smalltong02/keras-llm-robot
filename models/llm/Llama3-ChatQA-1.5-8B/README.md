---
license: llama3
language:
- en
pipeline_tag: text-generation
tags:
- nvidia
- chatqa-1.5
- chatqa
- llama-3
- pytorch
---


## Model Details
We introduce Llama3-ChatQA-1.5, which excels at conversational question answering (QA) and retrieval-augmented generation (RAG). Llama3-ChatQA-1.5 is developed using an improved training recipe from [ChatQA (1.0)](https://arxiv.org/abs/2401.10225), and it is built on top of [Llama-3 base model](https://huggingface.co/meta-llama/Meta-Llama-3-8B). Specifically, we incorporate more conversational QA data to enhance its tabular and arithmetic calculation capability. Llama3-ChatQA-1.5 has two variants: Llama3-ChatQA-1.5-8B and Llama3-ChatQA-1.5-70B. Both models were originally trained using [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), we converted the checkpoints to Hugging Face format. **For more information about ChatQA, check the [website](https://chatqa-project.github.io/)!**

## Other Resources
[Llama3-ChatQA-1.5-70B](https://huggingface.co/nvidia/Llama3-ChatQA-1.5-70B) &ensp; [Evaluation Data](https://huggingface.co/datasets/nvidia/ChatRAG-Bench) &ensp; [Training Data](https://huggingface.co/datasets/nvidia/ChatQA-Training-Data) &ensp; [Retriever](https://huggingface.co/nvidia/dragon-multiturn-query-encoder) &ensp; [Website](https://chatqa-project.github.io/) &ensp; [Paper](https://arxiv.org/abs/2401.10225)

## Benchmark Results
Results in [ChatRAG Bench](https://huggingface.co/datasets/nvidia/ChatRAG-Bench) are as follows:

| | ChatQA-1.0-7B | Command-R-Plus | Llama-3-instruct-70b | GPT-4-0613 | ChatQA-1.0-70B | ChatQA-1.5-8B | ChatQA-1.5-70B |
| -- |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Doc2Dial | 37.88 | 33.51 | 37.88 | 34.16 | 38.9 | 39.33 | 41.26 |
| QuAC | 29.69 | 34.16 | 36.96 | 40.29 | 41.82 | 39.73 | 38.82 |
| QReCC | 46.97 | 49.77 | 51.34 | 52.01 | 48.05 | 49.03 | 51.40 |
| CoQA | 76.61 | 69.71 | 76.98 | 77.42 | 78.57 | 76.46 | 78.44 |
| DoQA | 41.57 | 40.67 | 41.24 | 43.39 | 51.94 | 49.6 | 50.67 |
| ConvFinQA | 51.61 | 71.21 | 76.6 | 81.28 | 73.69 | 78.46 | 81.88 |
| SQA | 61.87 | 74.07 | 69.61 | 79.21 | 69.14 | 73.28 | 83.82 |
| TopioCQA | 45.45 | 53.77 | 49.72 | 45.09 | 50.98 | 49.96 | 55.63 |
| HybriDial* | 54.51 | 46.7 | 48.59 | 49.81 | 56.44 | 65.76 | 68.27 |
| INSCIT | 30.96 | 35.76 | 36.23 | 36.34 | 31.9 | 30.1 | 32.31 |
| Average (all) | 47.71 | 50.93 | 52.52 | 53.90 | 54.14 | 55.17 | 58.25 |
| Average (exclude HybriDial) | 46.96 | 51.40 | 52.95 | 54.35 | 53.89 | 53.99 | 57.14 |

Note that ChatQA-1.5 is built based on Llama-3 base model, and ChatQA-1.0 is built based on Llama-2 base model. ChatQA-1.5 used some samples from the HybriDial training dataset. To ensure fair comparison, we also compare average scores excluding HybriDial. The data and evaluation scripts for ChatRAG Bench can be found [here](https://huggingface.co/datasets/nvidia/ChatRAG-Bench).


## Prompt Format
**We highly recommend that you use the prompt format we provide, as follows:**
### when context is available
<pre>
System: {System}

{Context}

User: {Question}

Assistant: {Response}

User: {Question}

Assistant:
</pre>

### when context is not available
<pre>
System: {System}

User: {Question}

Assistant: {Response}

User: {Question}

Assistant:
</pre>
**The content of the system's turn (i.e., {System}) for both scenarios is as follows:**
<pre>
This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.
</pre>
**Note that our ChatQA-1.5 models are optimized for the capability with context, e.g., over documents or retrieved context.**

## How to use

### take the whole document as context 
This can be applied to the scenario where the whole document can be fitted into the model, so that there is no need to run retrieval over the document.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "nvidia/Llama3-ChatQA-1.5-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

messages = [
    {"role": "user", "content": "what is the percentage change of the net income from Q4 FY23 to Q4 FY24?"}
]

document = """NVIDIA (NASDAQ: NVDA) today reported revenue for the fourth quarter ended January 28, 2024, of $22.1 billion, up 22% from the previous quarter and up 265% from a year ago.\nFor the quarter, GAAP earnings per diluted share was $4.93, up 33% from the previous quarter and up 765% from a year ago. Non-GAAP earnings per diluted share was $5.16, up 28% from the previous quarter and up 486% from a year ago.\nQ4 Fiscal 2024 Summary\nGAAP\n| $ in millions, except earnings per share | Q4 FY24 | Q3 FY24 | Q4 FY23 | Q/Q | Y/Y |\n| Revenue | $22,103 | $18,120 | $6,051 | Up 22% | Up 265% |\n| Gross margin | 76.0% | 74.0% | 63.3% | Up 2.0 pts | Up 12.7 pts |\n| Operating expenses | $3,176 | $2,983 | $2,576 | Up 6% | Up 23% |\n| Operating income | $13,615 | $10,417 | $1,257 | Up 31% | Up 983% |\n| Net income | $12,285 | $9,243 | $1,414 | Up 33% | Up 769% |\n| Diluted earnings per share | $4.93 | $3.71 | $0.57 | Up 33% | Up 765% |"""

def get_formatted_input(messages, context):
    system = "System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context."
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            ## only apply this instruction for the first user turn
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join(["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for item in messages]) + "\n\nAssistant:"
    formatted_input = system + "\n\n" + context + "\n\n" + conversation
    
    return formatted_input

formatted_input = get_formatted_input(messages, document)
tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

### run retrieval to get top-n chunks as context
This can be applied to the scenario when the document is very long, so that it is necessary to run retrieval. Here, we use our [Dragon-multiturn](https://huggingface.co/nvidia/dragon-multiturn-query-encoder) retriever which can handle conversatinoal query. In addition, we provide a few [documents](https://huggingface.co/nvidia/Llama3-ChatQA-1.5-8B/tree/main/docs) for users to play with.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import json

## load ChatQA-1.5 tokenizer and model
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

## load retriever tokenizer and model
retriever_tokenizer = AutoTokenizer.from_pretrained('nvidia/dragon-multiturn-query-encoder')
query_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-query-encoder')
context_encoder = AutoModel.from_pretrained('nvidia/dragon-multiturn-context-encoder')

## prepare documents, we take landrover car manual document that we provide as an example
chunk_list = json.load(open("docs.json"))['landrover']

messages = [
    {"role": "user", "content": "how to connect the bluetooth in the car?"}
]

### running retrieval
## convert query into a format as follows:
## user: {user}\nagent: {agent}\nuser: {user}
formatted_query_for_retriever = '\n'.join([turn['role'] + ": " + turn['content'] for turn in messages]).strip()

query_input = retriever_tokenizer(formatted_query_for_retriever, return_tensors='pt')
ctx_input = retriever_tokenizer(chunk_list, padding=True, truncation=True, max_length=512, return_tensors='pt')
query_emb = query_encoder(**query_input).last_hidden_state[:, 0, :]
ctx_emb = context_encoder(**ctx_input).last_hidden_state[:, 0, :]

## Compute similarity scores using dot product and rank the similarity
similarities = query_emb.matmul(ctx_emb.transpose(0, 1)) # (1, num_ctx)
ranked_results = torch.argsort(similarities, dim=-1, descending=True) # (1, num_ctx)

## get top-n chunks (n=5)
retrieved_chunks = [chunk_list[idx] for idx in ranked_results.tolist()[0][:5]]
context = "\n\n".join(retrieved_chunks)

### running text generation
formatted_input = get_formatted_input(messages, context)
tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=128, eos_token_id=terminators)

response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```

## Correspondence to
Zihan Liu (zihanl@nvidia.com), Wei Ping (wping@nvidia.com)

## Citation
<pre>
@article{liu2024chatqa,
  title={ChatQA: Building GPT-4 Level Conversational QA Models},
  author={Liu, Zihan and Ping, Wei and Roy, Rajarshi and Xu, Peng and Lee, Chankyu and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2401.10225},
  year={2024}}
</pre>


## License
The use of this model is governed by the [META LLAMA 3 COMMUNITY LICENSE AGREEMENT](https://llama.meta.com/llama3/license/)

