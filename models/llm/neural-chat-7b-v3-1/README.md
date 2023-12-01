---
license: apache-2.0
---

## Fine-tuning on [Habana](https://habana.ai/) Gaudi2

This model is a fine-tuned model based on [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) on the open source dataset [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca). Then we align it with DPO algorithm. For more details, you can refer our blog: [The Practice of Supervised Fine-tuning and Direct Preference Optimization on Habana Gaudi2](https://medium.com/@NeuralCompressor/the-practice-of-supervised-finetuning-and-direct-preference-optimization-on-habana-gaudi2-a1197d8a3cd3).

## Model date
Neural-chat-7b-v3-1 was trained between September and October, 2023.

## Evaluation

We submit our model to [open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard), and the model performance has been **improved significantly** as we see from the average metric of 7 tasks from the leaderboard.

| Model | Average ⬆️| ARC (25-s) ⬆️ | HellaSwag (10-s) ⬆️ | MMLU (5-s) ⬆️| TruthfulQA (MC) (0-s) ⬆️ | Winogrande (5-s) | GSM8K (5-s) | DROP (3-s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|[mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 50.32 | 59.58  | 83.31  | 64.16  | 42.15 | 78.37 | 18.12 | 6.14 |
| [Intel/neural-chat-7b-v3](https://huggingface.co/Intel/neural-chat-7b-v3) | **57.31** | 67.15 | 83.29 | 62.26  | 58.77 | 78.06 | 1.21 | 50.43 |
| [Intel/neural-chat-7b-v3-1](https://huggingface.co/Intel/neural-chat-7b-v3-1) | **59.06** | 66.21 | 83.64 | 62.37  | 59.65 | 78.14 | 19.56 | 43.84 |

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-04
- train_batch_size: 1
- eval_batch_size: 2
- seed: 42
- distributed_type: multi-HPU
- num_devices: 8
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.03
- num_epochs: 2.0

### Training sample code
Here is the sample code to reproduce the model: [Sample Code](https://github.com/intel/intel-extension-for-transformers/blob/main/intel_extension_for_transformers/neural_chat/examples/finetuning/finetune_neuralchat_v3/README.md).

## Prompt Template

```
### System:
{system}
### User:
{usr}
### Assistant:

```


## Inference with transformers

```python
import transformers


model_name = 'Intel/neural-chat-7b-v3-1'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

def generate_response(system_input, user_input):

    # Format the input using the provided template
    prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"

    # Tokenize and encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)

    # Generate a response
    outputs = model.generate(inputs, max_length=1000, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the assistant's response
    return response.split("### Assistant:\n")[-1]


# Example usage
system_input = "You are a math expert assistant. Your mission is to help users understand and solve various math problems. You should provide step-by-step solutions, explain reasonings and give the correct answer."
user_input = "calculate 100 + 520 + 60"
response = generate_response(system_input, user_input)
print(response)

# expected response
"""
To calculate the sum of 100, 520, and 60, we will follow these steps:

1. Add the first two numbers: 100 + 520
2. Add the result from step 1 to the third number: (100 + 520) + 60

Step 1: Add 100 and 520
100 + 520 = 620

Step 2: Add the result from step 1 to the third number (60)
(620) + 60 = 680

So, the sum of 100, 520, and 60 is 680.
"""

```

## Ethical Considerations and Limitations
neural-chat-7b-v3-1 can produce factually incorrect output, and should not be relied on to produce factually accurate information. neural-chat-7b-v3-1 was trained on [Open-Orca/SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) based on [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1). Because of the limitations of the pretrained model and the finetuning datasets, it is possible that this model could generate lewd, biased or otherwise offensive outputs.

Therefore, before deploying any applications of neural-chat-7b-v3-1, developers should perform safety testing.

## Disclaimer

The license on this model does not constitute legal advice. We are not responsible for the actions of third parties who use this model. Please cosult an attorney before using this model for commercial purposes.

## Organizations developing the model

The NeuralChat team with members from Intel/DCAI/AISE/AIPT. Core team members: Kaokao Lv, Liang Lv, Chang Wang, Wenxin Zhang, Xuhui Ren, and Haihao Shen.

## Useful links
* Intel Neural Compressor [link](https://github.com/intel/neural-compressor)
* Intel Extension for Transformers [link](https://github.com/intel/intel-extension-for-transformers)

# [Open LLM Leaderboard Evaluation Results](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
Detailed results can be found [here](https://huggingface.co/datasets/open-llm-leaderboard/details_Intel__neural-chat-7b-v3-1)

| Metric                | Value                     |
|-----------------------|---------------------------|
| Avg.                  | 59.06   |
| ARC (25-shot)         | 66.21          |
| HellaSwag (10-shot)   | 83.64    |
| MMLU (5-shot)         | 62.37         |
| TruthfulQA (0-shot)   | 59.65   |
| Winogrande (5-shot)   | 78.14   |
| GSM8K (5-shot)        | 19.56        |
| DROP (3-shot)         | 43.84         |
