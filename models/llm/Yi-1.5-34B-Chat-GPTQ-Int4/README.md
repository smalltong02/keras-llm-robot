---
language:
- en
- zh
pipeline_tag: text-generation
tags:
- gptq
- int4
- yi1.5-34B-Chat
- pytorch
license: apache-2.0
license_name: apache-2.0
license_link: LICENSE
---

## About Quantization
我们使用modelscope [swift](https://github.com/modelscope/swift/)仓库进行GPTQ 4bit量化. 量化文档可以查看[这里](https://github.com/modelscope/swift/blob/main/docs/source/LLM/LLM%E9%87%8F%E5%8C%96%E6%96%87%E6%A1%A3.md). 量化命令如下:

We use the modelscope [swift](https://github.com/modelscope/swift/) repository to perform GPTQ 4bit quantization. Quantization documentation can be found [here](https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/LLM-quantization.md). The quantization command is as follows:

```bash
# Experimental Environment: A100
OMP_NUM_THREADS=14 \
	swift export \
	--quant_bits 4 \
	--model_type yi-1_5-34b-chat \
	--quant_method gptq \
	--dataset alpaca-zh alpaca-en sharegpt-gpt4-mini \
	--quant_seqlen 4096
```

Inference:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-1_5-34b-chat-gptq-int4
```

SFT:
```bash
CUDA_VISIBLE_DEVICES=0 swift sft --model_type yi-1_5-34b-chat-gptq-int4 --dataset leetcode-python-en
```

Original Model:

[YI1.5-34B-Chat](https://modelscope.cn/models/01ai/Yi-1.5-34B-Chat/summary)
