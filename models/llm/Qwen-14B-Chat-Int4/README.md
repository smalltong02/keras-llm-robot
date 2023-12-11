---
language:
- zh
- en
tags:
- qwen
pipeline_tag: text-generation
inference: false
---

# Qwen-14B-Chat-Int4

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg" width="400"/>
<p>
<br>

<p align="center">
        🤗 <a href="https://huggingface.co/Qwen">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/organization/qwen">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2309.16609">Paper</a> &nbsp&nbsp ｜ &nbsp&nbsp🖥️ <a href="https://modelscope.cn/studios/qwen/Qwen-14B-Chat-Demo/summary">Demo</a>
<br>
<a href="assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp<a href="https://discord.gg/z3GAxXZ9Ce">Discord</a>&nbsp&nbsp ｜  &nbsp&nbsp<a href="https://dashscope.aliyun.com">API</a> 
</p>
<br>

## 介绍（Introduction）

**通义千问-14B（Qwen-14B）**是阿里云研发的通义千问大模型系列的140亿参数规模的模型。Qwen-14B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-14B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-14B-Chat。本仓库为Qwen-14B-Chat的Int4量化模型的仓库。

如果您想了解更多关于通义千问-14B开源模型的细节，我们建议您参阅[GitHub代码库](https://github.com/QwenLM/Qwen)。

**Qwen-14B** is the 14B-parameter version of the large language model series, Qwen (abbr. Tongyi Qianwen), proposed by Alibaba Cloud. Qwen-14B is a Transformer-based large language model, which is pretrained on a large volume of data, including web texts, books, codes, etc. Additionally, based on the pretrained Qwen-14B, we release Qwen-14B-Chat, a large-model-based AI assistant, which is trained with alignment techniques. This repository is the one for the Int4 quantized model of Qwen-14B-Chat.

For more details about the open-source model of Qwen-14B, please refer to the [GitHub](https://github.com/QwenLM/Qwen) code repository.
<br>


## 要求（Requirements）

* python 3.8及以上版本
* pytorch 2.0及以上版本
* 建议使用CUDA 11.4及以上（GPU用户、flash-attention用户等需考虑此选项）
* python 3.8 and above
* pytorch 2.0 and above, 2.0 and above are recommended
* CUDA 11.4 and above are recommended (this is for GPU users, flash-attention users, etc.)
<br>


## 依赖项（Dependency）

运行Qwen-14B-Chat-Int4，请确保满足上述要求，再执行以下pip命令安装依赖库。如安装`auto-gptq`遇到问题，我们建议您到官方[repo](https://github.com/PanQiWei/AutoGPTQ)搜索合适的预编译wheel。

To run Qwen-14B-Chat-Int4, please make sure you meet the above requirements, and then execute the following pip commands to install the dependent libraries. If you meet problems installing `auto-gptq`, we advise you to check out the official [repo](https://github.com/PanQiWei/AutoGPTQ) to find a pre-build wheel.

```bash
pip install transformers==4.32.0 accelerate tiktoken einops scipy transformers_stream_generator==0.0.4 peft deepspeed
pip install auto-gptq optimum
```

另外，推荐安装`flash-attention`库（**当前已支持flash attention 2**），以实现更高的效率和更低的显存占用。

In addition, it is recommended to install the `flash-attention` library (**we support flash attention 2 now.**) for higher efficiency and lower memory usage.

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# 下方安装可选，安装可能比较缓慢。
# pip install csrc/layer_norm
# pip install csrc/rotary
```
<br>



## 快速使用（Quickstart）

下面我们展示了一个使用Qwen-14B-Chat-Int4模型的样例：

We show an example of how to use Qwen-14B-Chat-Int4 in the following code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-14B-Chat-Int4", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-14B-Chat-Int4",
    device_map="auto",
    trust_remote_code=True
).eval()
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
# 你好！很高兴为你提供帮助。
```

关于更多的使用说明，请参考我们的[GitHub repo](https://github.com/QwenLM/Qwen)获取更多信息。

For more information, please refer to our [GitHub repo](https://github.com/QwenLM/Qwen) for more information.
<br>



## 量化 (Quantization)

### 效果评测

我们对BF16，Int8和Int4模型在基准评测上做了测试（使用zero-shot设置），发现量化模型效果损失较小，结果如下所示：

We illustrate the zero-shot performance of both BF16, Int8 and Int4 models on the benchmark, and we find that the quantized model does not suffer from significant performance degradation. Results are shown below:

| Quantization | MMLU | CEval (val) | GSM8K | Humaneval |
|--------------|:----:|:-----------:|:-----:|:---------:|
| BF16         | 64.6 |    69.8     | 60.1  |   43.9    |
| Int8         | 63.6 |    68.6     | 60.0	|   48.2    |
| Int4         | 63.3 |    69.0     | 59.8  |   45.7    |

### 推理速度 (Inference Speed)

我们测算了不同精度模型以及不同FlashAttn库版本下模型生成2048和8192个token的平均推理速度。如图所示：

We measured the average inference speed of generating 2048 and 8192 tokens with different quantization levels and versions of flash-attention, respectively.

|  Quantization | FlashAttn | Speed (2048 tokens) | Speed (8192 tokens) |
| ------------- | :-------: | :------------------:| :------------------:|
|      BF16     |   v2      | 32.88               | 24.87               |
|      Int8     |   v2      | 29.28               | 24.22               |
|      Int4     |   v2      | 38.72               | 27.33               |
|      BF16     |   v1      | 32.76               | 28.89               |
|      Int8     |   v1      | 28.31               | 23.87               |
|      Int4     |   v1      | 37.81               | 26.46               |
|      BF16     |  Disabled | 29.32               | 22.91               |
|      Int8     |  Disabled | 31.12               | 24.60               |
|      Int4     |  Disabled | 37.65               | 26.00               |

具体而言，我们记录在长度为1的上下文的条件下生成8192个token的性能。评测运行于单张A100-SXM4-80G GPU，使用PyTorch 2.0.1和CUDA 11.8。推理速度是生成8192个token的速度均值。

In detail, the setting of profiling is generating 8192 new tokens with 1 context token. The profiling runs on a single A100-SXM4-80G GPU with PyTorch 2.0.1 and CUDA 11.8. The inference speed is averaged over the generated 8192 tokens.

注意：以上Int4/Int8模型生成速度使用autogptq库给出，当前``AutoModelForCausalLM.from_pretrained``载入的模型生成速度会慢大约20%。我们已经将该问题汇报给HuggingFace团队，若有解决方案将即时更新。

Note: The generation speed of the Int4/Int8 models mentioned above is provided by the autogptq library. The current speed of the model loaded using "AutoModelForCausalLM.from_pretrained" will be approximately 20% slower. We have reported this issue to the HuggingFace team and will update it promptly if a solution is available.

### 显存使用 (GPU Memory Usage)

我们还测算了不同模型精度编码2048个token及生成8192个token的峰值显存占用情况。（显存消耗在是否使用FlashAttn的情况下均类似。）结果如下所示：

We also profile the peak GPU memory usage for encoding 2048 tokens as context (and generating single token) and generating 8192 tokens (with single token as context) under different quantization levels, respectively. （The GPU memory usage is similar when using flash-attention or not.）The results are shown below.

| Quantization Level | Peak Usage for Encoding 2048 Tokens | Peak Usage for Generating 8192 Tokens |
| ------------------ | :---------------------------------: | :-----------------------------------: |
| BF16               | 30.15GB                             | 38.94GB                               |
| Int8               | 18.81GB                             | 27.54GB                               |
| Int4               | 13.01GB                             | 21.79GB                               |

上述性能测算使用[此脚本](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py)完成。

The above speed and memory profiling are conducted using [this script](https://qianwen-res.oss-cn-beijing.aliyuncs.com/profile.py).
<br>

## Tokenizer

> 注：作为术语的“tokenization”在中文中尚无共识的概念对应，本文档采用英文表达以利说明。

基于tiktoken的分词器有别于其他分词器，比如sentencepiece分词器。尤其在微调阶段，需要特别注意特殊token的使用。关于tokenizer的更多信息，以及微调时涉及的相关使用，请参阅[文档](https://github.com/QwenLM/Qwen/blob/main/tokenization_note_zh.md)。

Our tokenizer based on tiktoken is different from other tokenizers, e.g., sentencepiece tokenizer. You need to pay attention to special tokens, especially in finetuning. For more detailed information on the tokenizer and related use in fine-tuning, please refer to the [documentation](https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md).
<br>



## 模型细节（Model）

与Qwen-14B预训练模型相同，Qwen-14B-Chat模型规模基本情况如下所示

The details of the model architecture of Qwen-14B-Chat are listed as follows

| Hyperparameter  | Value  |
|:----------------|:------:|
| n_layers        |   40   |
| n_heads         |   40   |
| d_model         |  5120  |
| vocab size      | 151851 |
| sequence length |  2048  |

在位置编码、FFN激活函数和normalization的实现方式上，我们也采用了目前最流行的做法，
即RoPE相对位置编码、SwiGLU激活函数、RMSNorm（可选安装flash-attention加速）。

在分词器方面，相比目前主流开源模型以中英词表为主，Qwen-14B-Chat使用了约15万token大小的词表。
该词表在GPT-4使用的BPE词表`cl100k_base`基础上，对中文、多语言进行了优化，在对中、英、代码数据的高效编解码的基础上，对部分多语言更加友好，方便用户在不扩展词表的情况下对部分语种进行能力增强。
词表对数字按单个数字位切分。调用较为高效的[tiktoken分词库](https://github.com/openai/tiktoken)进行分词。

For position encoding, FFN activation function, and normalization calculation methods, we adopt the prevalent practices, i.e., RoPE relative position encoding, SwiGLU for activation function, and RMSNorm for normalization (optional installation of flash-attention for acceleration).

For tokenization, compared to the current mainstream open-source models based on Chinese and English vocabularies, Qwen-14B-Chat uses a vocabulary of over 150K tokens.
It first considers efficient encoding of Chinese, English, and code data, and is also more friendly to multilingual languages, enabling users to directly enhance the capability of some languages without expanding the vocabulary.
It segments numbers by single digit, and calls the [tiktoken](https://github.com/openai/tiktoken) tokenizer library for efficient tokenization.
<br>



## 评测效果（Evaluation）

对于Qwen-14B-Chat模型，我们同样评测了常规的中文理解（C-Eval）、英文理解（MMLU）、代码（HumanEval）和数学（GSM8K）等权威任务，同时包含了长序列任务的评测结果。由于Qwen-14B-Chat模型经过对齐后，激发了较强的外部系统调用能力，我们还进行了工具使用能力方面的评测。

提示：由于硬件和框架造成的舍入误差，复现结果如有波动属于正常现象。

For Qwen-14B-Chat, we also evaluate the model on C-Eval, MMLU, HumanEval, GSM8K, etc., as well as the benchmark evaluation for long-context understanding, and tool usage.

Note: Due to rounding errors caused by hardware and framework, differences in reproduced results are possible.

### 中文评测（Chinese Evaluation）

#### C-Eval

在[C-Eval](https://arxiv.org/abs/2305.08322)验证集上，我们评价了Qwen-14B-Chat模型的0-shot & 5-shot准确率

We demonstrate the 0-shot & 5-shot accuracy of Qwen-14B-Chat on C-Eval validation set

|              Model               | Avg. Acc. |
|:--------------------------------:|:---------:|
|          LLaMA2-7B-Chat          |   31.9    |
|         LLaMA2-13B-Chat          |   36.2    |
|         LLaMA2-70B-Chat          |   44.3    |
|         ChatGLM2-6B-Chat         |   52.6    |
|         InternLM-7B-Chat         |   53.6    |
|        Baichuan2-7B-Chat         |   55.6    |
|        Baichuan2-13B-Chat        |   56.7    |
| Qwen-7B-Chat (original) (0-shot) |   54.2    |
|    **Qwen-7B-Chat (0-shot)**     |   59.7    |
|    **Qwen-7B-Chat (5-shot)**     |   59.3    |
|    **Qwen-14B-Chat (0-shot)**    |   69.8    |
|    **Qwen-14B-Chat (5-shot)**    | **71.7**  |

C-Eval测试集上，Qwen-14B-Chat模型的zero-shot准确率结果如下：

The zero-shot accuracy of Qwen-14B-Chat on C-Eval testing set is provided below:

| Model                   |   Avg.   | STEM | Social Sciences | Humanities | Others |
| :---------------------- | :------: | :--: | :-------------: | :--------: | :----: |
| Chinese-Alpaca-Plus-13B |   41.5   | 36.6 |      49.7       |    43.1    |  41.2  |
| Chinese-Alpaca-2-7B     |   40.3   |  -   |        -        |     -      |   -    |
| ChatGLM2-6B-Chat        |   50.1   | 46.4 |      60.4       |    50.6    |  46.9  |
| Baichuan-13B-Chat       |   51.5   | 43.7 |      64.6       |    56.2    |  49.2  |
| Qwen-7B-Chat (original) |   54.6   | 47.8 |      67.6       |    59.3    |  50.6  |
| **Qwen-7B-Chat**        |   58.6   | 53.3 |      72.1       |    62.8    |  52.0  |
| **Qwen-14B-Chat**       | **69.1** | 65.1 |      80.9       |    71.2    |  63.4  |

在14B规模模型上，经过人类指令对齐的Qwen-14B-Chat模型，准确率在同类相近规模模型中仍然处于前列。

Compared with other pretrained models with comparable model size, the human-aligned Qwen-14B-Chat performs well in C-Eval accuracy.

### 英文评测（English Evaluation）

#### MMLU

[MMLU](https://arxiv.org/abs/2009.03300)评测集上，Qwen-14B-Chat模型的 0-shot & 5-shot 准确率如下，效果同样在同类对齐模型中同样表现较优。

The 0-shot & 5-shot accuracy of Qwen-14B-Chat on MMLU is provided below.
The performance of Qwen-14B-Chat still on the top between other human-aligned models with comparable size.

|              Model               | Avg. Acc. |
|:--------------------------------:|:---------:|
|         ChatGLM2-6B-Chat         |   46.0    |
|          LLaMA2-7B-Chat          |   46.2    |
|         InternLM-7B-Chat         |   51.1    |
|        Baichuan2-7B-Chat         |   52.9    |
|         LLaMA2-13B-Chat          |   54.6    |
|        Baichuan2-13B-Chat        |   57.3    |
|         LLaMA2-70B-Chat          |   63.8    |
| Qwen-7B-Chat (original) (0-shot) |   53.9    |
|    **Qwen-7B-Chat (0-shot)**     |   55.8    |
|    **Qwen-7B-Chat (5-shot)**     |   57.0    |
|    **Qwen-14B-Chat (0-shot)**    |   64.6    |
|    **Qwen-14B-Chat (5-shot)**    | **66.5**  |

### 代码评测（Coding Evaluation）

Qwen-14B-Chat在[HumanEval](https://github.com/openai/human-eval)的zero-shot Pass@1效果如下

The zero-shot Pass@1 of Qwen-14B-Chat on [HumanEval](https://github.com/openai/human-eval) is demonstrated below

|          Model          |  Pass@1  |
|:-----------------------:|:--------:|
|    ChatGLM2-6B-Chat     |   11.0   |
|     LLaMA2-7B-Chat      |   12.2   |
|    InternLM-7B-Chat     |   14.6   |
|    Baichuan2-7B-Chat    |   13.4   |
|     LLaMA2-13B-Chat     |   18.9   |
|   Baichuan2-13B-Chat    |   17.7   |
|     LLaMA2-70B-Chat     |   32.3   |
| Qwen-7B-Chat (original) |   24.4   |
|    **Qwen-7B-Chat**     |   37.2   |
|    **Qwen-14B-Chat**    | **43.9** |

### 数学评测（Mathematics Evaluation）

在评测数学能力的[GSM8K](https://github.com/openai/grade-school-math)上，Qwen-14B-Chat的准确率结果如下

The accuracy of Qwen-14B-Chat on GSM8K is shown below

|              Model               |   Acc.   |
|:--------------------------------:|:--------:|
|          LLaMA2-7B-Chat          |   26.3   |
|         ChatGLM2-6B-Chat         |   28.8   |
|        Baichuan2-7B-Chat         |   32.8   |
|         InternLM-7B-Chat         |   33.0   |
|         LLaMA2-13B-Chat          |   37.1   |
|        Baichuan2-13B-Chat        |   55.3   |
|         LLaMA2-70B-Chat          |   59.3   |
| Qwen-7B-Chat (original) (0-shot) |   41.1   |
|    **Qwen-7B-Chat (0-shot)**     |   50.3   |
|    **Qwen-7B-Chat (8-shot)**     |   54.1   |
|    **Qwen-14B-Chat (0-shot)**    | **60.1** |
|    **Qwen-14B-Chat (8-shot)**    |   59.3   |

### 长序列评测（Long-Context Understanding）

通过NTK插值，LogN注意力缩放可以扩展Qwen-14B-Chat的上下文长度。在长文本摘要数据集[VCSUM](https://arxiv.org/abs/2305.05280)上（文本平均长度在15K左右），Qwen-14B-Chat的Rouge-L结果如下：

**(若要启用这些技巧，请将config.json里的`use_dynamic_ntk`和`use_logn_attn`设置为true)**

We introduce NTK-aware interpolation, LogN attention scaling to extend the context length of Qwen-14B-Chat. The Rouge-L results of Qwen-14B-Chat on long-text summarization dataset [VCSUM](https://arxiv.org/abs/2305.05280) (The average length of this dataset is around 15K) are shown below:

**(To use these tricks, please set `use_dynamic_ntk` and `use_long_attn` to true in config.json.)**

| Model             | VCSUM (zh) |
|:------------------|:----------:|
| GPT-3.5-Turbo-16k |    16.0    |
| LLama2-7B-Chat    |    0.2     |
| InternLM-7B-Chat  |    13.0    |
| ChatGLM2-6B-Chat  |    16.3    |
| **Qwen-14B-Chat** |  **17.3**  |


### 工具使用能力的评测（Tool Usage）

#### ReAct Prompting

千问支持通过 [ReAct Prompting](https://arxiv.org/abs/2210.03629) 调用插件/工具/API。ReAct 也是 [LangChain](https://python.langchain.com/) 框架采用的主要方式之一。在我们开源的、用于评估工具使用能力的评测基准上，千问的表现如下：

Qwen-Chat supports calling plugins/tools/APIs through [ReAct Prompting](https://arxiv.org/abs/2210.03629). ReAct is also one of the main approaches used by the [LangChain](https://python.langchain.com/) framework. In our evaluation benchmark for assessing tool usage capabilities, Qwen-Chat's performance is as follows:

<table>
    <tr>
        <th colspan="4" align="center">Chinese Tool-Use Benchmark</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection (Acc.↑)</th><th align="center">Tool Input (Rouge-L↑)</th><th align="center">False Positive Error↓</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">95%</td><td align="center">0.90</td><td align="center">15.0%</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">85%</td><td align="center">0.88</td><td align="center">75.0%</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">98%</td><td align="center">0.91</td><td align="center">7.3%</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">98%</td><td align="center">0.93</td><td align="center">2.4%</td>
    </tr>
</table>

> 评测基准中出现的插件均没有出现在千问的训练集中。该基准评估了模型在多个候选插件中选择正确插件的准确率、传入插件的参数的合理性、以及假阳率。假阳率（False Positive）定义：在处理不该调用插件的请求时，错误地调用了插件。

> The plugins that appear in the evaluation set do not appear in the training set of Qwen. This benchmark evaluates the accuracy of the model in selecting the correct plugin from multiple candidate plugins, the rationality of the parameters passed into the plugin, and the false positive rate. False Positive: Incorrectly invoking a plugin when it should not have been called when responding to a query.

![](assets/react_showcase_001.png)
![](assets/react_showcase_002.png)

#### Code Interpreter

为了考察Qwen使用Python Code Interpreter完成数学解题、数据可视化、及文件处理与爬虫等任务的能力，我们专门建设并开源了一个评测这方面能力的[评测基准](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark)。

我们发现Qwen在生成代码的可执行率、结果正确性上均表现较好：

To assess Qwen's ability to use the Python Code Interpreter for tasks such as mathematical problem solving, data visualization, and other general-purpose tasks such as file handling and web scraping, we have created and open-sourced a benchmark specifically designed for evaluating these capabilities. You can find the benchmark at this [link](https://github.com/QwenLM/Qwen-Agent/tree/main/benchmark).

We have observed that Qwen performs well in terms of code executability and result accuracy when generating code:

<table>
    <tr>
        <th colspan="4" align="center">Executable Rate of Generated Code (%)</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Math↑</th><th align="center">Visualization↑</th><th align="center">General↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">91.9</td><td align="center">85.9</td><td align="center">82.8</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">89.2</td><td align="center">65.0</td><td align="center">74.1</td>
    </tr>
    <tr>
        <td>LLaMA2-7B-Chat</td>
        <td align="center">41.9</td>
        <td align="center">33.1</td>
        <td align="center">24.1 </td>
    </tr>
    <tr>
        <td>LLaMA2-13B-Chat</td>
        <td align="center">50.0</td>
        <td align="center">40.5</td>
        <td align="center">48.3 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-7B-Instruct</td>
        <td align="center">85.1</td>
        <td align="center">54.0</td>
        <td align="center">70.7 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-13B-Instruct</td>
        <td align="center">93.2</td>
        <td align="center">55.8</td>
        <td align="center">74.1 </td>
    </tr>
    <tr>
        <td>InternLM-7B-Chat-v1.1</td>
        <td align="center">78.4</td>
        <td align="center">44.2</td>
        <td align="center">62.1 </td>
    </tr>
    <tr>
        <td>InternLM-20B-Chat</td>
        <td align="center">70.3</td>
        <td align="center">44.2</td>
        <td align="center">65.5 </td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td>
        <td align="center">82.4</td>
        <td align="center">64.4</td>
        <td align="center">67.2 </td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td>
        <td align="center">89.2</td>
        <td align="center">84.1</td>
        <td align="center">65.5</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="4" align="center">Accuracy of Code Execution Results (%)</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Math↑</th><th align="center">Visualization-Hard↑</th><th align="center">Visualization-Easy↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">82.8</td><td align="center">66.7</td><td align="center">60.8</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">47.3</td><td align="center">33.3</td><td align="center">55.7</td>
    </tr>
    <tr>
        <td>LLaMA2-7B-Chat</td>
        <td align="center">3.9</td>
        <td align="center">14.3</td>
        <td align="center">39.2 </td>
    </tr>
    <tr>
        <td>LLaMA2-13B-Chat</td>
        <td align="center">8.3</td>
        <td align="center">8.3</td>
        <td align="center">40.5 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-7B-Instruct</td>
        <td align="center">14.3</td>
        <td align="center">26.2</td>
        <td align="center">60.8 </td>
    </tr>
    <tr>
        <td>CodeLLaMA-13B-Instruct</td>
        <td align="center">28.2</td>
        <td align="center">27.4</td>
        <td align="center">62.0 </td>
    </tr>
    <tr>
        <td>InternLM-7B-Chat-v1.1</td>
        <td align="center">28.5</td>
        <td align="center">4.8</td>
        <td align="center">40.5 </td>
    </tr>
    <tr>
        <td>InternLM-20B-Chat</td>
        <td align="center">34.6</td>
        <td align="center">21.4</td>
        <td align="center">45.6 </td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td>
        <td align="center">41.9</td>
        <td align="center">40.5</td>
        <td align="center">54.4 </td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td>
        <td align="center">58.4</td>
        <td align="center">53.6</td>
        <td align="center">59.5</td>
    </tr>
</table>

<p align="center">
    <br>
    <img src="assets/code_interpreter_showcase_001.jpg" />
    <br>
<p>

#### Huggingface Agent

千问还具备作为 [HuggingFace Agent](https://huggingface.co/docs/transformers/transformers_agents) 的能力。它在 Huggingface 提供的run模式评测基准上的表现如下：

Qwen-Chat also has the capability to be used as a [HuggingFace Agent](https://huggingface.co/docs/transformers/transformers_agents). Its performance on the run-mode benchmark provided by HuggingFace is as follows:

<table>
    <tr>
        <th colspan="4" align="center">HuggingFace Agent Benchmark- Run Mode</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection↑</th><th align="center">Tool Used↑</th><th align="center">Code↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">100</td><td align="center">100</td><td align="center">97.4</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">95.4</td><td align="center">96.3</td><td align="center">87.0</td>
    </tr>
    <tr>
        <td>StarCoder-Base-15B</td><td align="center">86.1</td><td align="center">87.0</td><td align="center">68.9</td>
    </tr>
    <tr>
        <td>StarCoder-15B</td><td align="center">87.0</td><td align="center">88.0</td><td align="center">68.9</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">87.0</td><td align="center">87.0</td><td align="center">71.5</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">93.5</td><td align="center">94.4</td><td align="center">87.0</td>
    </tr>
</table>

<table>
    <tr>
        <th colspan="4" align="center">HuggingFace Agent Benchmark - Chat Mode</th>
    </tr>
    <tr>
        <th align="center">Model</th><th align="center">Tool Selection↑</th><th align="center">Tool Used↑</th><th align="center">Code↑</th>
    </tr>
    <tr>
        <td>GPT-4</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">98.5</td>
    </tr>
    <tr>
        <td>GPT-3.5</td><td align="center">97.3</td><td align="center">96.8</td><td align="center">89.6</td>
    </tr>
    <tr>
        <td>StarCoder-Base-15B</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">91.1</td>
    </tr>
    <tr>
        <td>StarCoder-15B</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">89.6</td>
    </tr>
    <tr>
        <td>Qwen-7B-Chat</td><td align="center">94.7</td><td align="center">94.7</td><td align="center">85.1</td>
    </tr>
    <tr>
        <td>Qwen-14B-Chat</td><td align="center">97.9</td><td align="center">97.9</td><td align="center">95.5</td>
    </tr>
</table>

<br>

## FAQ

如遇到问题，敬请查阅[FAQ](https://github.com/QwenLM/Qwen/blob/main/FAQ_zh.md)以及issue区，如仍无法解决再提交issue。

If you meet problems, please refer to [FAQ](https://github.com/QwenLM/Qwen/blob/main/FAQ.md) and the issues first to search a solution before you launch a new issue.
<br>

## 引用 (Citation)

如果你觉得我们的工作对你有帮助，欢迎引用！

If you find our work helpful, feel free to give us a cite.

```
@article{qwen,
  title={Qwen Technical Report},
  author={Jinze Bai and Shuai Bai and Yunfei Chu and Zeyu Cui and Kai Dang and Xiaodong Deng and Yang Fan and Wenbin Ge and Yu Han and Fei Huang and Binyuan Hui and Luo Ji and Mei Li and Junyang Lin and Runji Lin and Dayiheng Liu and Gao Liu and Chengqiang Lu and Keming Lu and Jianxin Ma and Rui Men and Xingzhang Ren and Xuancheng Ren and Chuanqi Tan and Sinan Tan and Jianhong Tu and Peng Wang and Shijie Wang and Wei Wang and Shengguang Wu and Benfeng Xu and Jin Xu and An Yang and Hao Yang and Jian Yang and Shusheng Yang and Yang Yao and Bowen Yu and Hongyi Yuan and Zheng Yuan and Jianwei Zhang and Xingxuan Zhang and Yichang Zhang and Zhenru Zhang and Chang Zhou and Jingren Zhou and Xiaohuan Zhou and Tianhang Zhu},
  journal={arXiv preprint arXiv:2309.16609},
  year={2023}
}
```
<br>

## 使用协议（License Agreement）

我们的代码和模型权重对学术研究完全开放，并支持商用。请查看[LICENSE](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT)了解具体的开源协议细节。如需商用，请填写[问卷](https://dashscope.console.aliyun.com/openModelApply/Qwen-14B-Chat)申请。

Our code and checkpoints are open to research purpose, and they are allowed for commercial purposes. Check [LICENSE](https://github.com/QwenLM/Qwen/blob/main/Tongyi%20Qianwen%20LICENSE%20AGREEMENT) for more details about the license. If you have requirements for commercial use, please fill out the [form](https://dashscope.console.aliyun.com/openModelApply/Qwen-14B-Chat) to apply.
<br>



## 联系我们（Contact Us）

如果你想给我们的研发团队和产品团队留言，欢迎加入我们的微信群、钉钉群以及Discord！同时，也欢迎通过邮件（qianwen_opensource@alibabacloud.com）联系我们。

If you are interested to leave a message to either our research team or product team, join our Discord or WeChat groups! Also, feel free to send an email to qianwen_opensource@alibabacloud.com.

