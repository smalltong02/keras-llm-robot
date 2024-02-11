---
language:
- code
license: llama2
tags:
- llama-2
model_name: CodeLlama 34B Instruct
base_model: codellama/CodeLlama-34b-instruct-hf
inference: false
model_creator: Meta
model_type: llama
pipeline_tag: text-generation
prompt_template: '[INST] Write code to solve the following coding problem that obeys
  the constraints and passes the example test cases. Please wrap your code answer
  using ```:

  {prompt}

  [/INST]

  '
quantized_by: TheBloke
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# CodeLlama 34B Instruct - GGUF
- Model creator: [Meta](https://huggingface.co/meta-llama)
- Original model: [CodeLlama 34B Instruct](https://huggingface.co/codellama/CodeLlama-34b-instruct-hf)

<!-- description start -->
## Description

This repo contains GGUF format model files for [Meta's CodeLlama 34B Instruct](https://huggingface.co/codellama/CodeLlama-34b-instruct-hf).

<!-- description end -->
<!-- README_GGUF.md-about-gguf start -->
### About GGUF

GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp. GGUF offers numerous advantages over GGML, such as better tokenisation, and support for special tokens. It is also supports metadata, and is designed to be extensible.

Here is an incomplate list of clients and libraries that are known to support GGUF:

* [llama.cpp](https://github.com/ggerganov/llama.cpp). The source project for GGUF. Offers a CLI and a server option.
* [text-generation-webui](https://github.com/oobabooga/text-generation-webui), the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.
* [KoboldCpp](https://github.com/LostRuins/koboldcpp), a fully featured web UI, with GPU accel across all platforms and GPU architectures. Especially good for story telling.
* [LM Studio](https://lmstudio.ai/), an easy-to-use and powerful local GUI for Windows and macOS (Silicon), with GPU acceleration.
* [LoLLMS Web UI](https://github.com/ParisNeo/lollms-webui), a great web UI with many interesting and unique features, including a full model library for easy model selection.
* [Faraday.dev](https://faraday.dev/), an attractive and easy to use character-based chat GUI for Windows and macOS (both Silicon and Intel), with GPU acceleration.
* [ctransformers](https://github.com/marella/ctransformers), a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
* [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.
* [candle](https://github.com/huggingface/candle), a Rust ML framework with a focus on performance, including GPU support, and ease of use.

<!-- README_GGUF.md-about-gguf end -->
<!-- repositories-available start -->
## Repositories available

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF)
* [Meta's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/codellama/CodeLlama-34b-instruct-hf)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: CodeLlama

```
[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:
{prompt}
[/INST]

```

<!-- prompt-template end -->


<!-- compatibility_gguf start -->
## Compatibility

These quantised GGUFv2 files are compatible with llama.cpp from August 27th onwards, as of commit [d0cee0d36d5be95a0d9088b674dbb27354107221](https://github.com/ggerganov/llama.cpp/commit/d0cee0d36d5be95a0d9088b674dbb27354107221)

They are also compatible with many third party UIs and libraries - please see the list at the top of this README.

## Explanation of quantisation methods
<details>
  <summary>Click to see details</summary>

The new methods available are:
* GGML_TYPE_Q2_K - "type-1" 2-bit quantization in super-blocks containing 16 blocks, each block having 16 weight. Block scales and mins are quantized with 4 bits. This ends up effectively using 2.5625 bits per weight (bpw)
* GGML_TYPE_Q3_K - "type-0" 3-bit quantization in super-blocks containing 16 blocks, each block having 16 weights. Scales are quantized with 6 bits. This end up using 3.4375 bpw.
* GGML_TYPE_Q4_K - "type-1" 4-bit quantization in super-blocks containing 8 blocks, each block having 32 weights. Scales and mins are quantized with 6 bits. This ends up using 4.5 bpw.
* GGML_TYPE_Q5_K - "type-1" 5-bit quantization. Same super-block structure as GGML_TYPE_Q4_K resulting in 5.5 bpw
* GGML_TYPE_Q6_K - "type-0" 6-bit quantization. Super-blocks with 16 blocks, each block having 16 weights. Scales are quantized with 8 bits. This ends up using 6.5625 bpw

Refer to the Provided Files table below to see what files use which methods, and how.
</details>
<!-- compatibility_gguf end -->

<!-- README_GGUF.md-provided-files start -->
## Provided files

| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| [codellama-34b-instruct.Q2_K.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q2_K.gguf) | Q2_K | 2 | 14.21 GB| 16.71 GB | smallest, significant quality loss - not recommended for most purposes |
| [codellama-34b-instruct.Q3_K_S.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q3_K_S.gguf) | Q3_K_S | 3 | 14.61 GB| 17.11 GB | very small, high quality loss |
| [codellama-34b-instruct.Q3_K_M.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q3_K_M.gguf) | Q3_K_M | 3 | 16.28 GB| 18.78 GB | very small, high quality loss |
| [codellama-34b-instruct.Q3_K_L.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q3_K_L.gguf) | Q3_K_L | 3 | 17.77 GB| 20.27 GB | small, substantial quality loss |
| [codellama-34b-instruct.Q4_0.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q4_0.gguf) | Q4_0 | 4 | 19.05 GB| 21.55 GB | legacy; small, very high quality loss - prefer using Q3_K_M |
| [codellama-34b-instruct.Q4_K_S.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q4_K_S.gguf) | Q4_K_S | 4 | 19.15 GB| 21.65 GB | small, greater quality loss |
| [codellama-34b-instruct.Q4_K_M.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q4_K_M.gguf) | Q4_K_M | 4 | 20.22 GB| 22.72 GB | medium, balanced quality - recommended |
| [codellama-34b-instruct.Q5_0.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q5_0.gguf) | Q5_0 | 5 | 23.24 GB| 25.74 GB | legacy; medium, balanced quality - prefer using Q4_K_M |
| [codellama-34b-instruct.Q5_K_S.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q5_K_S.gguf) | Q5_K_S | 5 | 23.24 GB| 25.74 GB | large, low quality loss - recommended |
| [codellama-34b-instruct.Q5_K_M.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q5_K_M.gguf) | Q5_K_M | 5 | 23.84 GB| 26.34 GB | large, very low quality loss - recommended |
| [codellama-34b-instruct.Q6_K.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q6_K.gguf) | Q6_K | 6 | 27.68 GB| 30.18 GB | very large, extremely low quality loss |
| [codellama-34b-instruct.Q8_0.gguf](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/blob/main/codellama-34b-instruct.Q8_0.gguf) | Q8_0 | 8 | 35.86 GB| 38.36 GB | very large, extremely low quality loss - not recommended |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.



<!-- README_GGUF.md-provided-files end -->

<!-- README_GGUF.md-how-to-download start -->
## How to download GGUF files

**Note for manual downloaders:** You almost never want to clone the entire repo! Multiple different quantisation formats are provided, and most users only want to pick and download a single file.

The following clients/libraries will automatically download models for you, providing a list of available models to choose from:
- LM Studio
- LoLLMS Web UI
- Faraday.dev

### In `text-generation-webui`

Under Download Model, you can enter the model repo: TheBloke/CodeLlama-34B-Instruct-GGUF and below it, a specific filename to download, such as: codellama-34b-instruct.q4_K_M.gguf.

Then click Download.

### On the command line, including multiple files at once

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub>=0.17.1
```

Then you can download any individual model file to the current directory, at high speed, with a command like this:

```shell
huggingface-cli download TheBloke/CodeLlama-34B-Instruct-GGUF codellama-34b-instruct.q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

<details>
  <summary>More advanced huggingface-cli download usage</summary>

You can also download multiple files at once with a pattern:

```shell
huggingface-cli download TheBloke/CodeLlama-34B-Instruct-GGUF --local-dir . --local-dir-use-symlinks False --include='*Q4_K*gguf'
```

For more documentation on downloading with `huggingface-cli`, please see: [HF -> Hub Python Library -> Download files -> Download from the CLI](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli).

To accelerate downloads on fast connections (1Gbit/s or higher), install `hf_transfer`:

```shell
pip3 install hf_transfer
```

And set environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `1`:

```shell
HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/CodeLlama-34B-Instruct-GGUF codellama-34b-instruct.q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

Windows CLI users: Use `set HUGGINGFACE_HUB_ENABLE_HF_TRANSFER=1` before running the download command.
</details>
<!-- README_GGUF.md-how-to-download end -->

<!-- README_GGUF.md-how-to-run start -->
## Example `llama.cpp` command

Make sure you are using `llama.cpp` from commit [d0cee0d36d5be95a0d9088b674dbb27354107221](https://github.com/ggerganov/llama.cpp/commit/d0cee0d36d5be95a0d9088b674dbb27354107221) or later.

```shell
./main -ngl 32 -m codellama-34b-instruct.q4_K_M.gguf --color -c 4096 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "[INST] Write code to solve the following coding problem that obeys the constraints and passes the example test cases. Please wrap your code answer using ```:\n{prompt}\n[/INST]"
```

Change `-ngl 32` to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

Change `-c 4096` to the desired sequence length. For extended sequence models - eg 8K, 16K, 32K - the necessary RoPE scaling parameters are read from the GGUF file and set by llama.cpp automatically.

If you want to have a chat-style conversation, replace the `-p <PROMPT>` argument with `-i -ins`

For other parameters and how to use them, please refer to [the llama.cpp documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

## How to run in `text-generation-webui`

Further instructions here: [text-generation-webui/docs/llama.cpp.md](https://github.com/oobabooga/text-generation-webui/blob/main/docs/llama.cpp.md).

## How to run from Python code

You can use GGUF models from Python using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or [ctransformers](https://github.com/marella/ctransformers) libraries.

### How to load this model from Python using ctransformers

#### First install the package

```bash
# Base ctransformers with no GPU acceleration
pip install ctransformers>=0.2.24
# Or with CUDA GPU acceleration
pip install ctransformers[cuda]>=0.2.24
# Or with ROCm GPU acceleration
CT_HIPBLAS=1 pip install ctransformers>=0.2.24 --no-binary ctransformers
# Or with Metal GPU acceleration for macOS systems
CT_METAL=1 pip install ctransformers>=0.2.24 --no-binary ctransformers
```

#### Simple example code to load one of these GGUF models

```python
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/CodeLlama-34B-Instruct-GGUF", model_file="codellama-34b-instruct.q4_K_M.gguf", model_type="llama", gpu_layers=50)

print(llm("AI is going to"))
```

## How to use with LangChain

Here's guides on using llama-cpp-python or ctransformers with LangChain:

* [LangChain + llama-cpp-python](https://python.langchain.com/docs/integrations/llms/llamacpp)
* [LangChain + ctransformers](https://python.langchain.com/docs/integrations/providers/ctransformers)

<!-- README_GGUF.md-how-to-run end -->

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute

Thanks to the [chirper.ai](https://chirper.ai) team!

Thanks to Clay from [gpus.llm-utils.org](llm-utils)!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Alicia Loh, Stephen Murray, K, Ajan Kanaga, RoA, Magnesian, Deo Leter, Olakabola, Eugene Pentland, zynix, Deep Realms, Raymond Fosdick, Elijah Stavena, Iucharbius, Erik Bjäreholt, Luis Javier Navarrete Lozano, Nicholas, theTransient, John Detwiler, alfie_i, knownsqashed, Mano Prime, Willem Michiel, Enrico Ros, LangChain4j, OG, Michael Dempsey, Pierre Kircher, Pedro Madruga, James Bentley, Thomas Belote, Luke @flexchar, Leonard Tan, Johann-Peter Hartmann, Illia Dulskyi, Fen Risland, Chadd, S_X, Jeff Scroggin, Ken Nordquist, Sean Connelly, Artur Olbinski, Swaroop Kallakuri, Jack West, Ai Maven, David Ziegler, Russ Johnson, transmissions 11, John Villwock, Alps Aficionado, Clay Pascal, Viktor Bowallius, Subspace Studios, Rainer Wilmers, Trenton Dambrowitz, vamX, Michael Levine, 준교 김, Brandon Frisco, Kalila, Trailburnt, Randy H, Talal Aujan, Nathan Dryer, Vadim, 阿明, ReadyPlayerEmma, Tiffany J. Kim, George Stoitzev, Spencer Kim, Jerry Meng, Gabriel Tamborski, Cory Kujawski, Jeffrey Morgan, Spiking Neurons AB, Edmond Seymore, Alexandros Triantafyllidis, Lone Striker, Cap'n Zoog, Nikolai Manek, danny, ya boyyy, Derek Yates, usrbinkat, Mandus, TL, Nathan LeClaire, subjectnull, Imad Khwaja, webtim, Raven Klaugh, Asp the Wyvern, Gabriel Puliatti, Caitlyn Gatomon, Joseph William Delisle, Jonathan Leane, Luke Pendergrass, SuperWojo, Sebastain Graf, Will Dee, Fred von Graf, Andrey, Dan Guido, Daniel P. Andersen, Nitin Borwankar, Elle, Vitor Caleffi, biorpg, jjj, NimbleBox.ai, Pieter, Matthew Berman, terasurfer, Michael Davis, Alex, Stanislav Ovsiannikov


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

<!-- original-model-card start -->
# Original model card: Meta's CodeLlama 34B Instruct

# **Code Llama**
Code Llama is a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 34 billion parameters. This is the repository for the 34B instruct-tuned version in the Hugging Face Transformers format. This model is designed for general code synthesis and understanding. Links to other models can be found in the index at the bottom.

|     | Base Model                                                                    | Python                                                                                      | Instruct                                                                                        |
| --- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 7B  | [codellama/CodeLlama-7b-hf](https://huggingface.co/codellama/CodeLlama-7b-hf) | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf) | [codellama/CodeLlama-7b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) |
| 13B  | [codellama/CodeLlama-13b-hf](https://huggingface.co/codellama/CodeLlama-13b-hf) | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf) | [codellama/CodeLlama-13b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-13b-Instruct-hf) |
| 34B  | [codellama/CodeLlama-34b-hf](https://huggingface.co/codellama/CodeLlama-34b-hf) | [codellama/CodeLlama-34b-Python-hf](https://huggingface.co/codellama/CodeLlama-34b-Python-hf) | [codellama/CodeLlama-34b-Instruct-hf](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf) |

## Model Use

To use this model, please make sure to install transformers from `main` until the next version is released:

```bash
pip install git+https://github.com/huggingface/transformers.git@main accelerate
```

Model capabilities:

- [x] Code completion.
- [ ] Infilling.
- [x] Instructions / chat.
- [ ] Python specialist.

## Model Details
*Note: Use of this model is governed by the Meta license. Meta developed and publicly released the Code Llama family of large language models (LLMs).

**Model Developers** Meta

**Variations** Code Llama comes in three model sizes, and three variants:

* Code Llama: base models designed for general code synthesis and understanding
* Code Llama - Python: designed specifically for Python
* Code Llama - Instruct: for instruction following and safer deployment

All variants are available in sizes of 7B, 13B and 34B parameters.

**This repository contains the Instruct version of the 34B parameters model.**

**Input** Models input text only.

**Output** Models generate text only.

**Model Architecture** Code Llama is an auto-regressive language model that uses an optimized transformer architecture.

**Model Dates** Code Llama and its variants have been trained between January 2023 and July 2023.

**Status** This is a static model trained on an offline dataset. Future versions of Code Llama - Instruct will be released as we improve model safety with community feedback.

**License** A custom commercial license is available at: [https://ai.meta.com/resources/models-and-libraries/llama-downloads/](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

**Research Paper** More information can be found in the paper "[Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)" or its [arXiv page](https://arxiv.org/abs/2308.12950).

## Intended Use
**Intended Use Cases** Code Llama and its variants is intended for commercial and research use in English and relevant programming languages. The base model Code Llama can be adapted for a variety of code synthesis and understanding tasks, Code Llama - Python is designed specifically to handle the Python programming language, and Code Llama - Instruct is intended to be safer to use for code assistant and generation applications.

**Out-of-Scope Uses** Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in languages other than English. Use in any other way that is prohibited by the Acceptable Use Policy and Licensing Agreement for Code Llama and its variants.

## Hardware and Software
**Training Factors** We used custom training libraries. The training and fine-tuning of the released models have been performed Meta’s Research Super Cluster.

**Carbon Footprint** In aggregate, training all 9 Code Llama models required 400K GPU hours of computation on hardware of type A100-80GB (TDP of 350-400W). Estimated total emissions were 65.3 tCO2eq, 100% of which were offset by Meta’s sustainability program.

## Training Data

All experiments reported here and the released models have been trained and fine-tuned using the same data as Llama 2 with different weights (see Section 2 and Table 1 in the [research paper](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/) for details).

## Evaluation Results

See evaluations for the main models and detailed ablations in Section 3 and safety evaluations in Section 4 of the research paper.


## Ethical Considerations and Limitations

Code Llama and its variants are a new technology that carries risks with use. Testing conducted to date has been in English, and has not covered, nor could it cover all scenarios. For these reasons, as with all LLMs, Code Llama’s potential outputs cannot be predicted in advance, and the model may in some instances produce inaccurate or objectionable responses to user prompts. Therefore, before deploying any applications of Code Llama, developers should perform safety testing and tuning tailored to their specific applications of the model.

Please see the Responsible Use Guide available available at [https://ai.meta.com/llama/responsible-user-guide](https://ai.meta.com/llama/responsible-user-guide).

<!-- original-model-card end -->
