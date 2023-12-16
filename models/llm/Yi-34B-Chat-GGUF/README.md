---
base_model: 01-ai/Yi-34B-Chat
inference: false
license: other
license_link: LICENSE
license_name: yi-license
model_creator: 01-ai
model_name: Yi 34B Chat
model_type: yi
pipeline_tag: text-generation
prompt_template: '<|im_start|>system

  {system_message}<|im_end|>

  <|im_start|>user

  {prompt}<|im_end|>

  <|im_start|>assistant

  '
quantized_by: TheBloke
widget:
- example_title: Yi-34B-Chat
  output:
    text: ' Hello! How can I assist you today?'
  text: hi
- example_title: Yi-34B
  output:
    text: " an eerie sense that something is just not right\u2026\nBetween the two\
      \ worlds lies The Forgotten Kingdom - home to creatures long since thought extinct\
      \ and ancient magic so strong it defies belief! Only here can you find what\
      \ has been lost for centuries: An Elixir Of Life which will restore youth and\
      \ vitality if only those who seek its power are brave enough to face up against\
      \ all manner of dangers lurking in this mysterious land! But beware; some say\
      \ there may even exist powerful entities beyond our comprehension whose intentions\
      \ towards humanity remain unclear at best ---- they might want nothing more\
      \ than destruction itself rather then anything else from their quest after immortality\
      \ (and maybe someone should tell them about modern medicine)? In any event though\
      \ \u2013 one thing remains true regardless : whether or not success comes easy\
      \ depends entirely upon how much effort we put into conquering whatever challenges\
      \ lie ahead along with having faith deep down inside ourselves too ;) So let\u2019\
      s get started now shall We?"
  text: There's a place where time stands still. A place of breath taking wonder,
    but also
---
<!-- markdownlint-disable MD041 -->

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

# Yi 34B Chat - GGUF
- Model creator: [01-ai](https://huggingface.co/01-ai)
- Original model: [Yi 34B Chat](https://huggingface.co/01-ai/Yi-34B-Chat)

<!-- description start -->
## Description

This repo contains GGUF format model files for [01-ai's Yi 34B Chat](https://huggingface.co/01-ai/Yi-34B-Chat).

These files were quantised using hardware kindly provided by [Massed Compute](https://massedcompute.com/).

<!-- description end -->
<!-- README_GGUF.md-about-gguf start -->
### About GGUF

GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

Here is an incomplete list of clients and libraries that are known to support GGUF:

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

* [AWQ model(s) for GPU inference.](https://huggingface.co/TheBloke/Yi-34B-Chat-AWQ)
* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Yi-34B-Chat-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGUF models for CPU+GPU inference](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF)
* [01-ai's original unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/01-ai/Yi-34B-Chat)
<!-- repositories-available end -->

<!-- prompt-template start -->
## Prompt template: ChatML

```
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant

```

<!-- prompt-template end -->


<!-- compatibility_gguf start -->
## Compatibility

These quantised GGUFv2 files are compatible with llama.cpp from August 27th onwards, as of commit [d0cee0d](https://github.com/ggerganov/llama.cpp/commit/d0cee0d36d5be95a0d9088b674dbb27354107221)

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
| [yi-34b-chat.Q2_K.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q2_K.gguf) | Q2_K | 2 | 14.56 GB| 17.06 GB | smallest, significant quality loss - not recommended for most purposes |
| [yi-34b-chat.Q3_K_S.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q3_K_S.gguf) | Q3_K_S | 3 | 14.96 GB| 17.46 GB | very small, high quality loss |
| [yi-34b-chat.Q3_K_M.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q3_K_M.gguf) | Q3_K_M | 3 | 16.64 GB| 19.14 GB | very small, high quality loss |
| [yi-34b-chat.Q3_K_L.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q3_K_L.gguf) | Q3_K_L | 3 | 18.14 GB| 20.64 GB | small, substantial quality loss |
| [yi-34b-chat.Q4_0.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q4_0.gguf) | Q4_0 | 4 | 19.47 GB| 21.97 GB | legacy; small, very high quality loss - prefer using Q3_K_M |
| [yi-34b-chat.Q4_K_S.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q4_K_S.gguf) | Q4_K_S | 4 | 19.54 GB| 22.04 GB | small, greater quality loss |
| [yi-34b-chat.Q4_K_M.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q4_K_M.gguf) | Q4_K_M | 4 | 20.66 GB| 23.16 GB | medium, balanced quality - recommended |
| [yi-34b-chat.Q5_0.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q5_0.gguf) | Q5_0 | 5 | 23.71 GB| 26.21 GB | legacy; medium, balanced quality - prefer using Q4_K_M |
| [yi-34b-chat.Q5_K_S.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q5_K_S.gguf) | Q5_K_S | 5 | 23.71 GB| 26.21 GB | large, low quality loss - recommended |
| [yi-34b-chat.Q5_K_M.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q5_K_M.gguf) | Q5_K_M | 5 | 24.32 GB| 26.82 GB | large, very low quality loss - recommended |
| [yi-34b-chat.Q6_K.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q6_K.gguf) | Q6_K | 6 | 28.21 GB| 30.71 GB | very large, extremely low quality loss |
| [yi-34b-chat.Q8_0.gguf](https://huggingface.co/TheBloke/Yi-34B-Chat-GGUF/blob/main/yi-34b-chat.Q8_0.gguf) | Q8_0 | 8 | 36.54 GB| 39.04 GB | very large, extremely low quality loss - not recommended |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.



<!-- README_GGUF.md-provided-files end -->

<!-- README_GGUF.md-how-to-download start -->
## How to download GGUF files

**Note for manual downloaders:** You almost never want to clone the entire repo! Multiple different quantisation formats are provided, and most users only want to pick and download a single file.

The following clients/libraries will automatically download models for you, providing a list of available models to choose from:

* LM Studio
* LoLLMS Web UI
* Faraday.dev

### In `text-generation-webui`

Under Download Model, you can enter the model repo: TheBloke/Yi-34B-Chat-GGUF and below it, a specific filename to download, such as: yi-34b-chat.Q4_K_M.gguf.

Then click Download.

### On the command line, including multiple files at once

I recommend using the `huggingface-hub` Python library:

```shell
pip3 install huggingface-hub
```

Then you can download any individual model file to the current directory, at high speed, with a command like this:

```shell
huggingface-cli download TheBloke/Yi-34B-Chat-GGUF yi-34b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

<details>
  <summary>More advanced huggingface-cli download usage</summary>

You can also download multiple files at once with a pattern:

```shell
huggingface-cli download TheBloke/Yi-34B-Chat-GGUF --local-dir . --local-dir-use-symlinks False --include='*Q4_K*gguf'
```

For more documentation on downloading with `huggingface-cli`, please see: [HF -> Hub Python Library -> Download files -> Download from the CLI](https://huggingface.co/docs/huggingface_hub/guides/download#download-from-the-cli).

To accelerate downloads on fast connections (1Gbit/s or higher), install `hf_transfer`:

```shell
pip3 install hf_transfer
```

And set environment variable `HF_HUB_ENABLE_HF_TRANSFER` to `1`:

```shell
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/Yi-34B-Chat-GGUF yi-34b-chat.Q4_K_M.gguf --local-dir . --local-dir-use-symlinks False
```

Windows Command Line users: You can set the environment variable by running `set HF_HUB_ENABLE_HF_TRANSFER=1` before the download command.
</details>
<!-- README_GGUF.md-how-to-download end -->

<!-- README_GGUF.md-how-to-run start -->
## Example `llama.cpp` command

Make sure you are using `llama.cpp` from commit [d0cee0d](https://github.com/ggerganov/llama.cpp/commit/d0cee0d36d5be95a0d9088b674dbb27354107221) or later.

```shell
./main -ngl 32 -m yi-34b-chat.Q4_K_M.gguf --color -c 2048 --temp 0.7 --repeat_penalty 1.1 -n -1 -p "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
```

Change `-ngl 32` to the number of layers to offload to GPU. Remove it if you don't have GPU acceleration.

Change `-c 2048` to the desired sequence length. For extended sequence models - eg 8K, 16K, 32K - the necessary RoPE scaling parameters are read from the GGUF file and set by llama.cpp automatically.

If you want to have a chat-style conversation, replace the `-p <PROMPT>` argument with `-i -ins`

For other parameters and how to use them, please refer to [the llama.cpp documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/main/README.md)

## How to run in `text-generation-webui`

Further instructions can be found in the text-generation-webui documentation, here: [text-generation-webui/docs/04 ‐ Model Tab.md](https://github.com/oobabooga/text-generation-webui/blob/main/docs/04%20%E2%80%90%20Model%20Tab.md#llamacpp).

## How to run from Python code

You can use GGUF models from Python using the [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) or [ctransformers](https://github.com/marella/ctransformers) libraries.

### How to load this model in Python code, using ctransformers

#### First install the package

Run one of the following commands, according to your system:

```shell
# Base ctransformers with no GPU acceleration
pip install ctransformers
# Or with CUDA GPU acceleration
pip install ctransformers[cuda]
# Or with AMD ROCm GPU acceleration (Linux only)
CT_HIPBLAS=1 pip install ctransformers --no-binary ctransformers
# Or with Metal GPU acceleration for macOS systems only
CT_METAL=1 pip install ctransformers --no-binary ctransformers
```

#### Simple ctransformers example code

```python
from ctransformers import AutoModelForCausalLM

# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Yi-34B-Chat-GGUF", model_file="yi-34b-chat.Q4_K_M.gguf", model_type="yi", gpu_layers=50)

print(llm("AI is going to"))
```

## How to use with LangChain

Here are guides on using llama-cpp-python and ctransformers with LangChain:

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

**Patreon special mentions**: Brandon Frisco, LangChain4j, Spiking Neurons AB, transmissions 11, Joseph William Delisle, Nitin Borwankar, Willem Michiel, Michael Dempsey, vamX, Jeffrey Morgan, zynix, jjj, Omer Bin Jawed, Sean Connelly, jinyuan sun, Jeromy Smith, Shadi, Pawan Osman, Chadd, Elijah Stavena, Illia Dulskyi, Sebastain Graf, Stephen Murray, terasurfer, Edmond Seymore, Celu Ramasamy, Mandus, Alex, biorpg, Ajan Kanaga, Clay Pascal, Raven Klaugh, 阿明, K, ya boyyy, usrbinkat, Alicia Loh, John Villwock, ReadyPlayerEmma, Chris Smitley, Cap'n Zoog, fincy, GodLy, S_X, sidney chen, Cory Kujawski, OG, Mano Prime, AzureBlack, Pieter, Kalila, Spencer Kim, Tom X Nguyen, Stanislav Ovsiannikov, Michael Levine, Andrey, Trailburnt, Vadim, Enrico Ros, Talal Aujan, Brandon Phillips, Jack West, Eugene Pentland, Michael Davis, Will Dee, webtim, Jonathan Leane, Alps Aficionado, Rooh Singh, Tiffany J. Kim, theTransient, Luke @flexchar, Elle, Caitlyn Gatomon, Ari Malik, subjectnull, Johann-Peter Hartmann, Trenton Dambrowitz, Imad Khwaja, Asp the Wyvern, Emad Mostaque, Rainer Wilmers, Alexandros Triantafyllidis, Nicholas, Pedro Madruga, SuperWojo, Harry Royden McLaughlin, James Bentley, Olakabola, David Ziegler, Ai Maven, Jeff Scroggin, Nikolai Manek, Deo Leter, Matthew Berman, Fen Risland, Ken Nordquist, Manuel Alberto Morcote, Luke Pendergrass, TL, Fred von Graf, Randy H, Dan Guido, NimbleBox.ai, Vitor Caleffi, Gabriel Tamborski, knownsqashed, Lone Striker, Erik Bjäreholt, John Detwiler, Leonard Tan, Iucharbius


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

<!-- original-model-card start -->
# Original model card: 01-ai's Yi 34B Chat



<div align="center">

<p align="center">
<img width="200px" src="https://github.com/01-ai/Yi/raw/main/assets/img/Yi.svg?sanitize=true">
</p>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://github.com/01-ai/Yi/issues">
  <img src="https://img.shields.io/github/issues/01-ai/Yi?logo=github" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml">
<img src="https://github.com/01-ai/Yi/actions/workflows/build_docker_image.yml/badge.svg" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a href="https://huggingface.co/01-ai">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-01--ai-blue" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://www.modelscope.cn/organization/01ai/">
  <img src="https://img.shields.io/badge/ModelScope-01--ai-blue" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://wisemodel.cn/organization/01.AI">
  <img src="https://img.shields.io/badge/WiseModel-01--ai-blue" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://replicate.com/01-ai">
  <img src="https://img.shields.io/badge/Replicate-01--ai-blue?logo=data:image/svg%2bxml;base64,PHN2ZyB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4IiB2aWV3Qm94PSIwIDAgMTAwMCAxMDAwIiBjbGFzcz0ibG9nbyIgZmlsbD0iY3VycmVudENvbG9yIiB4bWw6c3BhY2U9InByZXNlcnZlIj4KICA8Zz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMTAwMCw0MjcuNiAxMDAwLDU0MC42IDYwMy40LDU0MC42IDYwMy40LDEwMDAgNDc3LDEwMDAgNDc3LDQyNy42IAkiPjwvcG9seWdvbj4KICAgIDxwb2x5Z29uIHBvaW50cz0iMTAwMCwyMTMuOCAxMDAwLDMyNyAzNjQuOCwzMjcgMzY0LjgsMTAwMCAyMzguNCwxMDAwIDIzOC40LDIxMy44IAkiPjwvcG9seWdvbj4KICAgIDxwb2x5Z29uIHBvaW50cz0iMTAwMCwwIDEwMDAsMTEzLjIgMTI2LjQsMTEzLjIgMTI2LjQsMTAwMCAwLDEwMDAgMCwwIAkiPjwvcG9seWdvbj4KICA8L2c+Cjwvc3ZnPg=="  style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://github.com/01-ai/Yi/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/Code_License-Apache_2.0-lightblue" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt">
  <img src="https://img.shields.io/badge/Model_License-Model_Agreement-lightblue" style="margin: 0 0;">
</a>
</div>

<div style="display: inline-block;">
<a rel="noopener nofollow" href="mailto:oss@01.ai">
  <img src="https://img.shields.io/badge/✉️-yi@01.ai-FFE01B" style="margin: 0 0;">
</a>
</div>

</div>

## Introduction

The **Yi** series models are large language models trained from scratch by
developers at [01.AI](https://01.ai/).

## News

<details open>
<summary>🎯 <b>2023/11/23</b>: The chat models are open to public.</summary>

This release contains two chat models based on previous released base models, two 8-bits models quntinized by GPTQ, two 4-bits models quantinized by AWQ.

- `Yi-34B-Chat`
- `Yi-34B-Chat-4bits`
- `Yi-34B-Chat-8bits`
- `Yi-6B-Chat`
- `Yi-6B-Chat-4bits`
- `Yi-6B-Chat-8bits`

You can try some of them interactively at:

- [HuggingFace](https://huggingface.co/spaces/01-ai/Yi-34B-Chat)
- [Replicate](https://replicate.com/01-ai)
</details>

<details open>
<summary>🔔 <b>2023/11/23</b>: The Yi Series Models Community License Agreement is updated to v2.1.</summary>
</details>

<details>
<summary>🔥 <b>2023/11/08</b>: Invited test of Yi-34B chat model.</summary>

Application form:

- [English](https://cn.mikecrm.com/l91ODJf)
- [Chinese](https://cn.mikecrm.com/gnEZjiQ)

</details>

<details>
<summary>🎯 <b>2023/11/05</b>: The base model of <code>Yi-6B-200K</code> and <code>Yi-34B-200K</code>.</summary>

This release contains two base models with the same parameter sizes of previous
release, except that the context window is extended to 200K.

</details>

<details>
<summary>🎯 <b>2023/11/02</b>: The base model of <code>Yi-6B</code> and <code>Yi-34B</code>.</summary>

The first public release contains two bilingual (English/Chinese) base models
with the parameter sizes of 6B and 34B.  Both of them are trained with 4K
sequence length and can be extended to 32K during inference time.

</details>

## Model Performance

### Base Model Performance

| Model         |   MMLU   |  CMMLU   |  C-Eval  |  GAOKAO  |   BBH    | Common-sense Reasoning | Reading Comprehension | Math & Code |
| :------------ | :------: | :------: | :------: | :------: | :------: | :--------------------: | :-------------------: | :---------: |
|               |  5-shot  |  5-shot  |  5-shot  |  0-shot  | 3-shot@1 |           -            |           -           |      -      |
| LLaMA2-34B    |   62.6   |    -     |    -     |    -     |   44.1   |          69.9          |         68.0          |    26.0     |
| LLaMA2-70B    |   68.9   |   53.3   |    -     |   49.8   |   51.2   |          71.9          |         69.4          |    36.8     |
| Baichuan2-13B |   59.2   |   62.0   |   58.1   |   54.3   |   48.8   |          64.3          |         62.4          |    23.0     |
| Qwen-14B      |   66.3   |   71.0   |   72.1   |   62.5   |   53.4   |          73.3          |         72.5          |  **39.8**   |
| Skywork-13B   |   62.1   |   61.8   |   60.6   |   68.1   |   41.7   |          72.4          |         61.4          |    24.9     |
| InternLM-20B  |   62.1   |   59.0   |   58.8   |   45.5   |   52.5   |          78.3          |           -           |    30.4     |
| Aquila-34B    |   67.8   |   71.4   |   63.1   |    -     |    -     |           -            |           -           |      -      |
| Falcon-180B   |   70.4   |   58.0   |   57.8   |   59.0   |   54.0   |          77.3          |         68.8          |    34.0     |
| Yi-6B         |   63.2   |   75.5   |   72.0   |   72.2   |   42.8   |          72.3          |         68.7          |    19.8     |
| Yi-6B-200K    |   64.0   |   75.3   |   73.5   |   73.9   |   42.0   |          72.0          |         69.1          |    19.0     |
| **Yi-34B**    | **76.3** | **83.7** |   81.4   |   82.8   | **54.3** |        **80.1**        |         76.4          |    37.1     |
| Yi-34B-200K   |   76.1   |   83.6   | **81.9** | **83.4** |   52.7   |          79.7          |       **76.6**        |    36.3     |

While benchmarking open-source models, we have observed a disparity between the
results generated by our pipeline and those reported in public sources (e.g.
OpenCompass). Upon conducting a more in-depth investigation of this difference,
we have discovered that various models may employ different prompts,
post-processing strategies, and sampling techniques, potentially resulting in
significant variations in the outcomes. Our prompt and post-processing strategy
remains consistent with the original benchmark, and greedy decoding is employed
during evaluation without any post-processing for the generated content. For
scores that were not reported by the original authors (including scores reported
with different settings), we try to get results with our pipeline.

To evaluate the model's capability extensively, we adopted the methodology
outlined in Llama2. Specifically, we included PIQA, SIQA, HellaSwag, WinoGrande,
ARC, OBQA, and CSQA to assess common sense reasoning. SquAD, QuAC, and BoolQ
were incorporated to evaluate reading comprehension. CSQA was exclusively tested
using a 7-shot setup, while all other tests were conducted with a 0-shot
configuration. Additionally, we introduced GSM8K (8-shot@1), MATH (4-shot@1),
HumanEval (0-shot@1), and MBPP (3-shot@1) under the category "Math & Code". Due
to technical constraints, we did not test Falcon-180 on QuAC and OBQA; the score
is derived by averaging the scores on the remaining tasks. Since the scores for
these two tasks are generally lower than the average, we believe that
Falcon-180B's performance was not underestimated.

### Chat Model Performance

| Model                   | MMLU      | MMLU      | CMMLU     | CMMLU     | C-Eval(val)<sup>*</sup> | C-Eval(val)<sup>*</sup> | Truthful QA | BBH       | BBH       | GSM8k     | GSM8k     |
| ----------------------- | --------- | --------- | --------- | --------- | ----------------------- | ----------------------- | ----------- | --------- | --------- | --------- | --------- |
|                         | 0-shot    | 5-shot    | 0-shot    | 5-shot    | 0-shot                  | 5-shot                  | 0-shot      | 0-shot    | 3-shot    | 0-shot    | 4-shot    |
| LLaMA2-13B-Chat         | 50.88     | 47.33     | 27.47     | 35.08     | 27.93                   | 35.88                   | 36.84       | 32.90     | 58.22     | 36.85     | 2.73      |
| LLaMA2-70B-Chat         | 59.42     | 59.86     | 36.10     | 40.99     | 34.99                   | 41.31                   | 53.95       | 42.36     | 58.53     | 47.08     | 58.68     |
| Baichuan2-13B-Chat      | 55.09     | 50.14     | 58.64     | 59.47     | 56.02                   | 54.75                   | 48.98       | 38.81     | 47.15     | 45.72     | 23.28     |
| Qwen-14B-Chat           | 63.99     | 64.98     | 67.73     | 70.57     | 66.12                   | 70.06                   | 52.49       | 49.65     | 54.98     | 59.51     | 61.18     |
| InternLM-Chat-20B       | 55.55     | 57.42     | 53.55     | 53.75     | 51.19                   | 53.57                   | 51.75       | 42.41     | 36.68     | 15.69     | 43.44     |
| AquilaChat2-34B v1.2    | 65.15     | 66.70     | 67.51     | 70.02     | **82.99**               | **89.38**               | **64.33**   | 20.12     | 34.28     | 11.52     | 48.45     |
| Yi-6B-Chat              | 58.24     | 60.99     | 69.44     | 74.71     | 68.80                   | 74.22                   | 50.58       | 39.70     | 47.15     | 38.44     | 44.88     |
| Yi-6B-Chat-8bits(GPTQ)  | 58.29     | 60.96     | 69.21     | 74.69     | 69.17                   | 73.85                   | 49.85       | 40.35     | 47.26     | 39.42     | 44.88     |
| Yi-6B-Chat-4bits(AWQ)   | 56.78     | 59.89     | 67.70     | 73.29     | 67.53                   | 72.29                   | 50.29       | 37.74     | 43.62     | 35.71     | 38.36     |
| Yi-34B-Chat             | **67.62** | 73.46     | **79.11** | **81.34** | 77.04                   | 78.53                   | 62.43       | 51.41     | **71.74** | **71.65** | **75.97** |
| Yi-34B-Chat-8bits(GPTQ) | 66.24     | **73.69** | 79.05     | 81.23     | 76.82                   | 78.97                   | 61.84       | **52.08** | 70.97     | 70.74     | 75.74     |
| Yi-34B-Chat-4bits(AWQ)  | 65.77     | 72.42     | 78.21     | 80.50     | 75.71                   | 77.27                   | 61.84       | 48.30     | 69.39     | 70.51     | 74.00     |

We evaluated various benchmarks using both zero-shot and few-shot methods, except for TruthfulQA. Generally, the zero-shot approach is more common in chat models. Our evaluation strategy involves generating responses while following instructions explicitly or implicitly (such as using few-shot examples). We then isolate relevant answers from the generated text. Some models are not well-suited to produce output in the specific format required by instructions in few datasets, which leads to suboptimal results.

<strong>*</strong>: C-Eval results are evaluated on the validation datasets

### Quantized Chat Model Performance

We also provide both 4-bit (AWQ) and 8-bit (GPTQ) quantized Yi chat models. Evaluation results on various benchmarks have shown that the quantized models have negligible losses. Additionally, they reduce the memory footprint size. After testing different configurations of prompts and generation lengths, we highly recommend following the guidelines in the memory footprint table below when selecting a device to run our models.

|                         | batch=1 | batch=4 | batch=16 | batch=32 |
| ----------------------- | ------- | ------- | -------- | -------- |
| Yi-34B-Chat             | 65GiB   | 68GiB   | 76GiB    | >80GiB   |
| Yi-34B-Chat-8bits(GPTQ) | 35GiB   | 37GiB   | 46GiB    | 58GiB    |
| Yi-34B-Chat-4bits(AWQ)  | 19GiB   | 20GiB   | 30GiB    | 40GiB    |
| Yi-6B-Chat              | 12GiB   | 13GiB   | 15GiB    | 18GiB    |
| Yi-6B-Chat-8bits(GPTQ)  | 7GiB    | 8GiB    | 10GiB    | 14GiB    |
| Yi-6B-Chat-4bits(AWQ)   | 4GiB    | 5GiB    | 7GiB     | 10GiB    |

Note: All the numbers in the table represent the minimum recommended memory for running models of the corresponding size.

### Limitations of Chat Model

The released chat model has undergone exclusive training using Supervised Fine-Tuning (SFT). Compared to other standard chat models, our model produces more diverse responses, making it suitable for various downstream tasks, such as creative scenarios. Furthermore, this diversity is expected to enhance the likelihood of generating higher quality responses, which will be advantageous for subsequent Reinforcement Learning (RL) training.

However, this higher diversity might amplify certain existing issues, including:

- **Hallucination**: This refers to the model generating factually incorrect or nonsensical information. With the model's responses being more varied, there's a higher chance of hallucination that are not based on accurate data or logical reasoning.
- **Non-determinism in re-generation**: When attempting to regenerate or sample responses, inconsistencies in the outcomes may occur. The increased diversity can lead to varying results even under similar input conditions.
- **Cumulative Error**: This occurs when errors in the model's responses compound over time. As the model generates more diverse responses, the likelihood of small inaccuracies building up into larger errors increases, especially in complex tasks like extended reasoning, mathematical problem-solving, etc.

To achieve more coherent and consistent responses, it is advisable to adjust generation configuration parameters such as`temperature`,`top_p`, or`top_k`. These adjustments can help in the balance between creativity and coherence in the model's outputs.



## Usage

Feel free to [create an issue](https://github.com/01-ai/Yi/issues/new) if you
encounter any problem when using the **Yi** series models.

### 1. Prepare development environment

#### 1.1 Docker
The best approach to try the **Yi** series models is through Docker with GPUs. We
provide the following docker images to help you get started.

- `registry.lingyiwanwu.com/ci/01-ai/yi:latest`
- `ghcr.io/01-ai/yi:latest`

Note that the `latest` tag always points to the latest code in the `main`
branch. To test a stable version, please replace it with a specific
[tag](https://github.com/01-ai/Yi/tags).

#### 1.2 Local development environment
We use [`conda-lock`](https://github.com/conda/conda-lock) to generate fully reproducible lock files for conda environments. You can refer to [conda-lock.yml](./conda-lock.yml) for the exact versions of the dependencies. Additionally, we utilize [`micromamba`](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) for installing these dependencies.

To install the dependencies, please follow these steps:
1. Install `micromamba` by following the instructions available [here](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html).
2. Execute `micromamba install -y -n yi -f conda-lock.yml` to create a conda environment named `yi` and install the necessary dependencies.

### 2. Download the model (optional)

By default, the model weights and tokenizer will be downloaded from
[HuggingFace](https://huggingface.co/01-ai) automatically in the next step. You
can also download them manually from the following places:

- [ModelScope](https://www.modelscope.cn/organization/01ai/)
- [WiseModel](https://wisemodel.cn/organization/01.AI)

### 3. Examples

#### 3.1 Use the chat model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '01-ai/Yi-34b-Chat'

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Since transformers 4.35.0, the GPT-Q/AWQ model can be loaded using AutoModelForCausalLM.
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()

# Prompt content: "hi"
messages = [
    {"role": "user", "content": "hi"}
]

input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
output_ids = model.generate(input_ids.to('cuda'))
response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

# Model response: "Hello! How can I assist you today?"
print(response)
```

#### 3.2 Use the base model

```bash
python demo/text_generation.py
```

To reuse the downloaded models in the previous step, you can provide the extra
`--model` argument:

```bash
python demo/text_generation.py  --model /path/to/model
```

Or if you'd like to get your hands dirty:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("01-ai/Yi-34B", device_map="auto", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("01-ai/Yi-34B", trust_remote_code=True)
inputs = tokenizer("There's a place where time stands still. A place of breath taking wonder, but also", return_tensors="pt")
max_length = 256

outputs = model.generate(
    inputs.input_ids.cuda(),
    max_length=max_length,
    eos_token_id=tokenizer.eos_token_id,
    do_sample=True,
    repetition_penalty=1.3,
    no_repeat_ngram_size=5,
    temperature=0.7,
    top_k=40,
    top_p=0.8,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

<details>

<summary>Output</summary>

**Prompt**: There's a place where time stands still. A place of breath taking wonder, but also

**Generation**: There's a place where time stands still. A place of breath taking wonder, but also of great danger. A place where the very air you breathe could kill you. A place where the only way to survive is to be prepared.
The place is called the Arctic.
The Arctic is a vast, frozen wilderness. It is a place of extremes. The temperatures can drop to -40 degrees Celsius. The winds can reach speeds of 100 kilometers per hour. The sun can shine for 24 hours a day, or not at all for weeks on end.
The Arctic is also a place of great beauty. The ice and snow are a pristine white. The sky is a deep blue. The sunsets are spectacular.
But the Arctic is also a place of great danger. The ice can be treacherous. The winds can be deadly. The sun can be blinding.
The Arctic is a place where the only way to survive is to be prepared.
The Arctic is a place of extremes. The temperatures can drop to -40 degrees Celsius. The winds can reach speeds of 100 kilometers per hour. The sun can shine for 24 hours a day, or not at all for weeks on end.
The Arctic is a place of great beauty. The ice and snow are a

</details>

For more advanced usage, please refer to the
[doc](https://github.com/01-ai/Yi/tree/main/demo).

#### 3.3 Finetuning from the base model:

```bash
bash finetune/scripts/run_sft_Yi_6b.sh
```

Once finished, you can compare the finetuned model and the base model with the following command:

```bash
bash finetune/scripts/run_eval.sh
```

For more advanced usage like fine-tuning based on your custom data, please refer
the [doc](https://github.com/01-ai/Yi/tree/main/finetune).

#### 3.4 Quantization

##### GPT-Q
```bash
python quantization/gptq/quant_autogptq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

Once finished, you can then evaluate the resulting model as follows:

```bash
python quantization/gptq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

For a more detailed explanation, please read the [doc](https://github.com/01-ai/Yi/tree/main/quantization/gptq)

##### AWQ
```bash
python quantization/awq/quant_autoawq.py \
  --model /base_model                      \
  --output_dir /quantized_model            \
  --trust_remote_code
```

Once finished, you can then evaluate the resulted model as follows:

```bash
python quantization/awq/eval_quantized_model.py \
  --model /quantized_model                       \
  --trust_remote_code
```

For more detailed explanation, please read the [doc](https://github.com/01-ai/Yi/tree/main/quantization/awq)

## Ecosystem

🤗 You are encouraged to create a PR and share your awesome work built on top of
the Yi series models.

- Serving
  - [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM#supported-models): Efficiently run Yi models locally.
- Quantization
  - [TheBloke/Yi-34B-GGUF](https://huggingface.co/TheBloke/Yi-34B-GGUF)
  - [TheBloke/Yi-34B-GPTQ](https://huggingface.co/TheBloke/Yi-34B-GPTQ)
- Finetuning
  - [NousResearch/Nous-Capybara-34B](https://huggingface.co/NousResearch/Nous-Capybara-34B)

## FAQ

1. **What dataset was this trained with?**

    The dataset we use contains Chinese & English only. We used approximately 3T
    tokens. The detailed number and its construction will be described in the
    upcoming technical report.

## Disclaimer

We use data compliance checking algorithms during the training process, to
ensure the compliance of the trained model to the best of our ability. Due to
complex data and the diversity of language model usage scenarios, we cannot
guarantee that the model will generate correct, and reasonable output in all
scenarios. Please be aware that there is still a risk of the model producing
problematic outputs. We will not be responsible for any risks and issues
resulting from misuse, misguidance, illegal usage, and related misinformation,
as well as any associated data security concerns.

## License

The source code in this repo is licensed under the [Apache 2.0
license](https://github.com/01-ai/Yi/blob/main/LICENSE). The Yi series models
are fully open for academic research and free commercial usage with permission
via applications. All usage must adhere to the [Model License
Agreement 2.0](https://github.com/01-ai/Yi/blob/main/MODEL_LICENSE_AGREEMENT.txt).
To apply for the official commercial license, please contact us
([yi@01.ai](mailto:yi@01.ai)).

<!-- original-model-card end -->
