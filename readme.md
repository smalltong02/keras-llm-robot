# Keras-llm-robot Web UI

🌍 [READ THIS IN CHINESE](readme-cn.md)

The project inherits from the Langchain-Chatchat project(https://github.com/chatchat-space/Langchain-Chatchat) The underlying architecture uses open-source frameworks such as Langchain and Fastchat, with the top layer implemented in Streamlit. The project is completely open-source, aiming for offline deployment and testing of most open-source models from the Hugging Face website. Additionally, it allows combining multiple models through configuration to achieve multimodal, RAG, Agent, and other functionalities.

---

## Table of Contents
* [Quick Start](readme.md#Quick-Start)
* [Video Demonstration](readme.md#Video-Demonstration)
* [Project Introduction](readme.md#Project-Introduction)
* [Environment Setup](readme.md#Environment-Setup)
* [Feature Overview](readme.md#Feature-Overview)
    * [Interface Overview](readme.md#Interface-Overview)
    * [Language Model Features](readme.md#Language-Model-Features)
      * [1. Load Model](readme.md#1-Load-Model)
      * [2. Quantization](readme.md#2-Quantization)
      * [3. Fine-tuning](readme.md#3-Fine-tuning)
      * [4. Prompt Templates](readme.md#4-Prompt-Templates)
    * [Auxiliary Model Features](readme.md#Auxiliary-Model-Features)
      * [1. Retrieval](readme.md#1-Retrieval)
      * [2. Code Interpreter](readme.md#2-Code-Interpreter)
      * [3. Speech Recognition](readme.md#3-Speech-Recognition)
      * [4. Image Recognition](readme.md#4-Image-Recognition)
      * [5. Function Calling](readme.md#5-Function-Calling)


## Quick Start
  
  Please first prepare the runtime environment, refer to [Environment Setup](readme.md#Environment-Setup)

  If deploying locally, you can start the Web UI using Python with an HTTP interface at http://127.0.0.1:8818
  ```bash
  python __webgui_server__.py --webui
  ```

  If deploying on a cloud server and accessing the Web UI locally, Please use reverse proxy and start the Web UI with HTTPS. Access using https://127.0.0.1:4480 on locally, and use the https interface at https://[server ip]:4480 on remotely:
  ```bash
  // By default, the batch file uses the virtual environment named keras-llm-robot,
  // Modify the batch file if using a different virtual environment name.

  // windows platform
  webui-startup-windows.bat

  // ubuntu(linux) platform
  python __webgui_server__.py --webui
  chmod +x ./tools/ssl-proxy-linux
  ./tools/ssl-proxy-linux -from 0.0.0.0:4480 -to 127.0.0.1:8818

  // MacOS platform
  python __webgui_server__.py --webui
  chmod +x ./tools/ssl-proxy-darwin
  ./tools/ssl-proxy-darwin -from 0.0.0.0:4480 -to 127.0.0.1:8818
  ```

## Video Demonstration

  1. The demonstration utilizes a multimodal online model GPT-4-vision-preview along with Azure Speech to Text services:

  [![Alt text](https://img.youtube.com/vi/7VzZqgg35Ak/0.jpg)](https://www.youtube.com/watch?v=7VzZqgg35Ak)

  2. The demonstration gpt-4-vision-preview VS Gemini-pro-vision：
   
  [![Alt text](https://img.youtube.com/vi/yFK62Tn_f4Q/0.jpg)](https://www.youtube.com/watch?v=yFK62Tn_f4Q)


## Project Introduction
Consists of three main interfaces: the chat interface for language models, the configuration interface for language models, and the tools and agent interface for auxiliary models.

Chat Interface:
![Image1](./img/WebUI.png)
The language model is the foundation model that can be used in chat mode after loading. It also serves as the brain in multimodal features. Auxiliary models, such as voice, image, and retrieval models, require language models to process their input or output text. The voice model like to ear and mouth, the image model like to eye, and the retrieval model provides long-term memory. The project currently supports dozens of language models.

Configuration Interface:
![Image1](./img/Configuration.png)
Models can be loaded based on requirements, categorized into general, multimodal, special, and online models.

Tools & Agent Interface：
![Image1](./img/Tools_Agent.png)
Auxiliary models, such as retrieval, code execution, text-to-speech, speech-to-text, image recognition, and image generation, it can be loaded based on requirements. The tools section includes settings for function calls (requires language model support for function calling).

## Environment Setup

  1. Install Anaconda or Miniconda and Git. Windows users also need to install the CMake tool, Ubuntu users need to install gcc tools.
   
  2. Create a virtual environment named keras-llm-robot using conda and install Python of 3.10 or 3.11:
  ```bash
  conda create -n keras-llm-robot python==3.11.5
  ```

  3. Clone the repository:
  ```bash
  git clone https://github.com/smalltong02/keras-llm-robot.git
  cd keras-llm-robot
  ```

  4. Activate the virtual environment:
  ```bash
  conda activate keras-llm-robot
  ```

  5. If you have an NVIDIA GPU, Please install the CUDA Toolkit from (https://developer.nvidia.com/cuda-toolkit-archive), and install the PyTorch CUDA version in the virtual environment (same to the CUDA Toolkit version https://pytorch.org/):
  ```bash
  // such as install version 12.1
  conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  ```

  6. Install dependencies, Please choose the appropriate requirements file based on your platform, On the Windows, if encounter compilation errors for llama-cpp-python or tts during the installation, please remove these two packages from the requirements:
  ```bash
  // windows
  pip install -r requirements-windows.txt
  // Ubuntu
  pip install -r requirements-ubuntu.txt
  // MacOS
  pip install -r requirements-macos.txt
  ```

  7. If speech feature is required, you also need to install the ffmpeg tool.

    // For Windows:
    Download the Windows binary package of ffmpeg from (https://www.gyan.dev/ffmpeg/builds/).
    Add the bin directory to the system PATH environment variable.

    // for ubuntu, install ffmpeg and pyaudio
    sudo apt install ffmpeg
    sudo apt-get install portaudio19-dev

    // For MacOS
    ```bash
    # Using libav
    brew install libav

    ####    OR    #####

    # Using ffmpeg
    brew install ffmpeg
    ```

  8. If you need to download models from Hugging Face for offline execution, please download the models yourself and place them in the "models" directory. If the models have not been downloaded in advance, the WebUI will automatically download them from the Hugging Face website to the local system cache.
  ```bash
  // such as the folder of llama-2-7b-chat model:
  models\llm\Llama-2-7b-chat-hf

  // such as the folder of XTTS-v2 speech-to-text model:
  models\voices\XTTS-v2

  // such as the folder of faster-whisper-large-v3 text-to-speech model:
  models\voices\faster-whisper-large-v3
  ```

  9. If run locally, start the Web UI using Python at http://127.0.0.1:8818:
  ```bash
  python __webgui_server__.py --webui
  ```

  10. If deploying on a cloud server and accessing the Web UI locally, use reverse proxy and start the Web UI with HTTPS. Access using https://127.0.0.1:4480 on locally, and use the https interface at https://[server ip]:4480 on remotely:
  ```bash
  // By default, the batch file uses the virtual environment named keras-llm-robot,
  // Modify the batch file if using a different virtual environment name.

  webui-startup-windows.bat
  
  // ubuntu(linux)平台

  python __webgui_server__.py --webui
  chmod +x ./tools/ssl-proxy-linux
  ./tools/ssl-proxy-linux -from 0.0.0.0:4480 -to 127.0.0.1:8818

  // MacOS平台

  python __webgui_server__.py --webui
  chmod +x ./tools/ssl-proxy-darwin
  ./tools/ssl-proxy-darwin -from 0.0.0.0:4480 -to 127.0.0.1:8818
  ```

## Feature Overview

### Interface Overview

- #### Configuration Interface

    In the configuration interface, you can choose suitable language models to load, categorized as `Foundation Models`, `Multimodal Models`, `Special Models`, and `Online Models`.

  1. **`Foundation Models`** Untouched models published on Hugging Face, supporting models with chat templates similar to OpenAI.
  2. **`Multimodal Models`** (`Not implemented`): Models supporting both voice and text or image and text at the lower level.
  3. **`Special Models`** Quantized models (GGUF) published on Hugging Face or models requiring special chat templates.
  4. **`Online Models`** Supports online language models from OpenAI and Google, such as GPT4-Turbo, Gemini-Pro, GPT4-vision, and Gemini-Pro-vision. Requires OpenAI API Key and Google API Key, which can be configured in the system environment variables or in the configuration interface.


![Image1](./img/Configuration.png)


- #### Tools & Agent Interface

  In the tools & agent interface, you can load auxiliary models such as retrieval, code execution, text-to-speech, speech-to-text, image recognition, image generation, or function calling.

  1. **`Retrieval`** Supports both local and online vector databases, local and online embedding models, and various document types. Can provide long-term memory for the Foundation model.
  2. **`Code Interpreter`** (`Not implemented`)
  3. **`Text-to-Speech`** Supports local model XTTS-v2 and Azure online text-to-speech service. Requires Azure API Key, which can be configured in the system environment variables `SPEECH_KEY` and `SPEECH_REGION`, or in the configuration interface.
  4. **`Speech-to-Text`** Supports local models whisper and fast-whisper and Azure online speech-to-text service. Requires Azure API Key, which can be configured in the system environment variables `SPEECH_KEY` and `SPEECH_REGION`, or in the configuration interface.
  5. **`Image Recognition`** (`Not implemented`)
  6. **`Image Generation`** (`Not implemented`)
  7. **`Function Calling`** (`Not implemented`)

![Image](./img/Tools_Agent.png)


  Once the speech-to-text model is loaded, voice and video chat controls will appear in the chat interface. Click the `START` button to record voice via the microphone and the `STOP` button to end the voice recording. The speech model will automatically convert the speech to text and engage in conversation with the language model. When the text-to-speech model is loaded, the text output by the language model will automatically be converted to speech and output through speakers and headphones.

![Image](./img/Chat_by_voice.png)

  Once the Multimodal model is loaded(such as Gemini-Pro-Vision)，upload controls will appear in the chat interface, The restrictions on uploading files depend on the loaded model. After sending text in the chatbox, both uploaded files and text will be forwarded to the multimodal model for processing.

![Image](./img/Chat_by_upload.png)


- ### Language Model Features

  1. **`Load Model`**
      
      **Foundation Models** can be loaded with CPU or GPU, and with 8-bits loading (`4-bits is invalid`). Set the appropriate CPU Threads to improve token output speed when using CPU. When encountering the error 'Using Exllama backend requires all the modules to be on GPU' while loading the GPTQ model, please add "'disable_exllama': true" in the 'quantization_config' section of the model's config.json.
      
      **Multimodal models** can be loaded with CPU or GPU. For Vision models, users can upload images and text for model interaction. For Voice models, users can interact with the model using a microphone (without the need for auxiliary models). (`Not implemented`)

      **Special models** can be loaded with CPU or GPU, Please prioritize CPU loading of GGUF models.

      **Online models** do not require additional local resources and currently support online language models from OpenAI and Google.


      **`NOTE`** When the TTS library is not installed, XTTS-2 local speech models cannot be loaded, but other online speech services can still be used. If the llama-cpp-python library is not installed, the GGUF model cannot be loaded. Without a GPU device, AWQ and GPTQ models cannot be loaded.

      | Supported Models | Model Type | Size |
      | :---- | :---- | :---- |
      | fastchat-t5-3b-v1.0 | LLM Model | 3B |
      | llama-2-7b-hf | LLM Model | 7B |
      | llama-2-7b-chat-hf | LLM Model | 7B |
      | chatglm2-6b | LLM Model | 7B |
      | chatglm2-6b-32k | LLM Model | 7B |
      | chatglm3-6b | LLM Model | 7B |
      | tigerbot-7b-chat | LLM Model | 7B |
      | openchat_3.5 | LLM Model | 7B |
      | Qwen-7B-Chat-Int4 | LLM Model | 7B |
      | fuyu-8b | LLM Model | 7B |
      | Yi-6B-Chat-4bits | LLM Model | 7B |
      | neural-chat-7b-v3-1 | LLM Model | 7B |
      | Mistral-7B-Instruct-v0.2 | LLM Model | 7B |
      | llama-2-13b-hf | LLM Model | 13B |
      | llama-2-13b-chat-hf | LLM Model | 13B |
      | tigerbot-13b-chat | LLM Model | 13B |
      | Qwen-14B-Chat | LLM Model | 13B |
      | Qwen-14B-Chat-Int4 | LLM Model | 13B |
      | Yi-34B-Chat-4bits | LLM Model | 34B |
      | llama-2-70b-hf | LLM Model | 70B |
      | llama-2-70b-chat-hf | LLM Model | 70B |
      | visualglm-6b| Multimodal Model (image) | 7B |
      | cogvlm-chat-hf | Multimodal Model (image) | 7B |
      | mplug-owl2-llama2-7b | Multimodal Model (image) | 7B |
      | Qwen-VL-Chat-Int4 | Multimodal Model (image) | 7B |
      | internlm-xcomposer-7b-4bit | Multimodal Model (image) | 7B |
      | phi-2-gguf | Special Model | 3B |
      | phi-2 | Special Model | 3B |
      | Yi-6B-Chat-gguf | Special Model | 7B |
      | OpenHermes-2.5-Mistral-7B | Special Model | 7B |
      | Yi-34B-Chat-gguf | Special Model | 34B |
      | Mixtral-8x7B-v0.1-gguf | Special Model | 8*7B |
      | gpt-3.5-turbo | Online Model | *B |
      | gpt-3.5-turbo-16k | Online Model | *B |
      | gpt-4 | Online Model | *B |
      | gpt-4-32k | Online Model | *B |
      | gpt-4-1106-preview | Online Model | *B |
      | gpt-4-vision-preview | Online Model | *B |
      | gemini-pro | Online Model | *B |
      | gemini-pro-vision | Online Model | *B |
      | chat-bison-001 | Online Model | *B |
      | text-bison-001 | Online Model | *B |
      | whisper-base | Voice Model | *B |
      | whisper-medium | Voice Model | *B |
      | whisper-large-v3 | Voice Model | *B |
      | faster-whisper-large-v3 | Voice Model | *B |
      | AzureVoiceService | Voice Model | *B |
      | XTTS-v2 | Speech Model | *B |
      | AzureSpeechService | Speech Model | *B |
      | OpenAISpeechService | Speech Model | *B |


  2. **`Quantization`**

      Use open-source tools like llama.cpp to create quantized versions of general models with 2, 3, 4, 5, 6, and 8 bits. `Not implemented`

  3. **`Fine-tuning`**

      You can fine-tune the language model using a private dataset. `Not implemented`

  4. **`Prompt Templates`**

      Set up a template for prompting the language model in specific scenarios. `Not implemented`
  
- ### Auxiliary Model Features

  1. **`Retrieval`**

      RAG functionality requires a vector database and embedding models to provide long-term memory capabilities to the language model. 

      Support the following Vector Database:

      | Databases | Type |
      | :---- | :---- |
      | Faiss | Local |
      | Milvus | Local |
      | PGVector | Local |
      | ElasticsearchStore | Local |
      | ZILLIZ | Online |

      Support the following Embedding Models:

      | Model | Type | Size |
      | :---- | :---- | :---- |
      | bge-small-en-v1.5 | Local | 130MB |
      | bge-base-en-v1.5 | Local | 430MB |
      | bge-large-en-v1.5 | Local | 1.3GB |
      | jina-embeddings-v2-small-en | Local | 63MB |
      | jina-embeddings-v2-base-en | Local | 260MB |
      | m3e-small | Local | 93MB |
      | m3e-base | Local | 400MB |
      | m3e-large | Local | 1.3GB |
      | text2vec-base-chinese | Local | 400MB |
      | text2vec-bge-large-chinese | Local | 1.3GB |
      | text-embedding-ada-002 | Online | *B |
      | embedding-gecko-001 | Online | *B |
      | embedding-001 | Online | *B |

      Support the following Documents:

      html, mhtml, md, json, jsonl, csv, pdf, png, jpg, jpeg, bmp, eml, msg, epub, xlsx, xls, xlsd, ipynb, odt, py, rst, rtf, srt, toml, tsv, docx, doc, xml, ppt, pptx, enex, txt

  2. **`Code Interpreter`**

      Enable code execution capability for the language model to empower it with actionable functionality for the mind. `Not implemented`

  3. **`Speech Recognition`**

      Provide the language model with speech input and output capabilities, adding the functions of listening and speaking to the mind. Support local models such as XTTS-v2 and Whisper, as well as integration with Azure online speech services.

  4. **`Image Recognition`**

      Provide the language model with input and output capabilities for images and videos, adding the functions of sight and drawing to the mind. `Not implemented`

  5. **`Function Calling`**

      Provide the language model with function calling capability, empowering the mind with the ability to use tools. Anticipated support for automation platforms such as Zapier, n8n, and others. `Not implemented`

## Note

Anaconda download：(https://www.anaconda.com/download)

Git download：(https://git-scm.com/downloads)

CMake download：(https://cmake.org/download/)

Langchain Project: (https://github.com/langchain-ai/langchain)

Fastchat Project: (https://github.com/lm-sys/FastChat)