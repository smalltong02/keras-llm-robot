---
license: apache-2.0
inference: false
---

# FastChat-T5 Model Card

## Model details

**Model type:**
FastChat-T5 is an open-source chatbot trained by fine-tuning Flan-t5-xl (3B parameters) on user-shared conversations collected from ShareGPT.
It is based on an encoder-decoder transformer architecture, and can autoregressively generate responses to users' inputs. 

**Model date:**
FastChat-T5 was trained on April 2023.

**Organizations developing the model:**
The FastChat developers, primarily Dacheng Li, Lianmin Zheng and Hao Zhang.

**Paper or resources for more information:**
https://github.com/lm-sys/FastChat#FastChat-T5

**License:**
Apache License 2.0

**Where to send questions or comments about the model:**
https://github.com/lm-sys/FastChat/issues

## Intended use
**Primary intended uses:**
The primary use of FastChat-T5 is the commercial usage of large language models and chatbots. It can also be used for research purposes.

**Primary intended users:**
The primary intended users of the model are entrepreneurs and researchers in natural language processing, machine learning, and artificial intelligence.

## Training dataset
70K conversations collected from ShareGPT.com.

## Training details
It processes the ShareGPT data in the form of question answering. Each ChatGPT response is processed as an answer, and previous conversations between the user and the ChatGPT are processed as the question.
The encoder bi-directionally encodes a question into a hidden representation. The decoder uses cross-attention to attend to this representation while generating an answer uni-directionally from a start token.
This model is fine-tuned for 3 epochs, with a max learning rate 2e-5, warmup ratio 0.03, and a cosine learning rate schedule. 

## Evaluation dataset
A preliminary evaluation of the model quality is conducted by creating a set of 80 diverse questions and utilizing GPT-4 to judge the model outputs. See https://vicuna.lmsys.org/ for more details.
