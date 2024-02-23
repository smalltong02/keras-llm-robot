---
library_name: transformers
tags: []
extra_gated_heading: "Access Gemma on Hugging Face"
extra_gated_prompt: "To access Gemma on Hugging Face, you’re required to review and agree to Google’s usage license. To do this, please ensure you’re logged-in to Hugging Face and click below. Requests are processed immediately."
extra_gated_button_content: "Acknowledge license"
license: other
license_name: gemma-terms-of-use
license_link: https://ai.google.dev/gemma/terms
---

# Gemma Model Card

**Model Page**: [Gemma](https://ai.google.dev/gemma/docs)

This model card corresponds to the 2B base version of the Gemma model. You can also visit the model card of the [7B base model](https://huggingface.co/google/gemma-7b), [7B instruct model](https://huggingface.co/google/gemma-7b-it), and [2B instruct model](https://huggingface.co/google/gemma-2b-it). 

**Resources and Technical Documentation**:

* [Responsible Generative AI Toolkit](https://ai.google.dev/responsible)
* [Gemma on Kaggle](https://www.kaggle.com/models/google/gemma)
* [Gemma on Vertex Model Garden](https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335?version=gemma-2b-gg-hf)

**Terms of Use**: [Terms](https://www.kaggle.com/models/google/gemma/license/consent)

**Authors**: Google

## Model Information

Summary description and brief definition of inputs and outputs.

### Description

Gemma is a family of lightweight, state-of-the-art open models from Google,
built from the same research and technology used to create the Gemini models.
They are text-to-text, decoder-only large language models, available in English,
with open weights, pre-trained variants, and instruction-tuned variants. Gemma
models are well-suited for a variety of text generation tasks, including
question answering, summarization, and reasoning. Their relatively small size
makes it possible to deploy them in environments with limited resources such as
a laptop, desktop or your own cloud infrastructure, democratizing access to
state of the art AI models and helping foster innovation for everyone.

### Usage

Below we share some code snippets on how to get quickly started with running the model. First make sure to `pip install -U transformers`, then copy the snippet from the section that is relevant for your usecase.


#### Fine-tuning the model

You can find fine-tuning scripts and notebook under the [`examples/` directory](https://huggingface.co/google/gemma-7b/tree/main/examples) of [`google/gemma-7b`](https://huggingface.co/google/gemma-7b) repository. To adapt it to this model, simply change the model-id to `google/gemma-2b`.
In that repository, we provide:

* A script to perform Supervised Fine-Tuning (SFT) on UltraChat dataset using QLoRA
* A script to perform SFT using FSDP on TPU devices
* A notebook that you can run on a free-tier Google Colab instance to perform SFT on English quotes dataset



#### Running the model on a CPU


```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```


#### Running the model on a single / multi GPU


```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```


#### Running the model on a GPU using different precisions

* _Using `torch.float16`_

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", torch_dtype=torch.float16)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

* _Using `torch.bfloat16`_

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto", torch_dtype=torch.bfloat16)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

#### Quantized Versions through `bitsandbytes`

* _Using 8-bit precision (int8)_

```python
# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

* _Using 4-bit precision_

```python
# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", quantization_config=quantization_config)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```


#### Other optimizations

* _Flash Attention 2_

First make sure to install `flash-attn` in your environment `pip install flash-attn`

```diff
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
+   attn_implementation="flash_attention_2"
).to(0)
```

### Inputs and outputs

*   **Input:** Text string, such as a question, a prompt, or a document to be
    summarized.
*   **Output:** Generated English-language text in response to the input, such
    as an answer to a question, or a summary of a document.

## Model Data

Data used for model training and how the data was processed.

### Training Dataset

These models were trained on a dataset of text data that includes a wide variety
of sources, totaling 6 trillion tokens. Here are the key components:

* Web Documents: A diverse collection of web text ensures the model is exposed
  to a broad range of linguistic styles, topics, and vocabulary. Primarily
  English-language content.
* Code: Exposing the model to code helps it to learn the syntax and patterns of
  programming languages, which improves its ability to generate code or
  understand code-related questions.
* Mathematics: Training on mathematical text helps the model learn logical
  reasoning, symbolic representation, and to address mathematical queries.

The combination of these diverse data sources is crucial for training a powerful
language model that can handle a wide variety of different tasks and text
formats.

### Data Preprocessing

Here are the key data cleaning and filtering methods applied to the training
data:

* CSAM Filtering: Rigorous CSAM (Child Sexual Abuse Material) filtering was
  applied at multiple stages in the data preparation process to ensure the
  exclusion of harmful and illegal content
* Sensitive Data Filtering: As part of making Gemma pre-trained models safe and
  reliable, automated techniques were used to filter out certain personal
  information and other sensitive data from training sets.
* Additional methods: Filtering based on content quality and safely in line with
  [our policies](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11).

## Implementation Information

Details about the model internals.

### Hardware

Gemma was trained using the latest generation of
[Tensor Processing Unit (TPU)](https://cloud.google.com/tpu/docs/intro-to-tpu) hardware (TPUv5e).

Training large language models requires significant computational power. TPUs,
designed specifically for matrix operations common in machine learning, offer
several advantages in this domain:

* Performance: TPUs are specifically designed to handle the massive computations
  involved in training LLMs. They can speed up training considerably compared to
  CPUs.
* Memory: TPUs often come with large amounts of high-bandwidth memory, allowing
  for the handling of large models and batch sizes during training. This can
  lead to better model quality.
* Scalability: TPU Pods (large clusters of TPUs) provide a scalable solution for
  handling the growing complexity of large foundation models. You can distribute
  training across multiple TPU devices for faster and more efficient processing.
* Cost-effectiveness: In many scenarios, TPUs can provide a more cost-effective
  solution for training large models compared to CPU-based infrastructure,
  especially when considering the time and resources saved due to faster
  training.
* These advantages are aligned with
  [Google's commitments to operate sustainably](https://sustainability.google/operating-sustainably/).

### Software

Training was done using [JAX](https://github.com/google/jax) and [ML Pathways](https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/ml-pathways).

JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models.

ML Pathways is Google's latest effort to build artificially intelligent systems
capable of generalizing across multiple tasks. This is specially suitable for
[foundation models](https://ai.google/discover/foundation-models/), including large language models like
these ones.

Together, JAX and ML Pathways are used as described in the
[paper about the Gemini family of models](https://arxiv.org/abs/2312.11805); "the 'single
controller' programming model of Jax and Pathways allows a single Python
process to orchestrate the entire training run, dramatically simplifying the
development workflow."

## Evaluation

Model evaluation metrics and results.

### Benchmark Results

These models were evaluated against a large collection of different datasets and
metrics to cover different aspects of text generation:

| Benchmark                      | Metric        | 2B Params | 7B Params |
| ------------------------------ | ------------- | ----------- | --------- |
| [MMLU](https://arxiv.org/abs/2009.03300)                   | 5-shot, top-1 | 42.3        | 64.3      |
| [HellaSwag](https://arxiv.org/abs/1905.07830)         | 0-shot        |71.4        | 81.2      |
| [PIQA](https://arxiv.org/abs/1911.11641)                   | 0-shot        | 77.3        | 81.2      |
| [SocialIQA](https://arxiv.org/abs/1904.09728)      | 0-shot        | 59.7        | 51.8      |
| [BooIQ](https://arxiv.org/abs/1905.10044)                | 0-shot        | 69.4        | 83.2      |
| [WinoGrande](https://arxiv.org/abs/1907.10641)       | partial score | 65.4        | 72.3      |
| [CommonsenseQA](https://arxiv.org/abs/1811.00937) | 7-shot        | 65.3        | 71.3      |
| [OpenBookQA](https://arxiv.org/abs/1809.02789)       |               | 47.8        | 52.8      |
| [ARC-e](https://arxiv.org/abs/1911.01547)                  |               | 73.2        | 81.5      |
| [ARC-c](https://arxiv.org/abs/1911.01547)                   |               | 42.1        | 53.2      |
| [TriviaQA](https://arxiv.org/abs/1705.03551)           | 5-shot        | 53.2        | 63.4      |
| [Natural Questions](https://github.com/google-research-datasets/natural-questions)  | 5-shot        | -       | 23        |
| [HumanEval](https://arxiv.org/abs/2107.03374)      | pass@1        | 22.0        | 32.3      |
| [MBPP](https://arxiv.org/abs/2108.07732)                   | 3-shot        | 29.2        | 44.4      |
| [GSM8K](https://arxiv.org/abs/2110.14168)                | maj@1         | 17.7        | 46.4      |
| [MATH](https://arxiv.org/abs/2108.07732)                   | 4-shot        | 11.8          | 24.3      |
| [AGIEval](https://arxiv.org/abs/2304.06364)           |               | 24.2        | 41.7      |
| [BIG-Bench](https://arxiv.org/abs/2206.04615)         |               | 35.2        | 55.1      |
| ------------------------------ | ------------- | ----------- | --------- |
| **Average**                    |               | **54.0**    | **56.4**  |

## Ethics and Safety

Ethics and safety evaluation approach and results.

### Evaluation Approach

Our evaluation methods include structured evaluations and internal red-teaming
testing of relevant content policies. Red-teaming was conducted by a number of
different teams, each with different goals and human evaluation metrics. These
models were evaluated against a number of different categories relevant to
ethics and safety, including:

* Text-to-Text Content Safety: Human evaluation on prompts covering safety
  policies including child sexual abuse and exploitation, harassment, violence
  and gore, and hate speech.
* Text-to-Text Representational Harms: Benchmark against relevant academic
  datasets such as [WinoBias](https://arxiv.org/abs/1804.06876) and [BBQ Dataset](https://arxiv.org/abs/2110.08193v2).
* Memorization: Automated evaluation of memorization of training data, including
  the risk of personally identifiable information exposure.
* Large-scale harm: Tests for "dangerous capabilities," such as chemical,
  biological, radiological, and nuclear (CBRN) risks.

### Evaluation Results

The results of ethics and safety evaluations are within acceptable thresholds
for meeting [internal policies](https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11) for categories such as child
safety, content safety, representational harms, memorization, large-scale harms.
On top of robust internal evaluations, the results of well known safety
benchmarks like BBQ, BOLD, Winogender, Winobias, RealToxicity, and TruthfulQA
are shown here.

| Benchmark                      | Metric        | 2B Params   | 7B Params |
| ------------------------------ | ------------- | ----------- | --------- |
| [RealToxicity](https://arxiv.org/abs/2009.11462)        | average       | 6.86        | 7.90      |
| [BOLD](https://arxiv.org/abs/2101.11718)                   |               | 45.57       | 49.08     |
| [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/)        | top-1         | 45.82       | 51.33     |
| [BBQ Ambig](https://arxiv.org/abs/2110.08193v2)               | 1-shot, top-1 | 62.58       | 92.54     |
| [BBQ Disambig](https://arxiv.org/abs/2110.08193v2)            | top-1         | 54.62       | 71.99     |
| [Winogender](https://arxiv.org/abs/1804.09301)       | top-1         | 51.25       | 54.17     |
| [TruthfulQA](https://arxiv.org/abs/2109.07958)       |               | 44.84       | 31.81     |
| [Winobias 1_2](https://arxiv.org/abs/1804.06876)       |               | 56.12       | 59.09     |
| [Winobias 2_2](https://arxiv.org/abs/1804.06876)       |               | 91.10       | 92.23     |
| [Toxigen](https://arxiv.org/abs/2203.09509)             |               | 29.77       | 39.59     |
| ------------------------------ | ------------- | ----------- | --------- |


## Usage and Limitations

These models have certain limitations that users should be aware of.

### Intended Usage

Open Large Language Models (LLMs) have a wide range of applications across
various industries and domains. The following list of potential uses is not
comprehensive. The purpose of this list is to provide contextual information
about the possible use-cases that the model creators considered as part of model
training and development.

* Content Creation and Communication
  * Text Generation: These models can be used to generate creative text formats
    such as poems, scripts, code, marketing copy, and email drafts.
  * Chatbots and Conversational AI: Power conversational interfaces for customer
    service, virtual assistants, or interactive applications.
  * Text Summarization: Generate concise summaries of a text corpus, research
    papers, or reports.
* Research and Education
  * Natural Language Processing (NLP) Research: These models can serve as a
    foundation for researchers to experiment with NLP techniques, develop
    algorithms, and contribute to the advancement of the field.
  * Language Learning Tools: Support interactive language learning experiences,
    aiding in grammar correction or providing writing practice.
  * Knowledge Exploration: Assist researchers in exploring large bodies of text
    by generating summaries or answering questions about specific topics.

### Limitations

* Training Data
  * The quality and diversity of the training data significantly influence the
    model's capabilities. Biases or gaps in the training data can lead to
    limitations in the model's responses.
  * The scope of the training dataset determines the subject areas the model can
    handle effectively.
* Context and Task Complexity
  * LLMs are better at tasks that can be framed with clear prompts and
    instructions. Open-ended or highly complex tasks might be challenging.
  * A model's performance can be influenced by the amount of context provided
    (longer context generally leads to better outputs, up to a certain point).
* Language Ambiguity and Nuance
  * Natural language is inherently complex. LLMs might struggle to grasp subtle
    nuances, sarcasm, or figurative language.
* Factual Accuracy
  * LLMs generate responses based on information they learned from their
    training datasets, but they are not knowledge bases. They may generate
    incorrect or outdated factual statements.
* Common Sense
  * LLMs rely on statistical patterns in language. They might lack the ability
    to apply common sense reasoning in certain situations.

### Ethical Considerations and Risks

The development of large language models (LLMs) raises several ethical concerns.
In creating an open model, we have carefully considered the following:

* Bias and Fairness
  * LLMs trained on large-scale, real-world text data can reflect socio-cultural
    biases embedded in the training material. These models underwent careful
    scrutiny, input data pre-processing described and posterior evaluations
    reported in this card.
* Misinformation and Misuse
  * LLMs can be misused to generate text that is false, misleading, or harmful.
  * Guidelines are provided for responsible use with the model, see the
    [Responsible Generative AI Toolkit](http://ai.google.dev/gemma/responsible).
* Transparency and Accountability:
  * This model card summarizes details on the models' architecture,
    capabilities, limitations, and evaluation processes.
  * A responsibly developed open model offers the opportunity to share
    innovation by making LLM technology accessible to developers and researchers
    across the AI ecosystem.

Risks identified and mitigations:

* Perpetuation of biases: It's encouraged to perform continuous monitoring
  (using evaluation metrics, human review) and the exploration of de-biasing
  techniques during model training, fine-tuning, and other use cases.
* Generation of harmful content: Mechanisms and guidelines for content safety
  are essential. Developers are encouraged to exercise caution and implement
  appropriate content safety safeguards based on their specific product policies
  and application use cases.
* Misuse for malicious purposes: Technical limitations and developer and
  end-user education can help mitigate against malicious applications of LLMs.
  Educational resources and reporting mechanisms for users to flag misuse are
  provided. Prohibited uses of Gemma models are outlined in the
  [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
* Privacy violations: Models were trained on data filtered for removal of PII
  (Personally Identifiable Information). Developers are encouraged to adhere to
  privacy regulations with privacy-preserving techniques.

### Benefits

At the time of release, this family of models provides high-performance open
large language model implementations designed from the ground up for Responsible
AI development compared to similarly sized models.

Using the benchmark evaluation metrics described in this document, these models
have shown to provide superior performance to other, comparably-sized open model
alternatives.

