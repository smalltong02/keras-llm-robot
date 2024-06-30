---
library_name: transformers
license: gemma
pipeline_tag: text-generation
extra_gated_heading: Access Gemma on Hugging Face
extra_gated_prompt: To access Gemma on Hugging Face, you’re required to review and
  agree to Google’s usage license. To do this, please ensure you’re logged in to Hugging
  Face and click below. Requests are processed immediately.
---

# Fork from google/gemma-2-27b-it

## 4-bit Quantization

```python
nf4_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_compute_dtype=torch.bfloat16,
                                bnb_4bit_quant_type="nf4")

```



# Gemma 2 model card

**Model Page**: [Gemma](https://ai.google.dev/gemma/docs)

**Resources and Technical Documentation**:

* [Responsible Generative AI Toolkit][rai-toolkit]
* [Gemma on Kaggle][kaggle-gemma]
* [Gemma on Vertex Model Garden][vertex-mg-gemma]

**Terms of Use**: [Terms](https://www.kaggle.com/models/google/gemma/license/consent/verify/huggingface?returnModelRepoId=google/gemma-2-27b-it)

**Authors**: Google

## Model Information

Summary description and brief definition of inputs and outputs.

### Description

Gemma is a family of lightweight, state-of-the-art open models from Google,
built from the same research and technology used to create the Gemini models.
They are text-to-text, decoder-only large language models, available in English,
with open weights for both pre-trained variants and instruction-tuned variants.
Gemma models are well-suited for a variety of text generation tasks, including
question answering, summarization, and reasoning. Their relatively small size
makes it possible to deploy them in environments with limited resources such as
a laptop, desktop or your own cloud infrastructure, democratizing access to
state of the art AI models and helping foster innovation for everyone.

### Usage

Below we share some code snippets on how to get quickly started with running the model. First make sure to `pip install -U transformers`, then copy the snippet from the section that is relevant for your usecase.


#### Running the model on a single / multi GPU


```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

<a name="precisions"></a>
#### Running the model on a GPU using different precisions

The native weights of this model were exported in `bfloat16` precision. You can use `float16`, which may be faster on certain hardware, indicating the `torch_dtype` when loading the model. For convenience, the `float16` revision of the repo contains a copy of the weights already converted to that precision.

You can also use `float32` if you skip the dtype, but no precision increase will occur (model weights will just be upcasted to `float32`). See examples below.

* _Using `torch.float16`_

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto",
    torch_dtype=torch.float16,
    revision="float16",
)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

* _Using `torch.bfloat16`_

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))
```

* _Upcasting to `torch.float32`_

```python
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    device_map="auto"
)

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

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    quantization_config=quantization_config)

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

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b-it",
    quantization_config=quantization_config)

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

### Chat Template

The instruction-tuned models use a chat template that must be adhered to for conversational use.
The easiest way to apply it is using the tokenizer's built-in chat template, as shown in the following snippet.

Let's load the model and apply the chat template to a conversation. In this example, we'll start with a single user interaction:

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_id = "google/gemma-2-27b-it"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype=dtype,
)

chat = [
    { "role": "user", "content": "Write a hello world program" },
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
```

At this point, the prompt contains the following text:

```
<bos><start_of_turn>user
Write a hello world program<end_of_turn>
<start_of_turn>model
```

As you can see, each turn is preceded by a `<start_of_turn>` delimiter and then the role of the entity
(either `user`, for content supplied by the user, or `model` for LLM responses). Turns finish with
the `<end_of_turn>` token.

You can follow this format to build the prompt manually, if you need to do it without the tokenizer's
chat template.

After the prompt is ready, generation can be performed like this:

```py
inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
print(tokenizer.decode(outputs[0]))
```

### Inputs and outputs

*   **Input:** Text string, such as a question, a prompt, or a document to be
    summarized.
*   **Output:** Generated English-language text in response to the input, such
    as an answer to a question, or a summary of a document.

### Citation

```none
@article{gemma_2024,
    title={Gemma},
    url={https://www.kaggle.com/m/3301},
    DOI={10.34740/KAGGLE/M/3301},
    publisher={Kaggle},
    author={Gemma Team},
    year={2024}
}
```

## Model Data

Data used for model training and how the data was processed.

### Training Dataset

These models were trained on a dataset of text data that includes a wide variety of sources. The 27B model was trained with 13 trillion tokens and the 9B model was trained with 8 trillion tokens. 
Here are the key components:

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
  exclusion of harmful and illegal content.
* Sensitive Data Filtering: As part of making Gemma pre-trained models safe and
  reliable, automated techniques were used to filter out certain personal
  information and other sensitive data from training sets.
* Additional methods: Filtering based on content quality and safety in line with
  [our policies][safety-policies].

## Implementation Information

Details about the model internals.

### Hardware

Gemma was trained using the latest generation of
[Tensor Processing Unit (TPU)][tpu] hardware (TPUv5p).

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
  [Google's commitments to operate sustainably][sustainability].

### Software

Training was done using [JAX][jax] and [ML Pathways][ml-pathways].

JAX allows researchers to take advantage of the latest generation of hardware,
including TPUs, for faster and more efficient training of large models.

ML Pathways is Google's latest effort to build artificially intelligent systems
capable of generalizing across multiple tasks. This is specially suitable for
[foundation models][foundation-models], including large language models like
these ones.

Together, JAX and ML Pathways are used as described in the
[paper about the Gemini family of models][gemini-2-paper]; "the 'single
controller' programming model of Jax and Pathways allows a single Python
process to orchestrate the entire training run, dramatically simplifying the
development workflow."

## Evaluation

Model evaluation metrics and results.

### Benchmark Results

These models were evaluated against a large collection of different datasets and
metrics to cover different aspects of text generation:

| Benchmark                      | Metric        | Gemma PT 9B | Gemma PT 27B |
| ------------------------------ | ------------- | ----------- | ------------ |
| [MMLU][mmlu]                   | 5-shot, top-1 | 71.3        | 75.2         |
| [HellaSwag][hellaswag]         | 10-shot       | 81.9        | 86.4         |
| [PIQA][piqa]                   | 0-shot        | 81.7        | 83.2         |
| [SocialIQA][socialiqa]         | 0-shot        | 53.4        | 53.7         |
| [BoolQ][boolq]                 | 0-shot        | 84.2        | 84.8         |
| [WinoGrande][winogrande]       | partial score | 80.6        | 83.7         |
| [ARC-e][arc]                   | 0-shot        | 88.0        | 88.6         |
| [ARC-c][arc]                   | 25-shot       | 68.4        | 71.4         |
| [TriviaQA][triviaqa]           | 5-shot        | 76.6        | 83.7         |
| [Natural Questions][naturalq]  | 5-shot        | 29.2        | 34.5         |
| [HumanEval][humaneval]         | pass@1        | 40.2        | 51.8         |
| [MBPP][mbpp]                   | 3-shot        | 52.4        | 62.6         |
| [GSM8K][gsm8k]                 | 5-shot, maj@1 | 68.6        | 74.0         |
| [MATH][math]                   | 4-shot        | 36.6        | 42.3         |
| [AGIEval][agieval]             | 3-5-shot      | 52.8        | 55.1         |
| [BIG-Bench][big-bench]         | 3-shot, CoT   | 68.2        | 74.9         |
| ------------------------------ | ------------- | ----------- | ------------ |

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
  datasets such as [WinoBias][winobias] and [BBQ Dataset][bbq].
* Memorization: Automated evaluation of memorization of training data, including
  the risk of personally identifiable information exposure.
* Large-scale harm: Tests for "dangerous capabilities," such as chemical,
  biological, radiological, and nuclear (CBRN) risks.

### Evaluation Results

The results of ethics and safety evaluations are within acceptable thresholds
for meeting [internal policies][safety-policies] for categories such as child
safety, content safety, representational harms, memorization, large-scale harms.
On top of robust internal evaluations, the results of well-known safety
benchmarks like BBQ, BOLD, Winogender, Winobias, RealToxicity, and TruthfulQA
are shown here.

#### Gemma 2.0

| Benchmark                | Metric        | Gemma 2 IT 9B | Gemma 2 IT 27B |
| ------------------------ | ------------- | --------------- | ---------------- |
| [RealToxicity][realtox]  | average       |  8.25           |  8.84            |
| [CrowS-Pairs][crows]     | top-1         | 37.47           | 36.67            |
| [BBQ Ambig][bbq]         | 1-shot, top-1 | 88.58           | 85.99            |
| [BBQ Disambig][bbq]      | top-1         | 82.67           | 86.94            |
| [Winogender][winogender] | top-1         | 79.17           | 77.22            |
| [TruthfulQA][truthfulqa] |               | 50.27           | 51.60            |
| [Winobias 1_2][winobias] |               | 78.09           | 81.94            |
| [Winobias 2_2][winobias] |               | 95.32           | 97.22            |
| [Toxigen][toxigen]       |               | 39.30           | 38.42            |
| ------------------------ | ------------- | --------------- | ---------------- |

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
    [Responsible Generative AI Toolkit][rai-toolkit].
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
  [Gemma Prohibited Use Policy][prohibited-use].
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

[rai-toolkit]: https://ai.google.dev/responsible
[kaggle-gemma]: https://www.kaggle.com/models/google/gemma-2
[terms]: https://ai.google.dev/gemma/terms
[vertex-mg-gemma]: https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335
[sensitive-info]: https://cloud.google.com/dlp/docs/high-sensitivity-infotypes-reference
[safety-policies]: https://storage.googleapis.com/gweb-uniblog-publish-prod/documents/2023_Google_AI_Principles_Progress_Update.pdf#page=11
[prohibited-use]: https://ai.google.dev/gemma/prohibited_use_policy
[tpu]: https://cloud.google.com/tpu/docs/intro-to-tpu
[sustainability]: https://sustainability.google/operating-sustainably/
[jax]: https://github.com/google/jax
[ml-pathways]: https://blog.google/technology/ai/introducing-pathways-next-generation-ai-architecture/
[sustainability]: https://sustainability.google/operating-sustainably/
[foundation-models]: https://ai.google/discover/foundation-models/
[gemini-2-paper]: https://goo.gle/gemma2report
[mmlu]: https://arxiv.org/abs/2009.03300
[hellaswag]: https://arxiv.org/abs/1905.07830
[piqa]: https://arxiv.org/abs/1911.11641
[socialiqa]: https://arxiv.org/abs/1904.09728
[boolq]: https://arxiv.org/abs/1905.10044
[winogrande]: https://arxiv.org/abs/1907.10641
[commonsenseqa]: https://arxiv.org/abs/1811.00937
[openbookqa]: https://arxiv.org/abs/1809.02789
[arc]: https://arxiv.org/abs/1911.01547
[triviaqa]: https://arxiv.org/abs/1705.03551
[naturalq]: https://github.com/google-research-datasets/natural-questions
[humaneval]: https://arxiv.org/abs/2107.03374
[mbpp]: https://arxiv.org/abs/2108.07732
[gsm8k]: https://arxiv.org/abs/2110.14168
[realtox]: https://arxiv.org/abs/2009.11462
[bold]: https://arxiv.org/abs/2101.11718
[crows]: https://aclanthology.org/2020.emnlp-main.154/
[bbq]: https://arxiv.org/abs/2110.08193v2
[winogender]: https://arxiv.org/abs/1804.09301
[truthfulqa]: https://arxiv.org/abs/2109.07958
[winobias]: https://arxiv.org/abs/1804.06876
[math]: https://arxiv.org/abs/2103.03874
[agieval]: https://arxiv.org/abs/2304.06364
[big-bench]: https://arxiv.org/abs/2206.04615
[toxigen]: https://arxiv.org/abs/2203.09509
