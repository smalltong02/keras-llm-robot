---
language: 
- en
- zh
- de
- es
- ru
- ko
- fr
- ja
- pt
- tr
- pl
- ca
- nl
- ar
- sv
- it
- id
- hi
- fi
- vi
- he
- uk
- el
- ms
- cs
- ro
- da
- hu
- ta
- no
- th
- ur
- hr
- bg
- lt
- la
- mi
- ml
- cy
- sk
- te
- fa
- lv
- bn
- sr
- az
- sl
- kn
- et
- mk
- br
- eu
- is
- hy
- ne
- mn
- bs
- kk
- sq
- sw
- gl
- mr
- pa
- si
- km
- sn
- yo
- so
- af
- oc
- ka
- be
- tg
- sd
- gu
- am
- yi
- lo
- uz
- fo
- ht
- ps
- tk
- nn
- mt
- sa
- lb
- my
- bo
- tl
- mg
- as
- tt
- haw
- ln
- ha
- ba
- jw
- su
tags:
- audio
- automatic-speech-recognition
- hf-asr-leaderboard
widget:
- example_title: Librispeech sample 1
  src: https://cdn-media.huggingface.co/speech_samples/sample1.flac
- example_title: Librispeech sample 2
  src: https://cdn-media.huggingface.co/speech_samples/sample2.flac
pipeline_tag: automatic-speech-recognition
license: apache-2.0
---

# Whisper

Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation. Trained on 680k hours 
of labelled data, Whisper models demonstrate a strong ability to generalise to many datasets and domains **without** the need 
for fine-tuning.

Whisper was proposed in the paper [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) 
by Alec Radford et al. from OpenAI. The original code repository can be found [here](https://github.com/openai/whisper).

Whisper `large-v3` has the same architecture as the previous large models except the following minor differences:

1. The input uses 128 Mel frequency bins instead of 80
2. A new language token for Cantonese

The Whisper `large-v3` model is trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper `large-v2`. 
The model was trained for 2.0 epochs over this mixture dataset.

The `large-v3` model shows improved performance over a wide variety of languages, showing 10% to 20% reduction of errors compared to Whisper `large-v2`.


**Disclaimer**: Content for this model card has partly been written by the Hugging Face team, and parts of it were 
copied and pasted from the original model card.

## Model details

Whisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model. 
It was trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper `large-v2`.

The models were trained on either English-only data or multilingual data. The English-only models were trained 
on the task of speech recognition. The multilingual models were trained on both speech recognition and speech 
translation. For speech recognition, the model predicts transcriptions in the *same* language as the audio. 
For speech translation, the model predicts transcriptions to a *different* language to the audio.

Whisper checkpoints come in five configurations of varying model sizes.
The smallest four are trained on either English-only or multilingual data.
The largest checkpoints are multilingual only. All ten of the pre-trained checkpoints 
are available on the [Hugging Face Hub](https://huggingface.co/models?search=openai/whisper). The 
checkpoints are summarised in the following table with links to the models on the Hub:

| Size     | Parameters | English-only                                         | Multilingual                                        |
|----------|------------|------------------------------------------------------|-----------------------------------------------------|
| tiny     | 39 M       | [✓](https://huggingface.co/openai/whisper-tiny.en)   | [✓](https://huggingface.co/openai/whisper-tiny)     |
| base     | 74 M       | [✓](https://huggingface.co/openai/whisper-base.en)   | [✓](https://huggingface.co/openai/whisper-base)     |
| small    | 244 M      | [✓](https://huggingface.co/openai/whisper-small.en)  | [✓](https://huggingface.co/openai/whisper-small)    |
| medium   | 769 M      | [✓](https://huggingface.co/openai/whisper-medium.en) | [✓](https://huggingface.co/openai/whisper-medium)   |
| large    | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large)    |
| large-v2 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v2) |
| large-v3 | 1550 M     | x                                                    | [✓](https://huggingface.co/openai/whisper-large-v3) |

## Usage

Whisper `large-v3` is supported in Hugging Face 🤗 Transformers through the `main` branch in the Transformers repo. To run the model, first 
install the Transformers library through the GitHub repo. For this example, we'll also install 🤗 Datasets to load toy 
audio dataset from the Hugging Face Hub:

```bash
pip install --upgrade pip
pip install --upgrade git+https://github.com/huggingface/transformers.git accelerate datasets[audio]
```

The model can be used with the [`pipeline`](https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline)
class to transcribe audio files of arbitrary length. Transformers uses a chunked algorithm to transcribe 
long-form audio files, which in-practice is 9x faster than the sequential algorithm proposed by OpenAI 
(see Table 7 of the [Distil-Whisper paper](https://arxiv.org/abs/2311.00430)). The batch size should 
be set based on the specifications of your device:

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
sample = dataset[0]["audio"]

result = pipe(sample)
print(result["text"])
```

To transcribe a local audio file, simply pass the path to your audio file when you call the pipeline:
```diff
- result = pipe(sample)
+ result = pipe("audio.mp3")
```

Whisper predicts the language of the source audio automatically. If the source audio language is known *a-priori*, it 
can be passed as an argument to the pipeline:

```python
result = pipe(sample, generate_kwargs={"language": "english"})
```

By default, Whisper performs the task of *speech transcription*, where the source audio language is the same as the target
text language. To perform *speech translation*, where the target text is in English, set the task to `"translate"`:

```python
result = pipe(sample, generate_kwargs={"task": "translate"})
```

Finally, the model can be made to predict timestamps. For sentence-level timestamps, pass the `return_timestamps` argument:

```python
result = pipe(sample, return_timestamps=True)
print(result["chunks"])
```

And for word-level timestamps:

```python
result = pipe(sample, return_timestamps="word")
print(result["chunks"])
```

The above arguments can be used in isolation or in combination. For example, to perform the task of speech transcription 
where the source audio is in French, and we want to return sentence-level timestamps, the following can be used:

```python
result = pipe(sample, return_timestamps=True, generate_kwargs={"language": "french", "task": "translate"})
print(result["chunks"])
```

## Additional Speed & Memory Improvements

You can apply additional speed and memory improvements to Whisper-large-v3 which we cover in the following.

### Flash Attention

We recommend using [Flash-Attention 2](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#flashattention-2) if your GPU allows for it.
To do so, you first need to install [Flash Attention](https://github.com/Dao-AILab/flash-attention):

```
pip install flash-attn --no-build-isolation
```

and then all you have to do is to pass `use_flash_attention_2=True` to `from_pretrained`:

```diff
- model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, use_flash_attention_2=True)
```

### Torch Scale-Product-Attention (SDPA)

If your GPU does not support Flash Attention, we recommend making use of [BetterTransformers](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#bettertransformer).
To do so, you first need to install optimum:

```
pip install --upgrade optimum
```

And then convert your model to a "BetterTransformer" model before using it:

```diff
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
+ model = model.to_bettertransformer()
```

## Fine-Tuning

The pre-trained Whisper model demonstrates a strong ability to generalise to different datasets and domains. However, 
its predictive capabilities can be improved further for certain languages and tasks through *fine-tuning*. The blog 
post [Fine-Tune Whisper with 🤗 Transformers](https://huggingface.co/blog/fine-tune-whisper) provides a step-by-step 
guide to fine-tuning the Whisper model with as little as 5 hours of labelled data.

### Evaluated Use

The primary intended users of these models are AI researchers studying robustness, generalization, capabilities, biases, and constraints of the current model. However, Whisper is also potentially quite useful as an ASR solution for developers, especially for English speech recognition. We recognize that once models are released, it is impossible to restrict access to only “intended” uses or to draw reasonable guidelines around what is or is not research.

The models are primarily trained and evaluated on ASR and speech translation to English tasks. They show strong ASR results in ~10 languages. They may exhibit additional capabilities, particularly if fine-tuned on certain tasks like voice activity detection, speaker classification, or speaker diarization but have not been robustly evaluated in these areas. We strongly recommend that users perform robust evaluations of the models in a particular context and domain before deploying them.

In particular, we caution against using Whisper models to transcribe recordings of individuals taken without their consent or purporting to use these models for any kind of subjective classification. We recommend against use in high-risk domains like decision-making contexts, where flaws in accuracy can lead to pronounced flaws in outcomes. The models are intended to transcribe and translate speech, use of the model for classification is not only not evaluated but also not appropriate, particularly to infer human attributes.


## Training Data

The models are trained on 1 million hours of weakly labeled audio and 4 million hours of pseudolabeled audio collected using Whisper `large-v2`. 

As discussed in [the accompanying paper](https://cdn.openai.com/papers/whisper.pdf), we see that performance on transcription in a given language is directly correlated with the amount of training data we employ in that language.


## Performance and Limitations

Our studies show that, over many existing ASR systems, the models exhibit improved robustness to accents, background noise, technical language, as well as zero shot translation from multiple languages into English; and that accuracy on speech recognition and translation is near the state-of-the-art level. 

However, because the models are trained in a weakly supervised manner using large-scale noisy data, the predictions may include texts that are not actually spoken in the audio input (i.e. hallucination). We hypothesize that this happens because, given their general knowledge of language, the models combine trying to predict the next word in audio with trying to transcribe the audio itself.

Our models perform unevenly across languages, and we observe lower accuracy on low-resource and/or low-discoverability languages or languages where we have less training data. The models also exhibit disparate performance on different accents and dialects of particular languages, which may include higher word error rate across speakers of different genders, races, ages, or other demographic criteria. Our full evaluation results are presented in [the paper accompanying this release](https://cdn.openai.com/papers/whisper.pdf). 

In addition, the sequence-to-sequence architecture of the model makes it prone to generating repetitive texts, which can be mitigated to some degree by beam search and temperature scheduling but not perfectly. Further analysis on these limitations are provided in [the paper](https://cdn.openai.com/papers/whisper.pdf). It is likely that this behavior and hallucinations may be worse on lower-resource and/or lower-discoverability languages.


## Broader Implications

We anticipate that Whisper models’ transcription capabilities may be used for improving accessibility tools. While Whisper models cannot be used for real-time transcription out of the box – their speed and size suggest that others may be able to build applications on top of them that allow for near-real-time speech recognition and translation. The real value of beneficial applications built on top of Whisper models suggests that the disparate performance of these models may have real economic implications.

There are also potential dual use concerns that come with releasing Whisper. While we hope the technology will be used primarily for beneficial purposes, making ASR technology more accessible could enable more actors to build capable surveillance technologies or scale up existing surveillance efforts, as the speed and accuracy allow for affordable automatic transcription and translation of large volumes of audio communication. Moreover, these models may have some capabilities to recognize specific individuals out of the box, which in turn presents safety concerns related both to dual use and disparate performance. In practice, we expect that the cost of transcription is not the limiting factor of scaling up surveillance projects.


### BibTeX entry and citation info
```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```