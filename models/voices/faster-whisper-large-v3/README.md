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
  - 'no'
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
  - yue
tags:
  - audio
  - automatic-speech-recognition
license: mit
library_name: ctranslate2
---

faster-whisper officially supports the large-v3 model now. The link is [Systran/faster-whisper-large-v3](https://huggingface.co/Systran/faster-whisper-large-v3)

___

**README.md file is based on "[guillaumekln/faster-whisper-large-v2](https://huggingface.co/guillaumekln/faster-whisper-large-v2)" and has been updated to version 3 content.**

# Whisper large-v3 model for CTranslate2

This repository contains the conversion of [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) to the [CTranslate2](https://github.com/OpenNMT/CTranslate2) model format.

This model can be used in CTranslate2 or projects based on CTranslate2 such as [faster-whisper](https://github.com/guillaumekln/faster-whisper).

## Example

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3")

segments, info = model.transcribe("audio.mp3")
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
```

## Conversion details

The original model was converted with the following command:

```
ct2-transformers-converter --model openai/whisper-large-v3 --output_dir faster-whisper-large-v3 \
    --copy_files added_tokens.json special_tokens_map.json tokenizer_config.json vocab.json  --quantization float16
```

Note that the model weights are saved in FP16. This type can be changed when the model is loaded using the [`compute_type` option in CTranslate2](https://opennmt.net/CTranslate2/quantization.html).

Note that while "openai/whisper-large-v3" does not come with a "tokenizer.json" file, you can generate it using AutoTokenizer.

```python
from transformers import AutoTokenizer
self.hf_tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3")
self.hf_tokenizer.save_pretrained("whisper-large-v3-test")
```

## How faster-whisper working with Whisper-large-v3

**In faster-whisper version 0.10.0, there is no need to perform this handling.**

~~[Working with Whisper-large-v3 #547](https://github.com/guillaumekln/faster-whisper/issues/547) by. UmarRamzan~~

```diff
- from faster_whisper import WhisperModel

- model = WhisperModel(model_url)

- if "large-v3" in model_url:
-     model.feature_extractor.mel_filters = model.feature_extractor.get_mel_filters(model.feature_extractor.sampling_rate, model.feature_extractor.n_fft, n_mels=128)
```

## More information

**For more information about the original model, see its [model card](https://huggingface.co/openai/whisper-large-v3).**
