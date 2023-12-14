---
license: other
license_name: coqui-public-model-license
license_link: https://coqui.ai/cpml
library_name: coqui
pipeline_tag: text-to-speech
widget:
  - text: "Once when I was six years old I saw a magnificent picture"
---

# ⓍTTS
ⓍTTS is a Voice generation model that lets you clone voices into different languages by using just a quick 6-second audio clip. There is no need for an excessive amount of training data that spans countless hours.

This is the same or similar model to what powers [Coqui Studio](https://coqui.ai/) and [Coqui API](https://docs.coqui.ai/docs).

### Features
- Supports 17 languages. 
- Voice cloning with just a 6-second audio clip.
- Emotion and style transfer by cloning. 
- Cross-language voice cloning.
- Multi-lingual speech generation.
- 24khz sampling rate.

### Updates over XTTS-v1
- 2 new languages; Hungarian and Korean
- Architectural improvements for speaker conditioning.
- Enables the use of multiple speaker references and interpolation between speakers.
- Stability improvements.
- Better prosody and audio quality across the board.

### Languages
XTTS-v2 supports 17 languages: **English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt),
Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu), Korean (ko)
Hindi (hi)**.

Stay tuned as we continue to add support for more languages. If you have any language requests, feel free to reach out!

### Code
The [code-base](https://github.com/coqui-ai/TTS) supports inference and [fine-tuning](https://tts.readthedocs.io/en/latest/models/xtts.html#training).

### Demo Spaces
- [XTTS Space](https://huggingface.co/spaces/coqui/xtts)  :  You can see how model performs on supported languages, and try with your own reference or microphone input
- [XTTS Voice Chat with Mistral or Zephyr](https://huggingface.co/spaces/coqui/voice-chat-with-mistral) : You can experience streaming voice chat with Mistral 7B Instruct or Zephyr 7B Beta

|                                 |                                         |
| ------------------------------- | --------------------------------------- |
| 🐸💬 **CoquiTTS**               | [coqui/TTS on Github](https://github.com/coqui-ai/TTS)|
| 💼 **Documentation**            | [ReadTheDocs](https://tts.readthedocs.io/en/latest/)
| 👩‍💻 **Questions**                | [GitHub Discussions](https://github.com/coqui-ai/TTS/discussions) |
| 🗯 **Community**         | [Discord](https://discord.gg/5eXr5seRrv)  |


### License
This model is licensed under [Coqui Public Model License](https://coqui.ai/cpml). There's a lot that goes into a license for generative models, and you can read more of [the origin story of CPML here](https://coqui.ai/blog/tts/cpml).

### Contact
Come and join in our 🐸Community. We're active on [Discord](https://discord.gg/fBC58unbKE) and [Twitter](https://twitter.com/coqui_ai).
You can also mail us at info@coqui.ai.

Using 🐸TTS API:

```python
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="/path/to/target/speaker.wav",
                language="en")

```

Using 🐸TTS Command line:

```console
 tts --model_name tts_models/multilingual/multi-dataset/xtts_v2 \
     --text "Bugün okula gitmek istemiyorum." \
     --speaker_wav /path/to/target/speaker.wav \
     --language_idx tr \
     --use_cuda true
```

Using the model directly:

```python
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json("/path/to/xtts/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/path/to/xtts/", eval=True)
model.cuda()

outputs = model.synthesize(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    config,
    speaker_wav="/data/TTS-public/_refclips/3.wav",
    gpt_cond_len=3,
    language="en",
)
```
