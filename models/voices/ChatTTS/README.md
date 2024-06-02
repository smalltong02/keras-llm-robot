---
license: cc-by-nc-4.0
---


**We are also training larger-scale models and need computational power and data support. If you can provide assistance, please contact OPEN-SOURCE@2NOISE.COM. Thank you very much.**

## Clone the Repository
First, clone the Git repository:
```bash
git clone https://github.com/2noise/ChatTTS.git
```

## Model Inference


```python
# Import necessary libraries and configure settings
import torch
import torchaudio
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
from IPython.display import Audio

# Initialize and load the model: 
chat = ChatTTS.Chat()
chat.load_models(compile=False) # Set to True for better performance

# Define the text input for inference (Support Batching)
texts = [
    "So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with.",
    ]

# Perform inference and play the generated audio
wavs = chat.infer(texts)
Audio(wavs[0], rate=24_000, autoplay=True)

# Save the generated audio 
torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
```
**For more usage examples, please refer to the [example notebook](https://github.com/2noise/ChatTTS/blob/main/example.ipynb), which includes parameters for finer control over the generated speech, such as specifying the speaker, adjusting speech speed, and adding laughter.**






### Disclaimer: For Academic Purposes Only

The information provided in this document is for academic purposes only. It is intended for educational and research use, and should not be used for any commercial or legal purposes. The authors do not guarantee the accuracy, completeness, or reliability of the information.