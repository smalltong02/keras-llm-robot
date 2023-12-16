---
base_model: 01-ai/yi-chat-6b-Chat
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
quantized_by: XeIaso
widget:
- example_title: yi-chat-6b-Chat
  output:
    text: 'Hello! How can I assist you today?'
  text: hi
---

# Yi 6B Chat - GGUF

- Model creator: [01-ai](https://huggingface.co/01-ai)
- Original model: [Yi 6B
  Chat](https://huggingface.co/01-ai/yi-chat-6b-Chat)

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

<!-- README_GGUF.md-provided-files start -->
## Provided files

| Name | Quant method | Bits | Size | Max RAM required | Use case |
| ---- | ---- | ---- | ---- | ---- | ----- |
| [yi-chat-6b.Q2_K.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q2_K.gguf) | Q2_K | 2 | 2.62 GB| 5.12 GB | smallest, significant quality loss - not recommended for most purposes |
| [yi-chat-6b.Q3_K_S.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q3_K_S.gguf) | Q3_K_S | 3 | 2.71 GB| 5.21 GB | very small, high quality loss |
| [yi-chat-6b.Q3_K_M.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q3_K_M.gguf) | Q3_K_M | 3 | 2.99 GB| 5.49 GB | very small, high quality loss |
| [yi-chat-6b.Q3_K_L.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q3_K_L.gguf) | Q3_K_L | 3 | 3.24 GB| 5.74 GB | small, substantial quality loss |
| [yi-chat-6b.Q4_0.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q4_0.gguf) | Q4_0 | 4 | 3.48 GB| 5.98 GB | legacy; small, very high quality loss - prefer using Q3_K_M |
| [yi-chat-6b.Q4_K_S.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q4_K_S.gguf) | Q4_K_S | 4 | 3.50 GB| 6.00 GB | small, greater quality loss |
| [yi-chat-6b.Q4_K_M.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q4_K_M.gguf) | Q4_K_M | 4 | 3.67 GB| 6.17 GB | medium, balanced quality - recommended |
| [yi-chat-6b.Q5_0.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q5_0.gguf) | Q5_0 | 5 | 4.20 GB| 6.70 GB | legacy; medium, balanced quality - prefer using Q4_K_M |
| [yi-chat-6b.Q5_K_S.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q5_K_S.gguf) | Q5_K_S | 5 | 4.20 GB| 6.70 GB | large, low quality loss - recommended |
| [yi-chat-6b.Q5_K_M.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q5_K_M.gguf) | Q5_K_M | 5 | 4.30 GB| 6.80 GB | large, very low quality loss - recommended |
| [yi-chat-6b.Q6_K.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q6_K.gguf) | Q6_K | 6 | 4.97 GB| 7.47 GB | very large, extremely low quality loss |
| [yi-chat-6b.Q8_0.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.Q8_0.gguf) | Q8_0 | 8 | 6.44 GB| 8.94 GB | very large, extremely low quality loss - not recommended |
| [yi-chat-6b.f16.gguf](https://huggingface.co/XeIaso/yi-chat-6b-GGUF/blob/main/yi-chat-6b.f16.gguf) | f16 | 16 | 12.2 GB | 14 GB | extremely large, minimal quality loss |

**Note**: the above RAM figures assume no GPU offloading. If layers are offloaded to the GPU, this will reduce RAM usage and use VRAM instead.

<!-- README_GGUF.md-provided-files end -->

If you want to support my efforts, check out my [Patreon](https://patreon.com/cadey).
