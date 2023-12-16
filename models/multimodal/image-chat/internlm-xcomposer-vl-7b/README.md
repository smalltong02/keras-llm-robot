---
license: apache-2.0
pipeline_tag: text-generation
---


<p align="center">
    <img src="logo.png" width="400"/>
<p>

<p align="center">
    <b><font size="6">InternLM-XComposer</font></b> 
<p>

<div align="center">

[💻Github Repo](https://github.com/InternLM/InternLM-XComposer)

</div>

**InternLM-XComposer** is a vision-language large model (VLLM) based on [InternLM](https://github.com/InternLM/InternLM/tree/main) for advanced text-image comprehension and composition. InternLM-XComposer has serveal appealing properties:

- **Interleaved Text-Image Composition**: InternLM-XComposer can effortlessly generate coherent and contextual articles that seamlessly integrate images, providing a more engaging and immersive reading experience. The interleaved text-image composition is implemented in following steps:

    1. **Text Generation**: It crafts long-form text based on human-provided instructions.
    2. **Image Spoting and Captioning**: It pinpoints optimal locations for image placement and furnishes image descriptions.
    3. **Image Retrieval and Selection**: It select image candidates and identify the image that optimally complements the content.

- **Comprehension with Rich Multilingual Knowledge**: The text-image comprehension is empowered by training on extensive multi-modal multilingual concepts with carefully crafted strategies, resulting in a deep understanding of visual content.
- **Strong performance**: It consistently achieves state-of-the-art results across various benchmarks for vision-language large models, including [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) (English), [MMBench](https://opencompass.org.cn/leaderboard-multimodal) (English), [Seed-Bench](https://huggingface.co/spaces/AILab-CVC/SEED-Bench_Leaderboard) (English), [CCBench](https://opencompass.org.cn/leaderboard-multimodal)(Chinese), and [MMBench-CN](https://opencompass.org.cn/leaderboard-multimodal) (Chineese).

We release InternLM-XComposer series in two versions:

- InternLM-XComposer-VL: The pretrained VLLM model with InternLM as the initialization of the LLM, achieving strong performance on various multimodal benchmarks, e.g., MME Benchmark, MMBench Seed-Bench, CCBench, and MMBench-CN.
- InternLM-XComposer: The finetuned VLLM for *Interleaved Text-Image Composition* and *LLM-based AI assistant*.
  <br>