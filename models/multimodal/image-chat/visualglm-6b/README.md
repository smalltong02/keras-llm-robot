---
language:
- zh
- en
tags:
- glm
- visualglm
- chatglm
- thudm
---
# VisualGLM-6B
<p align="center">
   💻 <a href="https://github.com/THUDM/VisualGLM-6B" target="_blank">Github Repo</a> • 🐦 <a href="https://twitter.com/thukeg" target="_blank">Twitter</a> • 📃 <a href="https://arxiv.org/abs/2103.10360" target="_blank">[GLM@ACL 22]</a> <a href="https://github.com/THUDM/GLM" target="_blank">[GitHub]</a> • 📃 <a href="https://arxiv.org/abs/2210.02414" target="_blank">[GLM-130B@ICLR 23]</a> <a href="https://github.com/THUDM/GLM-130B" target="_blank">[GitHub]</a> <br>
</p>

<p align="center">
    👋 Join our <a href="https://join.slack.com/t/chatglm/shared_invite/zt-1th2q5u69-7tURzFuOPanmuHy9hsZnKA" target="_blank">Slack</a> and <a href="https://github.com/THUDM/ChatGLM-6B/blob/main/resources/WECHAT.md" target="_blank">WeChat</a>
</p>

## 介绍
VisualGLM-6B 是一个开源的，支持**图像、中文和英文**的多模态对话语言模型，语言模型基于 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)，具有 62 亿参数；图像部分通过训练 [BLIP2-Qformer](https://arxiv.org/abs/2301.12597) 构建起视觉模型与语言模型的桥梁，整体模型共78亿参数。

VisualGLM-6B 依靠来自于 [CogView](https://arxiv.org/abs/2105.13290) 数据集的30M高质量中文图文对，与300M经过筛选的英文图文对进行预训练，中英文权重相同。该训练方式较好地将视觉信息对齐到ChatGLM的语义空间；之后的微调阶段，模型在长视觉问答数据上训练，以生成符合人类偏好的答案。

## 软件依赖

```shell
pip install SwissArmyTransformer>=0.3.6 torch>1.10.0 torchvision transformers>=4.27.1 cpm_kernels
```

## 代码调用 

可以通过如下代码调用 VisualGLM-6B 模型来生成对话：

```ipython
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("THUDM/visualglm-6b", trust_remote_code=True).half().cuda()
>>> image_path = "your image path"
>>> response, history = model.chat(tokenizer, image_path, "描述这张图片。", history=[])
>>> print(response)
>>> response, history = model.chat(tokenizer, image_path, "这张图片可能是在什么场所拍摄的？", history=history)
>>> print(response)
```

关于更多的使用说明，包括如何运行命令行和网页版本的 DEMO，以及使用模型量化以节省显存，请参考我们的 [Github Repo](https://github.com/THUDM/VisualGLM-6B)。

For more instructions, including how to run CLI and web demos, and model quantization, please refer to our [Github Repo](https://github.com/THUDM/VisualGLM-6B).

## 协议

本仓库的代码依照 [Apache-2.0](LICENSE) 协议开源，VisualGLM-6B 模型的权重的使用则需要遵循 [Model License](MODEL_LICENSE)。

## 引用

如果你觉得我们的工作有帮助的话，请考虑引用下列论文：

```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
```
@article{ding2021cogview,
  title={Cogview: Mastering text-to-image generation via transformers},
  author={Ding, Ming and Yang, Zhuoyi and Hong, Wenyi and Zheng, Wendi and Zhou, Chang and Yin, Da and Lin, Junyang and Zou, Xu and Shao, Zhou and Yang, Hongxia and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={19822--19835},
  year={2021}
}
```