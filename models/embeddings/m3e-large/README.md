---
language:
- zh
- en
tags:
- embedding
- text-embedding
library_name: sentence-transformers
---

# M3E Models

[m3e-small](https://huggingface.co/moka-ai/m3e-small) | [m3e-base](https://huggingface.co/moka-ai/m3e-base) | [m3e-large](https://huggingface.co/moka-ai/m3e-large)

M3E 是 Moka Massive Mixed Embedding 的缩写

- Moka，此模型由 MokaAI 训练，开源和评测，训练脚本使用 [uniem](https://github.com/wangyuxinwhy/uniem/blob/main/scripts/train_m3e.py) ，评测 BenchMark 使用 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh)
- Massive，此模型通过**千万级** (2200w+) 的中文句对数据集进行训练
- Mixed，此模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索
- Embedding，此模型是文本嵌入模型，可以将自然语言转换成稠密的向量

## 更新说明

- 2023.06.14，添加了三个中文开源文本嵌入模型到评测中，包括 UER, ErLangShen, DMetaSoul
- 2023.06.08，添加检索任务的评测结果，在 T2Ranking 1W 中文数据集上，m3e-base 在 ndcg@10 上达到了 0.8004，超过了 openai-ada-002 的 0.7786
- 2023.06.07，添加文本分类任务的评测结果，在 6 种文本分类数据集上，m3e-base 在 accuracy 上达到了 0.6157，超过了 openai-ada-002 的 0.5956

## 模型对比

|           | 参数数量 | 维度 | 中文 | 英文 | s2s | s2p | s2c | 开源 | 兼容性 | s2s Acc | s2p ndcg@10 |
| --------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | ---- | ---------- | ------------ | -------- |
| m3e-small | 24M      | 512      | 是       | 否       | 是       | 否       | 否       | 是   | 优         | 0.5834       | 0.7262   |
| m3e-base  | 110M     | 768      | 是       | 是       | 是       | 是       | 否       | 是   | 优         | 0.6157       | **0.8004**   |
| m3e-large | 340M | 768 | 是 | 否 | 是 | 是 | 否 | 是 | 优 | **0.6231** | 0.7974 |
| text2vec  | 110M     | 768      | 是       | 否       | 是       | 否       | 否       | 是   | 优         | 0.5755       | 0.6346   |
| openai-ada-002    | 未知     | 1536     | 是       | 是       | 是       | 是       | 是       | 否   | 优         | 0.5956       | 0.7786   |

说明：
- s2s, 即 sentence to sentence ，代表了同质文本之间的嵌入能力，适用任务：文本相似度，重复问题检测，文本分类等
- s2p, 即 sentence to passage ，代表了异质文本之间的嵌入能力，适用任务：文本检索，GPT 记忆模块等
- s2c, 即 sentence to code ，代表了自然语言和程序语言之间的嵌入能力，适用任务：代码检索
- 兼容性，代表了模型在开源社区中各种项目被支持的程度，由于 m3e 和 text2vec 都可以直接通过 sentence-transformers 直接使用，所以和 openai 在社区的支持度上相当
- ACC & ndcg@10，详情见下方的评测

Tips:
- 使用场景主要是中文，少量英文的情况，建议使用 m3e 系列的模型
- 多语言使用场景，并且不介意数据隐私的话，我建议使用 openai text-embedding-ada-002
- 代码检索场景，推荐使用 openai text-embedding-ada-002
- 文本检索场景，请使用具备文本检索能力的模型，只在 S2S 上训练的文本嵌入模型，没有办法完成文本检索任务

## 使用方式

您需要先安装 sentence-transformers

```bash
pip install -U sentence-transformers
```

安装完成后，您可以使用以下代码来使用 M3E Models

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('moka-ai/m3e-base')

#Our sentences we like to encode
sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]

#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)

#Print the embeddings
for sentence, embedding in zip(sentences, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")
```


M3E 系列的所有模型在设计的时候就考虑到完全兼容 [sentence-transformers](https://www.sbert.net/) ，所以你可以通过**替换名称字符串**的方式在所有支持 sentence-transformers 的项目中**无缝**使用 M3E Models，比如 [chroma](https://docs.trychroma.com/getting-started), [guidance](https://github.com/microsoft/guidance), [semantic-kernel](https://github.com/microsoft/semantic-kernel) 。


## 训练方案

M3E 使用 in-batch 负采样的对比学习的方式在句对数据集进行训练，为了保证 in-batch 负采样的效果，我们使用 A100 80G 来最大化 batch-size，并在共计 2200W+ 的句对数据集上训练了 1 epoch。训练脚本使用 [uniem](https://github.com/wangyuxinwhy/uniem/blob/main/scripts/train_m3e.py)，您可以在这里查看具体细节。

## 特性

- 中文训练集，M3E 在大规模句对数据集上的训练，包含中文百科，金融，医疗，法律，新闻，学术等多个领域共计 2200W 句对样本，数据集详见 [M3E 数据集](#M3E数据集)
- 英文训练集，M3E 使用 MEDI 145W 英文三元组数据集进行训练，数据集详见 [MEDI 数据集](https://drive.google.com/file/d/1vZ5c2oJNonGOvXzppNg5mHz24O6jcc52/view)，此数据集由 [instructor team](https://github.com/HKUNLP/instructor-embedding) 提供
- 指令数据集，M3E 使用了 300W + 的指令微调数据集，这使得 M3E 对文本编码的时候可以遵从指令，这部分的工作主要被启发于 [instructor-embedding](https://github.com/HKUNLP/instructor-embedding)
- 基础模型，M3E 使用 hfl 实验室的 [Roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext) 系列模型进行训练，目前提供  small、base和large三个版本，大家则需选用
- ALL IN ONE，M3E 旨在提供一个 ALL IN ONE 的文本嵌入模型，不仅支持同质句子相似度判断，还支持异质文本检索，你只需要一个模型就可以覆盖全部的应用场景，未来还会支持代码检索

## 评测

- 评测模型，[text2vec](https://github.com/shibing624/text2vec), m3e-base, m3e-small, openai text-embedding-ada-002, [DMetaSoul](https://huggingface.co/DMetaSoul/sbert-chinese-general-v2), [UER](https://huggingface.co/uer/sbert-base-chinese-nli), [ErLangShen](https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese)
- 评测脚本，具体参考 [MTEB-zh] (https://github.com/wangyuxinwhy/uniem/blob/main/mteb-zh)

### 文本分类

- 数据集选择，选择开源在 HuggingFace 上的 6 种文本分类数据集，包括新闻、电商评论、股票评论、长文本等
- 评测方式，使用 MTEB 的方式进行评测，报告 Accuracy。

|                   | text2vec | m3e-small | m3e-base | m3e-large | openai | DMetaSoul   | uer     | erlangshen  |
| ----------------- | -------- | --------- | -------- | ------ | ----------- | ------- | ----------- | ----------- |
| TNews             | 0.43     | 0.4443    | 0.4827   | **0.4866** | 0.4594 | 0.3084      | 0.3539  | 0.4361      |
| JDIphone          | 0.8214   | 0.8293    | 0.8533   | **0.8692** | 0.746  | 0.7972      | 0.8283  | 0.8356      |
| GubaEastmony      | 0.7472   | 0.712     | 0.7621   | 0.7663 | 0.7574 | 0.735       | 0.7534  | **0.7787**      |
| TYQSentiment      | 0.6099   | 0.6596    | 0.7188   | **0.7247** | 0.68   | 0.6437      | 0.6662  | 0.6444      |
| StockComSentiment | 0.4307   | 0.4291    | 0.4363   | 0.4475 | **0.4819** | 0.4309      | 0.4555  | 0.4482      |
| IFlyTek           | 0.414    | 0.4263    | 0.4409   | 0.4445 | **0.4486** | 0.3969      | 0.3762  | 0.4241      |
| Average           | 0.5755   | 0.5834    | 0.6157   | **0.6231** | 0.5956 | 0.552016667 | 0.57225 | 0.594516667 |

### 检索排序

#### T2Ranking 1W

- 数据集选择，使用 [T2Ranking](https://github.com/THUIR/T2Ranking/tree/main) 数据集，由于 T2Ranking 的数据集太大，openai 评测起来的时间成本和 api 费用有些高，所以我们只选择了 T2Ranking 中的前 10000 篇文章
- 评测方式，使用 MTEB 的方式进行评测，报告 map@1, map@10, mrr@1, mrr@10, ndcg@1, ndcg@10
- 注意！从实验结果和训练方式来看，除了 M3E 模型和 openai 模型外，其余模型都没有做检索任务的训练，所以结果仅供参考。

|         | text2vec | openai-ada-002 | m3e-small | m3e-base | m3e-large | DMetaSoul | uer     | erlangshen |
| ------- | -------- | -------------- | --------- | -------- | --------- | ------- | ---------- | ---------- |
| map@1   | 0.4684   | 0.6133         | 0.5574    | **0.626**    | 0.6256 | 0.25203   | 0.08647 | 0.25394    |
| map@10  | 0.5877   | 0.7423         | 0.6878    | **0.7656**   | 0.7627 | 0.33312   | 0.13008 | 0.34714    |
| mrr@1   | 0.5345   | 0.6931         | 0.6324    | 0.7047   | **0.7063** | 0.29258   | 0.10067 | 0.29447    |
| mrr@10  | 0.6217   | 0.7668         | 0.712     | **0.7841**   | 0.7827 | 0.36287   | 0.14516 | 0.3751     |
| ndcg@1  | 0.5207   | 0.6764         | 0.6159    | 0.6881   | **0.6884** | 0.28358   | 0.09748 | 0.28578    |
| ndcg@10 | 0.6346   | 0.7786         | 0.7262    | **0.8004**   | 0.7974 | 0.37468   | 0.15783 | 0.39329    |

#### T2Ranking

- 数据集选择，使用 T2Ranking，刨除 openai-ada-002 模型后，我们对剩余的三个模型，进行 T2Ranking 10W 和 T2Ranking 50W 的评测。（T2Ranking 评测太耗内存了... 128G 都不行）
- 评测方式，使用 MTEB 的方式进行评测，报告 ndcg@10

|         | text2vec | m3e-small | m3e-base |
| ------- | -------- | --------- | -------- |
| t2r-1w  | 0.6346   | 0.72621   | **0.8004**   |
| t2r-10w | 0.44644  | 0.5251    | **0.6263**   |
| t2r-50w | 0.33482  | 0.38626   | **0.47364**  |

说明：
- 检索排序对于 text2vec 并不公平，因为 text2vec 在训练的时候没有使用过检索相关的数据集，所以没有办法很好的完成检索任务也是正常的。

## M3E数据集

如果您想要使用这些数据集，你可以在 [uniem process_zh_datasets](https://github.com/wangyuxinwhy/uniem/blob/main/scripts/process_zh_datasets.py) 中找到加载 huggingface 数据集的脚本，非 huggingface 数据集需要您根据下方提供的链接自行下载和处理。

| 数据集名称           | 领域 | 数量      | 任务类型          | Prompt | 质量 | 数据提供者                                                   | 说明                                                         | 是否开源/研究使用 | 是否商用 | 脚本 | Done | URL                                                          | 是否同质 |
| -------------------- | ---- | --------- | ----------------- | ------ | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------- | -------- | ---- | ---- | ------------------------------------------------------------ | -------- |
| cmrc2018             | 百科 | 14,363    | 问答              | 问答   | 优   | Yiming Cui, Ting Liu, Wanxiang Che, Li Xiao, Zhipeng Chen, Wentao Ma, Shijin Wang, Guoping Hu | https://github.com/ymcui/cmrc2018/blob/master/README_CN.md 专家标注的基于维基百科的中文阅读理解数据集，将问题和上下文视为正例 | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/cmrc2018                     | 否       |
| belle_2m             | 百科 | 2,000,000 | 指令微调          | 无     | 优   | LianjiaTech/BELLE                                            | belle 的指令微调数据集，使用 self instruct 方法基于 gpt3.5 生成 | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/BelleGroup/train_2M_CN       | 否       |
| firefily             | 百科 | 1,649,399 | 指令微调          | 无     | 优   | YeungNLP                                                     | Firefly（流萤） 是一个开源的中文对话式大语言模型，使用指令微调（Instruction Tuning）在中文数据集上进行调优。使用了词表裁剪、ZeRO等技术，有效降低显存消耗和提高训练效率。 在训练中，我们使用了更小的模型参数量，以及更少的计算资源。 | 未说明            | 未说明   | 是   | 是   | https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M  | 否       |
| alpaca_gpt4          | 百科 | 48,818    | 指令微调          | 无     | 优   | Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao | 本数据集是参考Alpaca方法基于GPT4得到的self-instruct数据，约5万条。 | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/shibing624/alpaca-zh         | 否       |
| zhihu_kol            | 百科 | 1,006,218 | 问答              | 问答   | 优   | wangrui6                                                     | 知乎问答                                                     | 未说明            | 未说明   | 是   | 是   | https://huggingface.co/datasets/wangrui6/Zhihu-KOL           | 否       |
| hc3_chinese          | 百科 | 39,781    | 问答              | 问答   | 良   | Hello-SimpleAI                                               | 问答数据，包括人工回答和 GPT 回答                            | 是                | 未说明   | 是   | 是   | https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese   | 否       |
| amazon_reviews_multi | 电商 | 210,000   | 问答 文本分类     | 摘要   | 优   | 亚马逊                                                       | 亚马逊产品评论数据集                                         | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/amazon_reviews_multi/viewer/zh/train?row=8 | 否       |
| mlqa                 | 百科 | 85,853    | 问答              | 问答   | 良   | patrickvonplaten                                             | 一个用于评估跨语言问答性能的基准数据集                       | 是                | 未说明   | 是   | 是   | https://huggingface.co/datasets/mlqa/viewer/mlqa-translate-train.zh/train?p=2 | 否       |
| xlsum                | 新闻 | 93,404    | 摘要              | 摘要   | 良   | BUET CSE NLP Group                                           | BBC的专业注释文章摘要对                                      | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/chinese_simplified/train?row=259 | 否       |
| ocnli                | 口语 | 17,726    | 自然语言推理      | 推理   | 良   | Thomas Wolf                                                  | 自然语言推理数据集                                           | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/clue/viewer/ocnli            | 是       |
| BQ                   | 金融 | 60,000    | 文本分类          | 相似   | 良   | Intelligent Computing Research Center, Harbin Institute of Technology(Shenzhen) | http://icrc.hitsz.edu.cn/info/1037/1162.htm BQ 语料库包含来自网上银行自定义服务日志的 120，000 个问题对。它分为三部分：100，000 对用于训练，10，000 对用于验证，10，000 对用于测试。 数据提供者： 哈尔滨工业大学（深圳）智能计算研究中心 | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/shibing624/nli_zh/viewer/BQ  | 是       |
| lcqmc                | 口语 | 149,226   | 文本分类          | 相似   | 良   | Ming Xu                                                      | 哈工大文本匹配数据集，LCQMC 是哈尔滨工业大学在自然语言处理国际顶会 COLING2018 构建的问题语义匹配数据集，其目标是判断两个问题的语义是否相同 | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/shibing624/nli_zh/viewer/LCQMC/train | 是       |
| paws-x               | 百科 | 23,576    | 文本分类          | 相似   | 优   | Bhavitvya Malik                                              | PAWS Wiki中的示例                                            | 是                | 是       | 是   | 是   | https://huggingface.co/datasets/paws-x/viewer/zh/train       | 是       |
| wiki_atomic_edit     | 百科 | 1,213,780 | 平行语义          | 相似   | 优   | abhishek thakur                                              | 基于中文维基百科的编辑记录收集的数据集                       | 未说明            | 未说明   | 是   | 是   | https://huggingface.co/datasets/wiki_atomic_edits            | 是       |
| chatmed_consult      | 医药 | 549,326   | 问答              | 问答   | 优   | Wei Zhu                                                      | 真实世界的医学相关的问题，使用 gpt3.5 进行回答               | 是                | 否       | 是   | 是   | https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset | 否       |
| webqa                | 百科 | 42,216    | 问答              | 问答   | 优   | suolyer                                                      | 百度于2016年开源的数据集，数据来自于百度知道；格式为一个问题多篇意思基本一致的文章，分为人为标注以及浏览器检索；数据整体质量中，因为混合了很多检索而来的文章 | 是                | 未说明   | 是   | 是   | https://huggingface.co/datasets/suolyer/webqa/viewer/suolyer--webqa/train?p=3 | 否       |
| dureader_robust      | 百科 | 65,937    | 机器阅读理解 问答 | 问答   | 优   | 百度                                                         | DuReader robust旨在利用真实应用中的数据样本来衡量阅读理解模型的鲁棒性，评测模型的过敏感性、过稳定性以及泛化能力，是首个中文阅读理解鲁棒性数据集。 | 是                | 是       | 是   | 是   | https://huggingface.co/datasets/PaddlePaddle/dureader_robust/viewer/plain_text/train?row=96 | 否       |
| csl                  | 学术 | 395,927   | 语料              | 摘要   | 优   | Yudong Li, Yuqing Zhang, Zhe Zhao, Linlin Shen, Weijie Liu, Weiquan Mao and Hui Zhang | 提供首个中文科学文献数据集（CSL），包含 396,209 篇中文核心期刊论文元信息 （标题、摘要、关键词、学科、门类）。CSL 数据集可以作为预训练语料，也可以构建许多NLP任务，例如文本摘要（标题预测）、 关键词生成和文本分类等。 | 是                | 是       | 是   | 是   | https://huggingface.co/datasets/neuclir/csl                  | 否       |
| miracl-corpus        | 百科 | 4,934,368 | 语料              | 摘要   | 优   | MIRACL                                                       | The corpus for each language is prepared from a Wikipedia dump, where we keep only the plain text and discard images, tables, etc. Each article is segmented into multiple passages using WikiExtractor based on natural discourse units (e.g., \n\n in the wiki markup). Each of these passages comprises a "document" or unit of retrieval. We preserve the Wikipedia article title of each passage. | 是                | 是       | 是   | 是   | https://huggingface.co/datasets/miracl/miracl-corpus         | 否       |
| lawzhidao            | 法律 | 36,368    | 问答              | 问答   | 优   | 和鲸社区-Ustinian                                            | 百度知道清洗后的法律问答                                     | 是                | 是       | 否   | 是   | https://www.heywhale.com/mw/dataset/5e953ca8e7ec38002d02fca7/content | 否       |
| CINLID               | 成语 | 34,746    | 平行语义          | 相似   | 优   | 高长宽                                                       | 中文成语语义推理数据集（Chinese Idioms Natural Language Inference Dataset）收集了106832条由人工撰写的成语对（含少量歇后语、俗语等短文本），通过人工标注的方式进行平衡分类，标签为entailment、contradiction和neutral，支持自然语言推理（NLI）的任务。 | 是                | 否       | 否   | 是   | https://www.luge.ai/#/luge/dataDetail?id=39                  | 是       |
| DuSQL                | SQL  | 25,003    | NL2SQL            | SQL    | 优   | 百度                                                         | DuSQL是一个面向实际应用的数据集，包含200个数据库，覆盖了164个领域，问题覆盖了匹配、计算、推理等实际应用中常见形式。该数据集更贴近真实应用场景，要求模型领域无关、问题无关，且具备计算推理等能力。 | 是                | 否       | 否   | 是   | https://www.luge.ai/#/luge/dataDetail?id=13                  | 否       |
| Zhuiyi-NL2SQL        | SQL  | 45,918    | NL2SQL            | SQL    | 优   | 追一科技 刘云峰                                              | NL2SQL是一个多领域的简单数据集，其主要包含匹配类型问题。该数据集主要验证模型的泛化能力，其要求模型具有较强的领域泛化能力、问题泛化能力。 | 是                | 否       | 否   | 是   | https://www.luge.ai/#/luge/dataDetail?id=12                  | 否       |
| Cspider              | SQL  | 7,785     | NL2SQL            | SQL    | 优   | 西湖大学 张岳                                                | CSpider是一个多语言数据集，其问题以中文表达，数据库以英文存储，这种双语模式在实际应用中也非常常见，尤其是数据库引擎对中文支持不好的情况下。该数据集要求模型领域无关、问题无关，且能够实现多语言匹配。 | 是                | 否       | 否   | 是   | https://www.luge.ai/#/luge/dataDetail?id=11                  | 否       |
| news2016zh           | 新闻 | 2,507,549 | 语料              | 摘要   | 良   | Bright Xu                                                    | 包含了250万篇新闻。新闻来源涵盖了6.3万个媒体，含标题、关键词、描述、正文。 | 是                | 是       | 否   | 是   | https://github.com/brightmart/nlp_chinese_corpus             | 否       |
| baike2018qa          | 百科 | 1,470,142 | 问答              | 问答   | 良   | Bright Xu                                                    | 含有150万个预先过滤过的、高质量问题和答案，每个问题属于一个类别。总共有492个类别，其中频率达到或超过10次的类别有434个。 | 是                | 是       | 否   | 是   | https://github.com/brightmart/nlp_chinese_corpus             | 否       |
| webtext2019zh        | 百科 | 4,258,310 | 问答              | 问答   | 优   | Bright Xu                                                    | 含有410万个预先过滤过的、高质量问题和回复。每个问题属于一个【话题】，总共有2.8万个各式话题，话题包罗万象。 | 是                | 是       | 否   | 是   | https://github.com/brightmart/nlp_chinese_corpus             | 否       |
| SimCLUE              | 百科 | 775,593   | 平行语义          | 相似   | 良   | 数据集合，请在 simCLUE 中查看                                | 整合了中文领域绝大多数可用的开源的语义相似度和自然语言推理的数据集，并重新做了数据拆分和整理。 | 是                | 否       | 否   | 是   | https://github.com/CLUEbenchmark/SimCLUE                     | 是       |
| Chinese-SQuAD        | 新闻 | 76,449    | 机器阅读理解      | 问答   | 优   | junzeng-pluto                                                | 中文机器阅读理解数据集，通过机器翻译加人工校正的方式从原始Squad转换而来 | 是                | 否       | 否   | 是   | https://github.com/pluto-junzeng/ChineseSquad                | 否       |

## 计划表

- [x] 完成 MTEB 中文评测 BenchMark, [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh)
- [ ] 完成 Large 模型的训练和开源
- [ ] 完成支持代码检索的模型
- [ ] 对 M3E 数据集进行清洗，保留高质量的部分，组成 m3e-hq，并在 huggingface 上开源
- [ ] 在 m3e-hq 的数据集上补充 hard negative 的样本及相似度分数，组成 m3e-hq-with-score，并在 huggingface 上开源
- [ ] 在 m3e-hq-with-score 上通过 [cosent loss](https://github.com/wangyuxinwhy/uniem/blob/main/uniem/criteria.py#LL24C39-L24C39) loss 进行训练并开源模型，CoSent 原理参考这篇[博客](https://kexue.fm/archives/8847)
- [ ] 开源商用版本的 M3E models

## 致谢

感谢开源社区提供的中文语料，感谢所有在此工作中提供帮助的人们，希望中文社区越来越好，共勉！

## License

M3E models 使用的数据集中包括大量非商用的数据集，所以 M3E models 也是非商用的，仅供研究使用。不过我们已经在 M3E 数据集上标识了商用和非商用的数据集，您可以根据自己的需求自行训练。

## Citation
Please cite this model using the following format:
```
  @software {Moka Massive Mixed Embedding,  
  author = {Wang Yuxin,Sun Qingxuan,He sicheng},  
  title = {M3E: Moka Massive Mixed Embedding Model},  
  year = {2023}
  }
```