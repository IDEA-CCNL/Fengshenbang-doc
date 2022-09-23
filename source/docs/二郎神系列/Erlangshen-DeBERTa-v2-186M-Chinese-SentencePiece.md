# Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLU任务，采用sentence piece分词的，中文版的1.86亿参数DeBERTa-v2。

Good at solving NLU tasks, adopting sentence piece, Chinese DeBERTa-v2 with 186M parameters.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | DeBERTa-v2 |      186M      |    中文-分句 Chinese-SentencePiece     |

## 模型信息 Model Information

为了得到一个中文版的DeBERTa-v2（186M），我们用悟道语料库(180G版本)进行预训练。我们使用了Sentence Piece的方式分词（词表大小：约128000）。具体地，我们在预训练阶段中使用了[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)大概花费了8张3090TI（24G）约21天。

To get a Chinese DeBERTa-v2 (186M), we use WuDao Corpora (180 GB version) for pre-training. We employ the sentence piece as the tokenizer (vocabulary size: around 128,000). Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 21 days with 8 3090TI（24G） GPUs.

### 下游效果 Performance

我们展示了下列下游任务的结果（dev集）：

We present the results (dev set) on the following tasks:

| Model                                                | OCNLI  | CMNLI  |
| ---------------------------------------------------- | ------ | ------ |
| RoBERTa-base                                         | 0.743  | 0.7973 |
| **Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece** | 0.7625 | 0.8100 |

## 使用 Usage

### 模型下载地址
[Huggingface地址：Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece)

### 加载模型 Loading Models

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece', use_fast=False)
model=AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece')
text = '中国首都位于<mask>。'
fillmask_pipe = FillMaskPipeline(model, tokenizer)
print(fillmask_pipe(text, top_k=10))
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen and Ruyi Gan and Jiaxing Zhang},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```