# Erlangshen-DeBERTa-v2-320M-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLU任务，采用全词掩码的，中文版的3.2亿参数DeBERTa-v2-Large。

Good at solving NLU tasks, adopting Whole Word Masking, Chinese DeBERTa-v2-Large with 320M parameters.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | DeBERTa-v2 |      320M      |     Chinese     |

## 模型信息 Model Information

参考论文：[DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://readpaper.com/paper/3033187248)

为了得到一个中文版的DeBERTa-v2-large（320M），我们用悟道语料库(180G版本)进行预训练。我们在MLM中使用了全词掩码(wwm)的方式。具体地，我们在预训练阶段中使用了[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)大概花费了8张A100（80G）约7天。

To get a Chinese DeBERTa-v2-large (320M), we use WuDao Corpora (180 GB version) for pre-training. We employ the Whole Word Masking (wwm) in MLM. Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 7 days with 8 A100（80G） GPUs.

### 下游任务 Performance

我们展示了下列下游任务的结果（dev集）：

We present the results (dev set) on the following tasks:

| Model                                                                                                                            | AFQMC  | TNEWS1.1 | IFLYTEK | OCNLI  | CMNLI  |
| -------------------------------------------------------------------------------------------------------------------------------- | ------ | -------- | ------- | ------ | ------ |
| RoBERTa-base                                                                                                                     | 0.7406 | 0.575    | 0.6036  | 0.743  | 0.7973 |
| RoBERTa-large                                                                                                                    | 0.7488 | 0.5879   | 0.6152  | 0.777  | 0.814  |
| [IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-186M-Chinese-SentencePiece) | 0.7405 | 0.571    | 0.5977  | 0.7568 | 0.807  |
| **[IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese)**          | 0.7498 | 0.5817   | 0.6042  | 0.8022 | 0.8301 |
| [IDEA-CCNL/Erlangshen-Deberta-XLarge-710M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-710M-Chinese)          | 0.7549 | 0.5873   | 0.6177  | 0.8012 | 0.8389 |

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-DeBERTa-v2-320M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese)

### 加载模型 Loading Models

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese', use_fast=False)
model=AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese')
text = '桂林是世界闻名的旅游城市,它有[MASK]江。'
fillmask_pipe = FillMaskPipeline(model, tokenizer, device=0)
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