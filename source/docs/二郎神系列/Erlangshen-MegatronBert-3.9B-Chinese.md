# Erlangshen-MegatronBert-3.9B-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLU任务，现在最大的，拥有39亿的中文BERT模型。

Good at solving NLU tasks, the largest Chinese BERT (39B) currently.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | MegatronBERT |      3.9B      |    中文 Chinese     |

## 模型信息 Model Information

Erlangshen-MegatronBert-3.9B-Chinese是一个比[Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)拥有更多参数的版本（39亿）。我们遵循原来的预训练方式在悟道数据集（300G版本）上进行预训练。具体地，我们在预训练阶段中使用了封神框架大概花费了64张A100（40G）约30天。

Erlangshen-MegatronBert-3.9B-Chinese (3.9B) is a larger version of [Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B). By following the original instructions, we apply WuDao Corpora (300 GB version) as the pretraining dataset. Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 30 days with 64 A100 (40G) GPUs.

### 更多信息 More Information

[IDEA研究院中文预训练模型二郎神登顶FewCLUE榜单](https://mp.weixin.qq.com/s/bA_9n_TlBE9P-UzCn7mKoA)

2021年11月10日，Erlangshen-MegatronBERT-1.3B在FewCLUE上取得第一。其中，它在CHIDF(成语填空)和TNEWS(新闻分类)子任务中的表现优于人类表现。此外，它在CHIDF(成语填空), CSLDCP(学科文献分类), OCNLI(自然语言推理)任务中均名列前茅。

On November 10, 2021, Erlangshen-MegatronBert-1.3B topped the FewCLUE benchmark. Among them, our Erlangshen outperformed human performance in CHIDF (idiom fill-in-the-blank) and TNEWS (news classification) subtasks. In addition, our Erlangshen ranked the top in CHIDF (idiom fill-in-the-blank), CSLDCP (subject literature classification), and OCNLI (natural language inference) tasks.  

### 下游效果 Performance

下游中文任务的得分（没有做任何数据增强）:

Scores on downstream Chinese tasks (without any data augmentation):

|                                                 Model                                                 | afqmc  | tnews  | iflytek | ocnli  | cmnli  |  wsc   |  csl  |
| :---------------------------------------------------------------------------------------------------: | :----: | :----: | :-----: | :----: | :----: | :----: | :---: |
|                                         roberta-wwm-ext-large                                         | 0.7514 | 0.5872 | 0.6152  | 0.777  | 0.814  | 0.8914 | 0.86  |
|     [Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)     | 0.7608 | 0.5996 | 0.6234  | 0.7917 |  0.81  | 0.9243 | 0.872 |
| [Erlangshen-MegatronBert-3.9B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese) | 0.7561 | 0.6048 | 0.6204  | 0.8278 | 0.8517 |   -    |   -   |

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-MegatronBert-3.9B-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese)

### 加载模型 Loading Models

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer, FillMaskPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese', use_fast=False)
model=AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-3.9B-Chinese')
text = '生活的真谛是[MASK]。'
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