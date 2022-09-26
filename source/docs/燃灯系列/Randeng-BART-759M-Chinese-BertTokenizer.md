# Randeng-BART-759M-Chinese-BertTokenizer

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLT任务，使用BERT分词器，大规模的中文版的BART。

Good at solving NLT tasks, applying the BERT tokenizer, a large-scale Chinese BART.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | BART |      759M      |     中文-BERT分词器 Chinese-BERTTokenizer    |

## 模型信息 Model Information

为了得到一个大规模的中文版的BART(约BART-large的两倍)，我们用悟道语料库(180G版本)进行预训练。具体地，我们在预训练阶段中使用了[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)大概花费了8张A100约7天。值得注意的是，因为BERT分词器通常在中文任务中表现比其他分词器好，所以我们使用了它。我们也开放了我们预训练的代码：[pretrain_randeng_bart](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/pretrain_randeng_bart)。

To obtain a large-scale Chinese BART (around twice as large as BART-large), we use WuDao Corpora (180 GB version) for pre-training. Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 7 days with 8 A100 GPUs. Note that since the BERT tokenizer usually performs better than others for Chinese tasks, we employ it. We have also released our pre-training code: [pretrain_randeng_bart](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/pretrain_randeng_bart).

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Randeng-BART-759M-Chinese-BertTokenizer](https://huggingface.co/IDEA-CCNL/Randeng-BART-759M-Chinese-BertTokenizer)

### 加载模型 Loading Models

```python
from transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-BART-759M-Chinese-BertTokenizer', use_fast=false)
model=BartForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-BART-759M-Chinese-BertTokenizer')
text = '桂林是著名的[MASK]，它有很多[MASK]。'
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
print(text2text_generator(text, max_length=50, do_sample=False))
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
