# Randeng-T5-77M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLT任务，中文版的mT5-small。

Good at handling NLT tasks, Chinese mT5-small.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | mT5 |      77M      |     中文-Chinese    |

## 模型信息 Model Information

我们基于mT5-small，训练了它的中文版。为了加速训练，我们仅使用T5分词器(sentence piece)中的中英文对应的词表，并且使用了语料库自适应预训练(Corpus-Adaptive Pre-Training, CAPT)技术在悟道语料库(180G版本)继续预训练。预训练目标为破坏span。具体地，我们在预训练阶段中使用了[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)大概花费了8张A100约24小时。

Based on mT5-small, we implement its Chinese version. In order to accelerate training, we only retrain the vocabulary and embedding corresponding to Chinese and English in T5tokenizer (sentence piece), and Corpus-Adaptive Pre-Training (CAPT) on the WuDao Corpora (180 GB version). The pretraining objective is span corruption. Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 24 hours with 8 A100 GPUs.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Randeng-T5-77M](https://huggingface.co/IDEA-CCNL/Randeng-T5-77M)

### 加载模型 Loading Models

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-T5-77M', use_fast=false)
model=T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-T5-77M')
```

### 数据处理 Data Processing

通用的数据举例：
```
text: '运动,走势在什么时候结束是不可能有答案的。为了找到走势什么时候结束原来的运动方向而改变方向,必须引进新的概念:中枢。'
```
span corruption后的数据举例
```
input: '运动,走势在什么时候结束是不可能有答案的。为了 <extra_id_0>走势什么时候结束原来 <extra_id_1>必须引进新的概念:中枢。'

label: '<extra_id_0>找到 <extra_id_1>的运动方向而改变方向,\</s>'
```
对应的代码见：[地址](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/data/t5_dataloader/t5_datasets.py)。

### 模型训练 Training

模型利用封神框架在2张A100训练17小时，最后loss收敛到2.3左右，训练脚本见：[地址](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/pretrain_t5/pretrain_mt5_small.sh)。

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
