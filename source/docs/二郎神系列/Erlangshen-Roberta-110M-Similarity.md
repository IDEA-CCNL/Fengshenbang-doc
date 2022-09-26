# Erlangshen-Roberta-110M-Similarity

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

中文的RoBERTa-wwm-ext-base在数个相似度任务微调后的版本

This is the fine-tuned version of the Chinese RoBERTa-wwm-ext-base model on several similarity datasets.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | Roberta |      110M      |    相似度 Similarity     |

## 模型信息 Model Information

基于[chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext-base)，我们在收集的20个中文领域的改写数据集，总计2773880个样本上微调了一个Similarity版本。

Based on [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext-base), we fine-tuned a similarity version on 20 Chinese paraphrase datasets, with totaling 2,773,880 samples.

### 下游效果 Performance

|    Model   | BQ    |  BUSTM  | AFQMC    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-Similarity | 85.41     |   95.18    | 81.72     |
| Erlangshen-Roberta-330M-Similarity | 86.21      |   99.29    | 93.89      |  
| Erlangshen-MegatronBert-1.3B-Similarity | 86.31      |   -    | -      |   

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-Roberta-110M-Similarity](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Similarity)

### 加载模型 Loading Models

``` python
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Similarity')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Similarity')

texta='今天的饭不好吃'
textb='今天心情不好'

output=model(torch.tensor([tokenizer.encode(texta,textb)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
```

### 数据样本示例 Data Examples

```
{
  "texta": "可以換其他银行卡吗？", 
  "textb": "分期的如何用别的银行卡还钱", 
  "label": 1, 
  "id": 0
  }
```

标签映射：模型输出0表示不相似，输出1表示相似

```
"id2label":{
    "0":"not similarity",
    "1":"similarity"
     }
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