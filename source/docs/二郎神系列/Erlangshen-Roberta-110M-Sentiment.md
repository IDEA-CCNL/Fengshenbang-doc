# Erlangshen-Roberta-110M-Sentiment

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

中文的RoBERTa-wwm-ext-base在数个情感分析任务微调后的版本

This is the fine-tuned version of the Chinese RoBERTa-wwm-ext-base model on several sentiment analysis datasets.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | Roberta |      110M      |    情感分析 Sentiment     |

## 模型信息 Model Information

基于[chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext-base)，我们在收集的8个中文领域的情感分析数据集，总计227347个样本上微调了一个Semtiment版本。

Based on [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext-base), we fine-tuned a sentiment analysis version on 8 Chinese sentiment analysis datasets, with totaling 227,347 samples.

### 下游效果 Performance

|    模型 Model   | ASAP-SENT    |  ASAP-ASPECT  | ChnSentiCorp    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-Sentiment | 97.77     |   97.31    | 96.61     |
| Erlangshen-Roberta-330M-Sentiment | 97.9      |   97.51    | 96.66      |  
| Erlangshen-MegatronBert-1.3B-Sentiment | 98.1     |   97.8    | 97      | 

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment)

### 加载模型 Loading Models

``` python
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')

text='今天心情不好'

output=model(torch.tensor([tokenizer.encode(text)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
```

### 数据样本示例

```
{
  "texta": "外形还OK,用了2天了在VISTA下玩游戏还行的.发热量有时大有时小不知道为什么,不过总体上来说还不是很大,4600买的还送个大礼包.", 
  "textb": "", 
  "label": 1, 
  "id": "33"
    }
```

标签映射：模型输出0表示消极，输出1表示积极

```
"id2label":{
      "0":"Negative",
      "1":"Positive"
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