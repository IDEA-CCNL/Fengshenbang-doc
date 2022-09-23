# Erlangshen-MegatronBert-1.3B-Similarity

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

2021年登顶FewCLUE和ZeroCLUE的中文BERT，在数个相似度任务上微调后的版本。

This is the fine-tuned version of the Chinese BERT model on several similarity datasets, which topped FewCLUE and ZeroCLUE benchmark in 2021.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | MegatronBert |      1.3B      |    相似度 Similarity     |

## 模型信息 Model Information

基于[Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)，我们在收集的20个中文领域的改写数据集，总计2,773,880个样本上微调了一个Similarity版本。

Based on [Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B), we fine-tuned a similarity version on 20 Chinese paraphrase datasets, with totaling 2,773,880 samples.

### 成就 Achievements

我们于2022年7月10日登顶CLUE语义匹配榜，详情见[Towards No.1 in CLUE Semantic Matching Challenge: Pre-trained Language Model Erlangshen with Propensity-Corrected Loss](https://arxiv.org/abs/2208.02959)。

We topped the CLUE benchmark semantic matching task on July 10, 2022, as detailed in [Towards No.1 in CLUE Semantic Matching Challenge: Pre-trained Language Model Erlangshen with Propensity-Corrected Loss](https://arxiv.org/abs/2208.02959).

### 下游效果 Performance

|    Model   | BQ    |  BUSTM  | AFQMC    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-Similarity | 85.41     |   95.18    | 81.72     |
| Erlangshen-Roberta-330M-Similarity | 86.21      |   99.29    | 93.89      |  
| Erlangshen-MegatronBert-1.3B-Similarity | 86.31      |   -    | -      |  

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-MegatronBert-1.3B-Similarity](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity)

### 加载模型 Loading Models

``` python
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity')
model=AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity')

texta='今天的饭不好吃'
textb='今天心情不好'

output=model(torch.tensor([tokenizer.encode(texta,textb)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
```

### 数据样本示例

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

如果您在您的工作中使用了我们的模型，可以引用我们的对该模型的论文：

If you are using the resource for your work, please cite the our paper for this model:

```text
@article{fengshenbang/erlangshen-megatronbert-sim,
  author    = {Junjie Wang and
               Yuxiang Zhang and
               Ping Yang and
               Ruyi Gan},
  title     = {Towards No.1 in {CLUE} Semantic Matching Challenge: Pre-trained Language
               Model Erlangshen with Propensity-Corrected Loss},
  journal   = {CoRR},
  volume    = {abs/2208.02959},
  year      = {2022}
}
```

如果您在您的工作中使用了我们的模型，也可以引用我们的[总论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [overview paper](https://arxiv.org/abs/2209.02970):

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