# Erlangshen-MegatronBert-1.3B-NLI

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

2021年登顶FewCLUE和ZeroCLUE的中文BERT，在数个推理任务微调后的版本

This is the fine-tuned version of the Chinese BERT model on several NLI datasets, which topped FewCLUE and ZeroCLUE benchmark in 2021

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | MegatronBert |      1.3B      |    自然语言推断 NLI     |

## 模型信息 Model Information

基于[Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)，我们在收集的4个中文领域的NLI（自然语言推理）数据集，总计1014787个样本上微调了一个NLI版本。

Based on [Erlangshen-MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B), we fine-tuned a NLI version on 4 Chinese Natural Language Inference (NLI) datasets, with totaling 1,014,787 samples.

### 下游效果 Performance

|    模型 Model   | cmnli    |  ocnli  | snli    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-NLI | 80.83     |   78.56    | 88.01      |
| Erlangshen-Roberta-330M-NLI | 82.25      |   79.82    | 88.00 |
| Erlangshen-MegatronBert-1.3B-NLI | 84.52      |   84.17    | 88.67      |  

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-MegatronBert-1.3B-NLI](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI)

### 加载模型 Loading Models

``` python
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
import torch
tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
model=AutoModelForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI')
texta='今天的饭不好吃'
textb='今天心情不好'
output=model(torch.tensor([tokenizer.encode(texta,textb)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
```

### 数据样本示例  Data Examples

```
{
  "texta": "身上裹一件工厂发的棉大衣,手插在袖筒里",
  "textb": "身上至少一件衣服", 
  "label": 2, 
  "id": 0
 }
```

标签映射：模型输出0表示两个句子矛盾，1表示没有关系，2表示蕴含关系

```
"id2label": {
    "0": "CONTRADICTION",
    "1": "NEUTRAL",
    "2": "ENTAILMENT"
  },
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
