## Erlangshen-Roberta-110M-Similarity

Erlangshen-Roberta-110M-Similarity 是使用文本相似度数据集 fine-tune 过的模型，可以直接用于文本相似度任务。110M参数和330M参数的模型主要基于 [roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，1.3B参数的模型主要基于 [MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)，共收集了 20 份共 2773880 条样本。


### Finetune 样本示例
```
{
  "texta": "可以換其他银行卡吗？", 
  "textb": "分期的如何用别的银行卡还钱", 
  "label": 1, 
  "id": 0
  }
```

### 标签映射
模型输出0表示不相似，输出1表示相似
```
"id2label":{
    "0":"not similarity",
    "1":"similarity"
     }
```

### 模型下载
我们共训练了3个不同参数的模型（点击可跳转到模型下载地址页面）
- [Erlangshen-Roberta-110M-Similarity](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Similarity)
- [Erlangshen-Roberta-330M-Similarity](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Similarity)
- [Erlangshen-MegatronBert-1.3B-Similarity](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Similarity)


### 测评结果（dev集,BUSTM和AFQMC任务的dev集有些样本可能在训练集出现过）

|                  Model                  |  BQ   | BUSTM | AFQMC |
| :-------------------------------------: | :---: | :---: | :---: |
|   Erlangshen-Roberta-110M-Similarity    | 85.41 | 95.18 | 81.72 |
|   Erlangshen-Roberta-330M-Similarity    | 86.21 | 99.29 | 93.89 |
| Erlangshen-MegatronBert-1.3B-Similarity | 86.31 |   -   |   -   |


### 使用示例
```python
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


### Citation
如果你觉得我们的模型对你有用，你可以使用下面的引用方式进行引用。
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```