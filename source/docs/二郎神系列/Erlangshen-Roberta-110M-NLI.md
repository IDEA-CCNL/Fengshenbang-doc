## Erlangshen-Roberta-110M-NLI

Erlangshen-Roberta-110M-NLI 是使用 NLI（自然语言推理） 数据集 fine-tune 过的模型，可以直接用于 NLI 任务。模型主要基于 [roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，共收集了 4 份共 1014787 条公开的 NLI 样本。

### 标签映射
模型输出0表示两个句子矛盾，1表示没有关系，2表示蕴含关系
```
"id2label": {
    "0": "CONTRADICTION",
    "1": "NEUTRAL",
    "2": "ENTAILMENT"
  },
```
### 模型下载
我们共训练了3个不同参数的模型（点击可跳转到模型下载地址页面）
- [Erlangshen-Roberta-110M-NLI](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-NLI)
- [Erlangshen-Roberta-330M-NLI](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-NLI)
- [Erlangshen-MegatronBert-1.3B-NLI](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-NLI)


### 测评结果（dev集）
|    Model   | cmnli    |  ocnli  | snli    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-NLI | 80.83     |   78.56    | 88.01      |
| Erlangshen-Roberta-330M-NLI | 82.25      |   79.82    | 88      |  
| Erlangshen-MegatronBert-1.3B-NLI | 84.52      |   84.17    | 88.67      |  


### 使用示例
```python
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-NLI')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-NLI')

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