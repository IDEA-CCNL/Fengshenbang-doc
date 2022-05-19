## Erlangshen-Roberta-110M-Sentiment

Erlangshen-Roberta-110M-Sentiment 是使用情感分类数据集 fine-tune 过的模型，可以直接用于情感分析任务。110M参数和330M参数的模型主要基于 [roberta](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，1.3B参数的模型主要基于 [MegatronBert-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)，共收集了 8 份共 227347 条样本。


### Finetune 样本示例
```
{
  "texta": "外形还OK,用了2天了在VISTA下玩游戏还行的.发热量有时大有时小不知道为什么,不过总体上来说还不是很大,4600买的还送个大礼包.", 
  "textb": "", 
  "label": 1, 
  "id": "33"
    }
```

### 标签映射
模型输出0表示消极，输出1表示积极
```
"id2label":{
      "0":"Negative",
      "1":"Positive"
       }
```

### 模型下载
我们共训练了3个不同参数的模型（点击可跳转到模型下载地址页面）
- [Erlangshen-Roberta-110M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment)
- [Erlangshen-Roberta-330M-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment)
- [Erlangshen-MegatronBert-1.3B-Sentiment](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B-Sentiment)


### 测评结果（dev集）


|    Model   | ASAP-SENT    |  ASAP-ASPECT  | ChnSentiCorp    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-Sentiment | 97.77     |   97.31    | 96.61     |
| Erlangshen-Roberta-330M-Sentiment | 97.9      |   97.51    | 96.66      |  
| Erlangshen-MegatronBert-1.3B-Sentiment | 98.1     |   97.8    | 97      |


### 使用示例
```python
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment')

text='今天心情不好'

output=model(torch.tensor([tokenizer.encode(text)]))
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