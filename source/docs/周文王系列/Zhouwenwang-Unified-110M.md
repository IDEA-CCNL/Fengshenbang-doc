## 周文王系列

IDEA研究院认知计算中心联合追一科技有限公司的新结构大模型。该模型在训练阶段就统一考虑LM（Language Model）和MLM（Mask Language Model）任务，增加了旋转位置编码技术，让模型同时具备生成和理解的能力。目前已有13亿参数的周文王-1.3B大模型，是中文领域同时做LM和MLM任务最大的模型，会持续在模型规模、知识融入、监督任务辅助等方向不断优化。


### 模型下载地址

[Huggingface 周文王-1.3B](https://huggingface.co/IDEA-CCNL/Zhouwenwang-Unified-1.3B)<br>
[Huggingface 周文王-110M](https://huggingface.co/IDEA-CCNL/Zhouwenwang-Unified-110M)
### 模型加载
由于我们现在的周文王结构是在追一科技之前的roformer结构进行的修改，而HuggingFace还没有周文王的模型结构。因此需要从本仓库的fengshen框架导入，需要将fengshen放在你的工程文件夹。按照下面的脚本从huggingface下载并加载对应的模型：

``` python
from fengshen import RoFormerConfig
from fengshen import RoFormerModel
from transformers import BertTokenizer 

tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Zhouwenwang-Unified-110M')
config = RoFormerConfig.from_pretrained('IDEA-CCNL/Zhouwenwang-Unified-110M')
model = RoFormerModel.from_pretrained('IDEA-CCNL/Zhouwenwang-Unified-110M')
```


### 使用示例
1、首先修改finetune示例脚本[fengshen/scripts/finetune_classification.sh](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh)中的model_type和pretrained_model_path参数。其他如batch_size、data_dir等参数可根据自己的设备修改。
``` sh
MODEL_TYPE=fengshen-roformer
PRETRAINED_MODEL_PATH=IDEA-CCNL/Zhouwenwang-Unified-110M
```
2、然后运行：
``` sh
sh finetune_classification.sh
```

### 下游效果

#### 自然语言理解
使用周文王-1.3B模型进行自然语言理解任务时，需要将token_type全部设置为0。周文王的下游任务表现如下：

|     模型   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| Zhouwenwang-Unified-1.3B | 0.7463     |   0.6036    | 0.6288     |   0.7654   | 0.7741    | 0.8849    | 0. 8777   |

#### 自然语言生成
使用周文王-1.3B模型进行自然语言生成任务时，需要将token_type全部设置为1。周文王的生成例子如下：

```python
from fengshen import RoFormerModel
from transformers import BertTokenizer 
import torch
import numpy as np

sentence = '清华大学位于'
max_length = 32

tokenizer = BertTokenizer.from_pretrained('IDEA-CCNL/Zhouwenwang-Unified-110M')
model = RoFormerModel.from_pretrained('IDEA-CCNL/Zhouwenwang-Unified-110M')

for i in range(max_length):
    encode = [tokenizer.cls_token_id]+tokenizer.encode(sentence, add_special_tokens=False)
    input_ids=torch.tensor([encode]).long()
    token_type_ids=torch.tensor([[1]*len(encode)]).long()
    logits = model(input_ids=input_ids, 
                   token_type_ids=token_type_ids)[0]
    logits = torch.nn.functional.linear(
        logits, model.embeddings.word_embeddings.weight)
    logits = torch.nn.functional.softmax(
        logits, dim=-1).cpu().detach().numpy()[0]
    sentence = sentence + \
        tokenizer.decode(int(np.random.choice(logits.shape[1], p=logits[-1])))
    if sentence[-1] == '。':
        break
print(sentence)

 ```