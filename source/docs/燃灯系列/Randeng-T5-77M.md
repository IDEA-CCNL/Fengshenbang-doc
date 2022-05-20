# Randeng-T5-77M

Randeng-T5-77M在[mt5-small](https://huggingface.co/google/mt5-small)的基础上，使用180G的中文数据进行span corruption目标的预训练得到的模型。


## 任务描述

Randeng-T5-77M在[mt5-small](https://huggingface.co/google/mt5-small)的基础上，只保留中英文对应的词表和embedding，在180G的中文通用预训练语料的基础上进行继续训练，无监督的目标采用 [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)中的span corruption方式。

## 数据处理

通用的数据举例：
```
text: '运动,走势在什么时候结束是不可能有答案的。为了找到走势什么时候结束原来的运动方向而改变方向,必须引进新的概念:中枢。'
```
span corruption后的数据举例
```
input: '运动,走势在什么时候结束是不可能有答案的。为了 <extra_id_0>走势什么时候结束原来 <extra_id_1>必须引进新的概念:中枢。'

label: '<extra_id_0>找到 <extra_id_1>的运动方向而改变方向,\</s>'
```
对应的代码见：
Fengshenbang-LM/fengshen/data/t5_dataloader/t5_datasets.py

## 模型训练
模型利用封神框架在2张A100训练17小时，最后loss收敛到2.3左右，训练脚本见:
Fengshenbang-LM/fengshen/examples/pretrain_t5/pretrain_mt5_small.sh


## 模型使用

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-T5-77M', use_fast=false)
model=T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-T5-77M')
```

## Citation
If you find the resource is useful, please cite the following website in your paper.
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2022},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```