# Taiyi (太乙)

Taiyi (太乙) 系列模型属于 Fengshenbang (封神榜) 大模型开源计划的一个分支。专注于利用多模态信息，使模型具有多模态语义理解能力。

# Taiyi-Roberta-124M-D

此模型与 Taiyi-vit-87M-D，一同构成多模态双流模型的两个特征提取器。

我们使用了特殊的多模态预训练策略增强了不仅仅是多模态下游任务的表现性能，也增强了单模态下游任务的能力。预训练策略以及代码将会在论文被接收后一同开源。

其中，所用的Roberta基于预训练的Roberta-base。"D"为特殊的预训练策略的名称缩写。此模型用于文本的特征抽取。

## 模型下载

[Huggingface Taiyi-Roberta-124M-D](https://huggingface.co/IDEA-CCNL/Taiyi-Roberta-124M-D)

## NLP下游任务

GLUE和WNLI任务。

| Task                   | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | WNLI |
|------------------------|------|------|------|-------|------|-------|------|------|------|
| Robert-base (official) | 87.6 | 91.9 | 92.8 | 94.8  | 63.6 | 91.2  | 90.2 | 78.7 |   -  |
| Roberta-base (local)   | 87.0 | 91.3 | 92.5 | 94.2  | 62.8 | 90.6  | 92.9 | 78.0 | 56.3 |
| Taiyi (local)          | 87.1 | 91.8 | 92.3 | 94.5  | 62.6 | 90.4  | 92.4 | 78.7 | 56.3 |

local的设置:

Sequence length: 128, Batch size: 32, Learning rate: 3e-5

## 快速开始

```python
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("IDEA-CCNL/Taiyi-Roberta-124M-D")
model = RobertaModel.from_pretrained("IDEA-CCNL/Taiyi-Roberta-124M-D")
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