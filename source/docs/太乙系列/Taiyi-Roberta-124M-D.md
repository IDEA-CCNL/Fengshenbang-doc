# Taiyi-Roberta-124M-D

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

COCO和VG上特殊预训练的，英文版的MAP（名称暂定）的文本端RoBERTa-base。

Special pre-training on COCO and VG, the textual encoder for MAP (temporary) in English, RoBERTa-base.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 多模态 Multimodal | 太乙 Taiyi | 待定 TBD |      124M      |     特殊预训练方法-英文 D-English     |

## 模型信息 Model Information

基于Roberta-base，我们使用特殊的训练任务引入一些多模态信息。"D"表示这是一种新的预训练方法。对于特殊的多模态表征，在论文中我们设计了集中不同的训练目标。预训练数据集为MSCOCO和VG。我们的代码和预训练任务的细节将在论文接受后公开。

Based on pre-trained Roberta-base, we apply some multimodal information with special pre-training tasks. "D" implies a special training method. For special multimodal representations, we design several special training objectives in our paper. The pre-training datasets are MSCOCO and VG. Our code and details of pre-training tasks will be made publicly available upon paper acceptance.

### 下游效果 Performance

**GLUE**

| Task                   | MNLI | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | WNLI |
|------------------------|------|------|------|-------|------|-------|------|------|------|
| Robert-base (official) | 87.6 | 91.9 | 92.8 | 94.8  | 63.6 | 91.2  | 90.2 | 78.7 |   -  |
| Roberta-base (local)   | 87.0 | 91.3 | 92.5 | 94.2  | 62.8 | 90.6  | 92.9 | 78.0 | 56.3 |
| Taiyi-Roberta-124M-D (local) | 87.1 | 91.8 | 92.3 | 94.5  | 62.6 | 90.4  | 92.4 | 78.7 | 56.3 |

The local test settings are:
Sequence length: 128, Batch size: 32, Learning rate: 3e-5

An additional dataset WNLI is tested.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Taiyi-Roberta-124M-D](https://huggingface.co/IDEA-CCNL/Taiyi-Roberta-124M-D)

### 加载模型 Loading Models

```python
from transformers import RobertaTokenizer, RobertaModel

tokenizer = RobertaTokenizer.from_pretrained("IDEA-CCNL/Taiyi-Roberta-124M-D")
model = RobertaModel.from_pretrained("IDEA-CCNL/Taiyi-Roberta-124M-D")
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
