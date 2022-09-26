# Erlangshen-Longformer-110M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理长文本，采用旋转位置编码的中文版1.1亿参数的Longformer-Base。

The Chinese Longformer-Base (110M), which uses rotating positional encoding, is adept at handling lengthy text.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | Longformeer |      110M      |     中文 Chinese     |

## 模型信息 Model Information

遵循Longformer-Base的设计，我们基于[chinese_roformer_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/roformer)，在悟道语料库(180 GB版本)上进行了继续预训练。特别的，我们采用旋转位置嵌入(RoPE)来避免预训练语料库的不均匀序列长度问题。

Following the design of Longformer-Base, we performed continual pre-training on the WuDao corpus (180 GB) based on [chinese_roformer_L-12_H-768_A-12](https://github.com/ZhuiyiTechnology/roformer). Particularly, we employed rotational position embedding (RoPE) to avoid the uneven sequence length of the pre-trained corpus.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-Longformer-110M](https://huggingface.co/IDEA-CCNL/Erlangshen-Longformer-110M)

### 加载模型 Loading Models

因为[transformers](https://github.com/huggingface/transformers)库中是没有Longformer-Base相关的模型结构的，所以你可以在我们的[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)中找到并且运行代码。

Since there is no structure of Longformer-Base in [transformers library](https://github.com/huggingface/transformers), you can find the structure of Longformer-Base and run the codes in [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM).

 ```shell
 git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
 ```

### 加载模型 Loading Models
```python
from fengshen import LongformerModel    
from fengshen import LongformerConfig
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-Longformer-110M")
config = LongformerConfig.from_pretrained("IDEA-CCNL/Erlangshen-Longformer-110M")
model = LongformerModel.from_pretrained("IDEA-CCNL/Erlangshen-Longformer-110M")
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