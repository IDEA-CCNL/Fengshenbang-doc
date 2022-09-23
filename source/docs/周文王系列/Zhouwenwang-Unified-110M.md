---
language: 
  - zh
license: apache-2.0
widget:
- text: "生活的真谛是[MASK]。"
---

# Zhouwenwang-Unified-110M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

与追一科技合作探索的中文统一模型，1.1亿参数的编码器结构模型。

The Chinese unified model explored in cooperation with Zhuiyi Technology, the encoder structure model with 110M parameters.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 探索 Exploration | 周文王 Zhouwenwang | 待定 TBD |      110M      |     中文 Chinese     |

## 模型信息 Model Information

IDEA研究院认知计算中心联合追一科技有限公司提出的具有新结构的大模型。该模型在预训练阶段时考虑统一LM和MLM的任务，这让其同时具备生成和理解的能力，并且增加了旋转位置编码技术。我们后续会持续在模型规模、知识融入、监督辅助任务等方向不断优化。

A large-scale model (Zhouwenwang-Unified-1.3B) with a new structure proposed by IDEA CCNL and Zhuiyi Technology. The model considers the task of unifying LM (Language Modeling) and MLM (Masked Language Modeling) during the pre-training phase, which gives it both generative and comprehension capabilities, and applys rotational position encoding. In the future, we will continue to optimize it in the direction of model size, knowledge incorporation, and supervisory assistance tasks.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Zhouwenwang-Unified-110M](https://huggingface.co/IDEA-CCNL/Zhouwenwang-Unified-110M)

### 加载模型 Loading Models

因为[transformers](https://github.com/huggingface/transformers)库中是没有 Zhouwenwang-Unified-110M相关的模型结构的，所以你可以在我们的[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)中找到并且运行代码。

Since there is no structure of Zhouwenwang-Unified-110M in [transformers library](https://github.com/huggingface/transformers), you can find the structure of Zhouwenwang-Unified-110M and run the codes in [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM).

 ```shell
 git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
 ```

```python
from fengshen import RoFormerModel    
from fengshen import RoFormerConfig
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-110M")
config = RoFormerConfig.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-110M")
model = RoFormerModel.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-110M")
```

### 使用示例 Usage Examples

你可以使用该模型进行续写任务。

You can use the model for continuation writing tasks.

```python
from fengshen import RoFormerModel
from transformers import AutoTokenizer
import torch
import numpy as np

sentence = '清华大学位于'
max_length = 32

tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-110M")
model = RoFormerModel.from_pretrained("IDEA-CCNL/Zhouwenwang-Unified-110M")

for i in range(max_length):
    encode = torch.tensor(
        [[tokenizer.cls_token_id]+tokenizer.encode(sentence, add_special_tokens=False)]).long()
    logits = model(encode)[0]
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
