# Yuyuan-Bart-400M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)


## 简介 Brief Introduction

生物医疗领域的生成语言模型，英文的BioBART-large。

A generative language model for biomedicine, BioBART-large in English.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 领域 Domain | 余元 Yuyuan | BioBART |      400M      |     英文 English     |

## 模型信息 Model Information

Paper: [BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model](https://arxiv.org/pdf/2204.03905.pdf)

Yuyuan-Bart-139M由清华大学和IDEA研究院一起提供的生物医疗领域的生成语言模型。我们使用PubMed上的生物医学研究论文摘要（约41G）作为预训练语料。使用开源框架DeepSpeed的情况下，我们在2个带有16个40GB A100 GPU的DGX结点上对BioBART-large（400M参数）进行了约168小时的训练。

The Yuyuan-Bart-139M is a biomedical generative language model jointly produced by Tsinghua University and International Digital Economy Academy (IDEA). We use biomedical research paper abstracts on PubMed (41G) as the pretraining corpora. We train the base version of BioBART(139M parameters) on 2 DGX with 16 40GB A100 GPUs for about 168 hours with the help of the open-resource framework DeepSpeed.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Yuyuan-Bart-400M](https://huggingface.co/IDEA-CCNL/Yuyuan-Bart-400M)

### 加载模型 Loading Models

```python
from transformers import BartForConditionalGeneration, BartTokenizer
tokenizer = BartTokenizer.from_pretrained('IDEA-CCNL/Yuyuan-Bart-400M')
model = BartForConditionalGeneration.from_pretrained('IDEA-CCNL/Yuyuan-Bart-400M')

text = 'Influenza is a <mask> disease.'
input_ids = tokenizer([text], return_tensors="pt")['input_ids']
model.eval()
generated_ids = model.generate(
    input_ids=input_ids,
)
preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
print(preds)
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的对该模型的论文：

If you are using the resource for your work, please cite the our paper for this model:

```
@misc{BioBART,
  title={BioBART: Pretraining and Evaluation of A Biomedical Generative Language Model},
  author={Hongyi Yuan and Zheng Yuan and Ruyi Gan and Jiaxing Zhang and Yutao Xie and Sheng Yu},
  year={2022},
  eprint={2204.03905},
  archivePrefix={arXiv}
}
```

如果您在您的工作中使用了我们的模型，也可以引用我们的[总论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [overview paper](https://arxiv.org/abs/2209.02970):

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