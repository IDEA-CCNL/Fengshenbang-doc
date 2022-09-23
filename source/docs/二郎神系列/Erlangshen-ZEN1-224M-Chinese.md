# Erlangshen-ZEN1-224M-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLU任务，使用了N-gram编码增强文本语义，2.24亿参数量的ZEN1

ZEN1 model, which uses N-gram to enhance text semantic and has 224M parameters, is adept at NLU tasks.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | ZEN1 |      224M      |     中文-Chinese     |

## 模型信息 Model Information

我们与[ZEN团队](https://github.com/sinovation/ZEN)合作，使用我们的封神框架，开源发布了ZEN1模型。具体而言，通过引入无监督学习中提取的知识，ZEN通过N-gram方法学习不同的文本粒度信息。ZEN1可以通过仅在单个小语料库(低资源场景)上进行训练来获得良好的性能增益。下一步，我们将继续与ZEN团队一起探索PLM的优化，并提高下游任务的性能

We open source and publicly release ZEN1 using our Fengshen Framework in collaboration with the [ZEN team](https://github.com/sinovation/ZEN). More precisely, by bringing together knowledge extracted by unsupervised learning, ZEN learns different textual granularity information through N-gram methods. ZEN1 can obtain good performance gains by training only on a single small corpus (low-resource scenarios). In the next step, we continue with the ZEN team to explore the optimization of PLM and improve the performance on downstream tasks.

### 下游效果 Performance

**分类任务 Classification**

|  model   | dataset  | Acc |
|  ----  | ----  | ---- |
| IDEA-CCNL/Erlangshen-ZEN1-224M-Chinese | Tnews | 56.82% |

**抽取任务 Extraction**

|  model   | dataset  | F1 |
|  ----  | ----  | ---- |
| IDEA-CCNL/Erlangshen-ZEN1-224M-Chinese | OntoNote4.0 | 80.8% | 


## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-ZEN1-224M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-ZEN1-224M-Chinese)

### 加载模型 Loading Models

因为[transformers](https://github.com/huggingface/transformers)库中是没有ZEN1相关的模型结构的，所以你可以在我们的[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)中找到并且运行代码。

Since there is no structure of ZEN1 in [transformers library](https://github.com/huggingface/transformers), you can find the structure of ZEN1 and run the codes in [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM).

 ```shell
 git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
 ```

```python
from fengshen.models.zen1.ngram_utils import ZenNgramDict
from fengshen.models.zen1.tokenization import BertTokenizer
from fengshen.models.zen1.modeling import ZenForSequenceClassification, ZenForTokenClassification

pretrain_path = 'IDEA-CCNL/Erlangshen-ZEN1-224M-Chinese'

tokenizer = BertTokenizer.from_pretrained(pretrain_path)
model_classification = ZenForSequenceClassification.from_pretrained(pretrain_path)
model_extraction = ZenForTokenClassification.from_pretrained(pretrain_path)
ngram_dict = ZenNgramDict.from_pretrained(pretrain_path, tokenizer=tokenizer)

```

你可以从下方的链接获得我们做分类和抽取的详细示例。

You can get classification and extraction examples below.

[分类 classification example on fengshen](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/zen1_finetune/fs_zen1_tnews.sh)

[抽取 extraction example on fengshen](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/zen1_finetune/ner_zen1_ontonotes4.sh)

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的对该模型的论文：

If you are using the resource for your work, please cite the our paper for this model:

```text
@inproceedings{fengshenbang/zen1,
  author    = {Shizhe Diao and
               Jiaxin Bai and
               Yan Song and
               Tong Zhang and
               Yonggang Wang},
  title     = {{ZEN:} Pre-training Chinese Text Encoder Enhanced by N-gram Representations},
  booktitle = {{EMNLP} (Findings)},
  series    = {Findings of {ACL}},
  volume    = {{EMNLP} 2020},
  pages     = {4729--4740},
  publisher = {Association for Computational Linguistics},
  year      = {2020}
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