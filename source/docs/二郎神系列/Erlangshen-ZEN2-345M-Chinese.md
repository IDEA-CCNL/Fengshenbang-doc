# Erlangshen-ZEN2-345M-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLU任务，使用了N-gram编码增强文本语义，3.45亿参数量的ZEN2

ZEN2 model, which uses N-gram to enhance text semantic and has 345M parameters, is adept at NLU tasks.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | ZEN2 |      345M      |    中文-Chinese     |

## 模型信息 Model Information

我们与[ZEN团队](https://github.com/sinovation/ZEN)合作，使用我们的封神框架，开源发布了ZEN2模型。具体而言，通过引入无监督学习中提取的知识，ZEN通过N-gram方法学习不同的文本粒度信息。ZEN2使用大规模数据集和特殊的预训练策略对N-gram增强编码器进行预训练。下一步，我们将继续与ZEN团队一起探索PLM的优化，并提高下游任务的性能。

We open source and publicly release ZEN2 using our Fengshen Framework in collaboration with the [ZEN team](https://github.com/sinovation/ZEN). More precisely, by bringing together knowledge extracted by unsupervised learning, ZEN learns different textual granularity information through N-gram methods. ZEN2 pre-trains the N-gram-enhanced encoders with large-scale datasets and special pre-training strategies. In the next step, we continue with the ZEN team to explore the optimization of PLM and improve the performance on downstream tasks.

### 下游效果 Performance

**分类任务 Classification**

|    Model(Acc)   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: |
| Erlangshen-ZEN2-345M-Chinese | 0.741      |   0.584    | 0.599      |   0.788    | 0.80    |
| Erlangshen-ZEN2-668M-Chinese | 0.75      |   0.60    | 0.589      |   0.81    | 0.82    |

**抽取任务 Extraction**

|    Model(F1)   | WEIBO(test) |  Resume(test)  | MSRA(test) |  OntoNote4.0(test) |  CMeEE(dev)  | CLUENER(dev) |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: |
| Erlangshen-ZEN2-345M-Chinese | 65.26 | 96.03 | 95.15 | 78.93 | 62.81 | 79.27 |
| Erlangshen-ZEN2-668M-Chinese | 70.02 | 96.08 | 95.13 | 80.89 | 63.37 | 79.22 |

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Erlangshen-ZEN2-345M-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese)

### 加载模型 Loading Models

因为[transformers](https://github.com/huggingface/transformers)库中是没有ZEN2相关的模型结构的，所以你可以在我们的[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)中找到并且运行代码。

Since there is no structure of ZEN2 in [transformers library](https://github.com/huggingface/transformers), you can find the structure of ZEN2 and run the codes in [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM).

 ```shell
 git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
 ```

```python
from fengshen.models.zen2.ngram_utils import ZenNgramDict
from fengshen.models.zen2.tokenization import BertTokenizer
from fengshen.models.zen2.modeling import ZenForSequenceClassification, ZenForTokenClassification

pretrain_path = 'IDEA-CCNL/Erlangshen-ZEN2-345M-Chinese'

tokenizer = BertTokenizer.from_pretrained(pretrain_path)
model_classification = ZenForSequenceClassification.from_pretrained(pretrain_path)
model_extraction = ZenForTokenClassification.from_pretrained(pretrain_path)
ngram_dict = ZenNgramDict.from_pretrained(pretrain_path, tokenizer=tokenizer)

```

你可以从下方的链接获得我们做分类和抽取的详细示例。

You can get classification and extraction examples below.

[分类 classification example on fengshen](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/zen2_finetune/fs_zen2_base_tnews.sh)

[抽取 extraction example on fengshen](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/zen2_finetune/ner_zen2_base_ontonotes4.sh)


## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的对该模型的论文：

If you are using the resource for your work, please cite the our paper for this model:

```text
@article{Sinovation2021ZEN2,
  title="{ZEN 2.0: Continue Training and Adaption for N-gram Enhanced Text Encoders}",
  author={Yan Song, Tong Zhang, Yonggang Wang, Kai-Fu Lee},
  journal={arXiv preprint arXiv:2105.01279},
  year={2021},
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