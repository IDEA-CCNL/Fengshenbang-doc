# Erlangshen-MegatronBert-1.3B

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction
2021登顶FewCLUE和ZeroCLUE，处理NLU任务，开源时最大的中文BERT模型

Erlangshen topped FewCLUE and ZeroCLUE benchmark in 2021, designed to slove NLU tasks, is the largest BERT when publicly released.

## 模型分类 Model Taxonomy
|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | MegatronBert |      1.3B      |     -     |

## 模型信息 Model Information
Encoder结构为主的双向语言模型，专注于解决各种自然语言理解任务。
我们跟进了[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)的工作，使用了32张A100，总共耗时14天在悟道语料库（180 GB版本）上训练了十亿级别参数量的BERT。同时，鉴于中文语法和大规模训练的难度，我们使用四种预训练策略来改进BERT：1) 整词掩码, 2) 知识动态遮掩, 3) 句子顺序预测, 4) 层前归一化.

A bidirectional language model based on the Encoder structure, focusing on solving various NLU tasks.
We follow [Megatron-LM](https://github.com/NVIDIA/Megatron-LM), using 32 A100s and spending 14 days training a billion-level BERT on WuDao Corpora (180 GB version). Given Chinese grammar and the difficulty of large-scale training, we use four pre-training procedures to improve BERT: 1) Whole Word Masking (WWM), 2) Knowledge-based Dynamic Masking (KDM), 3) Sentence Order Prediction (SOP), 4) Pre-layer Normalization (Pre-LN).


### 成就 Achievement
1.2021年11月10日，二郎神在FewCLUE上取得第一。其中，它在CHIDF(成语填空)和TNEWS(新闻分类)子任务中的表现优于人类表现。此外，它在CHIDF(成语填空), CSLDCP(学科文献分类), OCNLI(自然语言推理)任务中均名列前茅。  
2.2022年1月24日，二郎神在CLUE基准测试中的ZeroCLUE中取得第一。具体到子任务，我们在CSLDCP(主题文献分类), TNEWS(新闻分类), IFLYTEK(应用描述分类), CSL(抽象关键字识别)和CLUEWSC(参考消歧)任务中取得第一。  
3.在2022年7月10日，我们在CLUE基准的语义匹配任务中取得第一。

1.On November 10, 2021, Erlangshen topped the FewCLUE benchmark. Among them, our Erlangshen outperformed human performance in CHIDF (idiom fill-in-the-blank) and TNEWS (news classification) subtasks. In addition, our Erlangshen ranked the top in CHIDF (idiom fill-in-the-blank), CSLDCP (subject literature classification), and OCNLI (natural language inference) tasks.  
2.On January 24, 2022, Erlangshen-MRC topped the ZeroCLUE benchmark. For each of these tasks, we rank the top ones in CSLDCP (Subject Literature Classification), TNEWS (News Classification), IFLYTEK (Application Description Classification), CSL (Abstract Keyword Recognition), and CLUEWSC (Referential Disambiguation) tasks.  
3.Erlangshen topped the CLUE benchmark semantic matching task on July 10, 2022.

2021年11月10日在中文语言理解权威评测基准FewCLUE 榜单上登顶。

![image](https://user-images.githubusercontent.com/4384420/141752311-d15c2a7f-cd83-4e9e-99a5-cb931088845e.png)

2022年1月24日，二郎神-MRC在中文语言理解评测零样本ZeroCLUE榜单上登顶。

![image](https://user-images.githubusercontent.com/4384420/151319156-e20ba252-b531-4779-8099-ef60c7954f76.png)


### 下游效果

|     模型   | afqmc    |  tnews  | iflytek    |  ocnli  |  cmnli  | wsc  | csl  |
| :--------:    | :-----:  | :----:  | :-----:   | :----: | :----: | :----: | :----: |
| roberta-wwm-ext-large | 0.7514      |   0.5872    | 0.6152      |   0.777    | 0.814    | 0.8914    | 0.86    |
| Erlangshen-MegatronBert-1.3B | 0.7608      |   0.5996    | 0.6234      |   0.7917    | 0.81    | 0.9243    | 0.872    |


## 使用 Usage

### 模型下载地址
[Huggingface 二郎神-1.3B](https://huggingface.co/IDEA-CCNL/Erlangshen-MegatronBert-1.3B)

### 模型加载

``` python
from transformers import MegatronBertConfig, MegatronBertModel
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
config = MegatronBertConfig.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
model = MegatronBertModel.from_pretrained("IDEA-CCNL/Erlangshen-MegatronBert-1.3B")
```

### 使用示例

为了便于开发者快速使用我们的开源模型，这里提供了一个下游任务的[finetune示例脚本](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh)，使用的[CLUE](https://github.com/CLUEbenchmark/CLUE)上的tnews新闻分类任务数据，运行脚本如下。其中DATA_PATH为数据路径，tnews任务数据的[下载地址](https://github.com/CLUEbenchmark/CLUE).

使用步骤如下：

1、首先修改finetune示例脚本[fengshen/scripts/finetune_classification.sh](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh)中的model_type和pretrained_model_path参数。其他如batch_size、data_dir等参数可根据自己的设备修改。
``` sh
MODEL_TYPE=huggingface-megatron_bert
PRETRAINED_MODEL_PATH=IDEA-CCNL/Erlangshen-MegatronBert-1.3B
```
2、然后运行：
``` sh
sh finetune_classification.sh
```


## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):
```
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
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
