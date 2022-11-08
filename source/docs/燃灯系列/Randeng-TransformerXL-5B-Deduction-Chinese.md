---
language: zh
license: apache-2.0
---

# Randeng-TransformerXL-5B-Deduction-Chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)
- Demo: [Reasoning Tree](https://idea.edu.cn/ccnl-act/reasoning/)

## 简介 Brief Introduction

基于Transformer-XL的中文因果推理生成模型。

Chinese deductive reasoning model based on Transformer-XL.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言生成 NLG | 燃灯 Randeng | TransformerXL |      5.0B      |     中文-因果推理 Chinese-Reasoning    |

## 模型信息 Model Information

**数据准备 Corpus Preparation**

* 悟道语料库（280G版本）
* 因果语料库（2.3M个样本）：基于悟道语料库（280G版本），通过关联词匹配、人工标注 + [GTSFactory](https://gtsfactory.com/)筛选、数据清洗等步骤获取的具有因果关系的句子对

* Wudao Corpus (with 280G samples) 
* Wudao Causal Corpus (with 2.3 million samples): Based on the Wudao corpus (280G version), sentence pairs with causality were obtained through logic indicator matching, manual annotation + [GTSFactory](https://gtsfactory.com/), and data cleaning.

**训练流程 Model Training**
1. 在悟道语料库（280G版本）上进行预训练
2. 在1.5M因果语料上进行因果生成任务的训练
3. 基于其余0.8M因果语料，协同[Randeng-TransformerXL-5B-Abduction-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-5B-Abduction-Chinese)和[Erlangshen-Roberta-330M-Causal-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Causal-Chinese)进行Self-consistent闭环迭代训练
    * 两个生成模型基于核采样和贪心的方式进行因果推理和反绎推理，产生大量伪样本；
    * Erlangshen-Roberta-330M-Causal-Chinese模型对伪样本句子对的因果关系进行打分，筛选供自身以及生成模型训练的样本

First, the Transformer-XL model was pre-trained on the Wudao Corpus (with 280G samples) and annotated similar-sentence pair dataset (same as [Randeng-TransformerXL-1.1B-Paraphrasing-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-1.1B-Paraphrasing-Chinese)).
Then, the model was trained on our causal corpus (about 1.5 million samples) for the deductive reasoning task.
At last, based on the remaining 0.8 million samples of the causal corpus, we conducted self-consistent learning on this model, cooperating with [Randeng-TransformerXL-5B-Abduction-Chinese](https://huggingface.co/IDEA-CCNL/Randeng-TransformerXL-5B-Abduction-Chinese) and [Erlangshen-Roberta-330M-Causal-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Causal-Chinese).
Specifically, two generative models performed deductive reasoning and abductive reasoning based on each sample respectively, generating a large number of pseudo-samples; [Erlangshen-Roberta-330M-Causal-Chinese](https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-330M-Causal-Chinese) scored the causality of the pseudo-samples and selected the training data for itself and the generative models in the next iteration.

## 加载模型 Loading Models

```shell
git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
cd Fengshenbang-LM
```

```python 
from fengshen.models.transfo_xl_reasoning import TransfoXLModel
from transformers import T5Tokenizer as TransfoXLTokenizer

model = TransfoXLModel.from_pretrained('IDEA-CCNL/Randeng-TransformerXL-5B-Deduction-Chinese')
tokenizer = TransfoXLTokenizer.from_pretrained(
    "IDEA-CCNL/Randeng-TransformerXL-5B-Deduction-Chinese",
    eos_token='<|endoftext|>',
    pad_token='<|endoftext|>',
    extra_ids=0
)
tokenizer.add_special_tokens({'bos_token': '<bos>'})
```

## 使用示例 Usage Example

```python 
from fengshen.models.transfo_xl_reasoning import deduction_generate

input_text = "机器人统治世界"
input_texts = ["机器人统治世界", "玉米价格持续上涨"]
print(deduction_generate(model, tokenizer, input_text, device=0))
print(deduction_generate(model, tokenizer, input_texts, device=0))
```

## 一些例子 Some Cases

（以下内容为模型生成，不代表本单位观点）

1. 鲸鱼是哺乳动物，哺乳动物是恒温动物
    - 鲸鱼拥有极强的恒温能力
    - 鲸鱼需要消耗大量的能量
    - 鲸鱼能在海洋中生存下来，完全依靠其体温恒定
2. 鲸鱼是哺乳动物，哺乳动物不是恒温动物
    - 鲸鱼不是恒温动物，鲸鱼是变温动物
    - 鲸鱼会随温度变化而变化
    - 鲸鱼有冬眠现象
3. 机器人统治世界
    - 机器人取代人类做大部分工作
    - 机器人世界的法律必须能够适用于机器人统治的现实世界
    - 机器人必须能够相互沟通，并能够处理人类无法处理的复杂情况
    - 未来将不再需要大量工人，机器人将接管工厂
4. 玉米价格持续上涨
    - 玉米淀粉价格也呈现上涨趋势
    - 玉米种植效益不断攀升
    - 在玉米深加工行业引起了一阵骚动
5. 实体经济融资难、融资贵
    - 急需发展互联网金融等金融业态，为实体经济提供融资服务
    - 融资需求向金融资产转移，增加了金融资产供给
    - 必须大力发展资本市场，使资本市场成为经济转型的助推器
6. 影响华北地区的冷空气势力偏弱
    - 冷空气的影响时间将偏短
    - 冷空气影响结束后，华北地区气温会继续缓慢回升
    - 华北地区气温较常年同期偏高


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