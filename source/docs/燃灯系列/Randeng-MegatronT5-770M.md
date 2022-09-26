# Randeng-MegatronT5-770M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLT任务，中文版的T5-large。

Good at solving NLT tasks, Chinese T5-large.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | MegatronT5 |      770M      |     -    |

## 模型信息 Model Information

为了得到一个大规模的中文版的T5，我们使用了Megatron-LM的方法和悟道语料库(180G版本)用于预训练。具体地，我们在预训练阶段中使用了[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)大概花费了16张A100约14天。

To get a large-scale Chinese T5, we use of Megatron-LM and WuDao Corpora (180 GB version) for pre-training. Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 14 days with 16 A100 GPUs.

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Randeng-MegatronT5-770M](https://huggingface.co/IDEA-CCNL/Randeng-MegatronT5-770M)

### 加载模型 Loading Models

因为[transformers](https://github.com/huggingface/transformers)库中是没有 Zhouwenwang-Unified-1.3B相关的模型结构的，所以你可以在我们的[Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)中找到并且运行代码。

Since there is no structure of Randeng-MegatronT5-770M in [transformers library](https://github.com/huggingface/transformers), you can find the structure of Randeng-MegatronT5-770M and run the codes in [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM).

 ```shell
 git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git
 ```

```python
from fengshen import T5ForConditionalGeneration
from fengshen import T5Config
from fengshen import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
config = T5Config.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
model = T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
```

之所以要进行上述操作是因为T5结构的Randeng-MegatronT5-770M模型是基于Megatron进行训练的，而Megatron的T5模型结构与HuggingFace的T5模型结构有略微的区别，不能直接使用HuggingFace的T5模型进行导入。因此需要从本仓库的fengshen框架导入，需要将fengshen放在你的工程文件夹。导入之后，即可按照下面的脚本从HuggingFace下载并加载对应的模型：

### 使用示例 Usage Examples

1、首先修改finetune示例脚本[fengshen/scripts/finetune_classification.sh](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh)中的model_type和pretrained_model_path参数。其他如batch_size、data_dir等参数可根据自己的设备修改。
``` sh
MODEL_TYPE=fengshen-megatron_t5
PRETRAINED_MODEL_PATH=IDEA-CCNL/Randeng-MegatronT5-770M
```
2、然后运行：
``` sh
sh finetune_classification.sh
```

### 生成任务使用示例 Generation Examples

```python
from fengshen import T5ForConditionalGeneration
from fengshen import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
model = T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')

output = model.generate(tokenizer.encode(tokenizer.encode('北京是中国的<extra_id_0>')))
print(tokenizer.decode(output))

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
