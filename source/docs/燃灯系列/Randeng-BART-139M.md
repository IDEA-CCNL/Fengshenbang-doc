# Randeng-BART-139M

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

善于处理NLT任务，中文版的BART-base。

Good at solving NLT tasks, Chinese BART-base.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General | 自然语言转换 NLT | 燃灯 Randeng | BART |      139M      |     中文-Chinese    |

## 模型信息 Model Information

参考论文：[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/pdf/1910.13461.pdf)

为了得到一个中文版的BART-base，我们用悟道语料库(180G版本)进行预训练。具体地，我们在预训练阶段中使用了[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen)大概花费了8张A100约3天。

To get a Chinese BART-base, we use WuDao Corpora (180 GB version) for pre-training. Specifically, we use the [fengshen framework](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen) in the pre-training phase which cost about 3 days with 8 A100 GPUs.

### 更多信息 More Information

BART模型在原论文中采用了5种Denoise的方式，我们在预训练的时候采用论文中效果比较好的text infilling的方式。同时在原论文的基础上，我们使用sentence piece tokenizer，使得模型具备更长文本的生成能力。

## 使用 Usage

### 模型下载地址 Download Address

[Huggingface地址：Randeng-BART-139M](https://huggingface.co/IDEA-CCNL/Randeng-BART-139M)

### 加载模型 Loading Models

因为transformers下BartTokenizer不支持sentence piece，所以这里借用的是T5Tokenizer，在使用时需要在句首手动添加\<s\> (bos_token) ^ ^

```python
from transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-BART-139M', use_fast=false)
model=BartForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-BART-139M')
text = '<s>桂林市是世界闻名<mask> ，它有悠久的<mask>'
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
print(text2text_generator(text, max_length=50, do_sample=False))
```

## 如何预训练自己的BART模型

我们的训练代码、训练脚本都已开源，可以在[fengshen/examples/pretrain_bart](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/hf-ds/fengshen/examples/pretrain_bart)找到。

### 数据预处理

我们的原始数据并没有经过太多的数据处理，原始数据仅仅过简单的分句、分词。
一个输入的sample如下：
```
[['▁在', '网上', '找', '了很久', '都没有', '关于', 'j', 'ava', '的', '网页', '超', '链接', '跳', '转', '方式', ',', '故', '写', '篇', '经验', '供', '大家', '分享', '。'], 
['▁下面', '将', '介绍', '如何在', '文本', '消息', '中使用', '网页', '超', '链接', ':', '▁其实', ',', '不知道', '如何在', '文本', '消息', '中使用', '网页', '超', '链接', '的开发', '者', '几乎', '100%', '都', '熟悉', 'l', ',', '特别是', '对', 'l', '中的', 'a', '标签', '再', '熟悉', '不过', '了', '。'], 
['▁那', '到底', '在', '微信', '公众', '帐', '号的', '文本', '消息', '中使用', '超', '链接', '要注意', ',', '在', '微信', '上', ',', 'l', '的', 'a', '标签', '属性', '值', '不用', '引', '号', '引起', ',', '或者', '使用', '单', '引', '号', '引起', ',', '都是', '错误的', '写', '法', '(', '在', 'iphone', '上', ',', 'a', '标签', '属性', 'h', 're', 'f', '的', '值', '用', '单', '引', '号', '是', '正常的', ')。'], 
['▁', '正确的', '用法', '是将', 'a', '标签', 'h', 're', 'f', '属', '性的', '值', '用', '双', '引', '号', '引起', ',', '代码', '如下', ':', '▁a', '▁h', 're', 'f', '="', '▁.', 'com', '"', '百度', '经验', 'a', '▁如果', '要', '使用', '超', '链接', '调用', 'action', '类', ',', '可以在', '要', '输出的', 't', 'ext', '文本', '中', '拼接', '如下', ':', '▁', '一定要在', 'action', '前', '加入', '在', '微信', '发布的', '▁r', 'l', '。'], 
['▁str', 'ing', '▁m', 'sg', '="', '超', '链接', ':', 'a', '▁h', 're', 'f', '=', '▁工程', '名', 'we', 'ix', 'in', '.', 'do', '?'], 
['▁act', 'ion', '=', 'xx', 'x', '&', 'a', '=', '2', '▁', '跳', '转', 'a', '";']]
```

最终由上述数据转成模型输入的函数可以在TextFillingCollator中找到。

### 脚本修改

用户仅需要简单修改[script脚本](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/hf-ds/fengshen/examples/pretrain_bart/pretrain_bart_base.sh)，即可快速分布式的训练我们的BART。

其中参数主要分成下面五个参数:

```
数据类参数，主要用于配制数据集Batch Size等参数，在后续我们开源数据集工程后，能大幅减少数据预处理的工作量。

DATA_ARGS="\
        --datasets_name wudao_180g_spbpe_tokenized \
        --pretrain_sp_tokenizer /cognitive_comp/common_data/tokenizers/sentence_piece_bpe/bpe_v40000_s42_cov0.9995_max6_corpus1M.model \
        --num_workers 30 \
        --train_batchsize $MICRO_BATCH_SIZE \
        --val_batchsize 32 \
        --test_batchsize 32  \
        --max_seq_length 1024 \
        --masked_lm_prob 0.15 \
        --val_datasets_field test \
        "
```

```
模型的参数会从model_path中自动获取，可以参考我们Huggingface的模型配置，在上面做修改。

learning_rate、weight_decay这些参数如果配置了Deepspeed配置，会从Deepspeed的配置中获取。

MODEL_ARGS="\
        --model_path IDEA-CCNL/Randeng-BART-139M \
        --learning_rate 1e-5 \
        --weight_decay 0.1 \
        --warmup 0.001 \
        "
```

```
模型保存类参数，用户可以根据自己需要设定

MODEL_CHECKPOINT_ARGS="\
        --monitor train_loss \
        --save_top_k 3 \
        --mode min \
        --save_last \
        --every_n_train_steps 50000 \
        --dirpath /cognitive_comp/gaoxinyu/ln_model/ckpt/fengshen-$MODEL_NAME \
        --filename model-{step:02d}-{train_loss:.4f} \

```

```
训练相关的参数，这里可以根据自己的机器需要，调整gpus、nodes、strategy等等，如果使用Deepspeed，Deepspeed的配置会从下面的环境变量中获取
export PL_DEEPSPEED_CONFIG_PATH=$config_json

TRAINER_ARGS="\
        --gradient_clip_val 1.0 \
        --max_epochs 1 \
        --gpus 1 \
        --num_nodes 1 \
        --strategy deepspeed_stage_1 \
        --log_every_n_steps 100 \
        --val_check_interval 0.1 \
        --accumulate_grad_batches 1 \
        --resume_from_checkpoint /cognitive_comp/gaoxinyu/ln_model/ckpt/fengshen-${MODEL_NAME}/last.ckpt \
        --default_root_dir /cognitive_comp/gaoxinyu/ln_model/fengshen-$MODEL_NAME \
        "
```

## 如何进行下游任务

这里我们提供利用BART做summary任务的示例。[代码脚本](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/hf-ds/fengshen/examples/summary/bart_summary.py)同样开源了。

在脚本中仅需要修改一下LSCTC数据的地址即可。

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