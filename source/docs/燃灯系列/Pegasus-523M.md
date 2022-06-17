# Rangdeng-Pegasus-523M-Pretrain
Pegasus预训练模型是专门为摘要任务而设计的预训练模型，相比于其它通用预训练模型，Pegasus 模型的架构设计更贴近下游的摘要任务，在摘要抽取的效果上的表现相比其他通用模型表现更好

## 模型架构和任务描述
Pegasus的模型架构是标准的encoder-decoder的Transformer结构，训练任务是用的是GSG（ Gap Sentences Generation）任务。GSG任务主要是通过对文本中的重要的句子进行mask，然后再通过decoder恢复。模型详细参数可看config.json

1. base版本

| 配置 | 参数 |
| ---- | ---- |
| encoder layers | 12 |
| encoder_attention_heads | 12 |
| encoder_ffn_dim | 3072 |
| decoder layers | 12 |
| decoder_attention_heads| 12 |
| decoder_ffn_dim | 3072 |
| max_encode_length | 512 |

2. large 版本
   
| 配置 | 参数 |
| ---- | ---- |
| encoder layers | 16 |
| encoder_attention_heads | 16 |
| encoder_ffn_dim | 4096 |
| decoder layers | 16 |
| decoder_attention_heads| 16 |
| decoder_ffn_dim | 4096 |
| max_encode_length | 1024 |


## 模型

我们提供了两个不同大小版本的pegasus模型，下载链接如下：<br>
pegasus-base: [Randeng_pegasus_238M_summary](https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_238M_Summary) <br/>
pegasus-large: [Randeng_pegasus_523M_summary](https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M_Summary)

### 使用方式

由于中文的sentence piece在中文上分词存在问题，因此在训练中文的pegasus模型时，我们才用的jieba + Bertokenizer的方法实现的PegasusTokenizer。需要提前下载[tokenizers_pegasus.py](https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M_Summary/blob/main/tokenizers_pegasus.py) 和 [data_utils.py](https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M_Summary/blob/main/data_utils.py)两个文件放在代码运行目录下。然后便可依照下面示例使用模型。

```python
from transformers import PegasusForConditionalGeneration
# Need to download tokenizers_pegasus.py and other Python script from Fengshenbang-LM github repo in advance,
# or you can mv download in tokenizers_pegasus.py and data_utils.py in https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_238M_Summary/tree/main
# Strongly recommend you git clone the Fengshenbang-LM repo to have a better experience

from tokenizers_pegasus import PegasusTokenizer

model = PegasusForConditionalGeneration.from_pretrained("IDEA-CCNL/randeng_pegasus_238M_summary")
tokenizer = PegasusTokenizer.from_pretrained("path/to/vocab.txt")

text = "在北京冬奥会自由式滑雪女子坡面障碍技巧决赛中，中国选手谷爱凌夺得银牌。祝贺谷爱凌！今天上午，自由式滑雪女子坡面障碍技巧决赛举行。决赛分三轮进行，取选手最佳成绩排名决出奖牌。第一跳，中国选手谷爱凌获得69.90分。在12位选手中排名第三。完成动作后，谷爱凌又扮了个鬼脸，甚是可爱。第二轮中，谷爱凌在道具区第三个障碍处失误，落地时摔倒。获得16.98分。网友：摔倒了也没关系，继续加油！在第二跳失误摔倒的情况下，谷爱凌顶住压力，第三跳稳稳发挥，流畅落地！获得86.23分！此轮比赛，共12位选手参赛，谷爱凌第10位出场。网友：看比赛时我比谷爱凌紧张，加油！"
inputs = tokenizer(text, max_length=1024, return_tensors="pt")

# Generate Summary
summary_ids = model.generate(inputs["input_ids"])
tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

## 下游效果

### LCSTS摘要数据finetune后效果

| model | rouge-1 | rouge-2 | rouge-L |
| ---- | ---- | ---- | ---- |
| Pegasus-base  | 44.13 | 31.31 | 41.06 | 
| Pegasus-large | 49.42 | 37.91 | 46.63 |

## 如何预训练自己的Pegasus模型

我们的训练代码以及脚本都已经开源到[封神](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/hf-ds)github仓库的fengshen/examples/pegasus路径下。

### 数据预处理

训练数据使用的是wudao 180g数据。并对数据进行了简单的预处理，包括：
1. 过滤过长单句（这样的句子通常会包括一些乱码句，无上下文语义的列表句、各种符号句，歌词句等）
2. 过滤句子数过少文本，如句子数少于3句则抛弃

### 脚本修改

#### pegasus代码介绍
pegasus主要包括以下四个文件：
- tokenizers_pegasus.py 中文版pegasus的tokenize实现
- pretrain_pegasus.py 模型预训练的核心实现文件
- pretrain_pegasusu.sh 预训练脚本，具体参数可通过此脚本修改
- data_utils.py 模型的一些工具

#### 预训练步骤
1. git clone https://github.com/IDEA-CCNL/Fengshenbang-LM.git 到指定目录
2. 修改 pretrain_pegasusu.sh 脚本中的参数，参数介绍可参考 [Randeng-BART-139M](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E7%87%83%E7%81%AF%E7%B3%BB%E5%88%97/BART-139M.html) 
3. 运行预训练脚本 sbatch pretrain_pegasus.sh

## 如何进行下游任务

同时我们也提供了Pegasus做下游摘要预训练任务的示例。代码和脚本请跳转[封神框架](https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/hf-ds)下，参考fengshen/examples/summary/finetune_pegasus_summary.py 以及 randeng_pegasus_523M_summary.sh 两个脚本

实际 finetune 时只需在 randeng_pegasus_523M_summary.sh 脚本中修改一下数据的地址。详细文档请参考 [Randeng-Pegasus-523M-summary](https://fengshenbang-doc.readthedocs.io/zh/latest/docs/%E7%87%83%E7%81%AF%E7%B3%BB%E5%88%97/Randeng-Pegasus-523M-Summary.html)

## Citation
If you find the resource is useful, please cite the following website in your paper.
```
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2022},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```