## Randeng-MegatronT5-770M
Transformer结构为主的编解码语言模型，7.7亿参数的燃灯-770M大模型，采用280G数据，16张A100训练14天。

### 模型下载地址
[Huggingface 燃灯-770M](https://huggingface.co/IDEA-CCNL/Randeng-MegatronT5-770M/)

### 模型加载
由于T5结构的燃灯-770M模型是基于Megatron进行训练的，而Megatron的T5模型结构与HuggingFace的T5模型结构有略微的区别，不能直接使用HuggingFace的T5模型进行导入。因此需要从本仓库的fengshen框架导入，需要将fengshen放在你的工程文件夹。导入之后，即可按照下面的脚本从huggingface下载并加载对应的模型：

``` python
from fengshen import T5Config
from fengshen import T5EncoderModel
from fengshen import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
config = T5Config.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
model = T5EncoderModel.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
```

### 使用示例
1、首先修改finetune示例脚本[fengshen/scripts/finetune_classification.sh](https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/scripts/finetune_classification.sh)中的model_type和pretrained_model_path参数。其他如batch_size、data_dir等参数可根据自己的设备修改。
``` sh
MODEL_TYPE=fengshen-megatron_t5
PRETRAINED_MODEL_PATH=IDEA-CCNL/Randeng-MegatronT5-770M
```
2、然后运行：
``` sh
sh finetune_classification.sh
```

#### 生成任务使用示例

```python
from fengshen import T5ForConditionalGeneration
from fengshen import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')
model = T5ForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-MegatronT5-770M')

output = model.generate(tokenizer.encode(tokenizer.encode('北京是中国的<extra_id_0>')))
print(tokenizer.decode(output))

```